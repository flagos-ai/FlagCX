/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "core.h"
#include "flagcx_common.h"
#include "flagcx_net.h"
#include "ib_common.h"
#include "ibuc_retrans.h"
#include "ibvwrap.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "timer.h"
#include "utils.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

char flagcxIbIfName[MAX_IF_NAME_SIZE + 1];
union flagcxSocketAddress flagcxIbIfAddr;

FLAGCX_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
FLAGCX_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
FLAGCX_PARAM(IbTimeout, "IB_TIMEOUT", 18);
FLAGCX_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
FLAGCX_PARAM(IbPkey, "IB_PKEY", 0);
FLAGCX_PARAM(IbUseInline, "IB_USE_INLINE", 0);
FLAGCX_PARAM(IbSl, "IB_SL", 0);
FLAGCX_PARAM(IbTc, "IB_TC", 0);
FLAGCX_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
FLAGCX_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
FLAGCX_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
FLAGCX_PARAM(IbDisable, "IB_DISABLE", 0);
FLAGCX_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
FLAGCX_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);
FLAGCX_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

int flagcxNMergedIbDevs = -1;
int flagcxNIbDevs = -1;

// Define global arrays
struct flagcxIbMergedDev flagcxIbMergedDevs[MAX_IB_VDEVS];
struct flagcxIbDev flagcxIbDevs[MAX_IB_DEVS];
pthread_mutex_t flagcxIbLock = PTHREAD_MUTEX_INITIALIZER;
int flagcxIbRelaxedOrderingEnabled = 0;

pthread_t flagcxIbAsyncThread;
void *flagcxIbAsyncThreadMain(void *args) {
  struct flagcxIbDev *dev = (struct flagcxIbDev *)args;
  while (1) {
    struct ibv_async_event event;
    if (flagcxSuccess != flagcxWrapIbvGetAsyncEvent(dev->context, &event)) {
      break;
    }
    char *str;
    if (flagcxSuccess != flagcxWrapIbvEventTypeStr(&str, event.event_type)) {
      break;
    }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : %s:%d Got async event : %s", dev->devName, dev->portNum,
           str);
    if (flagcxSuccess != flagcxWrapIbvAckAsyncEvent(&event)) {
      break;
    }
  }
  return NULL;
}

sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char *env = flagcxGetEnv("FLAGCX_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(FLAGCX_ENV, "FLAGCX_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

void *envIbAddrRange(sa_family_t af, int *mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char *env = flagcxGetEnv("FLAGCX_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(FLAGCX_ENV, "FLAGCX_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = {0};
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/") + 1;
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr - 1) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    WARN("NET/IB: Ip address '%s' is invalid for family %s, ignoring address",
         addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
         *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
         *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

sa_family_t getGidAddrFamily(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) |
                       (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
      (a->s6_addr32[0] == htonl(0xff0e0000) &&
       ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

bool matchGidAddrPrefix(sa_family_t af, void *prefix, int prefixlen,
                        union ibv_gid *gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;
  ;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

bool configuredGid(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) ||
      ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

bool linkLocalGid(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

bool validGid(union ibv_gid *gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

flagcxResult_t flagcxIbRoceGetVersionNum(const char *deviceName, int portNum,
                                         int gidIndex, int *version) {
  char gidRoceVerStr[16] = {0};
  char roceTypePath[PATH_MAX] = {0};
  sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
          deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    return flagcxSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    return flagcxSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
        strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxUpdateGidIndex(struct ibv_context *context,
                                    uint8_t portNum, sa_family_t af,
                                    void *prefix, int prefixlen, int roceVer,
                                    int gidIndexCandidate, int *gidIndex) {
  union ibv_gid gid, gidCandidate;
  FLAGCXCHECK(flagcxWrapIbvQueryGid(context, portNum, *gidIndex, &gid));
  FLAGCXCHECK(flagcxWrapIbvQueryGid(context, portNum, gidIndexCandidate,
                                    &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet =
      matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam &&
      gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) ||
        !gidCandidateMatchSubnet) {
      return flagcxSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate;
    const char *deviceName = flagcxWrapIbvGetDeviceName(context->device);
    FLAGCXCHECK(flagcxIbRoceGetVersionNum(deviceName, portNum, *gidIndex,
                                          &gidRoceVerNum));
    FLAGCXCHECK(flagcxIbRoceGetVersionNum(
        deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) &&
        gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbGetGidIndex(struct ibv_context *context, uint8_t portNum,
                                   int gidTblLen, int *gidIndex) {
  *gidIndex = flagcxParamIbGidIndex();
  if (*gidIndex >= 0) {
    return flagcxSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = flagcxParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    FLAGCXCHECK(flagcxUpdateGidIndex(context, portNum, userAddrFamily, prefix,
                                     prefixlen, userRoceVersion, gidIndexNext,
                                     gidIndex));
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbGetPciPath(char *devName, char **path, int *realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char *p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p) - 1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (flagcxParamIbMergeVfs())
      p[strlen(p) - 3] = p[strlen(p) - 4] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d = 0; d < flagcxNIbDevs; d++) {
      if (strcmp(p, flagcxIbDevs[d].pciPath) == 0)
        (*realPort)++;
    }
  }
  *path = p;
  return flagcxSuccess;
}

int ibvWidths[] = {1, 4, 8, 12, 2};
int ibvSpeeds[] = {2500,  /* SDR */
                   5000,  /* DDR */
                   10000, /* QDR */
                   10000, /* QDR */
                   14000, /* FDR */
                   25000, /* EDR */
                   50000, /* HDR */
                   100000 /* NDR */};

int firstBitSet(int val, int max) {
  int i = 0;
  while (i < max && ((val & (1 << i)) == 0))
    i++;
  return i;
}
int flagcxIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths) / sizeof(int) - 1)];
}
int flagcxIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds) / sizeof(int) - 1)];
}

int flagcxIbRelaxedOrderingCapable(void) {
  int roMode = flagcxParamIbPciRelaxedOrdering();
  flagcxResult_t r = flagcxInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = flagcxWrapIbvRegMrIova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == flagcxInternalError ? 0 : 1;
}

int flagcxIbFindMatchingDev(int dev) {
  for (int i = 0; i < flagcxNMergedIbDevs; i++) {
    if (flagcxIbMergedDevs[i].ndevs < FLAGCX_IB_MAX_DEVS_PER_NIC) {
      int compareDev = flagcxIbMergedDevs[i].devs[0];
      if (strcmp(flagcxIbDevs[dev].pciPath, flagcxIbDevs[compareDev].pciPath) ==
              0 &&
          (flagcxIbDevs[dev].guid == flagcxIbDevs[compareDev].guid) &&
          (flagcxIbDevs[dev].link == flagcxIbDevs[compareDev].link)) {
        TRACE(FLAGCX_NET,
              "NET/IB: Matched name1=%s pciPath1=%s guid1=0x%lx link1=%u "
              "name2=%s pciPath2=%s guid2=0x%lx link2=%u",
              flagcxIbDevs[dev].devName, flagcxIbDevs[dev].pciPath,
              flagcxIbDevs[dev].guid, flagcxIbDevs[dev].link,
              flagcxIbDevs[compareDev].devName,
              flagcxIbDevs[compareDev].pciPath, flagcxIbDevs[compareDev].guid,
              flagcxIbDevs[compareDev].link);
        return i;
      }
    }
  }

  return flagcxNMergedIbDevs;
}

flagcxResult_t flagcxIbInit() {
  flagcxResult_t ret;
  if (flagcxParamIbDisable())
    return flagcxInternalError;
  static int shownIbHcaEnv = 0;
  if (flagcxWrapIbvSymbols() != flagcxSuccess) {
    return flagcxInternalError;
  }

  if (flagcxNIbDevs == -1) {
    pthread_mutex_lock(&flagcxIbLock);
    flagcxWrapIbvForkInit();
    if (flagcxNIbDevs == -1) {
      flagcxNIbDevs = 0;
      flagcxNMergedIbDevs = 0;
      if (flagcxFindInterfaces(flagcxIbIfName, &flagcxIbIfAddr,
                               MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = flagcxInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device **devices;

      // Check if user defined which IB device:port to use
      char *userIbEnv = getenv("FLAGCX_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0)
        INFO(FLAGCX_NET | FLAGCX_ENV, "FLAGCX_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot)
        userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact)
        userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (flagcxSuccess != flagcxWrapIbvGetDeviceList(&devices, &nIbDevs)) {
        ret = flagcxInternalError;
        goto fail;
      }

      for (int d = 0; d < nIbDevs && flagcxNIbDevs < MAX_IB_DEVS; d++) {
        struct ibv_context *context;
        if (flagcxSuccess != flagcxWrapIbvOpenDevice(&context, devices[d]) ||
            context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (flagcxSuccess != flagcxWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (flagcxSuccess != flagcxWrapIbvCloseDevice(context)) {
            ret = flagcxInternalError;
            goto fail;
          }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (flagcxSuccess !=
              flagcxWrapIbvQueryPort(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE)
            continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
              portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
            continue;

          // check against user specified HCAs/ports
          if (!(matchIfList(devices[d]->name, port_num, userIfs, nUserIfs,
                            searchExact) ^
                searchNot)) {
            continue;
          }
          pthread_mutex_init(&flagcxIbDevs[flagcxNIbDevs].lock, NULL);
          flagcxIbDevs[flagcxNIbDevs].device = d;
          flagcxIbDevs[flagcxNIbDevs].guid = devAttr.sys_image_guid;
          flagcxIbDevs[flagcxNIbDevs].portAttr = portAttr;
          flagcxIbDevs[flagcxNIbDevs].portNum = port_num;
          flagcxIbDevs[flagcxNIbDevs].link = portAttr.link_layer;
          flagcxIbDevs[flagcxNIbDevs].speed =
              flagcxIbSpeed(portAttr.active_speed) *
              flagcxIbWidth(portAttr.active_width);
          flagcxIbDevs[flagcxNIbDevs].context = context;
          flagcxIbDevs[flagcxNIbDevs].pdRefs = 0;
          flagcxIbDevs[flagcxNIbDevs].pd = NULL;
          strncpy(flagcxIbDevs[flagcxNIbDevs].devName, devices[d]->name,
                  MAXNAMESIZE);
          FLAGCXCHECK(
              flagcxIbGetPciPath(flagcxIbDevs[flagcxNIbDevs].devName,
                                 &flagcxIbDevs[flagcxNIbDevs].pciPath,
                                 &flagcxIbDevs[flagcxNIbDevs].realPort));
          flagcxIbDevs[flagcxNIbDevs].maxQp = devAttr.max_qp;
          flagcxIbDevs[flagcxNIbDevs].mrCache.capacity = 0;
          flagcxIbDevs[flagcxNIbDevs].mrCache.population = 0;
          flagcxIbDevs[flagcxNIbDevs].mrCache.slots = NULL;

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          flagcxIbDevs[flagcxNIbDevs].ar =
              (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (flagcxParamIbAdaptiveRouting() != -2)
            flagcxIbDevs[flagcxNIbDevs].ar = flagcxParamIbAdaptiveRouting();

          TRACE(FLAGCX_NET,
                "NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d",
                d, devices[d]->name, devices[d]->dev_name,
                flagcxIbDevs[flagcxNIbDevs].portNum,
                portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                 : "RoCE",
                flagcxIbDevs[flagcxNIbDevs].speed, context,
                flagcxIbDevs[flagcxNIbDevs].pciPath,
                flagcxIbDevs[flagcxNIbDevs].ar);

          pthread_create(&flagcxIbAsyncThread, NULL, flagcxIbAsyncThreadMain,
                         flagcxIbDevs + flagcxNIbDevs);
          flagcxSetThreadName(flagcxIbAsyncThread, "FLAGCX IbAsync %2d",
                              flagcxNIbDevs);
          pthread_detach(flagcxIbAsyncThread); // will not be pthread_join()'d

          int mergedDev = flagcxNMergedIbDevs;
          if (flagcxParamIbMergeNics()) {
            mergedDev = flagcxIbFindMatchingDev(flagcxNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if
          // there's only one dev inside)
          if (mergedDev == flagcxNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IB device
            flagcxIbMergedDevs[mergedDev].ndevs = 1;
            flagcxIbMergedDevs[mergedDev].devs[0] = flagcxNIbDevs;
            flagcxNMergedIbDevs++;
            strncpy(flagcxIbMergedDevs[mergedDev].devName,
                    flagcxIbDevs[flagcxNIbDevs].devName, MAXNAMESIZE);
            // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IB device
            int ndevs = flagcxIbMergedDevs[mergedDev].ndevs;
            flagcxIbMergedDevs[mergedDev].devs[ndevs] = flagcxNIbDevs;
            flagcxIbMergedDevs[mergedDev].ndevs++;
            snprintf(flagcxIbMergedDevs[mergedDev].devName +
                         strlen(flagcxIbMergedDevs[mergedDev].devName),
                     MAXNAMESIZE + 1, "+%s",
                     flagcxIbDevs[flagcxNIbDevs].devName);
          }

          // Aggregate speed
          flagcxIbMergedDevs[mergedDev].speed +=
              flagcxIbDevs[flagcxNIbDevs].speed;
          flagcxNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && flagcxSuccess != flagcxWrapIbvCloseDevice(context)) {
          ret = flagcxInternalError;
          goto fail;
        }
      }
      if (nIbDevs && (flagcxSuccess != flagcxWrapIbvFreeDeviceList(devices))) {
        ret = flagcxInternalError;
        goto fail;
      };
    }
    if (flagcxNIbDevs == 0) {
      INFO(FLAGCX_INIT | FLAGCX_NET, "NET/IB : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      flagcxIbRelaxedOrderingEnabled = flagcxIbRelaxedOrderingCapable();
      for (int d = 0; d < flagcxNMergedIbDevs; d++) {
        struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibDev = mergedDev->devs[i];
            snprintf(
                line + strlen(line), 2047 - strlen(line), "[%d] %s:%d/%s%s",
                ibDev, flagcxIbDevs[ibDev].devName, flagcxIbDevs[ibDev].portNum,
                flagcxIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                      : "RoCE",
                // Insert comma to delineate
                i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line + strlen(line), 2047 - strlen(line), "}");
        } else {
          int ibDev = mergedDev->devs[0];
          snprintf(
              line + strlen(line), 2047 - strlen(line), " [%d]%s:%d/%s", ibDev,
              flagcxIbDevs[ibDev].devName, flagcxIbDevs[ibDev].portNum,
              flagcxIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                    : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN + 1];
      INFO(FLAGCX_NET, "NET/IB : Using%s %s; OOB %s:%s", line,
           flagcxIbRelaxedOrderingEnabled ? "[RO]" : "", flagcxIbIfName,
           flagcxSocketToString(&flagcxIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&flagcxIbLock);
  }
  return flagcxSuccess;
fail:
  pthread_mutex_unlock(&flagcxIbLock);
  return ret;
}

const char *reqTypeStr[] = {"Unused", "Send", "Recv", "Flush"};

static void flagcxIbAddEvent(struct flagcxIbRequest *req, int devIndex,
                             struct flagcxIbNetCommDevBase *base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

flagcxResult_t flagcxIbInitCommDevBase(int ibDevN,
                                       struct flagcxIbNetCommDevBase *base) {
  base->ibDevN = ibDevN;
  flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == ibDev->pdRefs++) {
    flagcxResult_t res;
    FLAGCXCHECKGOTO(flagcxWrapIbvAllocPd(&ibDev->pd, ibDev->context), res,
                    failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibDev->lock);
      return res;
    }
  }
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for
  // the Recv).
  FLAGCXCHECK(flagcxWrapIbvCreateCq(
      &base->cq, ibDev->context, 2 * MAX_REQUESTS * flagcxParamIbQpsPerConn(),
      NULL, NULL, 0));

  return flagcxSuccess;
}

flagcxResult_t flagcxIbDestroyBase(struct flagcxIbNetCommDevBase *base) {
  flagcxResult_t res;

  // Poll any remaining completions before destroying CQ
  if (base->cq) {
    struct ibv_wc wcs[64];
    int n_cqe = 0;
    // Poll multiple times to drain all pending completions
    for (int i = 0; i < 16; i++) {
      flagcxWrapIbvPollCq(base->cq, 64, wcs, &n_cqe);
      if (n_cqe == 0)
        break;
    }
  }

  FLAGCXCHECK(flagcxWrapIbvDestroyCq(base->cq));

  pthread_mutex_lock(&flagcxIbDevs[base->ibDevN].lock);
  if (0 == --flagcxIbDevs[base->ibDevN].pdRefs) {
    flagcxResult_t pd_result =
        flagcxWrapIbvDeallocPd(flagcxIbDevs[base->ibDevN].pd);
    if (pd_result != flagcxSuccess) {
      // Log but don't fail - PD deallocation errors are often non-fatal
      // (e.g., "Device or resource busy" when there are still resources using
      // the PD)
      if (flagcxDebugNoWarn == 0)
        INFO(FLAGCX_ALL,
             "Failed to deallocate PD: %d (non-fatal, may have remaining "
             "resources)",
             pd_result);
      res = flagcxSuccess; // Continue cleanup even if PD deallocation fails
    } else {
      res = flagcxSuccess;
    }
  } else {
    res = flagcxSuccess;
  }
  pthread_mutex_unlock(&flagcxIbDevs[base->ibDevN].lock);
  return res;
}

flagcxResult_t flagcxIbCreateQp(uint8_t ib_port,
                                struct flagcxIbNetCommDevBase *base,
                                int access_flags, struct flagcxIbQp *qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2 * MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data =
      flagcxParamIbUseInline() ? sizeof(struct flagcxIbSendFifo) : 0;
  FLAGCXCHECK(flagcxWrapIbvCreateQp(&qp->qp, base->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = flagcxParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  FLAGCXCHECK(flagcxWrapIbvModifyQp(qp->qp, &qpAttr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbRtrQp(struct ibv_qp *qp, uint8_t sGidIndex,
                             uint32_t dest_qp_num,
                             struct flagcxIbDevInfo *info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = flagcxParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = flagcxParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  FLAGCXCHECK(flagcxWrapIbvModifyQp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbRtsQp(struct ibv_qp *qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = flagcxParamIbTimeout();
  qpAttr.retry_cnt = flagcxParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  FLAGCXCHECK(flagcxWrapIbvModifyQp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbListen(int dev, void *opaqueHandle, void **listenComm) {
  struct flagcxIbListenComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  struct flagcxIbHandle *handle = (struct flagcxIbHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct flagcxIbHandle));
  comm->dev = dev;
  handle->magic = FLAGCX_SOCKET_MAGIC;
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &flagcxIbIfAddr, handle->magic,
                               flagcxSocketTypeNetIb, NULL, 1));
  FLAGCXCHECK(flagcxSocketListen(&comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbConnect(int dev, void *opaqueHandle, void **sendComm) {
  struct flagcxIbHandle *handle = (struct flagcxIbHandle *)opaqueHandle;
  struct flagcxIbCommStage *stage = &handle->stage;
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)stage->comm;
  int ready;
  *sendComm = NULL;

  if (stage->state == flagcxIbCommStateConnect)
    goto ib_connect_check;
  if (stage->state == flagcxIbCommStateSend)
    goto ib_send;
  if (stage->state == flagcxIbCommStateConnecting)
    goto ib_connect;
  if (stage->state == flagcxIbCommStateConnected)
    goto ib_send_ready;
  if (stage->state != flagcxIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return flagcxInternalError;
  }

  FLAGCXCHECK(flagcxIbMalloc((void **)&comm, sizeof(struct flagcxIbSendComm)));
  FLAGCXCHECK(flagcxSocketInit(&comm->base.sock, &handle->connectAddr,
                               handle->magic, flagcxSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = flagcxIbCommStateConnect;
  FLAGCXCHECK(flagcxSocketConnect(&comm->base.sock));

ib_connect_check:
  /* since flagcxSocketConnect is async, we must check if connection is complete
   */
  FLAGCXCHECK(flagcxSocketReady(&comm->base.sock, &ready));
  if (!ready)
    return flagcxSuccess;

  // IB Setup
  struct flagcxIbMergedDev *mergedDev;
  mergedDev = flagcxIbMergedDevs + dev;
  comm->base.ndevs = mergedDev->ndevs;
  comm->base.nqps = flagcxParamIbQpsPerConn() *
                    comm->base.ndevs; // We must have at least 1 qp per-device
  comm->base.isSend = true;

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDevN = mergedDev->devs[i];
    FLAGCXCHECK(flagcxIbInitCommDevBase(ibDevN, &comm->devs[i].base));
    comm->ar = comm->ar &&
               flagcxIbDevs[dev]
                   .ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  struct flagcxIbConnectionMetadata meta;
  meta.ndevs = comm->base.ndevs;

  // IBRC retransmission: default disabled, can be enabled via
  // FLAGCX_IB_RETRANS_ENABLE=1
  meta.retransEnabled = flagcxParamIbRetransEnable() ? 1 : 0;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    flagcxIbSendCommDev *commDev = comm->devs + devIndex;
    flagcxIbDev *ibDev = flagcxIbDevs + commDev->base.ibDevN;
    FLAGCXCHECK(flagcxIbCreateQp(ibDev->portNum, &commDev->base,
                                 IBV_ACCESS_REMOTE_WRITE, comm->base.qps + q));
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // Query ece capabilities (enhanced connection establishment)
    FLAGCXCHECK(flagcxWrapIbvQueryEce(comm->base.qps[q].qp, &meta.qpInfo[q].ece,
                                      &meta.qpInfo[q].ece_supported));
    devIndex = (devIndex + 1) % comm->base.ndevs;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    flagcxIbSendCommDev *commDev = comm->devs + i;
    flagcxIbDev *ibDev = flagcxIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    flagcxIbDevInfo *devInfo = meta.devs + i;
    devInfo->ib_port = ibDev->portNum;
    devInfo->mtu = ibDev->portAttr.active_mtu;
    devInfo->lid = ibDev->portAttr.lid;

    // Prepare my fifo
    FLAGCXCHECK(
        flagcxWrapIbvRegMr(&commDev->fifoMr, commDev->base.pd, comm->fifo,
                           sizeof(struct flagcxIbSendFifo) * MAX_REQUESTS *
                               FLAGCX_NET_IB_MAX_RECVS,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ));
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // RoCE support
    devInfo->link_layer = commDev->base.gidInfo.link_layer =
        ibDev->portAttr.link_layer;
    if (devInfo->link_layer == IBV_LINK_LAYER_ETHERNET) {
      FLAGCXCHECK(flagcxIbGetGidIndex(ibDev->context, ibDev->portNum,
                                      ibDev->portAttr.gid_tbl_len,
                                      &commDev->base.gidInfo.localGidIndex));
      FLAGCXCHECK(flagcxWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                        commDev->base.gidInfo.localGidIndex,
                                        &commDev->base.gidInfo.localGid));
      devInfo->spn = commDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo->iid = commDev->base.gidInfo.localGid.global.interface_id;
    } else {
      commDev->base.gidInfo.localGidIndex = 0;
      memset(&commDev->base.gidInfo.localGid, 0, sizeof(union ibv_gid));
    }

    // Create control QP for retransmission if enabled
    if (meta.retransEnabled) {
      FLAGCXCHECK(flagcxIbCreateCtrlQp(ibDev->context, commDev->base.pd,
                                       ibDev->portNum, &commDev->ctrlQp));
      meta.ctrlQpn[i] = commDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibDev->portAttr.lid;
      meta.ctrlGid[i] = commDev->base.gidInfo.localGid;

      size_t ack_buf_size =
          (sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING) *
          FLAGCX_IB_ACK_BUF_COUNT;
      commDev->ackBuffer = malloc(ack_buf_size);
      FLAGCXCHECK(flagcxWrapIbvRegMr(&commDev->ackMr, commDev->base.pd,
                                     commDev->ackBuffer, ack_buf_size,
                                     IBV_ACCESS_LOCAL_WRITE));

      TRACE(FLAGCX_NET,
            "Send: Created control QP for dev %d: qpn=%u, link_layer=%d, "
            "lid=%u, gid=%lx:%lx",
            i, commDev->ctrlQp.qp->qp_num, devInfo->link_layer, meta.ctrlLid[i],
            (unsigned long)meta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)meta.ctrlGid[i].global.interface_id);
    }

    if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(FLAGCX_NET,
               "NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d "
               "fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "FLAGCX MergedDev" : "FLAGCX Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, devInfo->lid, devInfo->fifoRkey,
               commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(FLAGCX_NET,
               "NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d "
               "query_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
               "comp_mask=0x%x} GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "FLAGCX MergedDev" : "FLAGCX Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, meta.qpInfo[q].ece_supported,
               meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options,
               meta.qpInfo[q].ece.comp_mask,
               (int64_t)commDev->base.gidInfo.localGidIndex, devInfo->spn,
               devInfo->iid, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    }
  }
  meta.fifoAddr = (uint64_t)comm->fifo;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = flagcxIbCommStateSend;
  stage->offset = 0;
  FLAGCXCHECK(flagcxIbMalloc((void **)&stage->buffer, sizeof(meta)));

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_SEND, &comm->base.sock,
                                   stage->buffer, sizeof(meta),
                                   &stage->offset));
  if (stage->offset != sizeof(meta))
    return flagcxSuccess;

  stage->state = flagcxIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct flagcxIbConnectionMetadata remMeta;
  FLAGCXCHECK(
      flagcxSocketProgress(FLAGCX_SOCKET_RECV, &comm->base.sock, stage->buffer,
                           sizeof(flagcxIbConnectionMetadata), &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return flagcxSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(flagcxIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;
  if (comm->base.nRemDevs != comm->base.ndevs) {
    mergedDev = flagcxIbMergedDevs + dev;
    WARN("NET/IB : Local mergedDev=%s has a different number of devices=%d as "
         "remoteDev=%s nRemDevs=%d",
         mergedDev->devName, comm->base.ndevs, remMeta.devName,
         comm->base.nRemDevs);
  }

  int link_layer;
  link_layer = remMeta.devs[0].link_layer;
  for (int i = 1; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].link_layer != link_layer) {
      WARN("NET/IB : Can't merge net devices with different link_layer. i=%d "
           "remMeta.ndevs=%d link_layer=%d rem_link_layer=%d",
           i, remMeta.ndevs, link_layer, remMeta.devs[i].link_layer);
      return flagcxInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id =
        comm->base.remDevs[i].iid;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix =
        comm->base.remDevs[i].spn;

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    FLAGCXCHECK(
        flagcxWrapIbvRegMr(comm->remSizesFifo.mrs + i, comm->devs[i].base.pd,
                           &comm->remSizesFifo.elems,
                           sizeof(int) * MAX_REQUESTS * FLAGCX_NET_IB_MAX_RECVS,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ));
  }
  comm->base.nRemDevs = remMeta.ndevs;

  // Initialize retransmission state
  FLAGCXCHECK(flagcxIbRetransInit(&comm->retrans));
  comm->last_timeout_check_us = 0;
  comm->outstanding_sends = 0;
  comm->outstanding_retrans = 0;
  comm->max_outstanding = flagcxParamIbMaxOutstanding();

  if (comm->retrans.enabled) {
    INFO(FLAGCX_NET,
         "NET/IBRC Sender: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         comm->retrans.minRtoUs, comm->retrans.maxRetry,
         comm->retrans.ackInterval);
  } else {
    INFO(FLAGCX_NET, "NET/IBRC Sender: Retransmission DISABLED");
  }

  for (int q = 0; q < comm->base.nqps; q++) {
    struct flagcxIbQpInfo *remQpInfo = remMeta.qpInfo + q;
    struct flagcxIbDevInfo *remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    flagcxIbSendCommDev *commDev = comm->devs + devIndex;
    uint8_t gidIndex = commDev->base.gidInfo.localGidIndex;

    struct ibv_qp *qp = comm->base.qps[q].qp;
    if (remQpInfo->ece_supported && remQpInfo->ece_supported)
      FLAGCXCHECK(
          flagcxWrapIbvSetEce(qp, &remQpInfo->ece, &remQpInfo->ece_supported));

    FLAGCXCHECK(flagcxIbRtrQp(qp, gidIndex, remQpInfo->qpn, remDevInfo));
    FLAGCXCHECK(flagcxIbRtsQp(qp));
  }

  if (link_layer == IBV_LINK_LAYER_ETHERNET) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct flagcxIbQp *qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;
      INFO(FLAGCX_NET,
           "NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, "
           "vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
           ibDevN, ibDev->portNum, remMeta.qpInfo[q].qpn,
           remMeta.qpInfo[q].ece_supported, remMeta.qpInfo[q].ece.vendor_id,
           remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  // Setup control QP for retransmission if enabled
  if (comm->retrans.enabled && remMeta.retransEnabled) {
    bool all_ah_success = true;

    for (int i = 0; i < comm->base.ndevs; i++) {
      flagcxIbSendCommDev *commDev = &comm->devs[i];
      flagcxIbDev *ibDev = flagcxIbDevs + commDev->base.ibDevN;

      TRACE(FLAGCX_NET,
            "Send: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibDev->portAttr.link_layer);

      flagcxResult_t ah_result = flagcxIbSetupCtrlQpConnection(
          ibDev->context, commDev->base.pd, &commDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibDev->portNum, ibDev->portAttr.link_layer,
          commDev->base.gidInfo.localGidIndex);

      if (ah_result != flagcxSuccess || !commDev->ctrlQp.ah) {
        all_ah_success = false;
        break;
      }

      size_t buf_entry_size =
          sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING;
      for (int r = 0; r < 32; r++) {
        struct ibv_sge sge;
        sge.addr = (uint64_t)((char *)commDev->ackBuffer + r * buf_entry_size);
        sge.length = buf_entry_size;
        sge.lkey = commDev->ackMr->lkey;

        struct ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id = r;
        recv_wr.next = NULL;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;

        struct ibv_recv_wr *bad_wr;
        FLAGCXCHECK(
            flagcxWrapIbvPostRecv(commDev->ctrlQp.qp, &recv_wr, &bad_wr));
      }

      TRACE(FLAGCX_NET,
            "Control QP ready for dev %d: local_qpn=%u, remote_qpn=%u, posted "
            "32 recv WRs",
            i, commDev->ctrlQp.qp->qp_num, remMeta.ctrlQpn[i]);
    }

    if (!all_ah_success) {
      comm->retrans.enabled = 0;

      for (int i = 0; i < comm->base.ndevs; i++) {
        if (comm->devs[i].ackMr)
          flagcxWrapIbvDeregMr(comm->devs[i].ackMr);
        if (comm->devs[i].ackBuffer)
          free(comm->devs[i].ackBuffer);
        flagcxIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
      }
    }

    if (all_ah_success && comm->retrans.enabled) {
      flagcxResult_t mr_result = flagcxWrapIbvRegMr(
          &comm->retrans_hdr_mr, comm->devs[0].base.pd, comm->retrans_hdr_pool,
          sizeof(comm->retrans_hdr_pool), IBV_ACCESS_LOCAL_WRITE);

      if (mr_result != flagcxSuccess || !comm->retrans_hdr_mr) {
        WARN("Failed to register retrans_hdr_mr, disabling retransmission");
        comm->retrans.enabled = 0;
        // Clean up already created resources
        for (int i = 0; i < comm->base.ndevs; i++) {
          if (comm->devs[i].ackMr)
            flagcxWrapIbvDeregMr(comm->devs[i].ackMr);
          if (comm->devs[i].ackBuffer)
            free(comm->devs[i].ackBuffer);
          flagcxIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
        }
      } else {
        TRACE(
            FLAGCX_NET,
            "Sender: Initialized SEND retransmission (header pool MR created)");
      }
    }
  }

  comm->base.ready = 1;
  stage->state = flagcxIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_SEND, &comm->base.sock,
                                   &comm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return flagcxSuccess;

  free(stage->buffer);
  stage->state = flagcxIbCommStateStart;

  *sendComm = comm;
  return flagcxSuccess;
}

FLAGCX_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

flagcxResult_t flagcxIbAccept(void *listenComm, void **recvComm) {
  struct flagcxIbListenComm *lComm = (struct flagcxIbListenComm *)listenComm;
  struct flagcxIbCommStage *stage = &lComm->stage;
  struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)stage->comm;
  int ready;
  *recvComm = NULL;
  // Pre-declare variables because of goto
  struct ibv_srq *srq = NULL;

  if (stage->state == flagcxIbCommStateAccept)
    goto ib_accept_check;
  if (stage->state == flagcxIbCommStateRecv)
    goto ib_recv;
  if (stage->state == flagcxIbCommStateSend)
    goto ib_send;
  if (stage->state == flagcxIbCommStatePendingReady)
    goto ib_recv_ready;
  if (stage->state != flagcxIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return flagcxInternalError;
  }

  FLAGCXCHECK(flagcxIbMalloc((void **)&rComm, sizeof(struct flagcxIbRecvComm)));
  stage->comm = rComm;
  stage->state = flagcxIbCommStateAccept;
  FLAGCXCHECK(flagcxSocketInit(&rComm->base.sock));
  FLAGCXCHECK(flagcxSocketAccept(&rComm->base.sock, &lComm->sock));

ib_accept_check:
  FLAGCXCHECK(flagcxSocketReady(&rComm->base.sock, &ready));
  if (!ready)
    return flagcxSuccess;

  struct flagcxIbConnectionMetadata remMeta;
  stage->state = flagcxIbCommStateRecv;
  stage->offset = 0;
  FLAGCXCHECK(flagcxIbMalloc((void **)&stage->buffer, sizeof(remMeta)));

ib_recv:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &rComm->base.sock,
                                   stage->buffer, sizeof(remMeta),
                                   &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return flagcxSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct flagcxIbConnectionMetadata));

  // IB setup
  // Pre-declare variables because of goto
  struct flagcxIbMergedDev *mergedDev;
  struct flagcxIbDev *ibDev;
  int ibDevN;
  struct flagcxIbRecvCommDev *rCommDev;
  struct flagcxIbDevInfo *remDevInfo;
  struct flagcxIbQp *qp;

  mergedDev = flagcxIbMergedDevs + lComm->dev;
  rComm->base.ndevs = mergedDev->ndevs;
  rComm->base.nqps = flagcxParamIbQpsPerConn() *
                     rComm->base.ndevs; // We must have at least 1 qp per-device
  rComm->base.isSend = false;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.ndevs) {
    WARN("NET/IB : Local mergedDev %s has a different number of devices=%d as "
         "remote %s %d",
         mergedDev->devName, rComm->base.ndevs, remMeta.devName,
         rComm->base.nRemDevs);
  }

  // Metadata to send back to requestor (sender)
  struct flagcxIbConnectionMetadata meta;
  memset(&meta, 0,
         sizeof(meta)); // Initialize meta, including meta.retransEnabled = 0

  // IBRC retransmission: default disabled, can be enabled via
  // FLAGCX_IB_RETRANS_ENABLE=1
  meta.retransEnabled = flagcxParamIbRetransEnable() ? 1 : 0;

  // Create SRQ if retransmission is enabled
  if (remMeta.retransEnabled && meta.retransEnabled) {
    ibDevN = mergedDev->devs[0];
    ibDev = flagcxIbDevs + ibDevN;

    flagcxResult_t srq_result = flagcxIbCreateSrq(
        ibDev->context, rComm->devs[0].base.pd, &rComm->srqMgr);
    if (srq_result == flagcxSuccess) {
      srq = (struct ibv_srq *)rComm->srqMgr.srq;
      TRACE(FLAGCX_NET, "Receiver: Created SRQ for retransmission: srq=%p",
            srq);
    } else {
      INFO(FLAGCX_NET,
           "Receiver: Failed to create SRQ (result=%d), disabling "
           "retransmission",
           srq_result);
      meta.retransEnabled = 0;
      srq = NULL;
    }
  }

  for (int i = 0; i < rComm->base.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = mergedDev->devs[i];
    FLAGCXCHECK(flagcxIbInitCommDevBase(ibDevN, &rCommDev->base));
    ibDev = flagcxIbDevs + ibDevN;
    FLAGCXCHECK(flagcxIbGetGidIndex(ibDev->context, ibDev->portNum,
                                    ibDev->portAttr.gid_tbl_len,
                                    &rCommDev->base.gidInfo.localGidIndex));
    FLAGCXCHECK(flagcxWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                      rCommDev->base.gidInfo.localGidIndex,
                                      &rCommDev->base.gidInfo.localGid));
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id =
        rComm->base.remDevs[i].iid;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix =
        rComm->base.remDevs[i].spn;
  }

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIdx;
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIdx = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIdx;
    qp = rComm->base.qps + q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIdx;

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = flagcxIbDevs + ibDevN;
    FLAGCXCHECK(flagcxIbCreateQp(ibDev->portNum, &rCommDev->base,
                                 IBV_ACCESS_REMOTE_WRITE, qp));
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].ece_supported) {
      FLAGCXCHECK(flagcxWrapIbvSetEce(qp->qp, &remMeta.qpInfo[q].ece,
                                      &meta.qpInfo[q].ece_supported));

      // Query the reduced ece for this QP (matching enhancements between the
      // requestor and the responder) Store this in our own qpInfo for returning
      // to the requestor
      if (meta.qpInfo[q].ece_supported)
        FLAGCXCHECK(flagcxWrapIbvQueryEce(qp->qp, &meta.qpInfo[q].ece,
                                          &meta.qpInfo[q].ece_supported));
    }

    FLAGCXCHECK(flagcxIbRtrQp(qp->qp, rCommDev->base.gidInfo.localGidIndex,
                              remMeta.qpInfo[q].qpn, remDevInfo));
    FLAGCXCHECK(flagcxIbRtsQp(qp->qp));
  }

  rComm->flushEnabled = 1;

  for (int i = 0; i < mergedDev->ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rCommDev->base.ibDevN;
    ibDev = flagcxIbDevs + ibDevN;

    // Retain remote fifo info and prepare my RDMA ops
    rCommDev->fifoRkey = remMeta.devs[i].fifoRkey;
    rComm->remFifo.addr = remMeta.fifoAddr;
    FLAGCXCHECK(flagcxWrapIbvRegMr(
        &rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems,
        sizeof(struct flagcxIbSendFifo) * MAX_REQUESTS *
            FLAGCX_NET_IB_MAX_RECVS,
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_READ));
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (flagcxParamIbUseInline())
      rComm->remFifo.flags = IBV_SEND_INLINE;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      FLAGCXCHECK(flagcxWrapIbvRegMr(&rCommDev->gpuFlush.hostMr,
                                     rCommDev->base.pd, &rComm->gpuFlushHostMem,
                                     sizeof(int), IBV_ACCESS_LOCAL_WRITE));
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      FLAGCXCHECK(
          flagcxIbCreateQp(ibDev->portNum, &rCommDev->base,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
                           &rCommDev->gpuFlush.qp));
      struct flagcxIbDevInfo devInfo;
      devInfo.lid = ibDev->portAttr.lid;
      devInfo.link_layer = ibDev->portAttr.link_layer;
      devInfo.ib_port = ibDev->portNum;
      devInfo.spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.iid = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu = ibDev->portAttr.active_mtu;
      FLAGCXCHECK(flagcxIbRtrQp(rCommDev->gpuFlush.qp.qp,
                                rCommDev->base.gidInfo.localGidIndex,
                                rCommDev->gpuFlush.qp.qp->qp_num, &devInfo));
      FLAGCXCHECK(flagcxIbRtsQp(rCommDev->gpuFlush.qp.qp));
    }

    // Fill Handle
    meta.devs[i].lid = ibDev->portAttr.lid;
    meta.devs[i].link_layer = rCommDev->base.gidInfo.link_layer =
        ibDev->portAttr.link_layer;
    meta.devs[i].ib_port = ibDev->portNum;
    meta.devs[i].spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].iid = rCommDev->base.gidInfo.localGid.global.interface_id;

    // Adjust the MTU
    remMeta.devs[i].mtu =
        (enum ibv_mtu)std::min(remMeta.devs[i].mtu, ibDev->portAttr.active_mtu);
    meta.devs[i].mtu = remMeta.devs[i].mtu;

    // Prepare sizes fifo
    FLAGCXCHECK(flagcxWrapIbvRegMr(
        &rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo,
        sizeof(int) * MAX_REQUESTS * FLAGCX_NET_IB_MAX_RECVS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;

    // Create control QP for retransmission if enabled
    if (remMeta.retransEnabled && meta.retransEnabled) {
      FLAGCXCHECK(flagcxIbCreateCtrlQp(ibDev->context, rCommDev->base.pd,
                                       ibDev->portNum, &rCommDev->ctrlQp));
      meta.ctrlQpn[i] = rCommDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibDev->portAttr.lid;
      meta.ctrlGid[i] = rCommDev->base.gidInfo.localGid;

      TRACE(FLAGCX_NET,
            "Receiver: Control QP created for dev %d, qpn=%u, lid=%u", i,
            meta.ctrlQpn[i], meta.ctrlLid[i]);

      size_t ack_buf_size =
          (sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING) *
          FLAGCX_IB_ACK_BUF_COUNT;
      rCommDev->ackBuffer = malloc(ack_buf_size);
      FLAGCXCHECK(flagcxWrapIbvRegMr(&rCommDev->ackMr, rCommDev->base.pd,
                                     rCommDev->ackBuffer, ack_buf_size,
                                     IBV_ACCESS_LOCAL_WRITE));

      TRACE(FLAGCX_NET,
            "Recv: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibDev->portAttr.link_layer);

      flagcxResult_t ah_result = flagcxIbSetupCtrlQpConnection(
          ibDev->context, rCommDev->base.pd, &rCommDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibDev->portNum, ibDev->portAttr.link_layer,
          rCommDev->base.gidInfo.localGidIndex);

      if (ah_result != flagcxSuccess || !rCommDev->ctrlQp.ah) {
        INFO(FLAGCX_NET,
             "Receiver Control QP setup failed for dev %d, disabling "
             "retransmission",
             i);
        meta.retransEnabled = 0;

        if (rCommDev->ackMr)
          flagcxWrapIbvDeregMr(rCommDev->ackMr);
        if (rCommDev->ackBuffer)
          free(rCommDev->ackBuffer);
        flagcxIbDestroyCtrlQp(&rCommDev->ctrlQp);
      } else {
        TRACE(FLAGCX_NET,
              "Receiver Control QP successfully initialized for dev %d (ah=%p)",
              i, rCommDev->ctrlQp.ah);
      }
    }
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = flagcxIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer)
    free(stage->buffer);
  FLAGCXCHECK(flagcxIbMalloc((void **)&stage->buffer,
                             sizeof(struct flagcxIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct flagcxIbConnectionMetadata));

ib_send:
  FLAGCXCHECK(flagcxSocketProgress(
      FLAGCX_SOCKET_SEND, &rComm->base.sock, stage->buffer,
      sizeof(struct flagcxIbConnectionMetadata), &stage->offset));
  if (stage->offset < sizeof(struct flagcxIbConnectionMetadata))
    return flagcxSuccess;

  stage->offset = 0;
  stage->state = flagcxIbCommStatePendingReady;

ib_recv_ready:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &rComm->base.sock,
                                   &rComm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return flagcxSuccess;

  // Initialize retransmission state
  FLAGCXCHECK(flagcxIbRetransInit(&rComm->retrans));

  if (rComm->retrans.enabled) {
    INFO(FLAGCX_NET,
         "NET/IBRC Receiver: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         rComm->retrans.minRtoUs, rComm->retrans.maxRetry,
         rComm->retrans.ackInterval);
  } else {
    INFO(FLAGCX_NET, "NET/IBRC Receiver: Retransmission DISABLED");
  }

  // Initialize SRQ with recv buffers if enabled
  if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
    rComm->srqMgr.postSrqCount = FLAGCX_IB_SRQ_SIZE;

    // Post in batches until all are posted
    while (rComm->srqMgr.postSrqCount > 0) {
      FLAGCXCHECK(flagcxIbSrqPostRecv(&rComm->srqMgr, FLAGCX_IB_ACK_BUF_COUNT));
    }

    INFO(FLAGCX_INIT | FLAGCX_NET,
         "NET/IBRC Receiver: Posted %d recv WRs to SRQ for retransmission",
         FLAGCX_IB_SRQ_SIZE);
  }

  free(stage->buffer);
  *recvComm = rComm;

  /* reset lComm stage */
  stage->state = flagcxIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbGetRequest(struct flagcxIbNetCommBase *base,
                                  struct flagcxIbRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct flagcxIbRequest *r = base->reqs + i;
    if (r->type == FLAGCX_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->devBases[0] = NULL;
      r->devBases[1] = NULL;
      r->events[0] = r->events[1] = 0;
      *req = r;
      return flagcxSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return flagcxInternalError;
}

flagcxResult_t flagcxIbFreeRequest(struct flagcxIbRequest *r) {
  r->type = FLAGCX_NET_IB_REQ_UNUSED;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbTest(void *request, int *done, int *size);

flagcxResult_t flagcxIbRegMrDmaBufInternal(flagcxIbNetCommDevBase *base,
                                           void *data, size_t size, int type,
                                           uint64_t offset, int fd,
                                           ibv_mr **mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0)
    pageSize = sysconf(_SC_PAGESIZE);
  struct flagcxIbMrCache *cache = &flagcxIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  flagcxResult_t res;
  pthread_mutex_lock(&flagcxIbDevs[base->ibDevN].lock);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population ||
        addr < cache->slots[slot].addr) {         // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        FLAGCXCHECKGOTO(
            flagcxRealloc(&cache->slots, cache->population, cache->capacity),
            res, returning);
      }
      // Deregister / register
      struct ibv_mr *mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
      if (flagcxIbRelaxedOrderingEnabled)
        flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        FLAGCXCHECKGOTO(flagcxWrapIbvRegDmabufMr(&mr, base->pd, offset,
                                                 pages * pageSize, addr, fd,
                                                 flags),
                        res, returning);
      } else {
        void *cpuptr = NULL;
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMmap(&cpuptr, (void *)addr, pages * pageSize);
        }
        if (flagcxIbRelaxedOrderingEnabled) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING
          // support
          FLAGCXCHECKGOTO(
              flagcxWrapIbvRegMrIova2(&mr, base->pd,
                                      cpuptr == NULL ? (void *)addr : cpuptr,
                                      pages * pageSize, addr, flags),
              res, returning);
        } else {
          FLAGCXCHECKGOTO(
              flagcxWrapIbvRegMr(&mr, base->pd,
                                 cpuptr == NULL ? (void *)addr : cpuptr,
                                 pages * pageSize, flags),
              res, returning);
        }
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMunmap(cpuptr, pages * pageSize);
        }
      }
      TRACE(FLAGCX_INIT | FLAGCX_NET,
            "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d",
            (unsigned long)addr, (long long)pages * pageSize, mr->rkey,
            mr->lkey, fd);
      if (slot != cache->population)
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(struct flagcxIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      res = flagcxSuccess;
      goto returning;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <=
                   cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      res = flagcxSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&flagcxIbDevs[base->ibDevN].lock);
  return res;
}

struct flagcxIbNetCommDevBase *
flagcxIbGetNetCommDevBase(flagcxIbNetCommBase *base, int devIndex) {
  if (base->isSend) {
    struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
flagcxResult_t flagcxIbRegMrDmaBuf(void *comm, void *data, size_t size,
                                   int type, uint64_t offset, int fd,
                                   void **mhandle) {
  assert(size > 0);
  struct flagcxIbNetCommBase *base = (struct flagcxIbNetCommBase *)comm;
  struct flagcxIbMrHandle *mhandleWrapper =
      (struct flagcxIbMrHandle *)malloc(sizeof(struct flagcxIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    // Each flagcxIbNetCommDevBase is at different offset in send and recv
    // netComms
    struct flagcxIbNetCommDevBase *devComm = flagcxIbGetNetCommDevBase(base, i);
    FLAGCXCHECK(flagcxIbRegMrDmaBufInternal(devComm, data, size, type, offset,
                                            fd, mhandleWrapper->mrs + i));
  }
  *mhandle = (void *)mhandleWrapper;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbRegMr(void *comm, void *data, size_t size, int type,
                             void **mhandle) {
  return flagcxIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mhandle);

  assert(size > 0);
  struct flagcxIbNetCommBase *base = (struct flagcxIbNetCommBase *)comm;
  struct flagcxIbMrHandle *mhandleWrapper =
      (struct flagcxIbMrHandle *)malloc(sizeof(struct flagcxIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    struct flagcxIbNetCommDevBase *devComm = flagcxIbGetNetCommDevBase(base, i);
    unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_READ;
    flagcxWrapIbvRegMr(&mhandleWrapper->mrs[i], devComm->pd, data, size, flags);
  }
  *mhandle = mhandleWrapper;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbDeregMrInternal(flagcxIbNetCommDevBase *base,
                                       ibv_mr *mhandle) {
  struct flagcxIbMrCache *cache = &flagcxIbDevs[base->ibDevN].mrCache;
  flagcxResult_t res;
  pthread_mutex_lock(&flagcxIbDevs[base->ibDevN].lock);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population],
                sizeof(struct flagcxIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        FLAGCXCHECKGOTO(flagcxWrapIbvDeregMr(mhandle), res, returning);
      }
      res = flagcxSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle,
       cache->population);
  res = flagcxInternalError;
returning:
  pthread_mutex_unlock(&flagcxIbDevs[base->ibDevN].lock);
  return res;
}

flagcxResult_t flagcxIbDeregMr(void *comm, void *mhandle) {
  struct flagcxIbMrHandle *mhandleWrapper = (struct flagcxIbMrHandle *)mhandle;
  struct flagcxIbNetCommBase *base = (struct flagcxIbNetCommBase *)comm;
  for (int i = 0; i < base->ndevs; i++) {
    // Each flagcxIbNetCommDevBase is at different offset in send and recv
    // netComms
    struct flagcxIbNetCommDevBase *devComm = flagcxIbGetNetCommDevBase(base, i);
    FLAGCXCHECK(flagcxIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return flagcxSuccess;
}

FLAGCX_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

flagcxResult_t flagcxIbMultiSend(struct flagcxIbSendComm *comm, int slot) {
  struct flagcxIbRequest **reqs = comm->fifoReqs[slot];
  volatile struct flagcxIbSendFifo *slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > FLAGCX_NET_IB_MAX_RECVS)
    return flagcxInternalError;

  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) {
    struct ibv_send_wr *wr = comm->wrs + r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge *sge = comm->sges + r;
    sge->addr = (uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r * 8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  uint32_t seq = 0;
  if (nreqs == 1) {
    if (comm->retrans.enabled) {
      seq = comm->retrans.sendSeq;
      comm->retrans.sendSeq = (comm->retrans.sendSeq + 1) & 0xFFFF;
      immData = flagcxIbEncodeImmData(seq, reqs[0]->send.size);
      TRACE(FLAGCX_NET,
            "Sending packet with SEQ: seq=%u, size=%u, immData=0x%x", seq,
            reqs[0]->send.size, immData);
    } else {
      immData = reqs[0]->send.size;
    }
  } else {
    int *sizes = comm->remSizesFifo.elems[slot];
    for (int r = 0; r < nreqs; r++)
      sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs * sizeof(int);
  }

  struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
  if (nreqs > 1 ||
      (comm->ar && reqs[0]->send.size > flagcxParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr =
          comm->remSizesFifo.addr +
          slot * FLAGCX_NET_IB_MAX_RECVS * sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128
  // protocols still work
  const int align = 128;
  int nqps = flagcxParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    flagcxIbQp *qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    for (int r = 0; r < nreqs; r++) {
      // Track this event for completion
      // flagcxIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length =
          std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges + r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    struct ibv_send_wr *bad_wr;
    FLAGCXCHECK(flagcxWrapIbvPostSend(qp->qp, comm->wrs, &bad_wr));

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }

  // Add packet to retransmission buffer if enabled
  if (comm->retrans.enabled && nreqs == 1) {
    FLAGCXCHECK(flagcxIbRetransAddPacket(
        &comm->retrans, seq, reqs[0]->send.size, reqs[0]->send.data,
        slots[0].addr, // remote_addr
        reqs[0]->send.lkeys, (uint32_t *)slots[0].rkeys));
  }

  comm->outstanding_sends++;

  return flagcxSuccess;
}

flagcxResult_t flagcxIbIsend(void *sendComm, void *data, size_t size, int tag,
                             void *mhandle, void *phandle, void **request) {
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: flagcxIbIsend() called when comm->base.ready == 0");
    return flagcxInternalError;
  }
  if (comm->base.ready == 0) {
    *request = NULL;
    return flagcxSuccess;
  }

  struct flagcxIbMrHandle *mhandleWrapper = (struct flagcxIbMrHandle *)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct flagcxIbSendFifo *slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct flagcxIbRequest **reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead + 1;
  if (slots[0].idx != idx) {
    *request = NULL;
    return flagcxSuccess;
  }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r = 1; r < nreqs; r++)
    while (slots[r].idx != idx)
      ;
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads
                        // below
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag)
      continue;

    if (size > slots[r].size)
      size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union flagcxSocketAddress addr;
      flagcxSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: "
           "size %ld addr %lx rkeys[0]=%x",
           r, nreqs, tag, flagcxSocketToString(&addr, line), slots[r].size,
           slots[r].addr, slots[r].rkeys[0]);
      return flagcxInternalError;
    }

    struct flagcxIbRequest *req;
    FLAGCXCHECK(flagcxIbGetRequest(&comm->base, &req));
    req->type = FLAGCX_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents =
        flagcxParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      flagcxIbQp *qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      flagcxIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same
      // set of QPs inside flagcxIbMultiSend()
      qpIndex = (qpIndex + 1) % comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r] == NULL)
        return flagcxSuccess;
    }

    TIME_START(0);
    FLAGCXCHECK(flagcxIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and
    // sanity checks
    memset((void *)slots, 0, sizeof(struct flagcxIbSendFifo));
    memset(reqs, 0, FLAGCX_NET_IB_MAX_RECVS * sizeof(struct flagcxIbRequest *));
    comm->fifoHead++;
    TIME_STOP(0);
    return flagcxSuccess;
  }

  *request = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbPostFifo(struct flagcxIbRecvComm *comm, int n,
                                void **data, size_t *sizes, int *tags,
                                void **mhandles, struct flagcxIbRequest *req) {
  return flagcxIbCommonPostFifo(comm, n, data, sizes, tags, mhandles, req,
                                flagcxIbAddEvent);
}

flagcxResult_t flagcxIbIrecv(void *recvComm, int n, void **data, size_t *sizes,
                             int *tags, void **mhandles, void **phandles,
                             void **request) {
  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: flagcxIbIrecv() called when comm->base.ready == 0");
    return flagcxInternalError;
  }
  if (comm->base.ready == 0) {
    *request = NULL;
    return flagcxSuccess;
  }
  if (n > FLAGCX_NET_IB_MAX_RECVS)
    return flagcxInternalError;

  struct flagcxIbRequest *req;
  FLAGCXCHECK(flagcxIbGetRequest(&comm->base, &req));
  req->type = FLAGCX_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;

  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps =
      flagcxParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Post recvs
  struct ibv_recv_wr *bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct flagcxIbQp *qp = comm->base.qps + comm->base.qpIndex;
    flagcxIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    FLAGCXCHECK(flagcxWrapIbvPostRecv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  FLAGCXCHECK(flagcxIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbIflush(void *recvComm, int n, void **data, int *sizes,
                              void **mhandles, void **request) {
  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  int last = -1;
  for (int i = 0; i < n; i++)
    if (sizes[i])
      last = i;
  if (comm->flushEnabled == 0 || last == -1)
    return flagcxSuccess;

  // Only flush once using the last non-zero receive
  struct flagcxIbRequest *req;
  FLAGCXCHECK(flagcxIbGetRequest(&comm->base, &req));
  req->type = FLAGCX_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct flagcxIbMrHandle *mhandle = (struct flagcxIbMrHandle *)mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr *bad_wr;
    FLAGCXCHECK(
        flagcxWrapIbvPostSend(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    flagcxIbAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxIbrcTestPreCheck(struct flagcxIbRequest *r) {
  if (!r)
    return flagcxInternalError;

  if (r->type == FLAGCX_NET_IB_REQ_SEND && r->base->isSend) {
    struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)r->base;
    if (sComm->retrans.enabled) {
      // Check if control QP is still valid before accessing it
      bool ctrlQpValid = false;
      for (int i = 0; i < sComm->base.ndevs; i++) {
        if (sComm->devs[i].ctrlQp.qp && sComm->devs[i].ctrlQp.cq) {
          ctrlQpValid = true;
          break;
        }
      }

      if (ctrlQpValid) {
        for (int i = 0; i < sComm->base.ndevs; i++) {
          // Only poll if control QP is still valid
          if (sComm->devs[i].ctrlQp.qp && sComm->devs[i].ctrlQp.cq) {
            for (int p = 0; p < 4; p++) {
              flagcxResult_t poll_result =
                  flagcxIbRetransRecvAckViaUd(sComm, i);
              if (poll_result != flagcxSuccess)
                break;
            }
          }
        }

        uint64_t now_us = flagcxIbGetTimeUs();
        const uint64_t CHECK_INTERVAL_US = 1000;
        if (now_us - sComm->last_timeout_check_us >= CHECK_INTERVAL_US) {
          // Don't use FLAGCXCHECK here - retransmission errors shouldn't block
          // operation completion
          flagcxResult_t retrans_result =
              flagcxIbRetransCheckTimeout(&sComm->retrans, sComm);
          if (retrans_result != flagcxSuccess &&
              retrans_result != flagcxInProgress) {
            // Log error but don't fail the operation
            if (flagcxDebugNoWarn == 0)
              INFO(FLAGCX_ALL,
                   "%s:%d -> %d (retransmission check failed, continuing)",
                   __FILE__, __LINE__, retrans_result);
          }
          sComm->last_timeout_check_us = now_us;
        }
      }
    }
  }

  if (r->type == FLAGCX_NET_IB_REQ_RECV && !r->base->isSend) {
    struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)r->base;
    if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
      const int kPostThreshold = 16;
      if (rComm->srqMgr.postSrqCount >= kPostThreshold) {
        FLAGCXCHECK(
            flagcxIbSrqPostRecv(&rComm->srqMgr, FLAGCX_IB_ACK_BUF_COUNT));
      }
    }
  }

  return flagcxSuccess;
}

static flagcxResult_t flagcxIbrcProcessWc(struct flagcxIbRequest *r,
                                          struct ibv_wc *wc, int devIndex,
                                          bool *handled) {
  if (!r || !wc || !handled)
    return flagcxInternalError;

  *handled = false;

  if (r->type == FLAGCX_NET_IB_REQ_RECV && !r->base->isSend) {
    struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)r->base;

    if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL &&
        wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      uint32_t seq, size;
      flagcxIbDecodeImmData(wc->imm_data, &seq, &size);
      r->recv.sizes[0] = size;

      struct flagcxIbAckMsg ack_msg = {0};
      int should_ack = 0;
      FLAGCXCHECK(flagcxIbRetransRecvPacket(&rComm->retrans, seq, &ack_msg,
                                            &should_ack));

      if (should_ack) {
        flagcxResult_t ack_result =
            flagcxIbRetransSendAckViaUd(rComm, &ack_msg, devIndex);
        if (ack_result != flagcxSuccess) {
          TRACE(FLAGCX_NET, "Failed to send ACK for seq=%u (result=%d)", seq,
                ack_result);
        } else {
          TRACE(FLAGCX_NET, "Sent ACK for seq=%u, ack_seq=%u", seq,
                ack_msg.ackSeq);
        }
      } else {
        TRACE(FLAGCX_NET, "No ACK needed for seq=%u (expect=%u)", seq,
              rComm->retrans.recvSeq);
      }
    } else if (!rComm->retrans.enabled &&
               wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      r->recv.sizes[0] = wc->imm_data;
    }
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbTest(void *request, int *done, int *sizes) {
  static const struct flagcxIbCommonTestOps kIbrcTestOps = {
      .component = "NET/IBRC",
      .pre_check = flagcxIbrcTestPreCheck,
      .process_wc = flagcxIbrcProcessWc};
  return flagcxIbCommonTestDataQp((struct flagcxIbRequest *)request, done,
                                  sizes, &kIbrcTestOps);
}

flagcxResult_t flagcxIbCloseSend(void *sendComm) {
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)sendComm;
  if (comm) {
    FLAGCXCHECK(flagcxSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbSendCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int n_cqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          flagcxWrapIbvPollCq(commDev->base.cq, 64, wcs, &n_cqe);
          if (n_cqe == 0)
            break;
        }
      }
    }

    // 清理重传使用的资源：即使 runtime 期间禁用了 retrans，也可能已经
    // 分配了 MR/ctrl QP，必须在 PD 释放前全部销毁
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbSendCommDev *commDev = comm->devs + i;
      if (commDev->ackMr) {
        flagcxWrapIbvDeregMr(commDev->ackMr);
        commDev->ackMr = NULL;
      }
      if (commDev->ackBuffer) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
      flagcxIbDestroyCtrlQp(&commDev->ctrlQp);
      if (i == 0 && comm->retrans_hdr_mr) {
        flagcxWrapIbvDeregMr(comm->retrans_hdr_mr);
        comm->retrans_hdr_mr = NULL;
      }
    }
    flagcxIbRetransDestroy(&comm->retrans);

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        FLAGCXCHECK(flagcxWrapIbvDestroyQp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbSendCommDev *commDev = comm->devs + i;
      if (commDev->fifoMr != NULL)
        FLAGCXCHECK(flagcxWrapIbvDeregMr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL)
        FLAGCXCHECK(flagcxWrapIbvDeregMr(comm->remSizesFifo.mrs[i]));
      FLAGCXCHECK(flagcxIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return flagcxSuccess;
}

flagcxResult_t flagcxIbCloseRecv(void *recvComm) {
  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  if (comm) {
    FLAGCXCHECK(flagcxSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbRecvCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int n_cqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          flagcxWrapIbvPollCq(commDev->base.cq, 64, wcs, &n_cqe);
          if (n_cqe == 0)
            break;
        }
      }
    }

    // Clean up retransmission resources regardless of enable flag
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbRecvCommDev *commDev = comm->devs + i;
      if (commDev->ackMr) {
        flagcxWrapIbvDeregMr(commDev->ackMr);
        commDev->ackMr = NULL;
      }
      if (commDev->ackBuffer) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
      flagcxIbDestroyCtrlQp(&commDev->ctrlQp);
    }

    if (comm->srqMgr.srq != NULL) {
      flagcxIbDestroySrq(&comm->srqMgr);
    }
    flagcxIbRetransDestroy(&comm->retrans);

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        FLAGCXCHECK(flagcxWrapIbvDestroyQp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct flagcxIbRecvCommDev *commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL)
          FLAGCXCHECK(flagcxWrapIbvDestroyQp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL)
          FLAGCXCHECK(flagcxWrapIbvDeregMr(commDev->gpuFlush.hostMr));
      }
      if (commDev->fifoMr != NULL)
        FLAGCXCHECK(flagcxWrapIbvDeregMr(commDev->fifoMr));
      if (commDev->sizesFifoMr != NULL)
        FLAGCXCHECK(flagcxWrapIbvDeregMr(commDev->sizesFifoMr));
      FLAGCXCHECK(flagcxIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxIbCloseListen(void *listenComm) {
  struct flagcxIbListenComm *comm = (struct flagcxIbListenComm *)listenComm;
  if (comm) {
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxIbGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < flagcxNMergedIbDevs; i++) {
    if (strcmp(flagcxIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return flagcxSuccess;
    }
  }
  return flagcxSystemError;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// flagcxSuccess : GDR works
// flagcxSystemError : no module or module loaded but not supported by GPU
flagcxResult_t flagcxIbGdrSupport() {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    // Check for the nv_peer_mem module being loaded
    moduleLoaded =
        ((access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) &&
         // Also support the new nvidia-peermem module
         (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) ==
          -1))
            ? 0
            : 1;
  }
  if (moduleLoaded == 0)
    return flagcxSystemError;
  return flagcxSuccess;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// flagcxSuccess : DMA-BUF support is available
// flagcxSystemError : DMA-BUF is not supported by the kernel
flagcxResult_t flagcxIbDmaBufSupport(int dev) {
  static int dmaBufSupported = -1;
  if (dmaBufSupported == -1) {
    flagcxResult_t res;
    struct ibv_pd *pd;
    struct ibv_context *ctx;
    struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + dev;

    // Test each dev
    for (int i = 0; i < mergedDev->ndevs; i++) {
      int ibDev = mergedDev->devs[i];
      ctx = flagcxIbDevs[ibDev].context;
      FLAGCXCHECKGOTO(flagcxWrapIbvAllocPd(&pd, ctx), res, failure);
      // Test kernel DMA-BUF support with a dummy call (fd=-1)
      (void)flagcxWrapDirectIbvRegDmabufMr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                           0ULL /*iova*/, -1 /*fd*/,
                                           0 /*flags*/);
      // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
      // supported (EBADF otherwise)
      dmaBufSupported =
          (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
      FLAGCXCHECKGOTO(flagcxWrapIbvDeallocPd(pd), res, failure);
    }
  }
  if (dmaBufSupported == 0)
    return flagcxSystemError;
  return flagcxSuccess;
failure:
  dmaBufSupported = 0;
  return flagcxSystemError;
}

flagcxResult_t flagcxIbDevices(int *ndev) {
  *ndev = flagcxNMergedIbDevs;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbGetProperties(int dev, void *props) {
  struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + dev;
  flagcxNetProperties_t *properties = (flagcxNetProperties_t *)props;

  properties->name = mergedDev->devName;
  properties->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the
  // same)
  struct flagcxIbDev *ibDev = flagcxIbDevs + mergedDev->devs[0];
  properties->pciPath = ibDev->pciPath;
  properties->guid = ibDev->guid;
  properties->ptrSupport = FLAGCX_PTR_HOST;

  if (flagcxIbGdrSupport() == flagcxSuccess) {
    properties->ptrSupport |= FLAGCX_PTR_CUDA; // GDR support via nv_peermem
  }
  properties->regIsGlobal = 1;
  if (flagcxIbDmaBufSupport(dev) == flagcxSuccess) {
    properties->ptrSupport |= FLAGCX_PTR_DMABUF;
  }
  properties->latency = 0; // Not set
  properties->port = ibDev->portNum + ibDev->realPort;
  properties->maxComms = ibDev->maxQp;
  properties->maxRecvs = FLAGCX_NET_IB_MAX_RECVS;
  properties->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  properties->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  return flagcxSuccess;
}

// Adapter wrapper functions

struct flagcxNetAdaptor flagcxNetIb = {
    // Basic functions
    "IB", flagcxIbInit, flagcxIbDevices, flagcxIbGetProperties,
    NULL, // reduceSupport
    NULL, // getDeviceMr
    NULL, // irecvConsumed

    // Setup functions
    flagcxIbListen, flagcxIbConnect, flagcxIbAccept, flagcxIbCloseSend,
    flagcxIbCloseRecv, flagcxIbCloseListen,

    // Memory region functions
    flagcxIbRegMr, flagcxIbRegMrDmaBuf, flagcxIbDeregMr,

    // Two-sided functions
    flagcxIbIsend, flagcxIbIrecv, flagcxIbIflush, flagcxIbTest,

    // One-sided functions
    NULL, // write
    NULL, // read
    NULL, // signal

    // Device name lookup
    flagcxIbGetDevFromName};
