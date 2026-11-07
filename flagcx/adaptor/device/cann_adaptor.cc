#include "ascend_adaptor.h"

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "check.h"
#include <ctype.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

/* Keep the DCMI ABI local to the CANN adaptor. */
struct flagcxCannDcmiPcieInfo {
  unsigned int vendorId;
  unsigned int subVendorId;
  unsigned int deviceId;
  unsigned int subDeviceId;
  int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int function;
  unsigned char reserved[32];
};
typedef char flagcxCannDcmiPcieInfoSizeCheck
    [sizeof(struct flagcxCannDcmiPcieInfo) == 64 ? 1 : -1];

struct flagcxCannPciApi {
  int (*deviceCount)(uint32_t *);
  int (*aclGetBusId)(int32_t, char *, int32_t);
  int (*aclGetDeviceByBusId)(const char *, int32_t *);
  int (*userToLogicId)(int32_t, int32_t *);
  int (*dcmiLogicToCardDevice)(int *, int *, unsigned int);
  int (*dcmiGetPcieInfo)(int, int, struct flagcxCannDcmiPcieInfo *);
};

enum flagcxCannPciProvider {
  CANN_PCI_PROVIDER_ACL,
  CANN_PCI_PROVIDER_DCMI,
};

#define CANN_PCI_BUS_ID_SIZE 32
#define CANN_PCI_ACL_FEATURE_NOT_SUPPORTED 207000

static flagcxResult_t cannAdaptorGetAddressRange(const void *ptr, void **base,
                                                 size_t *size);

static int cannPciNormalizeBusId(const char *input, char *normalized,
                                 size_t capacity) {
  size_t length;
  size_t domainDigits;
  unsigned int domain = 0, bus = 0, device = 0, function = 0;
  int written;

  if (input == NULL || normalized == NULL || capacity == 0)
    return 0;
  length = strlen(input);
  domainDigits = length == 12 ? 4 : (length == 16 ? 8 : 0);
  if (domainDigits == 0 || input[domainDigits] != ':' ||
      input[domainDigits + 3] != ':' || input[domainDigits + 6] != '.' ||
      input[domainDigits + 8] != '\0')
    return 0;
  for (size_t i = 0; i < length; ++i) {
    if (i == domainDigits || i == domainDigits + 3 || i == domainDigits + 6)
      continue;
    if (!isxdigit((unsigned char)input[i]))
      return 0;
  }
  if (sscanf(input, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4 ||
      bus > 0xff || device > 0x1f || function > 7)
    return 0;
  written =
      snprintf(normalized, capacity,
               domain <= 0xffff ? "%04x:%02x:%02x.%x" : "%08x:%02x:%02x.%x",
               domain, bus, device, function);
  return written >= 0 && (size_t)written < capacity;
}

static flagcxResult_t cannPciMapAclError(int result, const char *operation) {
  if (result == 0)
    return flagcxSuccess;
  WARN("CANN PCI: %s returned ACL error %d", operation, result);
  return result == CANN_PCI_ACL_FEATURE_NOT_SUPPORTED
             ? flagcxNotSupported
             : flagcxUnhandledDeviceError;
}

static flagcxResult_t
cannPciVisibleDeviceCount(const struct flagcxCannPciApi *api, uint32_t *count) {
  if (api == NULL || api->deviceCount == NULL)
    return flagcxNotSupported;
  return cannPciMapAclError(api->deviceCount(count), "aclrtGetDeviceCount");
}

static flagcxResult_t cannPciGetDcmiBusId(const struct flagcxCannPciApi *api,
                                          int userDeviceId, char *busId,
                                          size_t capacity) {
  int32_t logicId = -1;
  int cardId = -1, chipId = -1;
  int result;
  struct flagcxCannDcmiPcieInfo info;

  if (api->userToLogicId == NULL || api->dcmiLogicToCardDevice == NULL ||
      api->dcmiGetPcieInfo == NULL)
    return flagcxNotSupported;
  memset(&info, 0, sizeof(info));
  FLAGCXCHECK(cannPciMapAclError(api->userToLogicId(userDeviceId, &logicId),
                                 "aclrtGetLogicDevIdByUserDevId"));
  if (logicId < 0)
    return flagcxUnhandledDeviceError;
  result = api->dcmiLogicToCardDevice(&cardId, &chipId, (unsigned int)logicId);
  if (result != 0 || cardId < 0 || chipId < 0) {
    WARN("CANN PCI: DCMI logic-to-card failed: %d", result);
    return flagcxUnhandledDeviceError;
  }
  result = api->dcmiGetPcieInfo(cardId, chipId, &info);
  if (result != 0) {
    WARN("CANN PCI: DCMI PCI query failed: %d", result);
    return flagcxUnhandledDeviceError;
  }
  if (info.domain < 0 || info.bus > 0xff || info.device > 0x1f ||
      info.function > 7)
    return flagcxUnhandledDeviceError;
  result = snprintf(
      busId, capacity,
      info.domain <= 0xffff ? "%04x:%02x:%02x.%x" : "%08x:%02x:%02x.%x",
      (unsigned int)info.domain, info.bus, info.device, info.function);
  return result >= 0 && (size_t)result < capacity ? flagcxSuccess
                                                  : flagcxInvalidArgument;
}

static struct flagcxCannPciApi cannPciProductionApi;
static pthread_once_t cannPciInitOnce = PTHREAD_ONCE_INIT;
static pthread_once_t cannDcmiInitOnce = PTHREAD_ONCE_INIT;

static void cannPciInitializeProductionApi(void) {
  cannPciProductionApi.deviceCount =
      (int (*)(uint32_t *))dlsym(RTLD_DEFAULT, "aclrtGetDeviceCount");
  cannPciProductionApi.aclGetBusId = (int (*)(int32_t, char *, int32_t))dlsym(
      RTLD_DEFAULT, "aclrtDeviceGetPCIBusId");
  cannPciProductionApi.aclGetDeviceByBusId =
      (int (*)(const char *, int32_t *))dlsym(RTLD_DEFAULT,
                                              "aclrtDeviceGetByPCIBusId");
  cannPciProductionApi.userToLogicId = (int (*)(int32_t, int32_t *))dlsym(
      RTLD_DEFAULT, "aclrtGetLogicDevIdByUserDevId");
}

static void cannPciInitializeDcmiApi(void) {
  void *dcmi;
  int (*dcmiInit)(void);
  int (*logicToCard)(int *, int *, unsigned int);
  int (*pcieInfo)(int, int, struct flagcxCannDcmiPcieInfo *);
  int result;

  dcmi = dlopen("libdcmi.so", RTLD_NOW | RTLD_LOCAL);
  if (dcmi == NULL)
    dcmi = dlopen("libdcmi.so.1", RTLD_NOW | RTLD_LOCAL);
  if (dcmi == NULL) {
    WARN("CANN PCI: DCMI library unavailable: %s", dlerror());
    return;
  }
  dcmiInit = (int (*)(void))dlsym(dcmi, "dcmi_init");
  logicToCard = (int (*)(int *, int *, unsigned int))dlsym(
      dcmi, "dcmi_get_card_id_device_id_from_logicid");
  pcieInfo = (int (*)(int, int, struct flagcxCannDcmiPcieInfo *))dlsym(
      dcmi, "dcmi_get_device_pcie_info_v2");
  if (dcmiInit == NULL || logicToCard == NULL || pcieInfo == NULL) {
    WARN("CANN PCI: required DCMI symbols unavailable");
    dlclose(dcmi);
    return;
  }
  result = dcmiInit();
  if (result != 0) {
    WARN("CANN PCI: dcmi_init failed: %d", result);
    dlclose(dcmi);
    return;
  }
  /* Keep libdcmi loaded while its function pointers and state are in use. */
  cannPciProductionApi.dcmiLogicToCardDevice = logicToCard;
  cannPciProductionApi.dcmiGetPcieInfo = pcieInfo;
}

static const struct flagcxCannPciApi *cannPciGetProductionApi(void) {
  pthread_once(&cannPciInitOnce, cannPciInitializeProductionApi);
  return &cannPciProductionApi;
}

static void cannPciEnsureDcmi(const struct flagcxCannPciApi *api) {
  if (api == &cannPciProductionApi)
    pthread_once(&cannDcmiInitOnce, cannPciInitializeDcmiApi);
}

static flagcxResult_t cannPciGetBusId(const struct flagcxCannPciApi *api,
                                      char *busId, int length, int userDeviceId,
                                      enum flagcxCannPciProvider *provider) {
  uint32_t count = 0;
  char raw[CANN_PCI_BUS_ID_SIZE] = {0};
  char normalized[CANN_PCI_BUS_ID_SIZE] = {0};
  size_t normalizedLength;

  if (api == NULL || busId == NULL || length <= 0 || userDeviceId < 0)
    return flagcxInvalidArgument;
  busId[0] = '\0';
  FLAGCXCHECK(cannPciVisibleDeviceCount(api, &count));
  if ((uint32_t)userDeviceId >= count)
    return flagcxInvalidArgument;

  enum flagcxCannPciProvider selected = CANN_PCI_PROVIDER_DCMI;
  if (api->aclGetBusId != NULL && api->aclGetDeviceByBusId != NULL) {
    flagcxResult_t result =
        cannPciMapAclError(api->aclGetBusId(userDeviceId, raw, sizeof(raw)),
                           "aclrtDeviceGetPCIBusId");
    if (result == flagcxSuccess) {
      raw[sizeof(raw) - 1] = '\0';
      if (!cannPciNormalizeBusId(raw, normalized, sizeof(normalized)))
        return flagcxUnhandledDeviceError;
      selected = CANN_PCI_PROVIDER_ACL;
    } else if (result != flagcxNotSupported) {
      return result;
    }
  }
  if (selected == CANN_PCI_PROVIDER_DCMI) {
    cannPciEnsureDcmi(api);
    FLAGCXCHECK(
        cannPciGetDcmiBusId(api, userDeviceId, normalized, sizeof(normalized)));
  }
  normalizedLength = strlen(normalized) + 1;
  if (normalizedLength > (size_t)length)
    return flagcxInvalidArgument;
  memcpy(busId, normalized, normalizedLength);
  if (provider != NULL)
    *provider = selected;
  return flagcxSuccess;
}

static flagcxResult_t cannPciFindByDcmiBusId(const struct flagcxCannPciApi *api,
                                             const char *target, uint32_t count,
                                             int *userDeviceId) {
  int matched = -1;
  cannPciEnsureDcmi(api);
  for (uint32_t device = 0; device < count; ++device) {
    char candidate[CANN_PCI_BUS_ID_SIZE] = {0};
    FLAGCXCHECK(
        cannPciGetDcmiBusId(api, (int)device, candidate, sizeof(candidate)));
    if (strcmp(target, candidate) != 0)
      continue;
    if (matched >= 0)
      return flagcxInternalError;
    matched = (int)device;
  }
  if (matched < 0)
    return flagcxNotSupported;
  *userDeviceId = matched;
  return flagcxSuccess;
}

static flagcxResult_t
cannPciGetDeviceByBusId(const struct flagcxCannPciApi *api, int *userDeviceId,
                        const char *busId) {
  uint32_t count = 0;
  char target[CANN_PCI_BUS_ID_SIZE] = {0};
  int matched = -1;
  enum flagcxCannPciProvider matchedProvider = CANN_PCI_PROVIDER_DCMI;
  int32_t nativeFound = -1;
  flagcxResult_t nativeResult = flagcxNotSupported;

  if (api == NULL || userDeviceId == NULL || busId == NULL)
    return flagcxInvalidArgument;
  if (!cannPciNormalizeBusId(busId, target, sizeof(target)))
    return flagcxInvalidArgument;
  FLAGCXCHECK(cannPciVisibleDeviceCount(api, &count));
  if (api->aclGetBusId != NULL && api->aclGetDeviceByBusId != NULL)
    nativeResult =
        cannPciMapAclError(api->aclGetDeviceByBusId(busId, &nativeFound),
                           "aclrtDeviceGetByPCIBusId");
  for (uint32_t device = 0; device < count; ++device) {
    char candidate[CANN_PCI_BUS_ID_SIZE] = {0};
    enum flagcxCannPciProvider candidateProvider;
    FLAGCXCHECK(cannPciGetBusId(api, candidate, sizeof(candidate), (int)device,
                                &candidateProvider));
    if (strcmp(target, candidate) != 0)
      continue;
    if (matched >= 0)
      return flagcxInternalError; /* A duplicate BDF is not an identity. */
    matched = (int)device;
    matchedProvider = candidateProvider;
  }
  if (matched < 0)
    return flagcxNotSupported;
  if (matchedProvider == CANN_PCI_PROVIDER_ACL) {
    char raw[CANN_PCI_BUS_ID_SIZE] = {0};
    char normalized[CANN_PCI_BUS_ID_SIZE] = {0};
    flagcxResult_t result = cannPciMapAclError(
        api->aclGetBusId(matched, raw, sizeof(raw)), "aclrtDeviceGetPCIBusId");
    if (result == flagcxNotSupported)
      return cannPciFindByDcmiBusId(api, target, count, userDeviceId);
    FLAGCXCHECK(result);
    raw[sizeof(raw) - 1] = '\0';
    if (!cannPciNormalizeBusId(raw, normalized, sizeof(normalized)) ||
        strcmp(normalized, target) != 0)
      return flagcxUnhandledDeviceError;
    if (nativeResult != flagcxNotSupported &&
        (nativeResult != flagcxSuccess || nativeFound != matched)) {
      int32_t rawFound = -1;
      result = cannPciMapAclError(api->aclGetDeviceByBusId(raw, &rawFound),
                                  "aclrtDeviceGetByPCIBusId");
      FLAGCXCHECK(result);
      if (rawFound != matched)
        return flagcxUnhandledDeviceError;
    }
  }
  *userDeviceId = matched;
  return flagcxSuccess;
}

static flagcxResult_t cannAdaptorGetDevicePciBusId(char *busId, int length,
                                                   int userDeviceId) {
  return cannPciGetBusId(cannPciGetProductionApi(), busId, length, userDeviceId,
                         NULL);
}

static flagcxResult_t cannAdaptorGetDeviceByPciBusId(int *userDeviceId,
                                                     const char *busId) {
  return cannPciGetDeviceByBusId(cannPciGetProductionApi(), userDeviceId,
                                 busId);
}

std::map<flagcxMemcpyType_t, aclrtMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, ACL_MEMCPY_HOST_TO_DEVICE},
    {flagcxMemcpyDeviceToHost, ACL_MEMCPY_DEVICE_TO_HOST},
    {flagcxMemcpyDeviceToDevice, ACL_MEMCPY_DEVICE_TO_DEVICE},
};

flagcxResult_t cannAdaptorDeviceSynchronize() {
  DEVCHECK(aclrtSynchronizeDevice());
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(aclrtMemcpy(dst, size, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(aclrtMemcpyAsync(dst, size, src, size, memcpy_type_map[type],
                              stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(aclrtMemset(ptr, size, value, size));
    } else {
      DEVCHECK(aclrtMemsetAsync(ptr, size, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

struct cannIpcExport {
  char key[65];
  int closed;
  struct cannIpcExport *next;
};

struct cannIpcAllocation {
  void *ptr;
  size_t size;
  struct cannIpcExport *exports;
  struct cannIpcAllocation *next;
};

static pthread_mutex_t cannIpcAllocationLock = PTHREAD_MUTEX_INITIALIZER;
static struct cannIpcAllocation *cannIpcAllocations = NULL;

static flagcxResult_t cannIpcTrackAllocation(void *ptr, size_t size) {
  struct cannIpcAllocation *entry =
      (struct cannIpcAllocation *)calloc(1, sizeof(*entry));
  if (entry == NULL)
    return flagcxSystemError;
  entry->ptr = ptr;
  entry->size = size;
  pthread_mutex_lock(&cannIpcAllocationLock);
  entry->next = cannIpcAllocations;
  cannIpcAllocations = entry;
  pthread_mutex_unlock(&cannIpcAllocationLock);
  return flagcxSuccess;
}

static flagcxResult_t cannIpcFreeAllocation(void *ptr) {
  pthread_mutex_lock(&cannIpcAllocationLock);
  struct cannIpcAllocation **link = &cannIpcAllocations;
  while (*link != NULL && (*link)->ptr != ptr)
    link = &(*link)->next;
  struct cannIpcAllocation *entry = *link;
  if (entry != NULL) {
    for (struct cannIpcExport *item = entry->exports; item != NULL;
         item = item->next) {
      if (item->closed)
        continue;
      aclError result = aclrtIpcMemClose(item->key);
      if (result != ACL_SUCCESS) {
        WARN("CANN IPC export close failed: ACL error %d", result);
        pthread_mutex_unlock(&cannIpcAllocationLock);
        return flagcxUnhandledDeviceError;
      }
      item->closed = 1;
    }
  }
  aclError result = aclrtFree(ptr);
  if (result != ACL_SUCCESS) {
    WARN("CANN device free failed: ACL error %d", result);
    pthread_mutex_unlock(&cannIpcAllocationLock);
    return flagcxUnhandledDeviceError;
  }
  if (entry != NULL)
    *link = entry->next;
  pthread_mutex_unlock(&cannIpcAllocationLock);
  if (entry != NULL) {
    struct cannIpcExport *item = entry->exports;
    while (item != NULL) {
      struct cannIpcExport *next = item->next;
      free(item);
      item = next;
    }
  }
  free(entry);
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {

  if (type == flagcxMemHost) {
    DEVCHECK(aclrtMallocHost(ptr, size));
  } else {
    DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    flagcxResult_t result = cannIpcTrackAllocation(*ptr, size);
    if (result != flagcxSuccess) {
      (void)aclrtFree(*ptr);
      *ptr = NULL;
      return result;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(aclrtFreeHost(ptr));
  } else {
    FLAGCXCHECK(cannIpcFreeAllocation(ptr));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorSetDevice(int dev) {
  DEVCHECK(aclrtSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDevice(int *dev) {
  DEVCHECK(aclrtGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDeviceCount(int *count) {
  DEVCHECK(aclrtGetDeviceCount((uint32_t *)count));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return flagcxSuccess;
}
// TODO:unsupport
flagcxResult_t cannAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  flagcxResult_t result = cannIpcTrackAllocation(*ptr, size);
  if (result != flagcxSuccess) {
    (void)aclrtFree(*ptr);
    *ptr = NULL;
    return result;
  }
  return flagcxSuccess;
}

// TODO:unsupported
flagcxResult_t cannAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(cannIpcFreeAllocation(ptr));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(aclrtCreateStream((aclrtStream *)(*stream)));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtDestroyStream(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (aclrtStream)oldStream;
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtSynchronizeStream(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    aclrtStreamStatus status;
    DEVCHECK(aclrtStreamQuery(stream->base, &status));
    if (status == ACL_STREAM_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_STREAM_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t cannAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(aclrtStreamWaitEvent(stream->base, event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags =
      (eventType == flagcxEventDefault) ? ACL_EVENT_TIME_LINE : ACL_EVENT_SYNC;
  DEVCHECK(aclrtCreateEventWithFlag(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtDestroyEvent(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(aclrtRecordEvent(event->base, stream->base));
    } else {
      return flagcxUnhandledDeviceError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtSynchronizeEvent(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    aclrtEventRecordedStatus status;
    DEVCHECK(aclrtQueryEventStatus(event->base, &status));
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

struct cannIpcImport {
  void *ptr;
  char key[65];
  struct cannIpcImport *next;
};

static pthread_mutex_t cannIpcImportLock = PTHREAD_MUTEX_INITIALIZER;
static struct cannIpcImport *cannIpcImports = NULL;

flagcxResult_t cannAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  if (handle == NULL || size == NULL)
    return flagcxInvalidArgument;
  *handle = (flagcxIpcMemHandle_t)calloc(1, sizeof(**handle));
  if (*handle == NULL)
    return flagcxSystemError;
  *size = sizeof((*handle)->key);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL)
    return flagcxInvalidArgument;

  // A 65-byte CANN key must fit the existing 64-byte FlagCX wire payload.
  // Keep the terminating byte locally and reconstruct it at the importer.
  char key[66] = {};
  struct cannIpcExport *exportRecord =
      (struct cannIpcExport *)calloc(1, sizeof(*exportRecord));
  if (exportRecord == NULL)
    return flagcxSystemError;

  pthread_mutex_lock(&cannIpcAllocationLock);
  struct cannIpcAllocation *allocation = cannIpcAllocations;
  while (allocation != NULL) {
    uintptr_t address = (uintptr_t)devPtr;
    uintptr_t base = (uintptr_t)allocation->ptr;
    if (address >= base && address - base < allocation->size)
      break;
    allocation = allocation->next;
  }
  if (allocation == NULL) {
    pthread_mutex_unlock(&cannIpcAllocationLock);
    free(exportRecord);
    return flagcxNotSupported;
  }

  void *allocationBase = NULL;
  size_t allocationSize = 0;
  flagcxResult_t rangeResult =
      cannAdaptorGetAddressRange(devPtr, &allocationBase, &allocationSize);
  if (rangeResult != flagcxSuccess || allocationBase != allocation->ptr ||
      allocationSize < allocation->size) {
    pthread_mutex_unlock(&cannIpcAllocationLock);
    free(exportRecord);
    return rangeResult == flagcxSuccess ? flagcxNotSupported : rangeResult;
  }

  // The importer PID is not known at this adaptor callback boundary.
  aclError result = aclrtIpcMemGetExportKey(
      allocationBase, allocationSize, key, 65,
      ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION);
  if (result != ACL_SUCCESS) {
    WARN("CANN IPC export failed: ACL error %d", result);
    pthread_mutex_unlock(&cannIpcAllocationLock);
    free(exportRecord);
    return result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT
               ? flagcxNotSupported
               : flagcxUnhandledDeviceError;
  }
  if (key[0] == '\0' || key[64] != '\0') {
    WARN("CANN IPC key cannot fit the 64-byte FlagCX handle");
    (void)aclrtIpcMemClose(key);
    pthread_mutex_unlock(&cannIpcAllocationLock);
    free(exportRecord);
    return flagcxNotSupported;
  }

  memcpy(exportRecord->key, key, 65);
  exportRecord->next = allocation->exports;
  allocation->exports = exportRecord;
  memcpy(handle->key, key, sizeof(handle->key));
  pthread_mutex_unlock(&cannIpcAllocationLock);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL)
    return flagcxInvalidArgument;
  *devPtr = NULL;

  struct cannIpcImport *importRecord =
      (struct cannIpcImport *)calloc(1, sizeof(*importRecord));
  if (importRecord == NULL)
    return flagcxSystemError;
  memcpy(importRecord->key, handle->key, sizeof(handle->key));
  if (importRecord->key[0] == '\0') {
    free(importRecord);
    return flagcxInvalidArgument;
  }
  aclError result = aclrtIpcMemImportByKey(
      devPtr, importRecord->key, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
  if (result != ACL_SUCCESS || *devPtr == NULL) {
    WARN("CANN IPC import failed: ACL error %d", result);
    if (result == ACL_SUCCESS)
      (void)aclrtIpcMemClose(importRecord->key);
    *devPtr = NULL;
    free(importRecord);
    return result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT
               ? flagcxNotSupported
               : flagcxUnhandledDeviceError;
  }
  importRecord->ptr = *devPtr;
  pthread_mutex_lock(&cannIpcImportLock);
  importRecord->next = cannIpcImports;
  cannIpcImports = importRecord;
  pthread_mutex_unlock(&cannIpcImportLock);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL)
    return flagcxInvalidArgument;
  pthread_mutex_lock(&cannIpcImportLock);
  struct cannIpcImport **link = &cannIpcImports;
  while (*link != NULL && (*link)->ptr != devPtr)
    link = &(*link)->next;
  if (*link == NULL) {
    pthread_mutex_unlock(&cannIpcImportLock);
    return flagcxInvalidArgument;
  }
  struct cannIpcImport *importRecord = *link;
  aclError result = aclrtIpcMemClose(importRecord->key);
  if (result != ACL_SUCCESS) {
    WARN("CANN IPC import close failed: ACL error %d", result);
    pthread_mutex_unlock(&cannIpcImportLock);
    return flagcxUnhandledDeviceError;
  }
  *link = importRecord->next;
  pthread_mutex_unlock(&cannIpcImportLock);
  free(importRecord);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  // The export key remains live until its adaptor-owned allocation is freed.
  // Common IPC paths release this temporary wrapper before peers import it.
  free(handle);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    aclError aclStatus = aclrtLaunchHostFunc(stream->base, fn, args);
    if (aclStatus != ACL_SUCCESS) {
      WARN("CANN aclrtLaunchHostFunc failed: acl=%d", (int)aclStatus);
      return flagcxUnhandledDeviceError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                           flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorHostRegister(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorHostUnregister(void *) { return flagcxNotSupported; }

// Symmetric memory VMM stubs (not supported)
flagcxResult_t cannAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                       size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t cannAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                     void **) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastCreate(size_t, int, const int *, void **,
                                             int *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                           int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
}

static flagcxResult_t cannAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  aclrtPtrAttributes attrs = {};
  aclError result = aclrtPointerGetAttributes(ptr, &attrs);
  if (result == ACL_ERROR_INVALID_PARAM ||
      result == ACL_ERROR_RT_PARAM_INVALID) {
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT)
    return flagcxNotSupported;
  if (result != ACL_SUCCESS) {
    WARN("CANN pointer classification failed: ACL error %d", result);
    return flagcxUnhandledDeviceError;
  }

  switch (attrs.location.type) {
    case ACL_MEM_LOCATION_TYPE_DEVICE:
    case ACL_MEM_LOCATION_TYPE_MANAGED:
      *ptrType = FLAGCX_PTR_CUDA;
      return flagcxSuccess;
    case ACL_MEM_LOCATION_TYPE_HOST:
    case ACL_MEM_LOCATION_TYPE_UNREGISTERED:
      *ptrType = FLAGCX_PTR_HOST;
      return flagcxSuccess;
    default:
      WARN("CANN pointer classification returned unknown location %d",
           (int)attrs.location.type);
      return flagcxUnhandledDeviceError;
  }
}

static flagcxResult_t cannAdaptorGetAddressRange(const void *ptr, void **base,
                                                 size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;
  *base = NULL;
  *size = 0;
  aclError result = aclrtMemGetAddressRange((void *)ptr, base, size);
  if (result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT)
    return flagcxNotSupported;
  if (result != ACL_SUCCESS) {
    WARN("CANN address range query failed: ACL error %d", result);
    return flagcxUnhandledDeviceError;
  }
  if (*base == NULL || *size == 0 || (uintptr_t)ptr < (uintptr_t)*base ||
      (uintptr_t)ptr - (uintptr_t)*base >= *size)
    return flagcxInternalError;
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor cannAdaptor {
  "CANN",
      // Basic functions
      cannAdaptorDeviceSynchronize, cannAdaptorDeviceMemcpy,
      cannAdaptorDeviceMemset, cannAdaptorDeviceMalloc, cannAdaptorDeviceFree,
      cannAdaptorSetDevice, cannAdaptorGetDevice, cannAdaptorGetDeviceCount,
      cannAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cannAdaptorGdrMemAlloc, cannAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      cannAdaptorStreamCreate, cannAdaptorStreamDestroy, cannAdaptorStreamCopy,
      cannAdaptorStreamFree, cannAdaptorStreamSynchronize,
      cannAdaptorStreamQuery, cannAdaptorStreamWaitEvent,
      cannAdaptorStreamWaitValue64, cannAdaptorStreamWriteValue64,
      // Event functions
      cannAdaptorEventCreate, cannAdaptorEventDestroy, cannAdaptorEventRecord,
      cannAdaptorEventSynchronize, cannAdaptorEventQuery,
      cannAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cannAdaptorIpcMemHandleCreate, cannAdaptorIpcMemHandleGet,
      cannAdaptorIpcMemHandleOpen, cannAdaptorIpcMemHandleClose,
      cannAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      NULL, // flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps
            // *props, int dev);
      cannAdaptorGetDevicePciBusId, cannAdaptorGetDeviceByPciBusId,
      cannAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      cannAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      cannAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      cannAdaptorSymPhysAlloc, cannAdaptorSymPhysFree, cannAdaptorSymFlatMap,
      cannAdaptorSymFlatUnmap, cannAdaptorSymMulticastSupported,
      cannAdaptorSymMulticastCreate, cannAdaptorSymMulticastBind,
      cannAdaptorSymMulticastTeardown, cannAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      cannAdaptorGetPointerType, cannAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_NONE, FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, NULL, NULL,
      NULL, NULL, NULL,
      // Preserve the legacy collective receive acquire until this platform
      // explicitly documents a coherent GPUDirect WRITE path.
      FLAGCX_GDR_WRITE_REQUIRES_FLUSH, FLAGCX_GDR_DEVICE_ASCEND,
};

#endif // USE_ASCEND_ADAPTOR
