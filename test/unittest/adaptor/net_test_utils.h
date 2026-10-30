/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_TEST_UNITTEST_ADAPTOR_NET_TEST_UTILS_H_
#define FLAGCX_TEST_UNITTEST_ADAPTOR_NET_TEST_UTILS_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "p2p_topo.h"

namespace flagcx_test {

inline flagcxResult_t getLocalNetDevice(struct flagcxNetAdaptor *netAdaptor,
                                        int netDeviceCount, int *netDev,
                                        int *selectedGpu = nullptr) {
  if (netAdaptor == nullptr || netDeviceCount <= 0 || netDev == nullptr)
    return flagcxInvalidArgument;

  flagcxDeviceHandle_t deviceHandle = nullptr;
  flagcxResult_t result = flagcxDeviceHandleInit(&deviceHandle);
  if (result != flagcxSuccess)
    return result;
  if (deviceHandle == nullptr || deviceHandle->getDevice == nullptr) {
    flagcxDeviceHandleFree(deviceHandle);
    return flagcxInternalError;
  }

  const char *firstLocalDevice =
      std::getenv("FLAGCX_CI_FORCE_FIRST_LOCAL_DEVICE");
  const bool forceFirstLocalDevice =
      firstLocalDevice != nullptr && std::strcmp(firstLocalDevice, "1") == 0;
  const bool pciTopologyAvailable =
      deviceAdaptor != nullptr && deviceAdaptor->getDevicePciBusId != nullptr &&
      deviceAdaptor->getDeviceByPciBusId != nullptr;
  int gpuDev = 0;
  if (pciTopologyAvailable && !forceFirstLocalDevice) {
    result = deviceHandle->getDevice(&gpuDev);
    if (result != flagcxSuccess) {
      flagcxDeviceHandleFree(deviceHandle);
      return result;
    }
  }

  if (!pciTopologyAvailable || forceFirstLocalDevice) {
    int gpuCount = 0;
    result = deviceHandle->getDeviceCount(&gpuCount);
    if (result == flagcxSuccess && gpuCount > 0) {
      gpuDev = 0;
      *netDev = 0;
    } else if (result == flagcxSuccess) {
      result = flagcxInternalError;
    }
    const flagcxResult_t deviceFreeResult =
        flagcxDeviceHandleFree(deviceHandle);
    if (result == flagcxSuccess)
      result = deviceFreeResult;
    if (result == flagcxSuccess && selectedGpu != nullptr)
      *selectedGpu = gpuDev;
    if (result == flagcxSuccess)
      fprintf(stderr, "Adaptor tests selected first local accelerator/NIC; "
                      "PCI proximity is unavailable or disabled.\n");
    return result;
  }

  struct flagcxP2pTopoManager *topoMgr = nullptr;
  result = flagcxP2pTopoInit(netAdaptor, &topoMgr);
  if (result == flagcxSuccess)
    result = flagcxP2pTopoGetNetDev(topoMgr, gpuDev, netDev);

  const flagcxResult_t destroyResult = flagcxP2pTopoDestroy(topoMgr);
  if (result == flagcxSuccess)
    result = destroyResult;
  const flagcxResult_t deviceFreeResult = flagcxDeviceHandleFree(deviceHandle);
  if (result == flagcxSuccess)
    result = deviceFreeResult;
  if (result != flagcxSuccess)
    return result;
  if (*netDev < 0 || *netDev >= netDeviceCount)
    return flagcxInternalError;
  if (selectedGpu != nullptr)
    *selectedGpu = gpuDev;

  if (netAdaptor->getProperties != nullptr) {
    flagcxNetProperties_t properties = {};
    if (netAdaptor->getProperties(*netDev, &properties) == flagcxSuccess) {
      fprintf(stderr,
              "Adaptor test topology selected GPU %d -> netDev %d (%s, %s)\n",
              gpuDev, *netDev,
              properties.name != nullptr ? properties.name : "<unnamed>",
              properties.pciPath != nullptr ? properties.pciPath
                                            : "<unknown PCI path>");
    }
  }

  return flagcxSuccess;
}

} // namespace flagcx_test

#endif // FLAGCX_TEST_UNITTEST_ADAPTOR_NET_TEST_UTILS_H_
