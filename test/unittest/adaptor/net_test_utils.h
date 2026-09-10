/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_TEST_UNITTEST_ADAPTOR_NET_TEST_UTILS_H_
#define FLAGCX_TEST_UNITTEST_ADAPTOR_NET_TEST_UTILS_H_

#include <cstdio>

#include "flagcx.h"
#include "flagcx_net_adaptor.h"
#include "p2p_topo.h"

namespace flagcx_test {

inline flagcxResult_t getLocalNetDevice(struct flagcxNetAdaptor *netAdaptor,
                                        int netDeviceCount, int *netDev) {
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

  int gpuDev = 0;
  result = deviceHandle->getDevice(&gpuDev);
  if (result != flagcxSuccess) {
    flagcxDeviceHandleFree(deviceHandle);
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
