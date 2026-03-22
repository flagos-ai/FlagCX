/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_H_
#define FLAGCX_ADAPTOR_H_

#include "bootstrap.h"
#include "device_utils.h"
#include "flagcx.h"
#include "global_comm.h"
#include "topo.h"
#include "utils.h"

// Struct definitions are now in per-type public headers
#include "flagcx_ccl_adaptor.h"
#include "flagcx_device_adaptor.h"
#include "flagcx_net_adaptor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NCCLADAPTORS 2
#define flagcxCCLAdaptorHost 0
#define flagcxCCLAdaptorDevice 1

extern struct flagcxCCLAdaptor bootstrapAdaptor;
extern struct flagcxCCLAdaptor glooAdaptor;
extern struct flagcxCCLAdaptor mpiAdaptor;
extern struct flagcxCCLAdaptor ncclAdaptor;
extern struct flagcxCCLAdaptor hcclAdaptor;
extern struct flagcxCCLAdaptor ixncclAdaptor;
extern struct flagcxCCLAdaptor cnclAdaptor;
extern struct flagcxCCLAdaptor mcclAdaptor;
extern struct flagcxCCLAdaptor musa_mcclAdaptor;
extern struct flagcxCCLAdaptor xcclAdaptor;
extern struct flagcxCCLAdaptor duncclAdaptor;
extern struct flagcxCCLAdaptor rcclAdaptor;
extern struct flagcxCCLAdaptor tcclAdaptor;
extern struct flagcxCCLAdaptor ecclAdaptor;
extern struct flagcxCCLAdaptor *cclAdaptors[];

extern struct flagcxDeviceAdaptor cudaAdaptor;
extern struct flagcxDeviceAdaptor cannAdaptor;
extern struct flagcxDeviceAdaptor ixcudaAdaptor;
extern struct flagcxDeviceAdaptor mluAdaptor;
extern struct flagcxDeviceAdaptor macaAdaptor;
extern struct flagcxDeviceAdaptor musaAdaptor;
extern struct flagcxDeviceAdaptor kunlunAdaptor;
extern struct flagcxDeviceAdaptor ducudaAdaptor;
extern struct flagcxDeviceAdaptor hipAdaptor;
extern struct flagcxDeviceAdaptor tsmicroAdaptor;
extern struct flagcxDeviceAdaptor topsAdaptor;
extern struct flagcxDeviceAdaptor *deviceAdaptor;

extern struct flagcxNetAdaptor *netAdaptor;

// Network type enumeration
enum NetType {
  IBRC = 1,   // InfiniBand RC (or UCX when USE_UCX=1)
  SOCKET = 2, // Socket
#ifdef USE_IBUC
  IBUC = 3 // InfiniBand UC
#endif
};

// Unified network adaptor function declarations
struct flagcxNetAdaptor *getUnifiedNetAdaptor(int netType);

inline bool flagcxCCLAdaptorNeedSendrecv(size_t value) { return value != 0; }

const int MAX_VENDOR_LEN = 128;
typedef struct {
  char internal[MAX_VENDOR_LEN];
} flagcxVendor;

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
