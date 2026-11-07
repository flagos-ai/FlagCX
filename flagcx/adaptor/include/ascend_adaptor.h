#ifdef USE_ASCEND_ADAPTOR
#include "acl/acl.h"
#include "flagcx.h"
#include "hccl/hccl.h"
#include <map>
struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  HcclComm base;
};

struct flagcxStream {
  aclrtStream base;
};

struct flagcxEvent {
  aclrtEvent base;
};

struct flagcxIpcMemHandle {
  // The first 64 bytes are the transferable key payload. AscendCL uses a
  // 65-byte string buffer; the terminating byte is reconstructed on import.
  char key[64];
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != ACL_SUCCESS)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }
#endif // USE_ASCEND_ADAPTOR
