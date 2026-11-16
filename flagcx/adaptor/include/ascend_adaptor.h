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
  // AscendCL requires a 65-byte IPC key buffer, including its terminator.
  char key[65];
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != ACL_SUCCESS)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }
#endif // USE_ASCEND_ADAPTOR
