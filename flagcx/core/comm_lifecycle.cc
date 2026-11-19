#include "comm_lifecycle.h"
#include "global_comm.h"

flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return __atomic_load_n(&comm->finalizeState.lifecycle, __ATOMIC_ACQUIRE) ==
                 flagcxCommActive
             ? flagcxSuccess
             : flagcxInvalidUsage;
}
