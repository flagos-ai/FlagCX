#ifndef FLAGCX_UNIRUNNER_ALGORITHM_H_
#define FLAGCX_UNIRUNNER_ALGORITHM_H_

#include "flagcx.h"
#include <stdint.h>

// Selector values are separate from the persisted DAG cache algorithm IDs.
enum UniRunnerAlgorithm : int64_t {
  uniRunnerAlgorithmRingAR = 0,
  uniRunnerAlgorithmGroupedAR = 1,
  uniRunnerAlgorithmIpcRingAR = 2,
  uniRunnerAlgorithmLocRed = 3,
  uniRunnerAlgorithmRingAG = 4,
  uniRunnerAlgorithmHierarchicalSlicedAR = 5,
  uniRunnerAlgorithmSlicedAR = 6,
  uniRunnerAlgorithmGroupedAG = 7,
  uniRunnerAlgorithmP2pAG = 8,
  uniRunnerAlgorithmAllGatherGemmBuiltIn = 9,
  uniRunnerAlgorithmAllGatherGemmDsl = 10,
  uniRunnerAlgorithmGemmReduceScatterBuiltIn = 11,
  uniRunnerAlgorithmGemmReduceScatterDsl = 12
};

flagcxResult_t flagcxReadUniRunnerAlgorithm(int64_t *value);

inline flagcxResult_t resolveUniRunnerAlgorithm(UniRunnerAlgorithm defaultAlgo,
                                                UniRunnerAlgorithm first,
                                                UniRunnerAlgorithm last,
                                                UniRunnerAlgorithm *selected) {
  if (selected == nullptr)
    return flagcxInvalidArgument;
  int64_t value = -1;
  const flagcxResult_t result = flagcxReadUniRunnerAlgorithm(&value);
  if (result != flagcxSuccess)
    return result;
  if (value == -1) {
    *selected = defaultAlgo;
    return flagcxSuccess;
  }
  if (value < first || value > last)
    return flagcxInvalidArgument;
  *selected = static_cast<UniRunnerAlgorithm>(value);
  return flagcxSuccess;
}

#endif
