#ifndef FLAGCX_GLOBAL_COMM_H_
#define FLAGCX_GLOBAL_COMM_H_

#include "bootstrap.h"
#include "flagcx.h"
#include "flagcx_tuner.h"

#include <map>
#include <vector>

/* Opaque handle to flagcxInnerComm */
typedef struct flagcxInnerComm *flagcxInnerComm_t;

/* Opaque handle to flagcxHeteroComm */
typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

typedef enum {
  flagcxCommunicatorUnknown = 0,
  flagcxCommunicatorHomo = 1,  // Homogeneous Communicator
  flagcxCommunicatorHybrid = 2 // Hybrid Communicator
} flagcxCommunicatorType_t;

struct flagcxComm {
  // TODO: adjust code format
  int rank;
  int nranks;
  int nclusters;
  int homoRank;
  int homoRootRank;
  int homoRanks;
  int hasSingleRankHomoComm;
  flagcxCommunicatorType_t commType;
  uint64_t magic;
  volatile uint32_t *abortFlag;
  int *clusterSizes;
  int *clusterIds;
  int *globalRank2HomoRank;
  int *clusterInterRanks;
  bootstrapState *bootstrap;
  flagcxInnerComm_t hostComm;
  flagcxInnerComm_t homoComm;
  flagcxHeteroComm_t heteroComm;
  flagcxInnerComm_t homoInterComm;
  int homoInterRootRank;
  int homoInterMyRank;
  int homoInterRanks;
  std::vector<std::vector<int>> clusterInterRankList;
  std::vector<flagcxVendorType> clusterVendorMap;
  struct flagcxTuner *tuner;
  void *tunerContext;
  std::map<struct TunerCollCategory, flagcxInnerComm_t>
      homoCommMap; // key: commTag returned by tuner
  std::map<struct flagcxCommTag, flagcxInnerComm_t> commMap;
  std::map<struct TunerCollCategory, flagcxInnerComm_t>
      homoBestCommMap;              // key: commTag returned by tuner
  flagcxInnerComm_t tunerInnerComm; // innerComm selected by tuner
  flagcxUniqueId_t commId;
  flagcxUniqueId *uniqueIdData;
  bool isTuningWithFlagscale; // whether tuning with flagscale
  bool isTunningComm;         // whether tuning the communicator
  bool isUseSingleTunerComm;  // whether tuning with one communicator
  struct C2cSchedulePair {
    int sendCluster;
    int recvCluster;
  } * c2cSchedule; // C2C schedule for pairing send/recv operations
};

#endif // end include guard
