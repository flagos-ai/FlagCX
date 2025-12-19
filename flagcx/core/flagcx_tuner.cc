#include "flagcx_tuner.h"
#include "adaptor.h"
#include "check.h"
#include "param.h"
#include "timer.h"
#include "tuner_util.h"
#include "utils.h"
#include <cfloat>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
// A category of collective operation. the minimal unit for tuning.

bool operator<(const struct TunerCollCategory &lhs,
               const struct TunerCollCategory &rhs) {
  if (lhs.collType != rhs.collType) {
    return lhs.collType < rhs.collType;
  }
  return lhs.nBytes < rhs.nBytes;
}

static_assert(FLAGCX_PROFILE_KEY_MAX_LENGTH >= 20,
              "FLAGCX_PROFILE_KEY_MAX_LENGTH < 20, too short");

// Key used for time profiling
struct TunerProfileKey {
  size_t nBytes;
  uint32_t collType; // flagcxCommOp_t
  uint32_t seqId;    // sequence id of collective within this TunerCollCategory
  uint32_t commTagIdx; // index of commTag in configList

  // constructors
  TunerProfileKey() : nBytes(0), collType(0), seqId(0), commTagIdx(0) {}
  TunerProfileKey(size_t n, uint32_t c, uint32_t s, uint32_t i)
      : nBytes(n), collType(c), seqId(s), commTagIdx(i) {}
  TunerProfileKey(const struct flagcxProfileKey &k) {
    const char *ptr = k.key;
    memcpy(&nBytes, ptr, sizeof(nBytes));
    ptr += sizeof(nBytes);
    memcpy(&collType, ptr, sizeof(collType));
    ptr += sizeof(collType);
    memcpy(&seqId, ptr, sizeof(seqId));
    ptr += sizeof(seqId);
    memcpy(&commTagIdx, ptr, sizeof(commTagIdx));
  }

  // conversion function
  operator struct flagcxProfileKey() const {
    struct flagcxProfileKey k;
    memset(k.key, 0, FLAGCX_PROFILE_KEY_MAX_LENGTH);
    char *ptr = k.key;
    memcpy(ptr, &nBytes, sizeof(nBytes));
    ptr += sizeof(nBytes);
    memcpy(ptr, &collType, sizeof(collType));
    ptr += sizeof(collType);
    memcpy(ptr, &seqId, sizeof(seqId));
    ptr += sizeof(seqId);
    memcpy(ptr, &commTagIdx, sizeof(commTagIdx));
    return k;
  }

  bool operator<(const TunerProfileKey &other) const {
    if (nBytes != other.nBytes) {
      return nBytes < other.nBytes;
    } else if (collType != other.collType) {
      return collType < other.collType;
    } else if (seqId != other.seqId) {
      return seqId < other.seqId;
    }
    return commTagIdx < other.commTagIdx;
  }

  bool operator==(const TunerProfileKey &other) const {
    return (nBytes == other.nBytes) && (collType == other.collType) &&
           (seqId == other.seqId) && (commTagIdx == other.commTagIdx);
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "{nBytes=" << nBytes << ",collType=" << collType
        << ",seqId=" << seqId << ",commTagIdx=" << commTagIdx << "}";
    return oss.str();
  }
};

// (collType,nBytes,configIdx)
// Used for counting the number of configs corresponding to each Collective Op
struct TunerCommTagCounterKey {
  size_t nBytes;
  uint32_t collType;   // flagcxCommOp_t
  uint32_t commTagIdx; // index of commTag in configList
};

static bool operator<(const struct TunerCommTagCounterKey &lhs,
                      const struct TunerCommTagCounterKey &rhs) {
  if (lhs.nBytes != rhs.nBytes)
    return lhs.nBytes < rhs.nBytes;
  if (lhs.collType != rhs.collType)
    return lhs.collType < rhs.collType;
  return lhs.commTagIdx < rhs.commTagIdx;
}

// number loops of collectives call before using profiled data.
// Each loop will go thoroughly through all search space of all candidates.
#define TUNER_SEARCH_NLOOPS 5
#define PROFILE_ROUND                                                          \
  2 // Use data from the 3rd round, as it's likely more stable.

// customized context structure for internal use
struct flagcxTunerContext {
  void *bootstrap;

  int rank;
  int nranks;

  float *profilingResults;

  // configure related struct
  std::vector<struct flagcxEnvConfig> configList;
  flagcxDebugLogger_t logger = NULL;
  int envTagIdx = -1; // the index of envTag in configList
  uint32_t searchNLoops = TUNER_SEARCH_NLOOPS;

  // runtime related struct
  std::vector<int> activeCommList; // List of active communicator. Holds indices
                                   // of configList
  std::map<struct flagcxCommTag, int>
      commTagIdxMap; // map from commTag to configList index
  std::map<TunerCollCategory, uint32_t>
      collSeqMap; // record the sequence number of each collective category
  std::map<TunerCollCategory, int>
      collBestCommMap; // record the best communicator for each collective
                       // category. value is comm index in configList.
  std::map<struct TunerCommTagCounterKey, int>
      configCounterMap; // record per (collType,nBytes,configIdx) counter.

  int commConfigId = 0; // record the current communicator config id, used when
                        // tuning with FlagScale

  int bestConfigId = -1; // record the best communicator config id, used when
                         // tuning with FlagScale

  // for flagscale tuning
  bool tunerCommMatchingDone = false;
  int lastFlagscaleConfigId = -1;

  // timer
  flagcxTimer<TunerProfileKey> timer;
};

bool operator<(const struct flagcxCommTag &lhs,
               const struct flagcxCommTag &rhs) {
  return strcmp(lhs.tag, rhs.tag) < 0;
}

bool operator==(const struct flagcxCommTag &lhs,
                const struct flagcxCommTag &rhs) {
  return strcmp(lhs.tag, rhs.tag) == 0;
}

// A helper function set envs filtered by envType mask
static flagcxResult_t setEnvConfig(const struct flagcxEnvConfig &cfg,
                                   uint32_t mask) {
  for (int i = 0; i < cfg.envCount; i++) {
    const auto &item = cfg.envs[i];
    if (item.type & mask) {
      if (setenv(item.name, item.value, 1) != 0) {
        return flagcxInternalError;
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerInit(size_t nRanks, size_t rank,
                               flagcxDebugLogger_t logFunction, void **context,
                               void *commState) {
  struct flagcxTunerContext *ctx = new struct flagcxTunerContext;
  ctx->bootstrap = commState;
  ctx->rank = rank;
  ctx->nranks = nRanks;
  FLAGCXCHECK(generateCandidate(ctx->configList));
  INFO(FLAGCX_TUNING, "Candidate number: %ld.", ctx->configList.size());
  ctx->logger = logFunction;
  *context = ctx;

  // Initialize commTagIdxMap and activeCommList
  for (size_t i = 0; i < ctx->configList.size(); ++i) {
    const auto &cfg = ctx->configList[i];
    ctx->commTagIdxMap[cfg.commTag] = i;
    ctx->activeCommList.push_back(i);
  }

  // Whether comm tag specified by environment variable
  const char *tagEnv = flagcxGetEnv("FLAGCX_USE_COMM_TAG");
  if (tagEnv != nullptr) {
    struct flagcxCommTag envTag;
    snprintf(envTag.tag, FLAGCX_COMM_TAG_MAX_LENGTH, "%s", tagEnv);
    auto it = ctx->commTagIdxMap.find(envTag);
    if (it == ctx->commTagIdxMap.end()) {
      WARN("Communicator tag %s set by environment not found in config list.",
           envTag.tag);
      return flagcxInvalidArgument;
    }
    ctx->envTagIdx = it->second;
    INFO(FLAGCX_ENV | FLAGCX_TUNING,
         "Communicator tag set by environment to %s.", envTag.tag);
  }

  // Whether to change search nloops by environment variable
  const char *nLoopsEnv = flagcxGetEnv("FLAGCX_TUNER_SEARCH_NLOOPS");
  if (nLoopsEnv != nullptr) {
    try {
      int val = std::stoi(nLoopsEnv);
      if (val >= 1) {
        ctx->searchNLoops = val;
        INFO(FLAGCX_ENV | FLAGCX_TUNING,
             "Tuner search nloops set by environment to %d.",
             ctx->searchNLoops);
      }
    } catch (const std::exception &e) {
      WARN("Invalid value for FLAGCX_TUNER_SEARCH_NLOOPS: %s. Using default.",
           nLoopsEnv);
    }
  }

  // initialize profilingResults pointer
  FLAGCXCHECK(flagcxCalloc(&ctx->profilingResults, nRanks));
  // start timer
  ctx->timer.start();
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerGetCandidateNumber(void *context,
                                             uint32_t *nCandidates) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  *nCandidates = ctx->configList.size();
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerSetCandidate(void *context, uint32_t index,
                                       struct flagcxCommTag *commTag) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  if (index >= ctx->configList.size()) {
    WARN("invalid index, index %u must less than config size %zu.", index,
         ctx->configList.size());
    return flagcxInvalidArgument;
  }
  // Set env for that communicator index
  const auto &curCfg = ctx->configList[index];
  FLAGCXCHECK(setEnvConfig(curCfg, FLAGCX_ENV_TYPE_CREATION));
  *commTag = curCfg.commTag;
  return flagcxSuccess;
}

// Given a startup phase seqId, get the corresponding communicator index in
// configList. Logic must be consistent with getSeqIdForCommIdx.
static int getCommIdxFromSeqId(const struct flagcxTunerContext *ctx,
                               uint32_t seqId) {
  if (ctx->activeCommList.size() == 0) {
    return -1;
  }
  return ctx->activeCommList[seqId / ctx->searchNLoops];
}

// Given a communicator index in configList, get the corresponding startup phase
// seqId for specific round. Logic must be consistent with getCommIdxFromSeqId.
static int getSeqIdForCommIdx(const struct flagcxTunerContext *ctx, int commIdx,
                              uint32_t round) {
  int seqId = 0;
  bool found = false;
  for (const auto &idx : ctx->activeCommList) {
    if (idx != commIdx) {
      seqId++;
    } else {
      found = true;
      break;
    }
  }
  return (found ? (seqId * ctx->searchNLoops) + round : -1);
}

// add a small factor to avoid switching between two close communicators caused
// by measurement noise
const float tunerProfileFactor = 0.95f;

// Helper function to find the best communicator for a collective category based
// on profiling data Strategy: For each active communicator, check if we have
// profiling data for that collective category. If yes, use that data to
// calculate the time for that collective category. If no, skip that
// communicator. Finally, select the communicator with the minimum time as the
// best communicator.
static flagcxResult_t findBestComm(struct flagcxTunerContext *ctx,
                                   const struct TunerCollCategory &cat) {
  int bestCommIdx = -1; // index of best communicator in configList
  float minTime = FLT_MAX;
  // calculate the best communicator based on profiling data
  // get the profiling data for the 2/3th round for comparison
  // i.e. if searchNLoops = 5, this would be round 2
  const uint32_t profileDataRound = ctx->searchNLoops * 2 / 3 - 1;
  for (const auto &idx : ctx->activeCommList) {
    int seqId = getSeqIdForCommIdx(
        ctx, idx,
        std::min(profileDataRound,
                 static_cast<uint32_t>(ctx->searchNLoops - 1)));
    TunerProfileKey profileKey(cat.nBytes, static_cast<uint32_t>(cat.collType),
                               static_cast<uint32_t>(seqId), idx);
    struct flagcxRecordKey<TunerProfileKey> rkey(profileKey);
    float duration = ctx->timer.getRecord(rkey, true);

    if (duration <= 0) {
      // no profiling data for this communicator and collective category
      WARN("No profiling data for (commId=%d,coll=%d,size=%zu,seq=%u).", idx,
           cat.collType, cat.nBytes, seqId);
      continue;
    }

    memcpy(ctx->profilingResults + ctx->rank, &duration, sizeof(float));
    // get average duration across all ranks
    FLAGCXCHECK(bootstrapAllGather(
        ctx->bootstrap, (void *)ctx->profilingResults, sizeof(float)));
    FLAGCXCHECK(bootstrapBarrier(ctx->bootstrap, ctx->rank, ctx->nranks, 0));
    duration = 0.0f;
    for (int i = 0; i < ctx->nranks; ++i) {
      duration += ctx->profilingResults[i];
    }
    duration /= ctx->nranks;

    INFO(FLAGCX_TUNING,
         "Profiling data for (commId=%d,coll=%d,size=%zu,seq=%u) is %.3fms.",
         idx, cat.collType, cat.nBytes, seqId, duration);

    if (duration < minTime * tunerProfileFactor) {
      minTime = duration;
      bestCommIdx = idx;
    }
  }
  if (bestCommIdx == -1) {
    WARN("No best communicator found for (coll=%d, size=%zu).", cat.collType,
         cat.nBytes);
    return flagcxInternalError;
  }

  const flagcxEnvConfig &bestConfig = ctx->configList[bestCommIdx];
  std::stringstream msg;
  msg << "Best Envs: ";
  for (int i = 0; i < bestConfig.envCount; i++) {
    msg << bestConfig.envs[i].name << "=" << bestConfig.envs[i].value
        << "(default=" << bestConfig.envs[i].defaultValue << ")";
    if (i < bestConfig.envCount - 1)
      msg << "  ";
  }
  // Output the best config
  INFO(FLAGCX_TUNING, "Find (coll=%d,size=%zu) best CommId=%d. %s",
       cat.collType, cat.nBytes, bestCommIdx, msg.str().c_str());

  ctx->collBestCommMap[cat] = bestCommIdx;
  return flagcxSuccess;
}

flagcxResult_t
flagcxCreateOrReplaceHomoComm(flagcxComm_t *comm,
                              struct flagcxTunerContext *ctx, uint32_t seqId,
                              const struct TunerCollCategory &collCat,
                              flagcxStream_t stream, bool createBest) {

  // If a communicator has already been created for the corresponding collCat in
  // comm->homoCommMap, delete it before creating a new one to ensure that each
  // collCat has only one communicator.
  auto it = (*comm)->homoCommMap.find(collCat);
  if (it != (*comm)->homoCommMap.end()) {
    // make sure all operations on the comm stream is done before destroying
    // communicator
    deviceAdaptor->streamSynchronize(stream);
    // Destroy Comm of collCat
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(it->second));
    // Remove entry from map
    (*comm)->homoCommMap.erase(it);
  }

  uint32_t nConfigs = 0;
  uint32_t idx = getCommIdxFromSeqId(ctx, seqId);
  struct flagcxCommTag tag = {""};
  FLAGCXCHECK(flagcxTunerSetCandidate((*comm)->tunerContext, idx, &tag));
  FLAGCXCHECK(
      (*comm)->tuner->getCandidateNumber((*comm)->tunerContext, &nConfigs));
  if (createBest) {
    INFO(FLAGCX_INIT | FLAGCX_TUNING,
         "create the communicator of the best Config (CommId = %d)",
         ctx->collBestCommMap[collCat]);
  } else {
    INFO(FLAGCX_INIT | FLAGCX_TUNING,
         "start to prepare communicator tag=%s(%u/%u)", tag.tag, idx, nConfigs);
  }

  flagcxInnerComm_t innerComm = NULL;
  FLAGCXCHECK(flagcxHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                 (struct bootstrapState *)(ctx->bootstrap),
                                 *comm, &innerComm));
  // Store new communicator of collCat into homoCommMap
  (*comm)->homoCommMap[collCat] = innerComm;
  // For backward compatible, also assign homo_comm field.
  (*comm)->homo_comm = innerComm;
  return flagcxSuccess;
}

// Communicator selection logic:
// 1) Honor environment override when ctx->envTagIdx is set.
// 2) Otherwise, for the initial searchNLoops * activeCommCount invocations of
//    each {collType, nBytes}, cycle through ctx->activeCommList via seqId
//    (tuning phase).
// 3) After the tuning window, rely on the best communicator recorded in
//    ctx->collBestCommMap (populated via profiling). If no best entry exists,
//    return flagcxInternalError.
flagcxResult_t flagcxTunerGetCollInfo(void *context, flagcxCommOp_t collType,
                                      size_t nBytes, int numPipeOps,
                                      float **collCostTable, int regBuff,
                                      struct flagcxCommTag *commTag,
                                      flagcxComm_t *comm,
                                      flagcxStream_t stream) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  // Use env comm tag when possible.
  if (ctx->envTagIdx != -1) {
    FLAGCXCHECK(
        setEnvConfig(ctx->configList[ctx->envTagIdx], FLAGCX_ENV_TYPE_COLL));
    *commTag = ctx->configList[ctx->envTagIdx].commTag;
    INFO(FLAGCX_TUNING, "Use Communicator tag %s set by environment.",
         commTag->tag);
    return flagcxSuccess;
  }

  // get a seqId for {collType, nBytes}
  struct TunerCollCategory collCat = {collType, nBytes};
  auto it = ctx->collSeqMap.find(collCat);
  uint32_t seqId = 0;
  if (it == ctx->collSeqMap.end()) {
    ctx->collSeqMap[collCat] = 0;
  } else {
    it->second++;
    seqId = it->second;
  }

  if (seqId < ctx->searchNLoops * ctx->activeCommList.size()) {

    // Every {collType, nBytes, commTagIdx} will be profiled searchNLoops times.
    int cfgIdx = getCommIdxFromSeqId(ctx, seqId);
    if (cfgIdx == -1) {
      WARN("No active communicator found for startup phase seqId=%u.", seqId);
      return flagcxInternalError;
    }
    TunerCommTagCounterKey key{nBytes, static_cast<uint32_t>(collType),
                               static_cast<uint32_t>(cfgIdx)};
    auto cit = ctx->configCounterMap.find(key);
    if (cit == ctx->configCounterMap.end()) {
      // create a new communicator and destroy old communicator
      FLAGCXCHECK(flagcxCreateOrReplaceHomoComm(comm, ctx, seqId, collCat,
                                                stream, false));
      (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
      ctx->configCounterMap[key] = 1;
    } else {
      // use old communicator
      (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
      ctx->configCounterMap[key]++;
    }
    const auto &cfg = ctx->configList[cfgIdx];
    *commTag = cfg.commTag;
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_COLL));
    return flagcxSuccess;
  }

  // Select a communicator from active communicators based on profiling data
  // after searchNLoops * activeCommCount collectives. If we do not have a best
  // communicator recorded for this collective category, find it.
  if ((*comm)->homoBestCommMap[collCat] == nullptr) {
    // Find the best config
    FLAGCXCHECK(findBestComm(ctx, collCat));
    // Check whether the optimal config has been found; if not, return an error.
    auto it2 = ctx->collBestCommMap.find(collCat);
    if (it2 == ctx->collBestCommMap.end()) {
      WARN("No best communicator found for collective type %d with size %zu.",
           collType, nBytes);
      return flagcxInternalError;
    }
    // If the optimal config has been found, create a communicator of best
    // config
    const uint32_t profileDataRound = PROFILE_ROUND;
    uint32_t bestSeqId = getSeqIdForCommIdx(
        ctx, it2->second,
        std::min(profileDataRound,
                 static_cast<uint32_t>(ctx->searchNLoops - 1)));
    FLAGCXCHECK(flagcxCreateOrReplaceHomoComm(comm, ctx, bestSeqId, collCat,
                                              stream, true));
    auto &cfg = ctx->configList[it2->second];
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_COLL));
    *commTag = cfg.commTag;
    (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
    // Store the best communicator of collCat into homoBestCommMap
    (*comm)->homoBestCommMap[collCat] = (*comm)->homoCommMap[collCat];
  } else {
    // The best communicator has been created
    // get it in collBestCommMap directly
    auto it2 = ctx->collBestCommMap.find(collCat);
    if (it2 == ctx->collBestCommMap.end()) {
      WARN("No best communicator found for collective type %d with size %zu.",
           collType, nBytes);
      return flagcxInternalError;
    }
    auto &cfg = ctx->configList[it2->second];
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_COLL));
    *commTag = cfg.commTag;
    (*comm)->tunerInnerComm = (*comm)->homoBestCommMap[collCat];
    INFO(FLAGCX_TUNING,
         "Use Communicator tag %s based on profile data, seqId=%d.",
         commTag->tag, seqId);
  }
  return flagcxSuccess;
}

// Handle flagscale tuning logic
// This function processes flagscale tuning configuration:
// 1. Matches the collective operation and size against tune_objects to
// determine
//    if this comm needs tuning (sets comm->isTunningComm)
// 2. Reads FLAGCX_TUNER_CONFIG_ID and FLAGCX_TUNER_BEST_CONFIG_ID from
// environment
// 3. Switches communicator config if config_id increments by 1
// Returns flagcxSuccess on success, flagcxInternalError if config_id is invalid
flagcxResult_t flagcxHandleFlagscaleTuning(void *context, flagcxComm_t comm,
                                           flagcxCommOp_t commOp,
                                           size_t nBytes) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  // Execute matching only once when tune_objects has values
  const char *configIdEnv = getenv("FLAGCX_TUNER_CONFIG_ID");
  const int config_id = (configIdEnv != NULL) ? atoi(configIdEnv) : -1;
  // static bool matchingDone = false;
  if (!ctx->tunerCommMatchingDone && config_id == 0) {
    // Determine if this comm needs tuning
    FlagScaleConfig config = readFlagScaleJson();
    if (!config.tune_objects.empty()) {
      bool isTuningComm = false;
      flagcxCommOp_t currentCommOp = commOp;
      INFO(FLAGCX_TUNING, "flagcxTuner finding match for commOp=%d, nBytes=%zu",
           currentCommOp, nBytes);
      for (size_t idx = 0; idx < config.tune_objects.size(); ++idx) {
        const TuneObject &item = config.tune_objects[idx];
        std::string opStr = getTuneObjectCommOp(item);
        flagcxCommOp_t tuneCommOp = commOpStringToEnum(opStr);
        if (tuneCommOp == currentCommOp && item.nBytes == (int64_t)nBytes) {
          isTuningComm = true;
          break;
        }
      }
      comm->isTunningComm = isTuningComm;
      ctx->tunerCommMatchingDone = true;
    }
  }
  // If not tuning this comm, directly return
  if (!comm->isTunningComm) {
    return flagcxSuccess;
  }

  INFO(FLAGCX_TUNING, "comm->isTunningComm=%d", comm->isTunningComm);

  // Need tuning this comm
  // Handle config_id logic
  // static int lastFlagscaleConfigId = -1;
  const char *bestConfigIdEnv = getenv("FLAGCX_TUNER_BEST_CONFIG_ID");
  const int bestConfigId =
      (bestConfigIdEnv != NULL) ? atoi(bestConfigIdEnv) : -1;

  // if config_id is -1, use the default communicator config
  if (config_id == -1) {
    return flagcxSuccess;
  }

  // if config_id is greater than lastFlagscaleConfigId by 1,
  // switch to the new communicator config
  if (config_id - ctx->lastFlagscaleConfigId == 1) {
    ctx->lastFlagscaleConfigId = config_id;
    INFO(FLAGCX_TUNING, "call switchCommConfig with config_id=%d", config_id);
    FLAGCXCHECK(
        comm->tuner->switchCommConfig(comm->tunerContext, &comm, bestConfigId));
    return flagcxSuccess;
  }

  // if config_id is equal to lastFlagscaleConfigId, don't switch communicator
  // config
  if (config_id - ctx->lastFlagscaleConfigId == 0) {
    return flagcxSuccess; // Should call call() and return
  }

  // Invalid config_id
  WARN("config_id=%d is invalid", config_id);
  return flagcxInternalError;
}

flagcxResult_t flagcxTunerSwitchCommConfig(void *context, flagcxComm_t *comm,
                                           int bestConfigId) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);

  if (ctx->commConfigId < ctx->configList.size()) {
    if (bestConfigId != -1) {
      WARN("bestConfigId=%d is not -1, but commConfigId=%d is less than "
           "configList.size()=%zu",
           bestConfigId, ctx->commConfigId, ctx->configList.size());
      return flagcxInternalError;
    }

    const auto &cfg = ctx->configList[ctx->commConfigId];
    auto inner = (*comm)->tunerInnerComm;
    if (inner == nullptr) {
      WARN("comm->tunerInnerComm is null");
      return flagcxInternalError;
    }

    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(inner));
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_CREATION));
    flagcxInnerComm_t newInner = NULL;
    FLAGCXCHECK(flagcxHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                   (struct bootstrapState *)(ctx->bootstrap),
                                   *comm, &newInner));
    (*comm)->tunerInnerComm = newInner;
    (*comm)->homo_comm = newInner;
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_COLL));
    ctx->commConfigId += 1;
    // if all communicator configurations have been tested, set the environment
    // variable FLAGCX_TUNER_DONE to 1
    if (ctx->commConfigId >= ctx->configList.size()) {
      setenv("FLAGCX_TUNER_DONE", "1", 1);
      INFO(FLAGCX_TUNING,
           "Tuning completed: all %zu communicator configurations have been "
           "tested. ENV FLAGCX_TUNER_DONE=%s",
           ctx->configList.size(), getenv("FLAGCX_TUNER_DONE"));
    }
    return flagcxSuccess;
  }

  if (bestConfigId != -1 && ctx->bestConfigId == -1) {
    ctx->bestConfigId = bestConfigId;
    const auto &cfg = ctx->configList[ctx->bestConfigId];
    auto inner = (*comm)->tunerInnerComm;
    if (inner == nullptr) {
      WARN("comm->tunerInnerComm is null");
      return flagcxInternalError;
    }

    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(inner));
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_CREATION));
    flagcxInnerComm_t newInner = NULL;
    FLAGCXCHECK(flagcxHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                   (struct bootstrapState *)(ctx->bootstrap),
                                   *comm, &newInner));
    (*comm)->tunerInnerComm = newInner;
    (*comm)->homo_comm = newInner;
    FLAGCXCHECK(setEnvConfig(cfg, FLAGCX_ENV_TYPE_COLL));
    std::stringstream msg;
    msg << "Best Envs: ";
    for (int i = 0; i < cfg.envCount; i++) {
      msg << cfg.envs[i].name << "=" << cfg.envs[i].value
          << "(default=" << cfg.envs[i].defaultValue << ")";
      if (i < cfg.envCount - 1) {
        msg << "  ";
      }
    }
    INFO(FLAGCX_TUNING, "switch to the best config, config_id=%d. %s",
         ctx->bestConfigId, msg.str().c_str());
    return flagcxSuccess;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerStartProfiling(void *context, flagcxCommOp_t collType,
                                         size_t nBytes, flagcxStream_t stream,
                                         const struct flagcxCommTag *commTag,
                                         struct flagcxProfileKey *key) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  struct TunerCollCategory collCat = {collType, nBytes};

  auto it = ctx->collSeqMap.find(collCat);
  uint32_t seqId = 0;
  if (it != ctx->collSeqMap.end()) {
    seqId = it->second;
  } else {
    WARN("Collective category (coll=%d,size=%zu) not found in collSeqMap.",
         collType, nBytes);
    return flagcxInvalidArgument;
  }

  auto it2 = ctx->commTagIdxMap.find(*commTag);
  if (it2 == ctx->commTagIdxMap.end()) {
    WARN("Communicator tag %s not found in config list.", commTag->tag);
    return flagcxInvalidArgument;
  }
  uint32_t commTagIdx = it2->second;

  // Always generate the key, even if we do not do profiling for this
  // collective.
  TunerProfileKey profileKey(nBytes, static_cast<uint32_t>(collType), seqId,
                             commTagIdx);
  /*
  INFO(FLAGCX_TUNING, "Enter StartProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  *key = profileKey;

  // do profile only for startup collectives
  if (seqId < ctx->searchNLoops * ctx->activeCommList.size()) {
    struct flagcxRecordKey<TunerProfileKey> rkey(profileKey);
    FLAGCXCHECK(ctx->timer.begin(rkey, stream));
  }
  /*
  INFO(FLAGCX_TUNING, "Leave StartProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerStopProfiling(void *context,
                                        const struct flagcxProfileKey *key) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  TunerProfileKey profileKey(*key);
  /*
  INFO(FLAGCX_TUNING, "Enter StopProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  // do profile only for startup collectives
  if (profileKey.seqId < ctx->searchNLoops * ctx->activeCommList.size()) {
    struct flagcxRecordKey<TunerProfileKey> rkey(profileKey);
    FLAGCXCHECK(ctx->timer.end(rkey));
  }
  /*
  INFO(FLAGCX_TUNING, "Leave StopProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  return flagcxSuccess;
}

flagcxResult_t flagcxTunerDestroy(void *context) {
  struct flagcxTunerContext *ctx =
      static_cast<struct flagcxTunerContext *>(context);
  // INFO(FLAGCX_TUNING, "Enter flagcxTunerDestroy.");

  // stop timer
  ctx->timer.stop();
  free(ctx->profilingResults);
  delete ctx;
  return flagcxSuccess;
}

flagcxTuner_t internalTuner = {"internal tuner",
                               flagcxTunerInit,
                               flagcxTunerGetCandidateNumber,
                               flagcxTunerSetCandidate,
                               flagcxTunerGetCollInfo,
                               flagcxTunerStartProfiling,
                               flagcxTunerStopProfiling,
                               flagcxTunerDestroy,
                               flagcxCreateOrReplaceHomoComm,
                               flagcxTunerSwitchCommConfig,
                               flagcxHandleFlagscaleTuning};
