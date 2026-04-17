#include "uni_runner_impl.h"
#include "adaptor.h"
#include "coll_proxy_transport.h"
#include "comm.h"
#include "device_api/completion_word.h"
#include "flagcx_hetero.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
#include "runner_result.h"
#include "socket.h"
#include "transport.h"
#include "uni_runner_helper.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdint>
#include <math.h>
#include <new>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

FLAGCX_PARAM(UniRunnerNSlices, "UNIRUNNER_NSLICES", 1);
FLAGCX_PARAM(UniRunnerNThreads, "UNIRUNNER_NTHREADS", 32);
FLAGCX_PARAM(UniRunnerNBlocks, "UNIRUNNER_NBLOCKS", 1);
FLAGCX_PARAM(UniRunnerNRedSlices, "UNIRUNNER_NREDSLICES", 0);
FLAGCX_PARAM(UniRunnerRedSliceSize, "UNIRUNNER_REDSLICESIZE", 65536);

namespace {

constexpr char kUniRunnerDagCachePathEnv[] = "FLAGCX_UNIRUNNER_DAG_CACHE_PATH";

struct uniRunnerDagRuntimeBindings {
  const void *inputBase = NULL;
  void *outputBase = NULL;
  void *scratchBase = NULL;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t scratchBytes = 0;
};

std::mutex gUniRunnerDagCacheMutex;
std::unordered_map<size_t, std::vector<std::shared_ptr<uniRunnerDagTemplate>>>
    gUniRunnerDagCache;
std::unordered_set<std::string> gUniRunnerDagLoadedPaths;

static size_t hashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

static bool uniRunnerDagCacheKeysEqual(const uniRunnerDagCacheKey &lhs,
                                       const uniRunnerDagCacheKey &rhs) {
  return lhs.formatVersion == rhs.formatVersion &&
         lhs.algoType == rhs.algoType && lhs.commOp == rhs.commOp &&
         lhs.count == rhs.count && lhs.datatype == rhs.datatype &&
         lhs.redOp == rhs.redOp && lhs.rank == rhs.rank &&
         lhs.nranks == rhs.nranks && lhs.root == rhs.root &&
         lhs.groupSize == rhs.groupSize && lhs.numSlices == rhs.numSlices &&
         lhs.numRedSlices == rhs.numRedSlices &&
         lhs.redSliceSize == rhs.redSliceSize && lhs.nthreads == rhs.nthreads &&
         lhs.inputOutputAliased == rhs.inputOutputAliased &&
         lhs.inputScratchAliased == rhs.inputScratchAliased &&
         lhs.outputScratchAliased == rhs.outputScratchAliased;
}

static flagcxResult_t insertUniRunnerDagTemplateLocked(
    const std::shared_ptr<uniRunnerDagTemplate> &dagTemplate) {
  std::vector<std::shared_ptr<uniRunnerDagTemplate>> &bucket =
      gUniRunnerDagCache[dagTemplate->hashValue];
  for (std::shared_ptr<uniRunnerDagTemplate> &entry : bucket) {
    if (uniRunnerDagCacheKeysEqual(entry->key, dagTemplate->key)) {
      entry = dagTemplate;
      return flagcxSuccess;
    }
  }
  bucket.push_back(dagTemplate);
  return flagcxSuccess;
}

static bool findUniRunnerDagTemplateLocked(
    size_t hashValue, const uniRunnerDagCacheKey &key,
    std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::unordered_map<
      size_t, std::vector<std::shared_ptr<uniRunnerDagTemplate>>>::iterator it =
      gUniRunnerDagCache.find(hashValue);
  if (it == gUniRunnerDagCache.end()) {
    return false;
  }
  for (const std::shared_ptr<uniRunnerDagTemplate> &entry : it->second) {
    if (uniRunnerDagCacheKeysEqual(entry->key, key)) {
      *dagTemplate = entry;
      return true;
    }
  }
  return false;
}

static flagcxResult_t importUniRunnerDagTemplatesLocked(
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  for (const uniRunnerDagTemplate &dagTemplate : dagTemplates) {
    FLAGCXCHECK(insertUniRunnerDagTemplateLocked(
        std::make_shared<uniRunnerDagTemplate>(dagTemplate)));
  }
  return flagcxSuccess;
}

static std::string getUniRunnerDagEnvPath(const char *envName) {
  const char *path = flagcxGetEnv(envName);
  return path == NULL ? "" : std::string(path);
}

static std::string getUniRunnerDagCacheDirFromEnv() {
  return getUniRunnerDagEnvPath(kUniRunnerDagCachePathEnv);
}

static std::string makeUniRunnerDagFileName(size_t hashValue, int rank) {
  return "dag_hash_" + std::to_string(hashValue) + "_rank_" +
         std::to_string(rank) + ".json";
}

static std::string joinUniRunnerDagDirAndFile(const std::string &dir,
                                              const std::string &fileName) {
  if (dir.empty()) {
    return "";
  }
  if (dir.back() == '/') {
    return dir + fileName;
  }
  return dir + "/" + fileName;
}

static std::string makeUniRunnerDagFilePath(const std::string &dir,
                                            size_t hashValue, int rank) {
  return joinUniRunnerDagDirAndFile(dir,
                                    makeUniRunnerDagFileName(hashValue, rank));
}

static flagcxResult_t ensureUniRunnerDagDirExists(const std::string &dir) {
  if (dir.empty()) {
    return flagcxSuccess;
  }

  std::string current;
  if (dir[0] == '/') {
    current = "/";
  }

  size_t pos = 0;
  while (pos < dir.size()) {
    size_t next = dir.find('/', pos);
    std::string segment = dir.substr(
        pos, next == std::string::npos ? std::string::npos : next - pos);
    if (!segment.empty()) {
      if (!current.empty() && current.back() != '/') {
        current += "/";
      }
      current += segment;

      struct stat st;
      if (stat(current.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
          return flagcxSystemError;
        }
      } else if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        return flagcxSystemError;
      }
    }
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return flagcxSuccess;
}

static flagcxResult_t loadUniRunnerDagFileIntoCache(const std::string &path) {
  if (path.empty()) {
    return flagcxSuccess;
  }

  {
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    if (gUniRunnerDagLoadedPaths.count(path) != 0) {
      return flagcxSuccess;
    }
  }

  std::vector<uniRunnerDagTemplate> dagTemplates;
  FLAGCXCHECK(uniRunnerLoadDagJsonFileIfExists(path, &dagTemplates));
  if (dagTemplates.empty()) {
    return flagcxSuccess;
  }

  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  if (gUniRunnerDagLoadedPaths.count(path) != 0) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(importUniRunnerDagTemplatesLocked(dagTemplates));
  gUniRunnerDagLoadedPaths.insert(path);
  return flagcxSuccess;
}

static bool matchBindingRange(const void *ptr, const void *base, size_t bytes,
                              int64_t *offsetBytes) {
  if (ptr == NULL || base == NULL) {
    return false;
  }
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(base);
  if (ptrAddr < baseAddr) {
    return false;
  }
  uintptr_t delta = ptrAddr - baseAddr;
  if (delta > bytes) {
    return false;
  }
  *offsetBytes = static_cast<int64_t>(delta);
  return true;
}

static flagcxResult_t
captureBufferRef(const void *ptr, const uniRunnerDagRuntimeBindings &bindings,
                 uniRunnerDagBufferRef *ref) {
  if (ptr == NULL) {
    ref->bufferType = uniRunnerDagBufferTypeNone;
    ref->offsetBytes = 0;
    return flagcxSuccess;
  }

  int64_t offsetBytes = 0;
  if (matchBindingRange(ptr, bindings.inputBase, bindings.inputBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeInput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (matchBindingRange(ptr, bindings.outputBase, bindings.outputBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeOutput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (matchBindingRange(ptr, bindings.scratchBase, bindings.scratchBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeScratch;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  return flagcxInvalidArgument;
}

static flagcxResult_t
resolveBufferRef(const uniRunnerDagBufferRef &ref,
                 const uniRunnerDagRuntimeBindings &bindings, void **ptr) {
  const char *base = NULL;
  size_t bytes = 0;
  switch (ref.bufferType) {
    case uniRunnerDagBufferTypeNone:
      *ptr = NULL;
      return flagcxSuccess;
    case uniRunnerDagBufferTypeInput:
      base = static_cast<const char *>(bindings.inputBase);
      bytes = bindings.inputBytes;
      break;
    case uniRunnerDagBufferTypeOutput:
      base = static_cast<const char *>(bindings.outputBase);
      bytes = bindings.outputBytes;
      break;
    case uniRunnerDagBufferTypeScratch:
      base = static_cast<const char *>(bindings.scratchBase);
      bytes = bindings.scratchBytes;
      break;
    default:
      return flagcxInvalidArgument;
  }

  if (base == NULL || ref.offsetBytes < 0 ||
      static_cast<size_t>(ref.offsetBytes) > bytes) {
    return flagcxInvalidArgument;
  }

  *ptr = const_cast<char *>(base) + ref.offsetBytes;
  return flagcxSuccess;
}

static size_t
resolveEffectiveUniRunnerRedSlices(const flagcxUniRunnerState *runnerState,
                                   size_t count, int nranks) {
  if (runnerState->uniRunnerNRedSlices != 0) {
    return runnerState->uniRunnerNRedSlices;
  }
  if (count == 0 || runnerState->uniRunnerRedSliceSize == 0 ||
      runnerState->uniRunnerNSlices == 0 || nranks <= 0) {
    return 1;
  }

  size_t divisor = static_cast<size_t>(nranks) * runnerState->uniRunnerNSlices *
                   runnerState->uniRunnerRedSliceSize;
  if (divisor == 0) {
    return 1;
  }
  return std::max<size_t>(1, (count + divisor - 1) / divisor);
}

static uniRunnerDagCacheKey makeUniRunnerDagCacheKey(
    uniRunnerDagAlgoType algoType, flagcxCommOp_t commOp, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, int root, int groupSize,
    flagcxUniRunnerState *runnerState, flagcxComm_t comm, const void *sendbuff,
    void *recvbuff, void *scratchbuff) {
  uniRunnerDagCacheKey key{};
  key.formatVersion = kUniRunnerDagCacheFormatVersion;
  key.algoType = algoType;
  key.commOp = commOp;
  key.count = count;
  key.datatype = datatype;
  key.redOp = redOp;
  key.rank = comm->rank;
  key.nranks = comm->nranks;
  key.root = root;
  key.groupSize = groupSize;
  key.numSlices = runnerState->uniRunnerNSlices;
  key.numRedSlices = runnerState->uniRunnerNRedSlices;
  key.redSliceSize = runnerState->uniRunnerRedSliceSize;
  key.nthreads = runnerState->uniRunnerNThreads;
  key.inputOutputAliased = sendbuff == recvbuff;
  key.inputScratchAliased = sendbuff == scratchbuff;
  key.outputScratchAliased = recvbuff == scratchbuff;
  return key;
}

static bool
findUniRunnerDagTemplate(size_t hashValue, const uniRunnerDagCacheKey &key,
                         std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  return findUniRunnerDagTemplateLocked(hashValue, key, dagTemplate);
}

static flagcxResult_t loadUniRunnerDagFromCacheDir(size_t hashValue, int rank) {
  std::string cacheDir = getUniRunnerDagCacheDirFromEnv();
  if (cacheDir.empty()) {
    return flagcxSuccess;
  }
  return loadUniRunnerDagFileIntoCache(
      makeUniRunnerDagFilePath(cacheDir, hashValue, rank));
}

static flagcxResult_t cacheUniRunnerDagTemplate(
    const std::shared_ptr<uniRunnerDagTemplate> &dagTemplate) {
  size_t hashValue = getUniRunnerDagPatternHash(dagTemplate->key);
  std::string dagDir = getUniRunnerDagCacheDirFromEnv();
  std::string dagPath =
      makeUniRunnerDagFilePath(dagDir, hashValue, dagTemplate->key.rank);
  {
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    FLAGCXCHECK(insertUniRunnerDagTemplateLocked(dagTemplate));
  }
  if (dagDir.empty()) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(ensureUniRunnerDagDirExists(dagDir));
  FLAGCXCHECK(uniRunnerSaveDagJsonFile(dagPath, *dagTemplate));
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  gUniRunnerDagLoadedPaths.insert(dagPath);
  return flagcxSuccess;
}

} // namespace

size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key) {
  size_t hashValue = 0;
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.formatVersion));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.algoType));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.commOp));
  hashValue = hashCombine(hashValue, key.count);
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.datatype));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redOp));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.rank));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nranks));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.root + 1));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.groupSize + 1));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.numSlices));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.numRedSlices));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redSliceSize));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nthreads));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.inputOutputAliased));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.inputScratchAliased));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.outputScratchAliased));
  return hashValue;
}

static flagcxResult_t allocDagNodeDeps(uniRunnerDagNode *node) {
  node->pendingParents = 0;
  if (node->numParents > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->parents, node->numParents * sizeof(int)));
  }
  if (node->numChildren > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->children, node->numChildren * sizeof(int)));
  }
  return flagcxSuccess;
}

static flagcxResult_t setDagNodeParent(uniRunnerDagNode *node, int parentSlot,
                                       int parentIdx) {
  if (parentSlot < 0 || parentSlot >= node->numParents ||
      node->parents == NULL) {
    return flagcxInternalError;
  }
  node->parents[parentSlot] = parentIdx;
  node->pendingParents++;
  return flagcxSuccess;
}

// Validate that DAG construction filled every declared parent slot.
static flagcxResult_t validateDagNodes(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL || runnerState->dagNodes == NULL ||
      runnerState->numDagNodes == 0) {
    return flagcxSuccess;
  }

  const int numDagNodes = runnerState->numDagNodes;
  uniRunnerDagNode *dagNodes = runnerState->dagNodes;
  size_t numEdges = 0;

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    if (node->pendingParents != node->numParents) {
      return flagcxInternalError;
    }
    if (node->numParents < 0 || node->numChildren < 0) {
      return flagcxInternalError;
    }
    if ((node->numParents > 0 && node->parents == NULL) ||
        (node->numChildren > 0 && node->children == NULL)) {
      return flagcxInternalError;
    }
    numEdges += static_cast<size_t>(node->numParents);
  }

  std::unordered_set<uint64_t> dagEdges;
  dagEdges.reserve(numEdges);

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    for (int p = 0; p < node->numParents; p++) {
      int parentIdx = node->parents[p];
      if (parentIdx < 0 || parentIdx >= numDagNodes || parentIdx == i) {
        return flagcxInternalError;
      }
      uint64_t edge =
          (static_cast<uint64_t>(static_cast<uint32_t>(parentIdx)) << 32) |
          static_cast<uint32_t>(i);
      if (!dagEdges.emplace(edge).second) {
        return flagcxInternalError;
      }
    }
  }

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    for (int c = 0; c < node->numChildren; c++) {
      int childIdx = node->children[c];
      if (childIdx < 0 || childIdx >= numDagNodes || childIdx == i) {
        return flagcxInternalError;
      }
      uint64_t edge = (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
                      static_cast<uint32_t>(childIdx);
      std::unordered_set<uint64_t>::iterator it = dagEdges.find(edge);
      if (it == dagEdges.end()) {
        return flagcxInternalError;
      }
      dagEdges.erase(it);
    }
  }

  return dagEdges.empty() ? flagcxSuccess : flagcxInternalError;
}

static inline void *getDagNodeFlag(flagcxUniRunnerState *runnerState,
                                   int nodeIdx) {
  assert(nodeIdx >= 0 &&
         static_cast<size_t>(nodeIdx) < runnerState->streamFlagsSize);
  return runnerState->streamFlags[nodeIdx];
}

static void refreshStreamFlagAddressQueue(flagcxUniRunnerState *runnerState) {
  if (runnerState->streamFlagsPool == NULL ||
      runnerState->streamFlags == NULL) {
    return;
  }

  char *base = static_cast<char *>(runnerState->streamFlagsPool);
  for (size_t i = 0; i < runnerState->streamFlagsCapacity; ++i) {
    runnerState->streamFlags[i] =
        static_cast<void *>(base + i * sizeof(uint64_t));
  }
}

static flagcxResult_t
resizeStreamFlagAddressQueue(flagcxUniRunnerState *runnerState,
                             size_t newCapacity) {
  if (newCapacity <= runnerState->streamFlagsCapacity) {
    return flagcxSuccess;
  }

  if (runnerState->streamFlagsCapacity == 0) {
    if (runnerState->streamFlags != NULL) {
      free(runnerState->streamFlags);
      runnerState->streamFlags = NULL;
    }
    FLAGCXCHECK(flagcxCalloc(&runnerState->streamFlags, newCapacity));
  } else {
    FLAGCXCHECK(flagcxRealloc(&runnerState->streamFlags,
                              runnerState->streamFlagsCapacity, newCapacity));
  }

  return flagcxSuccess;
}

static flagcxResult_t
ensureStreamFlagQueueCapacity(flagcxUniRunnerState *runnerState,
                              size_t requiredFlags) {
  if (requiredFlags > runnerState->streamFlagsCapacity) {
    size_t newCapacity = runnerState->streamFlagsCapacity == 0
                             ? 1
                             : runnerState->streamFlagsCapacity;
    while (newCapacity < requiredFlags) {
      newCapacity *= 2;
    }

    FLAGCXCHECK(resizeStreamFlagAddressQueue(runnerState, newCapacity));

    void *newPool = NULL;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(
        &newPool, newCapacity * sizeof(uint64_t), flagcxMemDevice, NULL));
    if (runnerState->streamFlagsPool != NULL) {
      FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                            flagcxMemDevice, NULL));
    }

    runnerState->streamFlagsPool = newPool;
    runnerState->streamFlagsCapacity = newCapacity;
    refreshStreamFlagAddressQueue(runnerState);
  }

  return flagcxSuccess;
}

static flagcxResult_t prepareDagStreamFlags(flagcxUniRunnerState *runnerState) {
  size_t activeFlags = runnerState->numDagNodes > 0
                           ? static_cast<size_t>(runnerState->numDagNodes)
                           : 0;
  FLAGCXCHECK(ensureStreamFlagQueueCapacity(runnerState, activeFlags));
  runnerState->streamFlagsSize = activeFlags;

  if (runnerState->streamFlagsSize != 0) {
    FLAGCXCHECK(deviceAdaptor->deviceMemset(runnerState->streamFlagsPool, 0,
                                            runnerState->streamFlagsSize *
                                                sizeof(uint64_t),
                                            flagcxMemDevice, NULL));
  }

  return flagcxSuccess;
}

static flagcxResult_t
destroyStreamFlagQueue(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL) {
    return flagcxSuccess;
  }

  if (runnerState->streamFlagsPool != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                          flagcxMemDevice, NULL));
    runnerState->streamFlagsPool = NULL;
  }

  if (runnerState->streamFlags != NULL) {
    free(runnerState->streamFlags);
    runnerState->streamFlags = NULL;
  }
  runnerState->streamFlagsSize = 0;
  runnerState->streamFlagsCapacity = 0;
  return flagcxSuccess;
}

static inline flagcxStream_t
getDagNodeExecutionStream(flagcxUniRunnerState *runnerState,
                          const uniRunnerDagNode *node) {
  switch (node->nodeType) {
    case uniRunnerDagNodeTypeP2p:
      return runnerState->commStream;
    case uniRunnerDagNodeTypeRed:
      return runnerState->redStream;
    case uniRunnerDagNodeTypeCpy:
      return runnerState->cpyStream;
    default:
      return NULL;
  }
}

flagcxResult_t initUniRunnerStateDummy(flagcxUniRunnerState *runnerState) {
  return flagcxNotSupported;
}

static flagcxResult_t
buildUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 2) {
    return flagcxSuccess;
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateLocRed called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;
  size_t rankChunkCount =
      baseRankChunkCount + (rank < (int)rankChunkRemainder ? 1 : 0);

  const int numNodes = numSlices;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int s = 0; s < numSlices; s++) {
    size_t baseSliceCount = rankChunkCount / numSlices;
    size_t sliceRemainder = rankChunkCount % numSlices;
    // Calculate slice count with uneven distribution
    size_t sliceCount = baseSliceCount;
    if (s < sliceRemainder) {
      sliceCount++;
    }
    size_t sliceOffsetInChunk = s * baseSliceCount * typeSize;
    // Add offset for all previous slices that got the remainder
    sliceOffsetInChunk += std::min(s, (int)sliceRemainder) * typeSize;
    // Calculate offset accounting for rankChunkRemainder
    // First rankChunkRemainder ranks each have one extra element
    size_t rxOffset =
        (rank * baseRankChunkCount + std::min(rank, (int)rankChunkRemainder)) *
            typeSize +
        sliceOffsetInChunk;

    // Reduce Node
    int redNodeIdx = s;
    runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
    runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
    runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
    runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.input2 = static_cast<void *>(
        static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.output =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
    runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
        runnerState->uniRunnerNThreads;
    runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

    // Setup dependencies linearly within the slice chain
    runnerState->dagNodes[redNodeIdx].numParents = 0;
    runnerState->dagNodes[redNodeIdx].numChildren = 0;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
    // Enqueue the head of this slice chain to Ready Queue
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[redNodeIdx]);
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateGroupedAG(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxComm_t comm, int groupSize) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (groupSize <= 0 || groupSize > nranks || nranks % groupSize != 0) {
    return flagcxInvalidArgument;
  }

  int nGroups = nranks / groupSize;
  int groupIdx = rank / groupSize;
  int locRank = rank % groupSize;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateGroupedAG called, count=%lu, groupSize=%d, "
        "nGroups=%d",
        rank, count, groupSize, nGroups);

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t groupChunkCount = count * groupSize;
  const int numNodes = nGroups + 1;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  size_t localBaseOffset =
      static_cast<size_t>(groupIdx) * groupChunkCount * typeSize;
  for (int step = 0; step < nGroups; step++) {
    int nodeIdx = step;
    bool isLastStep = (step == nGroups - 1);
    int numIntraPeers = groupSize - 1;
    int numOps = isLastStep ? 2 * numIntraPeers : 2 * numIntraPeers + 2;

    runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
    runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
    runnerState->dagNodes[nodeIdx].nodeData.p2p.numOps = numOps;
    if (numOps > 0) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[nodeIdx].nodeData.p2p.ops,
                               numOps * sizeof(struct uniRunnerP2pOpData)));
    }

    for (int i = 0; i < numIntraPeers; i++) {
      int locSendPeer = (locRank + i + 1) % groupSize;
      int locRecvPeer = (locRank - i - 1 + groupSize) % groupSize;

      // Send
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].peerRank =
          groupIdx * groupSize + locSendPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].addr =
          step == 0 ? const_cast<void *>(sendbuff)
                    : static_cast<void *>(static_cast<char *>(recvbuff) +
                                          localBaseOffset +
                                          locRank * count * typeSize);
      // Recv
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].peerRank =
          groupIdx * groupSize + locRecvPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + localBaseOffset +
                              locRecvPeer * count * typeSize);
      TRACE(FLAGCX_UNIRUNNER,
            "Node %d: intra-group step %d, sendPeer=%d, recvPeer=%d, "
            "sendOffset=%lu, recvOffset=%lu",
            nodeIdx, i, groupIdx * groupSize + locSendPeer,
            groupIdx * groupSize + locRecvPeer,
            localBaseOffset + locRank * count * typeSize,
            localBaseOffset + locRecvPeer * count * typeSize);
    }

    if (!isLastStep) {
      size_t sendGroupIdx = (groupIdx + step + 1) % nGroups;
      size_t recvGroupIdx = (groupIdx - step - 1 + nGroups) % nGroups;
      size_t sendPeer = sendGroupIdx * groupSize + locRank;
      size_t recvPeer = recvGroupIdx * groupSize + locRank;
      size_t recvOffset = recvPeer * count * typeSize;
      int opIdx = 2 * numIntraPeers;

      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].peerRank =
          sendPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].addr =
          step == 0 ? const_cast<void *>(sendbuff)
                    : static_cast<void *>(static_cast<char *>(recvbuff) +
                                          localBaseOffset +
                                          locRank * count * typeSize);
      opIdx++;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].peerRank =
          recvPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + recvOffset);

      TRACE(FLAGCX_UNIRUNNER,
            "Node %d: inter-group step %d, sendPeer=%lu, recvPeer=%lu, "
            "recvOffset=%lu",
            nodeIdx, step, sendPeer, recvPeer, recvOffset);

      localBaseOffset = recvGroupIdx * groupChunkCount * typeSize;
    }
  }

  int nodeIdx = numNodes - 1;
  runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
  runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeCpy;
  runnerState->dagNodes[nodeIdx].nodeData.cpy.src =
      const_cast<void *>(sendbuff);
  runnerState->dagNodes[nodeIdx].nodeData.cpy.dst = static_cast<void *>(
      static_cast<char *>(recvbuff) + rank * count * typeSize);
  runnerState->dagNodes[nodeIdx].nodeData.cpy.count = count;
  runnerState->dagNodes[nodeIdx].nodeData.cpy.datatype = datatype;

  for (int s = 0; s < nGroups; s++) {
    runnerState->dagNodes[s].numParents = (s == 0) ? 0 : 1;
    runnerState->dagNodes[s].numChildren = (s == nGroups - 1) ? 0 : 1;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[s]));

    if (s == 0) {
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[s]);
    } else {
      runnerState->numPendingNodes++;
      FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[s], 0, s - 1));
    }
    if (s != nGroups - 1) {
      runnerState->dagNodes[s].children[0] = s + 1;
    }
  }

  runnerState->dagNodes[nodeIdx].numParents = 0;
  runnerState->dagNodes[nodeIdx].numChildren = 0;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[nodeIdx]));
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[nodeIdx]);

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Grouped AllGather topology",
        nranks);
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d",
          i,
          (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
          : (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeRed)
              ? "RED"
              : "CPY",
          runnerState->dagNodes[i].numParents,
          runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateP2p called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes + 1;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = i
   */
  for (int s = 0; s < numSlices; s++) {
    // All-Gather
    int sliceNodeBaseIdx = globalNodeIdx;
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, baseIdx %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, sliceNodeBaseIdx, txRankChunkCount, txSliceCount,
            rxRankChunkCount, rxSliceCount, txChunk, txOffset, rxChunk,
            rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    }

    // Setup dependencies linearly within the slice chain
    for (int i = 0; i < nodesPerSlice; i++) {
      int currIdx = sliceNodeBaseIdx + i;

      if (currIdx == 0) {
        runnerState->dagNodes[currIdx].numParents = 0;
      } else {
        runnerState->dagNodes[currIdx].numParents = 1;
      }
      if (currIdx == numNodes - 1) {
        runnerState->dagNodes[currIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[currIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[currIdx]));
      if (currIdx != 0) {
        int parentIdx = s == 0 ? (numSlices - 1) * nodesPerSlice + i - 1
                               : currIdx - nodesPerSlice;
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[currIdx], 0, parentIdx));
      }
      if (s == numSlices - 1) {
        if (currIdx != numNodes - 1) {
          runnerState->dagNodes[currIdx].children[0] = i + 1;
        }
      } else {
        runnerState->dagNodes[currIdx].children[0] = currIdx + nodesPerSlice;
      }
    }
  }
  // Copy local chunk from sendbuff to recvbuff before starting AG
  // Calculate offset accounting for rankChunkRemainder
  // First rankChunkRemainder ranks each have one extra element
  size_t localRankChunkCount =
      baseRankChunkCount + (rank < (int)rankChunkRemainder ? 1 : 0);
  size_t localChunkOffset =
      (rank * baseRankChunkCount + std::min(rank, (int)rankChunkRemainder)) *
      typeSize;
  int cpyNodeIdx = globalNodeIdx++;
  runnerState->dagNodes[cpyNodeIdx].nodeIdx = cpyNodeIdx;
  runnerState->dagNodes[cpyNodeIdx].nodeType = uniRunnerDagNodeTypeCpy;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.src = static_cast<void *>(
      static_cast<char *>(const_cast<void *>(sendbuff)) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.dst =
      static_cast<void *>(static_cast<char *>(recvbuff) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.count = localRankChunkCount;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.datatype = datatype;
  runnerState->dagNodes[cpyNodeIdx].numParents = 0;
  runnerState->dagNodes[cpyNodeIdx].numChildren = 0;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[cpyNodeIdx]));
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[cpyNodeIdx]);
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[0]);
  runnerState->numPendingNodes = numNodes - 1;

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingAR called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = 3 * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * 2
   * redNodeIdx = s * nodesPerSlice + i * 2 + 1
   * all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + (nranks - 1) * 2 + i
   */
  for (int s = 0; s < numSlices; s++) {
    // Phase 1: Scatter-Reduce
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 2;
        }
        runnerState->numPendingNodes++;
      }
      runnerState->dagNodes[p2pNodeIdx].numChildren = 2;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice + 2 * (i - 1);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 1,
                                       p2pNodeIdx - 1));
        }
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * (i + 1);
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, 2 * (i + 1));
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, p2pNodeIdx + nodesPerSlice);
      }
      runnerState->dagNodes[p2pNodeIdx].children[1] = p2pNodeIdx + 1;
      TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
            p2pNodeIdx, p2pNodeIdx + 1);

      // Reduce Node
      int redNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
      runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
      runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
      runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
          static_cast<void *>(
              static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.output =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.count = rxSliceCount;
      runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
          runnerState->uniRunnerNThreads;
      runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

      // Set up red node dependency
      runnerState->numPendingNodes++;
      runnerState->dagNodes[redNodeIdx].numParents = 1;
      runnerState->dagNodes[redNodeIdx].numChildren = 1;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0, p2pNodeIdx));
      runnerState->dagNodes[redNodeIdx].children[0] = redNodeIdx + 1;
      TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
            redNodeIdx, redNodeIdx + 1);
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up all-gather phase p2p node dependency
      runnerState->numPendingNodes++;
      if (i == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 2;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1;
      }
      if (p2pNodeIdx == numNodes - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      int parentIdx = p2pNodeIdx - nodesPerSlice;
      if (s == 0) {
        if (i == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice + 2 * (nranks - 2);
        } else {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + 2 * (nranks - 1) + i - 1;
        }
      }
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
      if (i == 0) {
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 1,
                                     p2pNodeIdx - 1));
      }
      if (s == numSlices - 1) {
        if (p2pNodeIdx != numNodes - 1) {
          runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * nranks + i - 1;
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
                p2pNodeIdx, 2 * nranks + i - 1);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
              p2pNodeIdx, p2pNodeIdx + nodesPerSlice);
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Ring AllReduce topology (%d "
        "slices)",
        nranks, numSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                            const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateSlicedAR called, count=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce * numRedSlices) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = (numRedSlices + 2) * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1
   * all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + (nranks - 1) * (1 + numRedSlices) + i
   */
  for (int s = 0; s < numSlices; s++) {
    // Phase 1: Scatter-Reduce
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      runnerState->dagNodes[p2pNodeIdx].numChildren = 1 + numRedSlices;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                         r + 1, p2pNodeIdx - numRedSlices + r));
          }
        }
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            (i + 1) * (1 + numRedSlices);
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[p2pNodeIdx].children[r + 1] = p2pNodeIdx + 1 + r;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, r + 1,
              runnerState->dagNodes[p2pNodeIdx].children[r + 1]);
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = rxSliceCount / numRedSlices;
      size_t redSliceRemainder = rxSliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(recvbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(
                static_cast<char *>(const_cast<void *>(sendbuff)) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            static_cast<void *>(static_cast<char *>(recvbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        runnerState->dagNodes[redNodeIdx].numChildren = 1;
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     p2pNodeIdx));
        runnerState->dagNodes[redNodeIdx].children[0] =
            redSliceStartIdx + numRedSlices;
        TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
              redNodeIdx, runnerState->dagNodes[redNodeIdx].children[0]);
      }
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up all-gather phase p2p node dependency
      runnerState->numPendingNodes++;
      if (i == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1;
      }
      if (p2pNodeIdx == numNodes - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      int parentIdx = p2pNodeIdx - nodesPerSlice;
      if (s == 0) {
        if (i == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (nranks - 2) * (1 + numRedSlices);
        } else {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (nranks - 1) * (1 + numRedSlices) + i - 1;
        }
      }
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
      if (i == 0) {
        for (int r = 0; r < numRedSlices; r++) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                       r + 1, p2pNodeIdx - numRedSlices + r));
        }
      }
      if (s == numSlices - 1) {
        if (p2pNodeIdx != numNodes - 1) {
          runnerState->dagNodes[p2pNodeIdx].children[0] =
              (1 + numRedSlices) * (nranks - 1) + i + 1;
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
                p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Sliced AllReduce topology (%d "
        "slices, %d redSlices)",
        nranks, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateRingRS(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    void *scratchbuff, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t op, flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingRS called, recvcount=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t baseRankChunkCount = count;

  // Nodes per slice chain:
  // (P2P + Reduce * numRedSlices) * (nranks - 1)
  const int nodesPerSlice = (numRedSlices + 1) * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1
   */
  for (int s = 0; s < numSlices; s++) {
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i - 1 + nranks) % nranks;
      int rxChunk = (rank - i - 2 + nranks) % nranks;

      size_t txRankChunkCount = baseRankChunkCount;
      size_t rxRankChunkCount = baseRankChunkCount;
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      size_t txOffset =
          (txChunk * baseRankChunkCount) * typeSize + txSliceOffsetInChunk;
      size_t rxOffset =
          (rxChunk * baseRankChunkCount) * typeSize + rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from scratchbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : scratchbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(scratchbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      if (i == nranks - 2 && s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = numRedSlices;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1 + numRedSlices;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                         r + 1, p2pNodeIdx - numRedSlices + r));
          }
        }
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[p2pNodeIdx].children[r] = p2pNodeIdx + 1 + r;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, r, runnerState->dagNodes[p2pNodeIdx].children[r]);
      }
      if (s == numSlices - 1) {
        if (i != nranks - 2) {
          runnerState->dagNodes[p2pNodeIdx].children[numRedSlices] =
              (i + 1) * (1 + numRedSlices);
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
                p2pNodeIdx, numRedSlices,
                runnerState->dagNodes[p2pNodeIdx].children[numRedSlices]);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[numRedSlices] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, numRedSlices,
              runnerState->dagNodes[p2pNodeIdx].children[numRedSlices]);
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = rxSliceCount / numRedSlices;
      size_t redSliceRemainder = rxSliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(scratchbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(
                static_cast<char *>(const_cast<void *>(sendbuff)) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            i == nranks - 2
                ? static_cast<void *>(
                      static_cast<char *>(recvbuff) + rxSliceOffsetInChunk +
                      r * baseRedSliceCount * typeSize +
                      std::min(r, (int)redSliceRemainder) * typeSize)
                : static_cast<void *>(static_cast<char *>(scratchbuff) +
                                      redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        if (i == nranks - 2) {
          runnerState->dagNodes[redNodeIdx].numChildren = 0;
        } else {
          runnerState->dagNodes[redNodeIdx].numChildren = 1;
        }
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     p2pNodeIdx));
        if (i != nranks - 2) {
          runnerState->dagNodes[redNodeIdx].children[0] =
              redSliceStartIdx + numRedSlices;
          TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
                redNodeIdx, runnerState->dagNodes[redNodeIdx].children[0]);
        }
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank ReduceScatter topology (%d "
        "slices, %d redSlices)",
        nranks, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateTreeRed(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    void *scratchbuff, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t op, int root, flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int algoRank = (rank - root + nranks) % nranks; // Rotate ranks so root is 0

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Nodes per slice chain:
  const int nTotalSteps = 8 * sizeof(int) - __builtin_clz(nranks - 1);
  int recvNodesPerSlice = algoRank ? __builtin_ctz(algoRank) : nTotalSteps;
  if (algoRank && recvNodesPerSlice &&
      nranks - algoRank <= (1 << (recvNodesPerSlice - 1))) {
    recvNodesPerSlice =
        nranks - algoRank - 1
            ? 8 * sizeof(int) - __builtin_clz(nranks - algoRank - 1)
            : 0;
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d (algoRank %d) adjusted recvNodesPerSlice to %d from %d",
          rank, algoRank, recvNodesPerSlice, __builtin_ctz(algoRank));
  }
  const int sendNodesPerSlice = algoRank ? 1 : 0;
  const int redNodesPerSlice = recvNodesPerSlice * numRedSlices;
  const int nodesPerSlice =
      sendNodesPerSlice + recvNodesPerSlice + redNodesPerSlice;
  const int numNodes = nodesPerSlice * numSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d (algoRank %d) initUniRunnerStateTreeReduce called, count=%lu, "
        "numSlices=%d, numRedSlices=%d, recvSteps %d, sendSteps %d",
        comm->rank, algoRank, count, numSlices, numRedSlices, recvNodesPerSlice,
        sendNodesPerSlice);

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* halving doubling tree reduce
   * slice = s, step = i
   * recvNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1..numRedSlices
   * sendNodeIdx = s * nodesPerSlice + recvNodesPerSlice + redNodesPerSlice
   */
  for (int s = 0; s < numSlices; s++) {
    size_t baseSliceCount = count / numSlices;
    size_t sliceRemainder = count % numSlices;
    size_t sliceCount = baseSliceCount + (s < sliceRemainder ? 1 : 0);
    size_t sliceOffset = s * baseSliceCount * typeSize;
    sliceOffset += std::min(s, (int)sliceRemainder) * typeSize;
    size_t rxOffset = count * typeSize + sliceOffset;

    TRACE(FLAGCX_UNIRUNNER,
          "Initializing rank %d (algoRank %d) slice %d, rxSliceCount %lu, "
          "rxSliceOffset %lu, rxOffset %lu",
          rank, algoRank, s, sliceCount, sliceOffset, rxOffset);

    // recv nodes and red nodes
    for (int i = 0; i < recvNodesPerSlice; i++) {
      int recvNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[recvNodeIdx].nodeIdx = recvNodeIdx;
      runnerState->dagNodes[recvNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps = 1;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops,
                       runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps *
                           sizeof(struct uniRunnerP2pOpData)));

      // Recv Node
      int peer = (rank + (1 << i)) % nranks;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].peerRank = peer;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].datatype =
          datatype;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(scratchbuff) + rxOffset);
      TRACE(FLAGCX_UNIRUNNER,
            "rank %d (algoRank %d) recvNode %d recv from peer %d, count %lu, "
            "offset %lu",
            rank, algoRank, recvNodeIdx, peer, sliceCount, rxOffset);

      // Set up p2p node dependency
      if (recvNodeIdx == 0) {
        runnerState->dagNodes[recvNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[recvNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[recvNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[recvNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      if (i == nTotalSteps - 1 && s == numSlices - 1) {
        runnerState->dagNodes[recvNodeIdx].numChildren = numRedSlices;
      } else {
        runnerState->dagNodes[recvNodeIdx].numChildren = 1 + numRedSlices;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[recvNodeIdx]));
      if (recvNodeIdx != 0) {
        int parentIdx = recvNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[recvNodeIdx], 0,
                                     parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[recvNodeIdx],
                                         r + 1,
                                         recvNodeIdx - numRedSlices + r));
          }
        }
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[recvNodeIdx].children[r] = recvNodeIdx + 1 + r;
      }
      if (s == numSlices - 1) {
        if (i != nTotalSteps - 1) {
          runnerState->dagNodes[recvNodeIdx].children[numRedSlices] =
              (i + 1) * (1 + numRedSlices);
        }
      } else {
        runnerState->dagNodes[recvNodeIdx].children[numRedSlices] =
            recvNodeIdx + nodesPerSlice;
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = sliceCount / numRedSlices;
      size_t redSliceRemainder = sliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(scratchbuff) + redOffset);
        void *redInput2Base =
            (i == 0) ? const_cast<void *>(sendbuff) : scratchbuff;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(static_cast<char *>(redInput2Base) + redOffset -
                                count * typeSize);
        void *redOutput = (i == nTotalSteps - 1) ? recvbuff : scratchbuff;
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            static_cast<void *>(static_cast<char *>(redOutput) + redOffset -
                                count * typeSize);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        if (i == nTotalSteps - 1) {
          runnerState->dagNodes[redNodeIdx].numChildren = 0;
        } else {
          runnerState->dagNodes[redNodeIdx].numChildren = 1;
        }
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     recvNodeIdx));
        if (i != nTotalSteps - 1) {
          runnerState->dagNodes[redNodeIdx].children[0] =
              redSliceStartIdx + numRedSlices;
        }
      }
    }

    // Send Node
    if (algoRank) {
      int sendNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[sendNodeIdx].nodeIdx = sendNodeIdx;
      runnerState->dagNodes[sendNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps = 1;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops,
                       runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps *
                           sizeof(struct uniRunnerP2pOpData)));

      int peer = (rank - (1 << (__builtin_ctz(algoRank))) + nranks) % nranks;

      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].peerRank = peer;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].datatype =
          datatype;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(
              static_cast<char *>(recvNodesPerSlice == 0
                                      ? const_cast<void *>(sendbuff)
                                      : scratchbuff) +
              sliceOffset);
      TRACE(FLAGCX_UNIRUNNER,
            "rank %d (algoRank %d) sendNode %d send to peer %d, count %lu, "
            "offset %lu",
            rank, algoRank, sendNodeIdx, peer, sliceCount, sliceOffset);
      // Set up p2p node dependency
      if (recvNodesPerSlice == 0) {
        if (s == 0) {
          runnerState->dagNodes[sendNodeIdx].numParents = 0;
        } else {
          runnerState->dagNodes[sendNodeIdx].numParents = 1;
          runnerState->numPendingNodes++;
        }
      } else {
        runnerState->dagNodes[sendNodeIdx].numParents = 1 + numRedSlices;
        runnerState->numPendingNodes++;
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[sendNodeIdx].numChildren = 0;

      } else {
        runnerState->dagNodes[sendNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[sendNodeIdx]));
      if (recvNodesPerSlice == 0) {
        if (s == 0) {
          flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                  &runnerState->dagNodes[sendNodeIdx]);
        } else {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx], 0,
                                       sendNodeIdx - nodesPerSlice));
        }
      } else {
        int parentIdx = sendNodeIdx - nodesPerSlice;
        if (s == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (recvNodesPerSlice - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx], 0,
                                     parentIdx));
        for (int r = 0; r < numRedSlices; r++) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx],
                                       r + 1, sendNodeIdx - numRedSlices + r));
        }
      }
      if (s != numSlices - 1) {
        runnerState->dagNodes[sendNodeIdx].children[0] =
            sendNodeIdx + nodesPerSlice;
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Reduce (root %d) topology (%d "
        "slices, %d redSlices)",
        nranks, root, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static void resetDagSchedulerRuntimeState(flagcxUniRunnerState *runnerState) {
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  runnerState->numPendingNodes = 0;
}

static flagcxResult_t captureUniRunnerDagTemplateFromState(
    const flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings,
    std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::shared_ptr<uniRunnerDagTemplate> captured =
      std::make_shared<uniRunnerDagTemplate>();
  captured->key = key;
  captured->hashValue = getUniRunnerDagPatternHash(key);
  captured->nodes.reserve(runnerState->numDagNodes);

  for (int i = 0; i < runnerState->numDagNodes; i++) {
    const uniRunnerDagNode *node = &runnerState->dagNodes[i];
    uniRunnerDagNodeDesc nodeDesc;
    nodeDesc.nodeType = node->nodeType;
    nodeDesc.nodeIdx = node->nodeIdx;
    if (node->numParents > 0) {
      nodeDesc.parents.assign(node->parents, node->parents + node->numParents);
    }
    if (node->numChildren > 0) {
      nodeDesc.children.assign(node->children,
                               node->children + node->numChildren);
    }

    if (node->nodeType == uniRunnerDagNodeTypeP2p) {
      for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerP2pOpData *op = &node->nodeData.p2p.ops[opIdx];
        uniRunnerDagP2pOpDesc opDesc;
        opDesc.count = op->count;
        opDesc.peerRank = op->peerRank;
        opDesc.datatype = op->datatype;
        opDesc.type = op->type;
        FLAGCXCHECK(captureBufferRef(op->addr, bindings, &opDesc.buffer));
        nodeDesc.p2pOps.push_back(opDesc);
      }
    } else if (node->nodeType == uniRunnerDagNodeTypeRed) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input1, bindings,
                                   &nodeDesc.red.input1));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input2, bindings,
                                   &nodeDesc.red.input2));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.output, bindings,
                                   &nodeDesc.red.output));
      nodeDesc.red.count = node->nodeData.red.count;
      nodeDesc.red.nthreads = node->nodeData.red.nthreads;
      nodeDesc.red.datatype = node->nodeData.red.datatype;
      nodeDesc.red.redOp = node->nodeData.red.redOp;
    } else if (node->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.src, bindings,
                                   &nodeDesc.cpy.src));
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.dst, bindings,
                                   &nodeDesc.cpy.dst));
      nodeDesc.cpy.count = node->nodeData.cpy.count;
      nodeDesc.cpy.datatype = node->nodeData.cpy.datatype;
    } else {
      return flagcxNotSupported;
    }

    captured->nodes.push_back(nodeDesc);
  }

  *dagTemplate = captured;
  return flagcxSuccess;
}

static flagcxResult_t
materializeUniRunnerDagTemplate(flagcxUniRunnerState *runnerState,
                                const uniRunnerDagTemplate &dagTemplate,
                                const uniRunnerDagRuntimeBindings &bindings) {
  resetDagSchedulerRuntimeState(runnerState);
  runnerState->numDagNodes = static_cast<int>(dagTemplate.nodes.size());
  if (runnerState->numDagNodes == 0) {
    runnerState->dagNodes = NULL;
    return flagcxSuccess;
  }

  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int i = 0; i < runnerState->numDagNodes; i++) {
    const uniRunnerDagNodeDesc &src = dagTemplate.nodes[i];
    uniRunnerDagNode *dst = &runnerState->dagNodes[i];
    dst->nodeType = src.nodeType;
    dst->nodeIdx = src.nodeIdx;
    dst->numParents = static_cast<int>(src.parents.size());
    dst->numChildren = static_cast<int>(src.children.size());
    FLAGCXCHECK(allocDagNodeDeps(dst));
    dst->pendingParents = dst->numParents;
    for (int parentIdx = 0; parentIdx < dst->numParents; parentIdx++) {
      dst->parents[parentIdx] = src.parents[parentIdx];
    }
    for (int childIdx = 0; childIdx < dst->numChildren; childIdx++) {
      dst->children[childIdx] = src.children[childIdx];
    }

    if (dst->nodeType == uniRunnerDagNodeTypeP2p) {
      dst->nodeData.p2p.numOps = static_cast<int>(src.p2pOps.size());
      if (dst->nodeData.p2p.numOps > 0) {
        FLAGCXCHECK(flagcxCalloc(&dst->nodeData.p2p.ops,
                                 dst->nodeData.p2p.numOps *
                                     sizeof(struct uniRunnerP2pOpData)));
      }
      for (int opIdx = 0; opIdx < dst->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerDagP2pOpDesc &srcOp = src.p2pOps[opIdx];
        uniRunnerP2pOpData *dstOp = &dst->nodeData.p2p.ops[opIdx];
        dstOp->count = srcOp.count;
        dstOp->peerRank = srcOp.peerRank;
        dstOp->datatype = srcOp.datatype;
        dstOp->type = srcOp.type;
        FLAGCXCHECK(resolveBufferRef(srcOp.buffer, bindings, &dstOp->addr));
      }
    } else if (dst->nodeType == uniRunnerDagNodeTypeRed) {
      dst->nodeData.red.triggerIdx = -1;
      FLAGCXCHECK(resolveBufferRef(src.red.input1, bindings,
                                   &dst->nodeData.red.input1));
      FLAGCXCHECK(resolveBufferRef(src.red.input2, bindings,
                                   &dst->nodeData.red.input2));
      FLAGCXCHECK(resolveBufferRef(src.red.output, bindings,
                                   &dst->nodeData.red.output));
      dst->nodeData.red.count = src.red.count;
      dst->nodeData.red.nthreads = src.red.nthreads;
      dst->nodeData.red.datatype = src.red.datatype;
      dst->nodeData.red.redOp = src.red.redOp;
    } else if (dst->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.src, bindings, &dst->nodeData.cpy.src));
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.dst, bindings, &dst->nodeData.cpy.dst));
      dst->nodeData.cpy.count = src.cpy.count;
      dst->nodeData.cpy.datatype = src.cpy.datatype;
    } else {
      return flagcxNotSupported;
    }

    if (dst->numParents == 0) {
      if (dst->nodeType == uniRunnerDagNodeTypeRed) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, dst);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, dst);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }

  return flagcxSuccess;
}

static flagcxResult_t tryLoadCachedUniRunnerDag(
    flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings, bool *cacheHit) {
  *cacheHit = false;
  std::shared_ptr<uniRunnerDagTemplate> dagTemplate;
  size_t hashValue = getUniRunnerDagPatternHash(key);
  if (!findUniRunnerDagTemplate(hashValue, key, &dagTemplate)) {
    flagcxResult_t loadRes = loadUniRunnerDagFromCacheDir(hashValue, key.rank);
    if (!findUniRunnerDagTemplate(hashValue, key, &dagTemplate)) {
      if (loadRes != flagcxSuccess) {
        return loadRes;
      }
      TRACE(FLAGCX_UNIRUNNER,
            "uniRunner DAG cache miss, algo=%s commOp=%s hash=%lu",
            uniRunnerDagAlgoTypeToString(key.algoType),
            uniRunnerCommOpToString(key.commOp), hashValue);
      return flagcxSuccess;
    }
  }

  TRACE(FLAGCX_UNIRUNNER, "uniRunner DAG cache hit, algo=%s commOp=%s hash=%lu",
        uniRunnerDagAlgoTypeToString(key.algoType),
        uniRunnerCommOpToString(key.commOp), hashValue);
  FLAGCXCHECK(
      materializeUniRunnerDagTemplate(runnerState, *dagTemplate, bindings));
  FLAGCXCHECK(validateDagNodes(runnerState));
  *cacheHit = true;
  return flagcxSuccess;
}

static flagcxResult_t
cacheBuiltUniRunnerDag(const flagcxUniRunnerState *runnerState,
                       const uniRunnerDagCacheKey &key,
                       const uniRunnerDagRuntimeBindings &bindings) {
  size_t hashValue = getUniRunnerDagPatternHash(key);
  std::shared_ptr<uniRunnerDagTemplate> dagTemplate;
  flagcxResult_t captureRes = captureUniRunnerDagTemplateFromState(
      runnerState, key, bindings, &dagTemplate);
  if (captureRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG capture skipped for hash=%lu, result=%d", hashValue,
          captureRes);
    return flagcxSuccess;
  }

  flagcxResult_t cacheRes = cacheUniRunnerDagTemplate(dagTemplate);
  if (cacheRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG cache persist skipped for hash=%lu, result=%d",
          hashValue, cacheRes);
  }
  return flagcxSuccess;
}

// Clean up DAG nodes
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_UNIRUNNER, "cleanupDagScheduler called");
  if (!runnerState) {
    return flagcxSuccess;
  }
  if (runnerState->dagNodes) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
          runnerState->dagNodes[i].nodeData.p2p.ops) {
        free(runnerState->dagNodes[i].nodeData.p2p.ops);
      }
      if (runnerState->dagNodes[i].parents) {
        free(runnerState->dagNodes[i].parents);
      }
      if (runnerState->dagNodes[i].children) {
        free(runnerState->dagNodes[i].children);
      }
    }
    free(runnerState->dagNodes);
    runnerState->dagNodes = NULL;
  }
  runnerState->numDagNodes = 0;
  runnerState->numPendingNodes = 0;
  return flagcxSuccess;
}

template <typename BuildFn>
static flagcxResult_t initUniRunnerStateCached(
    flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings, BuildFn buildFn) {
  bool cacheHit = false;
  flagcxResult_t cacheLoadRes =
      tryLoadCachedUniRunnerDag(runnerState, key, bindings, &cacheHit);
  if (cacheLoadRes == flagcxSuccess && cacheHit) {
    return flagcxSuccess;
  }
  if (cacheLoadRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG cache load failed, rebuild dag instead, result=%d",
          cacheLoadRes);
    (void)cleanupDagScheduler(runnerState);
    resetDagSchedulerRuntimeState(runnerState);
  }

  FLAGCXCHECK(buildFn());
  return cacheBuiltUniRunnerDag(runnerState, key, bindings);
}

flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoLocRed, flagcxCommOpAllReduce, count, datatype, op, -1, 0,
      runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateLocRed(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateGroupedAG(flagcxUniRunnerState *runnerState,
                                           const void *sendbuff, void *recvbuff,
                                           size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxComm_t comm, int groupSize) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * comm->nranks * typeSize;

  uniRunnerDagCacheKey key =
      makeUniRunnerDagCacheKey(uniRunnerDagAlgoGroupedAG, flagcxCommOpAllGather,
                               count, datatype, flagcxRedNoOp, -1, groupSize,
                               runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff, count,
                                        datatype, comm, groupSize);
  });
}

flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAG, flagcxCommOpAllReduce, count, datatype,
      flagcxRedNoOp, -1, 0, runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAG(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAR, flagcxCommOpAllReduce, count, datatype, op, -1, 0,
      runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAR(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoSlicedAR, flagcxCommOpAllReduce, count, datatype, op, -1,
      0, runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * comm->nranks * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = count * comm->nranks * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingRS, flagcxCommOpReduceScatter, count, datatype, op,
      -1, 0, runnerState, comm, sendbuff, recvbuff, scratchbuff);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                     scratchbuff, count, datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateTreeRed(flagcxUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         void *scratchbuff, size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, int root,
                                         flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = 2 * count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoTreeRed, flagcxCommOpReduce, count, datatype, op, root, 0,
      runnerState, comm, sendbuff, recvbuff, scratchbuff);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                      scratchbuff, count, datatype, op, root,
                                      comm);
  });
}

static flagcxResult_t launchP2pOps(flagcxUniRunnerState *runnerState,
                                   flagcxHeteroComm_t comm) {
  // Dequeue
  uniRunnerDagNode *current =
      flagcxIntruQueueDequeue(&runnerState->p2pReadyQueue);
  void *flag = getDagNodeFlag(runnerState, current->nodeIdx);
  flagcxStream_t currentStream =
      getDagNodeExecutionStream(runnerState, current);

  if (current->nodeType == uniRunnerDagNodeTypeP2p) {
    // Mark the node as submitted before wiring its completion dependency.
    TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWrite flag %d: PEND",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->commStream, flag,
                                                  flagcxStreamFlagPend, 0));
    for (int i = 0; i < current->numParents; i++) {
      int parentIdx = current->parents[i];
      uniRunnerDagNode *parent = &runnerState->dagNodes[parentIdx];
      if (getDagNodeExecutionStream(runnerState, parent) == currentStream) {
        TRACE(FLAGCX_UNIRUNNER,
              "rank %d p2p op %d skip same-stream wait for parent %d",
              comm->rank, current->nodeIdx, parentIdx);
        continue;
      }
      void *parentFlag = getDagNodeFlag(runnerState, parentIdx);
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          runnerState->commStream, parentFlag, flagcxStreamFlagDone, 0));
      TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWait flag %d: DONE",
            comm->rank, current->nodeIdx, parentIdx);
    }

    // Prepare ops list
    struct uniRunnerP2pOpData *ops = current->nodeData.p2p.ops;

    flagcxResult_t result = flagcxRunnerRunGroup(
        []() { return flagcxHeteroGroupStart(); },
        [&]() {
          for (int i = 0; i < current->nodeData.p2p.numOps; i++) {
            struct uniRunnerP2pOpData *op = &ops[i];
            flagcxResult_t opResult = flagcxSuccess;
            if (op->type == flagcxDevicePrimSend) {
              opResult = flagcxHeteroSendOnChannel(
                  op->addr, op->count, op->datatype, op->peerRank, comm,
                  runnerState->commStream,
                  flagcxCollProxyChannelForEdge(comm->rank, op->peerRank, 0));
            } else if (op->type == flagcxDevicePrimRecv) {
              opResult = flagcxHeteroRecvOnChannel(
                  op->addr, op->count, op->datatype, op->peerRank, comm,
                  runnerState->commStream,
                  flagcxCollProxyChannelForEdge(op->peerRank, comm->rank, 0));
            } else {
              return flagcxInvalidArgument;
            }
            if (!flagcxRunnerResultAccepted(opResult))
              return opResult;
          }
          return flagcxSuccess;
        },
        []() { return flagcxHeteroGroupEnd(); });
    if (result != flagcxSuccess) {
      return result;
    }

    TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWrite flag %d: DONE",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->commStream, flag,
                                                  flagcxStreamFlagDone, 0));
  } else if (current->nodeType == uniRunnerDagNodeTypeCpy) {
    TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWrite flag %d: PEND",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->cpyStream, flag,
                                                  flagcxStreamFlagPend, 0));
    for (int i = 0; i < current->numParents; i++) {
      int parentIdx = current->parents[i];
      uniRunnerDagNode *parent = &runnerState->dagNodes[parentIdx];
      if (getDagNodeExecutionStream(runnerState, parent) == currentStream) {
        TRACE(FLAGCX_UNIRUNNER,
              "rank %d cpy op %d skip same-stream wait for parent %d",
              comm->rank, current->nodeIdx, parentIdx);
        continue;
      }
      void *parentFlag = getDagNodeFlag(runnerState, parentIdx);
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          runnerState->cpyStream, parentFlag, flagcxStreamFlagDone, 0));
      TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWait flag %d: DONE",
            comm->rank, current->nodeIdx, parentIdx);
    }

    // Launch copy
    flagcxResult_t result = deviceAdaptor->deviceMemcpy(
        current->nodeData.cpy.dst, current->nodeData.cpy.src,
        current->nodeData.cpy.count *
            getFlagcxDataTypeSize(current->nodeData.cpy.datatype),
        flagcxMemcpyDeviceToDevice, runnerState->cpyStream, NULL);
    if (result != flagcxSuccess) {
      return result;
    }

    // Write flag to stream
    TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWrite flag %d: DONE",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->cpyStream, flag,
                                                  flagcxStreamFlagDone, 0));
  } else {
    return flagcxSystemError;
  }

  return flagcxSuccess;
}

static flagcxResult_t enqueueReadyQueue(flagcxUniRunnerState *runnerState,
                                        int nodeIdx) {
  if (runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeP2p ||
      runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeCpy) {
    flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else if (runnerState->dagNodes[nodeIdx].nodeType ==
             uniRunnerDagNodeTypeRed) {
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else {
    return flagcxNotSupported;
  }
  runnerState->numPendingNodes--;
  return flagcxSuccess;
}

static flagcxResult_t notifyChildrenScheduled(flagcxUniRunnerState *runnerState,
                                              uniRunnerDagNode *current) {
  for (int i = 0; i < current->numChildren; i++) {
    uniRunnerDagNode *child = &runnerState->dagNodes[current->children[i]];
    if (child->pendingParents <= 0) {
      return flagcxInternalError;
    }
    child->pendingParents--;
    if (child->pendingParents == 0) {
      FLAGCXCHECK(enqueueReadyQueue(runnerState, current->children[i]));
    }
  }
  return flagcxSuccess;
}

// Process ready queue: submit ready nodes to the corresponding execution
// stream/FIFO. Child readiness is host-scheduled immediately after submission;
// same-stream execution dependencies rely on launch order, while cross-stream
// dependencies are enforced via stream flags.
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  // process p2pReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue)) {
    uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->p2pReadyQueue);
    FLAGCXCHECK(launchP2pOps(runnerState, comm));
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  // process redReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->redReadyQueue)) {
    struct uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->redReadyQueue);
    uint64_t flagIn =
        current->numParents == 0
            ? 0
            : (uintptr_t)getDagNodeFlag(runnerState, current->parents[0]);
    uint64_t flagOut = (uintptr_t)getDagNodeFlag(runnerState, current->nodeIdx);
    // The current algorithms only create single-parent RED nodes. Multi-parent
    // dependencies are handled for P2P/CPY nodes by emitting one stream wait
    // per parent; RED nodes would need an explicit fan-in flag if that ever
    // changes.
    if (current->numParents > 1) {
      return flagcxInvalidArgument;
    }
    int idx = -1;
    FLAGCXCHECK(enqueue(
        (void *)runnerState->fifo->buffer,
        (uintptr_t)current->nodeData.red.input1,
        (uintptr_t)current->nodeData.red.input2,
        (uintptr_t)current->nodeData.red.output, current->nodeData.red.count,
        current->nodeData.red.nthreads, current->nodeData.red.datatype,
        current->nodeData.red.redOp, flagIn, flagOut, &idx));
    if (idx == -1) {
      sched_yield();
      break; // FIFO full, skip for now
    }
    // Dequeue
    flagcxIntruQueueDequeue(&runnerState->redReadyQueue);
    current->nodeData.red.triggerIdx = idx;
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  return flagcxSuccess;
}

flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  flagcxResult_t result = flagcxSuccess;
  flagcxStream_t redStream = nullptr;
  flagcxStream_t cpyStream = nullptr;
  runnerState->dagNodes = NULL;
  runnerState->numDagNodes = 0;
  runnerState->streamFlagsSize = 0;

  runnerState->uniRunnerNSlices = flagcxParamUniRunnerNSlices();
  runnerState->uniRunnerNThreads = flagcxParamUniRunnerNThreads();
  runnerState->uniRunnerNBlocks = flagcxParamUniRunnerNBlocks();
  runnerState->uniRunnerNRedSlices = flagcxParamUniRunnerNRedSlices();
  runnerState->uniRunnerRedSliceSize = flagcxParamUniRunnerRedSliceSize();

  // Set device context
  result = deviceAdaptor->setDevice(hcomm->cudaDev);
  if (result != flagcxSuccess)
    return result;

  runnerState->fifo = nullptr;
  runnerState->commStream = stream;
  runnerState->redStream = nullptr;
  runnerState->cpyStream = nullptr;
  runnerState->dagNodes = nullptr;
  runnerState->numDagNodes = 0;
  hcomm->uniRunnerFifoBuffer = nullptr;

  // Create FIFO
  runnerState->fifo = new (std::nothrow) flagcxFifo();
  if (runnerState->fifo == nullptr)
    return flagcxSystemError;
  result = runnerState->fifo->flagcxRedFifoInit();
  if (result != flagcxSuccess) {
    delete runnerState->fifo;
    runnerState->fifo = nullptr;
    return result;
  }
  // hcomm->proxyState->uniRunnerState.fifo->buffer is the host pointer
  // hcomm->uniRunnerFifoBuffer stores the device pointer to fifo buffer
  result = deviceAdaptor->hostGetDevicePointer(
      &hcomm->uniRunnerFifoBuffer, (void *)runnerState->fifo->buffer);
  if (result != flagcxSuccess)
    goto fail;

  // Initialize queues
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  runnerState->numPendingNodes = 0;

  // Create dedicated reduce and copy streams
  result = deviceAdaptor->streamCreate(&redStream);
  if (result != flagcxSuccess)
    goto fail;
  runnerState->redStream = redStream;
  result = deviceAdaptor->streamCreate(&cpyStream);
  if (result != flagcxSuccess)
    goto fail;
  runnerState->cpyStream = cpyStream;
  runnerState->commStream = stream;
  return flagcxSuccess;

fail:
  flagcxRunnerRecordFirstError(cleanupUniRunner(comm), &result);
  return result;
}

flagcxResult_t cleanupUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  flagcxResult_t result = flagcxSuccess;

  // Stream operations can still reference streamFlags after a runner error.
  // Drain them before marking the reusable pool inactive.
  if (runnerState->redStream != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->streamSynchronize(runnerState->redStream), &result);
  }
  if (runnerState->cpyStream != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->streamSynchronize(runnerState->cpyStream), &result);
  }
  if (runnerState->commStream != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->streamSynchronize(runnerState->commStream), &result);
  }
  flagcxRunnerRecordFirstError(cleanupDagScheduler(runnerState), &result);
  runnerState->streamFlagsSize = 0;

  if (runnerState->redStream != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->streamDestroy(runnerState->redStream), &result);
    runnerState->redStream = nullptr;
  }
  if (runnerState->cpyStream != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->streamDestroy(runnerState->cpyStream), &result);
    runnerState->cpyStream = nullptr;
  }
  runnerState->commStream = nullptr;

  // Destroy fifo
  if (runnerState->fifo != nullptr) {
    flagcxRunnerRecordFirstError(runnerState->fifo->flagcxRedFifoDestroy(),
                                 &result);
    delete runnerState->fifo;
    runnerState->fifo = nullptr;
  }
  hcomm->uniRunnerFifoBuffer = NULL;

  return result;
}

flagcxResult_t runUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxFifo_t fifo = hcomm->proxyState->uniRunnerState.fifo;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  flagcxResult_t result = flagcxSuccess;
  TRACE(FLAGCX_UNIRUNNER, "runUniRunner called");
  FLAGCXCHECK(prepareDagStreamFlags(runnerState));

#ifdef COMPILE_KERNEL_HOST
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(
      hcomm->uniRunnerFifoBuffer, runnerState->uniRunnerNThreads,
      runnerState->uniRunnerNBlocks, runnerState->redStream);
#endif

  // Main scheduling loop using DAG-based queue scheduling
  while (true) {
    if (flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->redReadyQueue) &&
        runnerState->numPendingNodes == 0) {
      TRACE(FLAGCX_UNIRUNNER,
            "runUniRunner: all queues empty, terminating runner loop");
      // set terminate flag
      __atomic_store_n(
          flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxTerminate),
          flagcxCompletionWord_t{1}, __ATOMIC_RELEASE);
      break;
    }

    // Step 1: Process ready queue - write triggers to FIFO
    result = processReadyQueue(runnerState, hcomm);
    if (result != flagcxSuccess && result != flagcxInProgress)
      goto fail;

  }
  result = deviceAdaptor->streamSynchronize(runnerState->redStream);
  flagcxRunnerRecordFirstError(
      deviceAdaptor->streamSynchronize(runnerState->cpyStream), &result);
  flagcxRunnerRecordFirstError(
      deviceAdaptor->streamSynchronize(runnerState->commStream), &result);
  if (result != flagcxSuccess && result != flagcxInProgress)
    goto fail;
  return result;

fail:
  // Publish termination before poisoning the proxy so the reduction kernel
  // and host waiters can leave their progress loops.
  if (fifo != nullptr && fifo->buffer != nullptr) {
    __atomic_store_n(flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxTerminate),
                     flagcxCompletionWord_t{1}, __ATOMIC_RELEASE);
  }
  return flagcxProxyRecordAsyncError(hcomm->proxyState, result);
}

flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState) {
  FLAGCXCHECK(cleanupDagScheduler(runnerState));
  return destroyStreamFlagQueue(runnerState);
}
