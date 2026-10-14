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
#include "device_api/flagcx_device.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <math.h>
#include <new>
#include <mutex>
#include <sched.h>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

FLAGCX_PARAM(UniRunnerNSlices, "UNIRUNNER_NSLICES", 1);
FLAGCX_PARAM(UniRunnerLocalNSlices, "UNIRUNNER_LOCAL_NSLICES", 1);
FLAGCX_PARAM(UniRunnerMaxThreads, "UNIRUNNER_MAX_THREADS", 512);
FLAGCX_PARAM(UniRunnerMaxBlocks, "UNIRUNNER_MAX_BLOCKS", 32);
FLAGCX_PARAM(UniRunnerNIpcBlocks, "UNIRUNNER_NIPCBLOCKS", 0);
FLAGCX_PARAM(UniRunnerIpcChunkSize, "UNIRUNNER_IPCCHUNKSIZE", 262144);
FLAGCX_PARAM(UniRunnerNRedSlices, "UNIRUNNER_NREDSLICES", 0);
FLAGCX_PARAM(UniRunnerRedSliceSize, "UNIRUNNER_REDSLICESIZE", 65536);

static flagcxResult_t loadUniRunnerLaunchConfig(uint64_t *maxThreads,
                                                uint64_t *maxBlocks,
                                                uint64_t *nIpcBlocks) {
  const int64_t rawMaxThreads = flagcxParamUniRunnerMaxThreads();
  const int64_t rawMaxBlocks = flagcxParamUniRunnerMaxBlocks();
  const int64_t rawNIpcBlocks = flagcxParamUniRunnerNIpcBlocks();
  const int64_t maxLaunchBlocks = std::numeric_limits<int>::max();

  if (rawMaxThreads < 32 || rawMaxBlocks <= 0 || rawNIpcBlocks < 0) {
    WARN("UniRunner requires MAX_THREADS >= 32, MAX_BLOCKS > 0 and "
         "NIPCBLOCKS >= 0");
    return flagcxInvalidArgument;
  }
  if (rawMaxBlocks > maxLaunchBlocks || rawNIpcBlocks > maxLaunchBlocks ||
      rawMaxBlocks > maxLaunchBlocks - rawNIpcBlocks) {
    WARN("UniRunner executor block count exceeds the supported launch range");
    return flagcxInvalidArgument;
  }

  *maxThreads = static_cast<uint64_t>(rawMaxThreads);
  *maxBlocks = static_cast<uint64_t>(rawMaxBlocks);
  *nIpcBlocks = static_cast<uint64_t>(rawNIpcBlocks);
  return flagcxSuccess;
}

static flagcxResult_t loadUniRunnerDagBuilderConfig(uint64_t *nSlices,
                                                    uint64_t *nRedSlices,
                                                    uint64_t *redSliceSize) {
  const int64_t rawNSlices = flagcxParamUniRunnerNSlices();
  const int64_t rawNRedSlices = flagcxParamUniRunnerNRedSlices();
  const int64_t rawRedSliceSize = flagcxParamUniRunnerRedSliceSize();
  const int64_t maxBuilderCount = std::numeric_limits<int>::max();
  if (nSlices == NULL || nRedSlices == NULL || redSliceSize == NULL) {
    return flagcxInvalidArgument;
  }
  if (rawNSlices <= 0 || rawNSlices > maxBuilderCount ||
      rawNRedSlices < 0 || rawNRedSlices > maxBuilderCount ||
      rawRedSliceSize < 0) {
    WARN("UniRunner requires 0 < NSLICES <= INT_MAX, "
         "0 <= NREDSLICES <= INT_MAX and REDSLICESIZE >= 0");
    return flagcxInvalidArgument;
  }
  *nSlices = static_cast<uint64_t>(rawNSlices);
  *nRedSlices = static_cast<uint64_t>(rawNRedSlices);
  *redSliceSize = static_cast<uint64_t>(rawRedSliceSize);
  return flagcxSuccess;
}

flagcxResult_t validateUniRunnerLaunchConfig(flagcxCommOp_t commOp) {
  uint64_t maxThreads = 0;
  uint64_t maxBlocks = 0;
  uint64_t nIpcBlocks = 0;
  FLAGCXCHECK(
      loadUniRunnerLaunchConfig(&maxThreads, &maxBlocks, &nIpcBlocks));
  uint64_t nSlices = 0;
  uint64_t nRedSlices = 0;
  uint64_t redSliceSize = 0;
  FLAGCXCHECK(loadUniRunnerDagBuilderConfig(&nSlices, &nRedSlices,
                                            &redSliceSize));

  (void)commOp;
  return flagcxSuccess;
}

static flagcxResult_t ensureUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    const void *sendbuff, void *recvbuff, size_t bytes,
    flagcxResult_t localPreparationResult);
static flagcxResult_t ensureUniRunnerIpcReadyStorage(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    size_t requiredSlots);
static flagcxResult_t agreeUniRunnerInvocationState(
    flagcxComm_t comm, uint64_t localSignature, flagcxResult_t localResult,
    flagcxResult_t *collectiveResult);

static bool isValidUniRunnerDataType(flagcxDataType_t datatype) {
  return static_cast<int>(datatype) >= 0 &&
         static_cast<int>(datatype) < flagcxNumTypes &&
         getFlagcxDataTypeSize(datatype) != 0;
}

static bool isValidUniRunnerRedOp(flagcxRedOp_t op) {
  return static_cast<int>(op) >= static_cast<int>(flagcxSum) &&
         static_cast<int>(op) < static_cast<int>(flagcxNumRedOps);
}

flagcxResult_t validateUniRunnerReduceArgs(size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxRedOp_t op) {
  if (!isValidUniRunnerDataType(datatype) || !isValidUniRunnerRedOp(op)) {
    return flagcxInvalidArgument;
  }
  if (count > std::numeric_limits<size_t>::max() /
                  getFlagcxDataTypeSize(datatype)) {
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

flagcxResult_t checkedUniRunnerTypeBytes(size_t count, size_t multiplier,
                                         flagcxDataType_t datatype,
                                         size_t *bytes) {
  if (bytes == NULL || validateUniRunnerReduceArgs(count, datatype,
                                                    flagcxSum) !=
                          flagcxSuccess) {
    return flagcxInvalidArgument;
  }
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  const size_t maxSize = std::numeric_limits<size_t>::max();
  if (multiplier != 0 && count > maxSize / multiplier) {
    return flagcxInvalidArgument;
  }
  const size_t elements = count * multiplier;
  if (typeSize != 0 && elements > maxSize / typeSize) {
    return flagcxInvalidArgument;
  }
  *bytes = elements * typeSize;
  return flagcxSuccess;
}

flagcxResult_t checkedUniRunnerDagNodeCount(size_t outerCount,
                                            size_t nodesPerOuter,
                                            size_t extraNodes,
                                            int *nodeCount) {
  if (nodeCount == NULL ||
      extraNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  const size_t maxProduct =
      static_cast<size_t>(std::numeric_limits<int>::max()) - extraNodes;
  if (outerCount != 0 && nodesPerOuter > maxProduct / outerCount) {
    return flagcxInvalidArgument;
  }
  *nodeCount = static_cast<int>(outerCount * nodesPerOuter + extraNodes);
  return flagcxSuccess;
}

namespace {

constexpr char kUniRunnerDagCachePathEnv[] = "FLAGCX_UNIRUNNER_DAG_CACHE_PATH";
constexpr uint64_t kUniRunnerDagAlgorithmHashSchemaVersion = 1;
constexpr uint64_t kUniRunnerDagFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kUniRunnerDagFnvPrime = 1099511628211ull;

struct uniRunnerDagRuntimeBindings {
  const void *inputBase = NULL;
  void *outputBase = NULL;
  void *scratchBase = NULL;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t scratchBytes = 0;
  int localRanks = 0;
};

static flagcxRedOp_t effectiveUniRunnerRedOp(flagcxRedOp_t requestedOp,
                                             bool isFinalReduction) {
  if (requestedOp == flagcxAvg && !isFinalReduction) {
    return flagcxSum;
  }
  return requestedOp;
}

static void setUniRunnerAvgDivisor(flagcxUniRunnerState *runnerState,
                                   flagcxRedOp_t op, uint64_t divisor) {
  runnerState->avgDivisor = (op == flagcxAvg && divisor != 0) ? divisor : 1;
}

// Compiled entries are process-lifetime objects: insertion is keep-first and
// there is deliberately no erase or replacement path. Runtime plans may
// therefore borrow topoOrder storage without per-invocation copies. Any future
// eviction policy must first add an owning cache pin to each active invocation.
std::mutex gUniRunnerDagCacheMutex;
std::unordered_map<
    size_t,
    std::vector<std::shared_ptr<const uniRunnerCompiledDagTemplate>>>
    gUniRunnerCompiledDagCache;
std::unordered_set<std::string> gUniRunnerDagLoadedPaths;

static size_t hashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

static uint64_t hashAlgorithmUint64(uint64_t hashValue, uint64_t value) {
  // Hash fixed-width integers byte-by-byte so algoHash is stable across host
  // word sizes and standard-library implementations.
  for (unsigned int byte = 0; byte < sizeof(value); ++byte) {
    hashValue ^= value & 0xffu;
    hashValue *= kUniRunnerDagFnvPrime;
    value >>= 8;
  }
  return hashValue;
}

static bool bufferMatchesElementOffset(const void *candidate, const void *base,
                                       size_t count, int rank,
                                       size_t typeSize) {
  if (candidate == NULL || base == NULL || rank < 0 || typeSize == 0) {
    return false;
  }
  const size_t rankValue = static_cast<size_t>(rank);
  const size_t maxSize = std::numeric_limits<size_t>::max();
  if (rankValue != 0 && count > maxSize / rankValue) {
    return false;
  }
  const size_t elementOffset = count * rankValue;
  if (elementOffset > maxSize / typeSize) {
    return false;
  }
  const size_t byteOffset = elementOffset * typeSize;
  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
  if (baseAddress > std::numeric_limits<uintptr_t>::max() - byteOffset) {
    return false;
  }
  return reinterpret_cast<uintptr_t>(candidate) == baseAddress + byteOffset;
}

static int positiveModulo(int64_t value, int modulus) {
  assert(modulus > 0);
  const int64_t remainder = value % static_cast<int64_t>(modulus);
  return static_cast<int>(remainder < 0 ? remainder + modulus : remainder);
}

static bool uniRunnerDagCacheKeysEqual(const uniRunnerDagCacheKey &lhs,
                                       const uniRunnerDagCacheKey &rhs) {
  return lhs.algoHash == rhs.algoHash && lhs.commOp == rhs.commOp &&
         lhs.count == rhs.count && lhs.datatype == rhs.datatype &&
         lhs.redOp == rhs.redOp && lhs.rank == rhs.rank &&
         lhs.nranks == rhs.nranks && lhs.root == rhs.root;
}

static flagcxResult_t insertUniRunnerCompiledDagTemplateLocked(
    const std::shared_ptr<const uniRunnerCompiledDagTemplate> &compiled,
    std::shared_ptr<const uniRunnerCompiledDagTemplate> *canonical) {
  const uniRunnerDagTemplate &dagTemplate = compiled->dagTemplate;
  std::vector<std::shared_ptr<const uniRunnerCompiledDagTemplate>> &bucket =
      gUniRunnerCompiledDagCache[dagTemplate.hashValue];
  for (const std::shared_ptr<const uniRunnerCompiledDagTemplate> &entry :
       bucket) {
    if (uniRunnerDagCacheKeysEqual(entry->dagTemplate.key, dagTemplate.key)) {
      if (canonical != NULL) {
        *canonical = entry;
      }
      return flagcxSuccess;
    }
  }
  bucket.push_back(compiled);
  if (canonical != NULL) {
    *canonical = compiled;
  }
  return flagcxSuccess;
}

static bool findUniRunnerCompiledDagTemplateLocked(
    size_t hashValue, const uniRunnerDagCacheKey &key,
    std::shared_ptr<const uniRunnerCompiledDagTemplate> *compiled) {
  std::unordered_map<
      size_t,
      std::vector<std::shared_ptr<const uniRunnerCompiledDagTemplate>>>::
      iterator it = gUniRunnerCompiledDagCache.find(hashValue);
  if (it == gUniRunnerCompiledDagCache.end()) {
    return false;
  }
  for (const std::shared_ptr<const uniRunnerCompiledDagTemplate> &entry :
       it->second) {
    if (uniRunnerDagCacheKeysEqual(entry->dagTemplate.key, key)) {
      *compiled = entry;
      return true;
    }
  }
  return false;
}

static flagcxResult_t importUniRunnerDagTemplates(
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  try {
    std::vector<std::shared_ptr<const uniRunnerCompiledDagTemplate>> compiled;
    compiled.reserve(dagTemplates.size());
    for (const uniRunnerDagTemplate &dagTemplate : dagTemplates) {
      std::shared_ptr<uniRunnerCompiledDagTemplate> entry =
          std::make_shared<uniRunnerCompiledDagTemplate>();
      FLAGCXCHECK(compileUniRunnerDagTemplate(dagTemplate, entry.get()));
      compiled.push_back(entry);
    }
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    for (const auto &entry : compiled) {
      FLAGCXCHECK(insertUniRunnerCompiledDagTemplateLocked(entry, NULL));
    }
  } catch (...) {
    return flagcxSystemError;
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

  FLAGCXCHECK(importUniRunnerDagTemplates(dagTemplates));
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  gUniRunnerDagLoadedPaths.insert(path);
  return flagcxSuccess;
}

static flagcxResult_t
captureBufferRef(const void *ptr, const uniRunnerDagRuntimeBindings &bindings,
                 size_t accessBytes, uniRunnerDagBufferRef *ref) {
  if (ptr == NULL) {
    ref->bufferType = uniRunnerDagBufferTypeNone;
    ref->offsetBytes = 0;
    return accessBytes == 0 ? flagcxSuccess : flagcxInvalidArgument;
  }

  int64_t offsetBytes = 0;
  if (uniRunnerDagBindingRangeContains(ptr, bindings.inputBase,
                                       bindings.inputBytes, accessBytes,
                                       &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeInput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (uniRunnerDagBindingRangeContains(ptr, bindings.outputBase,
                                       bindings.outputBytes, accessBytes,
                                       &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeOutput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (uniRunnerDagBindingRangeContains(ptr, bindings.scratchBase,
                                       bindings.scratchBytes, accessBytes,
                                       &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeScratch;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  return flagcxInvalidArgument;
}

static flagcxResult_t
resolveBufferRef(const uniRunnerDagBufferRef &ref,
                 const uniRunnerDagRuntimeBindings &bindings,
                 size_t accessBytes, void **ptr) {
  const char *base = NULL;
  size_t bytes = 0;
  switch (ref.bufferType) {
    case uniRunnerDagBufferTypeNone:
      *ptr = NULL;
      return accessBytes == 0 ? flagcxSuccess : flagcxInvalidArgument;
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
      static_cast<size_t>(ref.offsetBytes) > bytes ||
      accessBytes > bytes - static_cast<size_t>(ref.offsetBytes)) {
    return flagcxInvalidArgument;
  }

  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
  const uintptr_t offset = static_cast<uintptr_t>(ref.offsetBytes);
  if (baseAddress > std::numeric_limits<uintptr_t>::max() - offset) {
    return flagcxInvalidArgument;
  }
  *ptr = reinterpret_cast<void *>(baseAddress + offset);
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

  const size_t maxSize = std::numeric_limits<size_t>::max();
  size_t divisor = static_cast<size_t>(nranks);
  const size_t nSlices = static_cast<size_t>(runnerState->uniRunnerNSlices);
  const size_t redSliceSize =
      static_cast<size_t>(runnerState->uniRunnerRedSliceSize);
  // If the mathematical divisor exceeds SIZE_MAX, it is necessarily larger
  // than count and the positive-count ceil division is exactly one.
  if (divisor > maxSize / nSlices) {
    return 1;
  }
  divisor *= nSlices;
  if (divisor > maxSize / redSliceSize) {
    return 1;
  }
  divisor *= redSliceSize;
  return std::max<size_t>(1, count / divisor + (count % divisor != 0));
}

static flagcxResult_t setEffectiveUniRunnerRedSlices(
    flagcxUniRunnerState *runnerState, size_t count, int nranks) {
  if (runnerState == NULL || nranks <= 0) {
    return flagcxInvalidArgument;
  }
  const size_t resolved =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, nranks);
  if (resolved >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  runnerState->uniRunnerNRedSlices = static_cast<uint64_t>(resolved);
  return flagcxSuccess;
}

static uniRunnerDagCacheKey makeUniRunnerDagCacheKey(
    uniRunnerDagAlgoType algoType,
    const uniRunnerDagAlgorithmConfig &algorithmConfig,
    flagcxCommOp_t commOp, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t redOp, int root, flagcxComm_t comm) {
  uniRunnerDagCacheKey key{};
  key.algoHash = getUniRunnerDagAlgorithmHash(algoType, algorithmConfig);
  key.commOp = commOp;
  key.count = count;
  key.datatype = datatype;
  key.redOp = redOp;
  key.rank = comm->rank;
  key.nranks = comm->nranks;
  key.root = root;
  return key;
}

static bool
findUniRunnerCompiledDagTemplate(
    size_t hashValue, const uniRunnerDagCacheKey &key,
    std::shared_ptr<const uniRunnerCompiledDagTemplate> *compiled) {
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  return findUniRunnerCompiledDagTemplateLocked(hashValue, key, compiled);
}

static flagcxResult_t loadUniRunnerDagFromCacheDir(size_t hashValue, int rank) {
  std::string cacheDir = getUniRunnerDagCacheDirFromEnv();
  if (cacheDir.empty()) {
    return flagcxSuccess;
  }
  return loadUniRunnerDagFileIntoCache(
      makeUniRunnerDagFilePath(cacheDir, hashValue, rank));
}

static flagcxResult_t cacheUniRunnerCompiledDagTemplate(
    const std::shared_ptr<const uniRunnerCompiledDagTemplate> &compiled) {
  const uniRunnerDagTemplate &dagTemplate = compiled->dagTemplate;
  size_t hashValue = getUniRunnerDagPatternHash(dagTemplate.key);
  std::shared_ptr<const uniRunnerCompiledDagTemplate> canonical;
  std::string dagDir = getUniRunnerDagCacheDirFromEnv();
  std::string dagPath =
      makeUniRunnerDagFilePath(dagDir, hashValue, dagTemplate.key.rank);
  {
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    FLAGCXCHECK(
        insertUniRunnerCompiledDagTemplateLocked(compiled, &canonical));
  }
  if (dagDir.empty()) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(ensureUniRunnerDagDirExists(dagDir));
  FLAGCXCHECK(uniRunnerSaveDagJsonFile(dagPath, canonical->dagTemplate));
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  gUniRunnerDagLoadedPaths.insert(dagPath);
  return flagcxSuccess;
}

} // namespace

uint64_t getUniRunnerDagAlgorithmHash(
    uniRunnerDagAlgoType algoType,
    const uniRunnerDagAlgorithmConfig &algorithmConfig) {
  uint64_t hashValue = kUniRunnerDagFnvOffsetBasis;
  hashValue =
      hashAlgorithmUint64(hashValue, kUniRunnerDagAlgorithmHashSchemaVersion);
  hashValue = hashAlgorithmUint64(hashValue, static_cast<uint64_t>(algoType));
  switch (algoType) {
    case uniRunnerDagAlgoLocRed:
    case uniRunnerDagAlgoRingAG:
    case uniRunnerDagAlgoRingAR:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoGroupedAR:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numLocalSlices);
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numRedSlices);
      if (algorithmConfig.pipelineVersion != 0)
        hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.pipelineVersion);
      // The remaining fields are shared with grouped AG.
      [[fallthrough]];
    case uniRunnerDagAlgoGroupedAG:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.groupSize);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoSlicedAR:
    case uniRunnerDagAlgoRingRS:
    case uniRunnerDagAlgoTreeRed:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue =
          hashAlgorithmUint64(hashValue, algorithmConfig.numRedSlices);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoIpcAR:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue =
          hashAlgorithmUint64(hashValue, algorithmConfig.numRedSlices);
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.topologyHash);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoDummy:
    default:
      break;
  }
  return hashValue;
}

flagcxResult_t getUniRunnerIpcTopologyHash(int localRanks, int nranks,
                                           const int *localRankToRank,
                                           uint64_t *topologyHash) {
  if (localRanks <= 0 || nranks <= 0 || localRanks != nranks ||
      nranks > FLAGCX_MAX_LOCAL_RANKS || localRankToRank == NULL ||
      topologyHash == NULL) {
    return localRanks > 0 && nranks > 0 && localRanks != nranks
               ? flagcxNotSupported
               : flagcxInvalidArgument;
  }
  for (int localRank = 0; localRank < localRanks; ++localRank) {
    const int globalRank = localRankToRank[localRank];
    if (globalRank < 0 || globalRank >= nranks) {
      return flagcxInvalidArgument;
    }
    for (int previous = 0; previous < localRank; ++previous) {
      if (localRankToRank[previous] == globalRank) {
        return flagcxInvalidArgument;
      }
    }
  }

  uint64_t hashValue = kUniRunnerDagFnvOffsetBasis;
  hashValue = hashAlgorithmUint64(hashValue,
                                  static_cast<uint64_t>(localRanks));
  for (int localRank = 0; localRank < localRanks; ++localRank) {
    hashValue = hashAlgorithmUint64(
        hashValue, static_cast<uint64_t>(localRankToRank[localRank]));
  }
  *topologyHash = hashValue;
  return flagcxSuccess;
}

flagcxResult_t validateUniRunnerIpcDagTemplateBindings(
    const uniRunnerDagTemplate &dagTemplate, size_t inputBytes,
    size_t outputBytes, int localRanks) {
  size_t numIpcNodes = 0;
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    if (node.nodeType == uniRunnerDagNodeTypeIpc) {
      ++numIpcNodes;
    }
  }
  if (numIpcNodes == 0) {
    return flagcxSuccess;
  }
  if (localRanks <= 0 ||
      numIpcNodes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return flagcxInvalidArgument;
  }

  std::vector<uint8_t> readySlots;
  try {
    readySlots.assign(numIpcNodes, uint8_t{0});
  } catch (...) {
    return flagcxSystemError;
  }
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    if (node.nodeType != uniRunnerDagNodeTypeIpc) {
      continue;
    }
    const uniRunnerDagIpcOpDesc &ipc = node.ipc;
    if (ipc.srcBufferType != flagcxIpcBufferInput &&
        ipc.srcBufferType != flagcxIpcBufferOutput) {
      return flagcxInvalidArgument;
    }
    const size_t srcBytes = ipc.srcBufferType == flagcxIpcBufferInput
                                ? inputBytes
                                : outputBytes;
    if (ipc.srcOffsetBytes > srcBytes ||
        ipc.bytes > srcBytes - ipc.srcOffsetBytes ||
        ipc.dstOffsetBytes > outputBytes ||
        ipc.bytes > outputBytes - ipc.dstOffsetBytes ||
        ipc.peerLocalRank < 0 || ipc.peerLocalRank >= localRanks ||
        static_cast<size_t>(ipc.readySlot) >= numIpcNodes ||
        readySlots[ipc.readySlot] != 0) {
      return flagcxInvalidArgument;
    }
    readySlots[ipc.readySlot] = 1;
  }
  // There are exactly numIpcNodes unique slots in [0, numIpcNodes), which is
  // equivalent to the required dense range. Keep the explicit check as a
  // guard if validation above is later refactored.
  for (uint8_t present : readySlots) {
    if (present == 0) {
      return flagcxInvalidArgument;
    }
  }
  return flagcxSuccess;
}

size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key) {
  size_t hashValue = 0;
  // Cache format is serialization metadata rather than a collective semantic
  // key field. Salt the persisted pattern hash so incompatible formats never
  // share a cache filename.
  hashValue = hashCombine(
      hashValue, static_cast<size_t>(kUniRunnerDagCacheFormatVersion));
  hashValue = hashCombine(
      hashValue,
      static_cast<size_t>(static_cast<uint32_t>(key.algoHash & 0xffffffffu)));
  hashValue = hashCombine(
      hashValue, static_cast<size_t>(static_cast<uint32_t>(key.algoHash >> 32)));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.commOp));
  hashValue = hashCombine(hashValue, key.count);
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.datatype));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redOp));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.rank));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nranks));
  hashValue = hashCombine(
      hashValue,
      static_cast<size_t>(static_cast<int64_t>(key.root) + int64_t{1}));
  return hashValue;
}

static flagcxResult_t allocDagNodeDeps(uniRunnerDagNode *node) {
  node->pendingParents = 0;
  if (node->numParents > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->parents, node->numParents));
  }
  if (node->numChildren > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->children, node->numChildren));
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
  if (runnerState == NULL) {
    return flagcxInvalidArgument;
  }
  if (runnerState->numDagNodes < 0) {
    return flagcxInternalError;
  }
  if (runnerState->numDagNodes == 0) {
    return runnerState->dagNodes == NULL ? flagcxSuccess
                                         : flagcxInternalError;
  }
  if (runnerState->dagNodes == NULL) {
    return flagcxInternalError;
  }

  const int numDagNodes = runnerState->numDagNodes;
  uniRunnerDagNode *dagNodes = runnerState->dagNodes;
  size_t numEdges = 0;

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    if (node->nodeIdx != i) {
      return flagcxInternalError;
    }
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
    switch (node->nodeType) {
      case uniRunnerDagNodeTypeP2p:
        if (node->nodeData.p2p.numOps < 0 ||
            (node->nodeData.p2p.numOps > 0 &&
             node->nodeData.p2p.ops == NULL)) {
          return flagcxInternalError;
        }
        for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; ++opIdx) {
          if (!isValidUniRunnerDataType(
                  node->nodeData.p2p.ops[opIdx].datatype)) {
            return flagcxInvalidArgument;
          }
        }
        break;
      case uniRunnerDagNodeTypeRed:
        if (runnerState->uniRunnerNRedBlocks == 0) {
          WARN("UniRunner DAG contains RED work but NREDBLOCKS is zero");
          return flagcxInvalidArgument;
        }
        if (!isValidUniRunnerDataType(node->nodeData.red.datatype) ||
            !isValidUniRunnerRedOp(node->nodeData.red.redOp) ||
            node->nodeData.red.count >
                flagcxTriggerMask(flagcxReduceTriggerBitsCount) ||
            node->nodeData.red.nthreads == 0 ||
            node->nodeData.red.nthreads >
                flagcxTriggerMask(flagcxReduceTriggerBitsNThreads) ||
            node->nodeData.red.nthreads != runnerState->uniRunnerNThreads) {
          return flagcxInvalidArgument;
        }
        break;
      case uniRunnerDagNodeTypeCpy:
        if (!isValidUniRunnerDataType(node->nodeData.cpy.datatype)) {
          return flagcxInvalidArgument;
        }
        break;
      case uniRunnerDagNodeTypeIpc:
        break;
      default:
        return flagcxInternalError;
    }
    if (static_cast<size_t>(node->numParents) >
        std::numeric_limits<size_t>::max() - numEdges) {
      return flagcxInvalidArgument;
    }
    numEdges += static_cast<size_t>(node->numParents);
  }

  try {
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
        uint64_t edge =
            (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
            static_cast<uint32_t>(childIdx);
        std::unordered_set<uint64_t>::iterator it = dagEdges.find(edge);
        if (it == dagEdges.end()) {
          return flagcxInternalError;
        }
        dagEdges.erase(it);
      }
    }

    return dagEdges.empty() ? flagcxSuccess : flagcxInternalError;
  } catch (...) {
    return flagcxSystemError;
  }
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
  if (newCapacity >
      std::numeric_limits<size_t>::max() / sizeof(*runnerState->streamFlags)) {
    return flagcxInvalidArgument;
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
    const size_t maxSize = std::numeric_limits<size_t>::max();
    const size_t maxCapacity =
        std::min(maxSize / sizeof(*runnerState->streamFlags),
                 maxSize / sizeof(uint64_t));
    if (requiredFlags > maxCapacity) {
      return flagcxInvalidArgument;
    }
    size_t newCapacity = runnerState->streamFlagsCapacity == 0
                             ? 1
                             : runnerState->streamFlagsCapacity;
    while (newCapacity < requiredFlags) {
      if (newCapacity > maxCapacity / 2) {
        newCapacity = requiredFlags;
        break;
      }
      newCapacity *= 2;
    }

    void *newPool = NULL;
    flagcxResult_t res = deviceAdaptor->deviceMalloc(
        &newPool, newCapacity * sizeof(uint64_t), flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      return res;
    }
    res = resizeStreamFlagAddressQueue(runnerState, newCapacity);
    if (res != flagcxSuccess) {
      (void)deviceAdaptor->deviceFree(newPool, flagcxMemDevice, NULL);
      return res;
    }
    if (runnerState->streamFlagsPool != NULL) {
      res = deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                      flagcxMemDevice, NULL);
      if (res != flagcxSuccess) {
        (void)deviceAdaptor->deviceFree(newPool, flagcxMemDevice, NULL);
        return res;
      }
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
      return node->commDomain == uniRunnerDagCommNet
                 ? runnerState->netCommStream : runnerState->ipcCommStream;
    case uniRunnerDagNodeTypeRed:
      return runnerState->redStream;
    case uniRunnerDagNodeTypeCpy:
      return runnerState->cpyStream;
    case uniRunnerDagNodeTypeIpc:
      return runnerState->redStream;
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
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
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
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
        effectiveUniRunnerRedOp(op, true);

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

// Contiguous groups use the same rank/group mapping as grouped AG. Each
// local lane owns a disjoint chunk and drives its own inter-group ring.
static flagcxResult_t buildUniRunnerStateGroupedAR(
    flagcxUniRunnerState *state, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm, int groupSize, int nLocalSlices) {
  if (count == 0) return validateDagNodes(state);
  const int rank = comm->rank;
  const int nGroups = comm->nranks / groupSize;
  const int group = rank / groupSize;
  const int lane = rank % groupSize;
  const int nSlices = state->uniRunnerNSlices;
  const int nRedSlices = state->uniRunnerNRedSlices;
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  int localSteps = 0, perSlice = 0, capacity = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      groupSize - 1, nLocalSlices, 0, &localSteps));
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      static_cast<size_t>(localSteps) + nGroups - 1,
      static_cast<size_t>(nRedSlices) + 1,
      static_cast<size_t>(nGroups - 1) + 2, &perSlice));
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(nSlices, perSlice, 2, &capacity));
  FLAGCXCHECK(flagcxCalloc(&state->dagNodes, capacity));
  std::vector<std::vector<int>> parents(capacity), children(capacity);
  std::vector<int> ready;
  uniRunnerDagCommDomain domain = uniRunnerDagCommIpc;
  size_t scratchOffset = 0;

  struct Range { size_t offset; size_t count; };
  auto partition = [](Range r, int parts, int index) -> Range {
    const size_t base = r.count / parts;
    const size_t rem = r.count % parts;
    return {r.offset + index * base + std::min<size_t>(index, rem),
            base + (static_cast<size_t>(index) < rem)};
  };
  auto chunkSlice = [&](int localLane, int slice) -> Range {
    return partition(partition({0, count}, groupSize, localLane), nSlices,
                     slice);
  };
  auto ptr = [&](const void *base, size_t offset) -> void * {
    return static_cast<char *>(const_cast<void *>(base)) + offset * typeSize;
  };
  auto node = [&](uniRunnerDagNodeType type, const std::vector<int> &deps) {
    const int id = state->numDagNodes++;
    state->dagNodes[id].nodeIdx = id;
    state->dagNodes[id].nodeType = type;
    state->dagNodes[id].commDomain = domain;
    parents[id] = deps;
    for (int dep : deps) children[dep].push_back(id);
    return id;
  };
  auto exchange = [&](int sendPeer, Range tx, const void *src,
                      int recvPeer, Range rx, void *dst) -> flagcxResult_t {
    const int id = node(uniRunnerDagNodeTypeP2p, ready);
    auto &p2p = state->dagNodes[id].nodeData.p2p;
    p2p.numOps = 2;
    FLAGCXCHECK(flagcxCalloc(&p2p.ops, 2));
    p2p.ops[0] = {ptr(src, tx.offset), tx.count, sendPeer, datatype,
                  flagcxDevicePrimSend};
    p2p.ops[1] = {ptr(dst, rx.offset), rx.count, recvPeer, datatype,
                  flagcxDevicePrimRecv};
    ready = {id};
    return flagcxSuccess;
  };
  auto reduce = [&](Range output, bool final) {
    const std::vector<int> deps = ready;
    std::vector<int> reductions;
    // All RED nodes for a step run in parallel. Scratch is reused only after
    // every reduction completes, including when counts are not divisible.
    for (int r = 0; r < nRedSlices; ++r) {
      const Range sub = partition({0, output.count}, nRedSlices, r);
      if (sub.count == 0) continue;
      const int id = node(uniRunnerDagNodeTypeRed, deps);
      auto &red = state->dagNodes[id].nodeData.red;
      red.input1 = ptr(recvbuff, output.offset + sub.offset);
      red.input2 = ptr(state->groupedArScratch, scratchOffset + sub.offset);
      red.output = red.input1;
      red.count = sub.count;
      red.nthreads = state->uniRunnerNThreads;
      red.datatype = datatype;
      red.redOp = effectiveUniRunnerRedOp(op, final);
      red.triggerIdx = -1;
      reductions.push_back(id);
    }
    if (!reductions.empty()) ready = std::move(reductions);
  };

  // Publish the caller stream's preceding work before a copy on cpyStream.
  // This also orders asynchronous input producers with the first local copy.
  ready = {node(uniRunnerDagNodeTypeP2p, ready)};
  // Initialize the entire lane before communication. A later slice's copy
  // must not wait on earlier P2P work: stream-value waits are invisible to
  // CUDA scheduling and can block the copies needed to finish that P2P work.
  const Range ownLane = partition({0, count}, groupSize, lane);
  if (sendbuff != recvbuff && ownLane.count != 0) {
    const int id = node(uniRunnerDagNodeTypeCpy, ready);
    auto &cpy = state->dagNodes[id].nodeData.cpy;
    cpy.src = ptr(sendbuff, ownLane.offset);
    cpy.dst = ptr(recvbuff, ownLane.offset);
    cpy.count = ownLane.count;
    cpy.datatype = datatype;
    ready = {id};
  }
  const size_t maxChunk = count / groupSize + (count % groupSize != 0);
  const size_t maxSlice = maxChunk / nSlices + (maxChunk % nSlices != 0);
  const size_t netScratchOffset = groupSize > 1 ? maxSlice : 0;
  auto mergeDeps = [](std::vector<int> deps, const std::vector<int> &extra) {
    for (int id : extra)
      if (std::find(deps.begin(), deps.end(), id) == deps.end())
        deps.push_back(id);
    return deps;
  };
  auto localReduceScatter = [&](int s, const std::vector<int> &deps,
                                std::vector<int> *done) -> flagcxResult_t {
    ready = deps;
    domain = uniRunnerDagCommIpc;
    scratchOffset = 0;
    const Range own = chunkSlice(lane, s);

    // Each inner slice has a disjoint scratch/output range. Advance P2P
    // independently of RED, waiting only before reusing the same range.
    std::vector<int> commTail = ready;
    std::vector<std::vector<int>> redTails(groupSize > 1 ? nLocalSlices : 0);
    for (int step = 1; step < groupSize; ++step) {
      const int txLane = positiveModulo(static_cast<int64_t>(lane) + step,
                                        groupSize);
      const int rxLane = positiveModulo(static_cast<int64_t>(lane) - step,
                                        groupSize);
      for (int u = 0; u < nLocalSlices; ++u) {
        const Range output = partition(own, nLocalSlices, u);
        const Range tx = partition(chunkSlice(txLane, s), nLocalSlices, u);
        scratchOffset = output.offset - own.offset;
        ready = mergeDeps(commTail, redTails[u]);
        FLAGCXCHECK(exchange(group * groupSize + txLane, tx, sendbuff,
                             group * groupSize + rxLane,
                             {scratchOffset, output.count},
                             state->groupedArScratch));
        commTail = ready;
        // RED retains a single P2P parent; scratch reuse is gated above.
        reduce(output, nGroups == 1 && step == groupSize - 1);
        redTails[u] = ready;
      }
    }
    ready = std::move(commTail);
    for (const auto &tail : redTails)
      ready = mergeDeps(std::move(ready), tail);

    // Publish this slice's RED completion on the IPC stream so the network
    // handoff can use a CUDA-visible event dependency.
    if (nGroups > 1) ready = {node(uniRunnerDagNodeTypeP2p, ready)};
    *done = ready;
    return flagcxSuccess;
  };
  auto networkAllReduce = [&](int s, const std::vector<int> &deps,
                              std::vector<int> *done) -> flagcxResult_t {
    ready = deps;
    domain = uniRunnerDagCommNet;
    scratchOffset = netScratchOffset;
    const Range own = chunkSlice(lane, s);
    // Inter-group reduce-scatter on each independent NIC lane. Divide the
    // lane chunk again across groups to keep network volume bandwidth optimal.
    const int next = ((group + 1) % nGroups) * groupSize + lane;
    const int prev = positiveModulo(static_cast<int64_t>(group) - 1,
                                    nGroups) * groupSize + lane;
    for (int step = 0; step < nGroups - 1; ++step) {
      const Range tx = partition(own, nGroups, positiveModulo(
          static_cast<int64_t>(group) - step, nGroups));
      const Range rx = partition(own, nGroups, positiveModulo(
          static_cast<int64_t>(group) - step - 1, nGroups));
      FLAGCXCHECK(exchange(next, tx, recvbuff, prev, {scratchOffset, rx.count},
                           state->groupedArScratch));
      reduce(rx, step == nGroups - 2);
    }
    // Inter-group allgather completes the lane's globally reduced chunk.
    for (int step = 0; step < nGroups - 1; ++step) {
      const Range tx = partition(own, nGroups, positiveModulo(
          static_cast<int64_t>(group) - step + 1, nGroups));
      const Range rx = partition(own, nGroups, positiveModulo(
          static_cast<int64_t>(group) - step, nGroups));
      FLAGCXCHECK(exchange(next, tx, recvbuff, prev, rx, recvbuff));
    }

    *done = ready;
    return flagcxSuccess;
  };
  auto localAllGather = [&](int s, const std::vector<int> &deps,
                            std::vector<int> *done) -> flagcxResult_t {
    ready = deps;
    domain = uniRunnerDagCommIpc;
    const Range own = chunkSlice(lane, s);
    // Intra-group direct allgather, matching grouped AG's peer ordering.
    // All input sends for this slice precede these receives (also in-place).
    if (groupSize > 1) {
      const int id = node(uniRunnerDagNodeTypeP2p, ready);
      auto &p2p = state->dagNodes[id].nodeData.p2p;
      FLAGCXCHECK(checkedUniRunnerDagNodeCount(2, groupSize - 1, 0,
                                               &p2p.numOps));
      FLAGCXCHECK(flagcxCalloc(&p2p.ops, p2p.numOps));
      for (int step = 1; step < groupSize; ++step) {
        const int txLane = positiveModulo(static_cast<int64_t>(lane) + step,
                                          groupSize);
        const int rxLane = positiveModulo(static_cast<int64_t>(lane) - step,
                                          groupSize);
        const Range rx = chunkSlice(rxLane, s);
        p2p.ops[2 * (step - 1)] = {
            ptr(recvbuff, own.offset), own.count, group * groupSize + txLane,
            datatype, flagcxDevicePrimSend};
        p2p.ops[2 * (step - 1) + 1] = {
            ptr(recvbuff, rx.offset), rx.count, group * groupSize + rxLane,
            datatype, flagcxDevicePrimRecv};
      }
      ready = {id};
    }
    *done = ready;
    return flagcxSuccess;
  };

  std::vector<int> localDone, ipcTail, netTail;
  FLAGCXCHECK(localReduceScatter(0, ready, &localDone));
  ipcTail = localDone;
  for (int s = 0; s < nSlices; ++s) {
    std::vector<int> netDone, nextLocalDone;
    FLAGCXCHECK(networkAllReduce(s, mergeDeps(localDone, netTail), &netDone));
    netTail = netDone;
    // Keep one local slice ahead of the network. The local stream order is
    // RS(0), RS(1), AG(0), RS(2), AG(1), ...; each domain reuses only its
    // own scratch after its preceding reductions have completed.
    if (s + 1 < nSlices) {
      FLAGCXCHECK(localReduceScatter(s + 1, ipcTail, &nextLocalDone));
      ipcTail = nextLocalDone;
    }
    FLAGCXCHECK(localAllGather(s, mergeDeps(ipcTail, netDone), &ipcTail));
    localDone = std::move(nextLocalDone);
  }
  if (state->numDagNodes == 0) {
    free(state->dagNodes);
    state->dagNodes = NULL;
  }
  for (int id = 0; id < state->numDagNodes; ++id) {
    auto &n = state->dagNodes[id];
    n.numParents = parents[id].size();
    n.numChildren = children[id].size();
    FLAGCXCHECK(allocDagNodeDeps(&n));
    for (int i = 0; i < n.numParents; ++i)
      FLAGCXCHECK(setDagNodeParent(&n, i, parents[id][i]));
    std::copy(children[id].begin(), children[id].end(), n.children);
    if (n.numParents == 0)
      flagcxIntruQueueEnqueue(&state->p2pReadyQueue, &n);
    else
      ++state->numPendingNodes;
  }
  TRACE(FLAGCX_UNIRUNNER,
        "GroupedAR rank %d: groupSize=%d nGroups=%d slices=%d "
        "localSlices=%d redSlices=%d",
        rank, groupSize, nGroups, nSlices, nLocalSlices, nRedSlices);
  return validateDagNodes(state);
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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(1, nGroups, 1, &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  size_t localBaseOffset =
      static_cast<size_t>(groupIdx) * groupChunkCount * typeSize;
  for (int step = 0; step < nGroups; step++) {
    int nodeIdx = step;
    bool isLastStep = (step == nGroups - 1);
    int numIntraPeers = groupSize - 1;
    int numOps = 0;
    FLAGCXCHECK(checkedUniRunnerDagNodeCount(
        2, numIntraPeers, isLastStep ? 0 : 2, &numOps));

    runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
    runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
    runnerState->dagNodes[nodeIdx].nodeData.p2p.numOps = numOps;
    if (numOps > 0) {
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[nodeIdx].nodeData.p2p.ops, numOps));
    }

    for (int i = 0; i < numIntraPeers; i++) {
      int locSendPeer = positiveModulo(
          static_cast<int64_t>(locRank) + i + 1, groupSize);
      int locRecvPeer = positiveModulo(
          static_cast<int64_t>(locRank) - i - 1, groupSize);

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
      size_t sendGroupIdx = positiveModulo(
          static_cast<int64_t>(groupIdx) + step + 1, nGroups);
      size_t recvGroupIdx = positiveModulo(
          static_cast<int64_t>(groupIdx) - step - 1, nGroups);
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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 1,
                                           &runnerState->numDagNodes));
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  int nodesPerSlice = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(3, nranks - 1, 0, &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
          effectiveUniRunnerRedOp(op, i == nranks - 2);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i + 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);

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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateSlicedAR called, count=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce * numRedSlices) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  int nodesPerStep = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(1, numRedSlices, 2, &nodesPerStep));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(nodesPerStep, nranks - 1, 0,
                                           &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(op, i == nranks - 2);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i + 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);

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

static bool getIpcBufferOffset(const void *ptr, const void *base, size_t bytes,
                               size_t accessBytes, size_t *offset) {
  if (offset == NULL) {
    return false;
  }
  // One-past-end is valid for a zero-count slice; the trigger will carry
  // bytes==0 and only exchange its ready epoch.
  int64_t checkedOffset = 0;
  if (!uniRunnerDagBindingRangeContains(ptr, base, bytes, accessBytes,
                                        &checkedOffset)) {
    return false;
  }
  *offset = static_cast<size_t>(checkedOffset);
  return true;
}

static int globalRankToLocalRank(flagcxComm_t comm, int globalRank) {
  for (int localRank = 0; localRank < comm->localRanks; ++localRank) {
    if (comm->localRankToRank[localRank] == globalRank)
      return localRank;
  }
  return -1;
}

static flagcxResult_t buildUniRunnerStateIpcAR(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm) {
  if (comm->localRanks != comm->nranks)
    return flagcxNotSupported;
  if (comm->nranks <= 1 || count == 0) {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  }

  FLAGCXCHECK(buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                          count, datatype, op, comm));

  size_t totalBytes = 0;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 1, datatype, &totalBytes));
  uint32_t readySlot = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeP2p)
      continue;

    const uniRunnerP2pOpData *sendOp = NULL;
    const uniRunnerP2pOpData *recvOp = NULL;
    for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; ++opIdx) {
      const uniRunnerP2pOpData *candidate = &node->nodeData.p2p.ops[opIdx];
      if (candidate->type == flagcxDevicePrimSend)
        sendOp = candidate;
      else if (candidate->type == flagcxDevicePrimRecv)
        recvOp = candidate;
    }
    if (sendOp == NULL || recvOp == NULL) {
      return flagcxNotSupported;
    }

    size_t sendBytes = 0;
    FLAGCXCHECK(checkedUniRunnerTypeBytes(sendOp->count, 1,
                                          sendOp->datatype, &sendBytes));
    size_t recvBytes = 0;
    FLAGCXCHECK(checkedUniRunnerTypeBytes(recvOp->count, 1,
                                          recvOp->datatype, &recvBytes));

    size_t srcOffset = 0;
    flagcxIpcBufferType srcBufferType = flagcxIpcBufferInput;
    // Prefer output for aliased buffers: both handles describe the same local
    // storage and output is the destination surface carried around the ring.
    if (getIpcBufferOffset(sendOp->addr, recvbuff, totalBytes, sendBytes,
                           &srcOffset)) {
      srcBufferType = flagcxIpcBufferOutput;
    } else if (getIpcBufferOffset(sendOp->addr, sendbuff, totalBytes,
                                  sendBytes, &srcOffset)) {
      srcBufferType = flagcxIpcBufferInput;
    } else {
      return flagcxInvalidArgument;
    }

    size_t localRecvOffset = 0;
    if (!getIpcBufferOffset(recvOp->addr, recvbuff, totalBytes,
                            recvBytes, &localRecvOffset)) {
      return flagcxInvalidArgument;
    }
    (void)localRecvOffset;
    int peerLocalRank = globalRankToLocalRank(comm, sendOp->peerRank);
    if (peerLocalRank < 0)
      return flagcxNotSupported;

    uniRunnerP2pOpData *oldOps = node->nodeData.p2p.ops;
    uniRunnerIpcNodeData ipc = {};
    ipc.srcOffsetBytes = srcOffset;
    // Ring chunks retain their global output-buffer offset at every rank. The
    // local recv op refers to the chunk arriving from prevRank, while this
    // send op writes its own source chunk to nextRank at srcOffset.
    ipc.dstOffsetBytes = srcOffset;
    ipc.bytes = sendBytes;
    if (ipc.bytes > totalBytes - srcOffset ||
        recvBytes > totalBytes - localRecvOffset ||
        ipc.bytes > totalBytes - ipc.dstOffsetBytes) {
      return flagcxInvalidArgument;
    }
    ipc.srcBufferType = srcBufferType;
    ipc.peerLocalRank = peerLocalRank;
    if (readySlot == std::numeric_limits<uint32_t>::max()) {
      return flagcxInvalidArgument;
    }
    ipc.readySlot = readySlot++;
    ipc.parentFlagsOffset = 0;
    ipc.triggerIdx = -1;
    free(oldOps);
    node->nodeType = uniRunnerDagNodeTypeIpc;
    node->nodeData.ipc = ipc;
  }

  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
  runnerState->numPendingNodes = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    node->pendingParents = node->numParents;
    if (node->numParents == 0) {
      if (node->nodeType == uniRunnerDagNodeTypeRed) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, node);
      } else if (node->nodeType == uniRunnerDagNodeTypeIpc) {
        flagcxIntruQueueEnqueue(&runnerState->ipcReadyQueue, node);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }

  runnerState->ipcReadySlots = static_cast<size_t>(readySlot);
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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingRS called, recvcount=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t baseRankChunkCount = count;

  // Nodes per slice chain:
  // (P2P + Reduce * numRedSlices) * (nranks - 1)
  int nodesPerStep = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(1, numRedSlices, 1, &nodesPerStep));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(nodesPerStep, nranks - 1, 0,
                                           &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 2, nranks);

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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(op, i == nranks - 2);

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

  if (nranks < 1 || rank < 0 || rank >= nranks || root < 0 ||
      root >= nranks) {
    return flagcxInvalidArgument;
  }
  int algoRank = positiveModulo(static_cast<int64_t>(rank) - root,
                                nranks); // Rotate ranks so root is 0
  if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
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
  int redNodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      recvNodesPerSlice, numRedSlices, 0, &redNodesPerSlice));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      1, redNodesPerSlice,
      static_cast<size_t>(sendNodesPerSlice) + recvNodesPerSlice,
      &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d (algoRank %d) initUniRunnerStateTreeReduce called, count=%lu, "
        "numSlices=%d, numRedSlices=%d, recvSteps %d, sendSteps %d",
        comm->rank, algoRank, count, numSlices, numRedSlices, recvNodesPerSlice,
        sendNodesPerSlice);

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops,
          runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps));

      // Recv Node
      int peer = positiveModulo(
          static_cast<int64_t>(rank) + (int64_t{1} << i), nranks);
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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(
                op, algoRank == 0 && i == nTotalSteps - 1);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops,
          runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps));

      int peer = positiveModulo(
          static_cast<int64_t>(rank) -
              (int64_t{1} << __builtin_ctz(algoRank)),
          nranks);

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
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
  runnerState->numPendingNodes = 0;
}

static flagcxResult_t
prepareUniRunnerDagExecutionPlan(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL || runnerState->numDagNodes < 0) {
    return flagcxInvalidArgument;
  }
  return compileUniRunnerDagExecutionPlan(
      runnerState->dagNodes, static_cast<size_t>(runnerState->numDagNodes),
      &runnerState->dagPlan);
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
    nodeDesc.commDomain = node->commDomain;
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
        size_t opBytes = 0;
        FLAGCXCHECK(checkedUniRunnerTypeBytes(op->count, 1, op->datatype,
                                              &opBytes));
        FLAGCXCHECK(
            captureBufferRef(op->addr, bindings, opBytes, &opDesc.buffer));
        nodeDesc.p2pOps.push_back(opDesc);
      }
    } else if (node->nodeType == uniRunnerDagNodeTypeRed) {
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(node->nodeData.red.count, 1,
                                            node->nodeData.red.datatype,
                                            &opBytes));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input1, bindings,
                                   opBytes, &nodeDesc.red.input1));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input2, bindings,
                                   opBytes, &nodeDesc.red.input2));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.output, bindings,
                                   opBytes, &nodeDesc.red.output));
      nodeDesc.red.count = node->nodeData.red.count;
      nodeDesc.red.datatype = node->nodeData.red.datatype;
      nodeDesc.red.redOp = node->nodeData.red.redOp;
    } else if (node->nodeType == uniRunnerDagNodeTypeCpy) {
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(node->nodeData.cpy.count, 1,
                                            node->nodeData.cpy.datatype,
                                            &opBytes));
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.src, bindings,
                                   opBytes, &nodeDesc.cpy.src));
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.dst, bindings,
                                   opBytes, &nodeDesc.cpy.dst));
      nodeDesc.cpy.count = node->nodeData.cpy.count;
      nodeDesc.cpy.datatype = node->nodeData.cpy.datatype;
    } else if (node->nodeType == uniRunnerDagNodeTypeIpc) {
      nodeDesc.ipc.srcOffsetBytes = node->nodeData.ipc.srcOffsetBytes;
      nodeDesc.ipc.dstOffsetBytes = node->nodeData.ipc.dstOffsetBytes;
      nodeDesc.ipc.bytes = node->nodeData.ipc.bytes;
      nodeDesc.ipc.srcBufferType = node->nodeData.ipc.srcBufferType;
      nodeDesc.ipc.peerLocalRank = node->nodeData.ipc.peerLocalRank;
      nodeDesc.ipc.readySlot = node->nodeData.ipc.readySlot;
    } else {
      return flagcxNotSupported;
    }

    captured->nodes.push_back(nodeDesc);
  }

  FLAGCXCHECK(validateUniRunnerIpcDagTemplateBindings(
      *captured, bindings.inputBytes, bindings.outputBytes,
      bindings.localRanks));

  *dagTemplate = captured;
  return flagcxSuccess;
}

static flagcxResult_t
materializeUniRunnerDagTemplate(flagcxUniRunnerState *runnerState,
                                const uniRunnerDagTemplate &dagTemplate,
                                const uniRunnerDagRuntimeBindings &bindings) {
  // A rejected cache entry must not leave a ready-slot cardinality from a
  // previous invocation visible to the collective IPC preparation agreement.
  runnerState->ipcReadySlots = 0;
  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (dagTemplate.nodes.size() > maxInt) {
    return flagcxInvalidArgument;
  }
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    if (node.parents.size() > maxInt || node.children.size() > maxInt ||
        node.p2pOps.size() > maxInt) {
      return flagcxInvalidArgument;
    }
  }
  FLAGCXCHECK(validateUniRunnerIpcDagTemplateBindings(
      dagTemplate, bindings.inputBytes, bindings.outputBytes,
      bindings.localRanks));

  resetDagSchedulerRuntimeState(runnerState);
  runnerState->numDagNodes = static_cast<int>(dagTemplate.nodes.size());
  if (runnerState->numDagNodes == 0) {
    runnerState->dagNodes = NULL;
    return flagcxSuccess;
  }

  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int i = 0; i < runnerState->numDagNodes; i++) {
    const uniRunnerDagNodeDesc &src = dagTemplate.nodes[i];
    uniRunnerDagNode *dst = &runnerState->dagNodes[i];
    dst->nodeType = src.nodeType;
    dst->commDomain = src.commDomain;
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
        FLAGCXCHECK(
            flagcxCalloc(&dst->nodeData.p2p.ops, dst->nodeData.p2p.numOps));
      }
      for (int opIdx = 0; opIdx < dst->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerDagP2pOpDesc &srcOp = src.p2pOps[opIdx];
        uniRunnerP2pOpData *dstOp = &dst->nodeData.p2p.ops[opIdx];
        dstOp->count = srcOp.count;
        dstOp->peerRank = srcOp.peerRank;
        dstOp->datatype = srcOp.datatype;
        dstOp->type = srcOp.type;
        size_t opBytes = 0;
        FLAGCXCHECK(checkedUniRunnerTypeBytes(srcOp.count, 1, srcOp.datatype,
                                              &opBytes));
        FLAGCXCHECK(
            resolveBufferRef(srcOp.buffer, bindings, opBytes, &dstOp->addr));
      }
    } else if (dst->nodeType == uniRunnerDagNodeTypeRed) {
      dst->nodeData.red.triggerIdx = -1;
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(src.red.count, 1,
                                            src.red.datatype, &opBytes));
      FLAGCXCHECK(resolveBufferRef(src.red.input1, bindings, opBytes,
                                   &dst->nodeData.red.input1));
      FLAGCXCHECK(resolveBufferRef(src.red.input2, bindings, opBytes,
                                   &dst->nodeData.red.input2));
      FLAGCXCHECK(resolveBufferRef(src.red.output, bindings, opBytes,
                                   &dst->nodeData.red.output));
      dst->nodeData.red.count = src.red.count;
      // The reduce kernel uses nthreads as its per-trigger loop stride, so it
      // must match the block size used to launch the persistent RED kernel.
      // Loaded DAG JSON stores only operation DAG parameters; nthreads comes
      // from the runtime launch setting.
      dst->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
      dst->nodeData.red.datatype = src.red.datatype;
      dst->nodeData.red.redOp = src.red.redOp;
    } else if (dst->nodeType == uniRunnerDagNodeTypeCpy) {
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(src.cpy.count, 1,
                                            src.cpy.datatype, &opBytes));
      FLAGCXCHECK(resolveBufferRef(src.cpy.src, bindings, opBytes,
                                   &dst->nodeData.cpy.src));
      FLAGCXCHECK(resolveBufferRef(src.cpy.dst, bindings, opBytes,
                                   &dst->nodeData.cpy.dst));
      dst->nodeData.cpy.count = src.cpy.count;
      dst->nodeData.cpy.datatype = src.cpy.datatype;
    } else if (dst->nodeType == uniRunnerDagNodeTypeIpc) {
      dst->nodeData.ipc.srcOffsetBytes = src.ipc.srcOffsetBytes;
      dst->nodeData.ipc.dstOffsetBytes = src.ipc.dstOffsetBytes;
      dst->nodeData.ipc.bytes = src.ipc.bytes;
      dst->nodeData.ipc.srcBufferType = src.ipc.srcBufferType;
      dst->nodeData.ipc.peerLocalRank = src.ipc.peerLocalRank;
      dst->nodeData.ipc.readySlot = src.ipc.readySlot;
      dst->nodeData.ipc.parentFlagsOffset = 0;
      dst->nodeData.ipc.triggerIdx = -1;
    } else {
      return flagcxNotSupported;
    }

    if (dst->numParents == 0) {
      if (dst->nodeType == uniRunnerDagNodeTypeRed) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, dst);
      } else if (dst->nodeType == uniRunnerDagNodeTypeIpc) {
        flagcxIntruQueueEnqueue(&runnerState->ipcReadyQueue, dst);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, dst);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }

  return flagcxSuccess;
}

static flagcxResult_t bindUniRunnerCompiledDagPlan(
    const uniRunnerCompiledDagTemplate &compiled,
    uniRunnerDagExecutionPlan *plan) {
  if (plan == NULL || compiled.numNodes != compiled.dagTemplate.nodes.size() ||
      compiled.topoOrder.size() != compiled.numNodes ||
      compiled.numHostNodes + compiled.numRedNodes + compiled.numIpcNodes !=
          compiled.numNodes) {
    return flagcxInternalError;
  }
  destroyUniRunnerDagExecutionPlan(plan);
  plan->topoOrder = compiled.topoOrder.empty()
                        ? NULL
                        : compiled.topoOrder.data();
  plan->numNodes = compiled.numNodes;
  plan->numHostNodes = compiled.numHostNodes;
  plan->numRedNodes = compiled.numRedNodes;
  plan->numIpcNodes = compiled.numIpcNodes;
  plan->maxParallelRedNodes = compiled.maxParallelRedNodes;
  plan->maxRedNodeCount = compiled.maxRedNodeCount;
  plan->maxIpcNodeBytes = compiled.maxIpcNodeBytes;
  plan->ownsTopoOrder = false;
  plan->staticValidated = true;
  return flagcxSuccess;
}

static flagcxResult_t tryLoadCachedUniRunnerDag(
    flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings, bool *cacheHit) {
  *cacheHit = false;
  std::shared_ptr<const uniRunnerCompiledDagTemplate> compiled;
  size_t hashValue = getUniRunnerDagPatternHash(key);
  if (!findUniRunnerCompiledDagTemplate(hashValue, key, &compiled)) {
    flagcxResult_t loadRes = loadUniRunnerDagFromCacheDir(hashValue, key.rank);
    if (!findUniRunnerCompiledDagTemplate(hashValue, key, &compiled)) {
      if (loadRes != flagcxSuccess) {
        return loadRes;
      }
      TRACE(FLAGCX_UNIRUNNER,
            "uniRunner DAG cache miss, algo_hash=%llu commOp=%s "
            "pattern_hash=%lu",
            static_cast<unsigned long long>(key.algoHash),
            uniRunnerCommOpToString(key.commOp), hashValue);
      return flagcxSuccess;
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "uniRunner DAG cache hit, algo_hash=%llu commOp=%s pattern_hash=%lu",
        static_cast<unsigned long long>(key.algoHash),
        uniRunnerCommOpToString(key.commOp), hashValue);
  FLAGCXCHECK(
      materializeUniRunnerDagTemplate(runnerState, compiled->dagTemplate,
                                      bindings));
  FLAGCXCHECK(bindUniRunnerCompiledDagPlan(*compiled, &runnerState->dagPlan));
  // IPC ready slots are one-to-one with cached IPC nodes. Use the compiled
  // count instead of scanning the materialized DAG again.
  runnerState->ipcReadySlots = compiled->numIpcNodes;
  *cacheHit = true;
  return flagcxSuccess;
}

static flagcxResult_t
cacheBuiltUniRunnerDag(const flagcxUniRunnerState *runnerState,
                       const uniRunnerDagCacheKey &key,
                       const uniRunnerDagRuntimeBindings &bindings) {
  size_t hashValue = getUniRunnerDagPatternHash(key);
  if (!runnerState->dagPlan.staticValidated) {
    return flagcxInternalError;
  }

  try {
    std::shared_ptr<uniRunnerDagTemplate> dagTemplate;
    flagcxResult_t captureRes = captureUniRunnerDagTemplateFromState(
        runnerState, key, bindings, &dagTemplate);
    if (captureRes != flagcxSuccess) {
      TRACE(FLAGCX_UNIRUNNER,
            "uniRunner DAG capture skipped for hash=%lu, result=%d",
            hashValue, captureRes);
      return flagcxSuccess;
    }

    std::shared_ptr<uniRunnerCompiledDagTemplate> compiled =
        std::make_shared<uniRunnerCompiledDagTemplate>();
    compiled->dagTemplate = std::move(*dagTemplate);
    // The builder path has already completed validateDagNodes() and the
    // runtime plan compiler, so its cached copy is a validated static plan.
    compiled->numNodes = runnerState->dagPlan.numNodes;
    compiled->numHostNodes = runnerState->dagPlan.numHostNodes;
    compiled->numRedNodes = runnerState->dagPlan.numRedNodes;
    compiled->numIpcNodes = runnerState->dagPlan.numIpcNodes;
    compiled->maxParallelRedNodes =
        runnerState->dagPlan.maxParallelRedNodes;
    compiled->maxRedNodeCount = runnerState->dagPlan.maxRedNodeCount;
    compiled->maxIpcNodeBytes = runnerState->dagPlan.maxIpcNodeBytes;
    if (runnerState->dagPlan.numNodes != 0) {
      if (runnerState->dagPlan.topoOrder == NULL) {
        return flagcxInternalError;
      }
      compiled->topoOrder.assign(
          runnerState->dagPlan.topoOrder,
          runnerState->dagPlan.topoOrder + runnerState->dagPlan.numNodes);
    }
    flagcxResult_t cacheRes = cacheUniRunnerCompiledDagTemplate(compiled);
    if (cacheRes != flagcxSuccess) {
      TRACE(FLAGCX_UNIRUNNER,
            "uniRunner DAG cache persist skipped for hash=%lu, result=%d",
            hashValue, cacheRes);
    }
  } catch (...) {
    return flagcxSuccess;
  }
  return flagcxSuccess;
}

static flagcxResult_t applyUniRunnerDagLaunchShape(
    flagcxUniRunnerState *runnerState) {
  size_t nthreads = 0;
  size_t redBlocks = 0;
  size_t ipcBlocks = 0;
  FLAGCXCHECK(resolveUniRunnerDagLaunchShape(
      &runnerState->dagPlan,
      static_cast<size_t>(runnerState->uniRunnerIpcChunkSize),
      static_cast<size_t>(runnerState->uniRunnerMaxThreads),
      static_cast<size_t>(runnerState->uniRunnerMaxBlocks), &nthreads,
      &redBlocks, &ipcBlocks));
  runnerState->uniRunnerNThreads = nthreads;
  runnerState->uniRunnerNRedBlocks = redBlocks;
  runnerState->uniRunnerNIpcBlocks = ipcBlocks;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    if (runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeRed) {
      runnerState->dagNodes[nodeIdx].nodeData.red.nthreads = nthreads;
    }
  }
  return flagcxSuccess;
}

// Clean up DAG nodes
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_UNIRUNNER, "cleanupDagScheduler called");
  if (!runnerState) {
    return flagcxSuccess;
  }
  destroyUniRunnerDagExecutionPlan(&runnerState->dagPlan);
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
    FLAGCXCHECK(applyUniRunnerDagLaunchShape(runnerState));
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
  FLAGCXCHECK(prepareUniRunnerDagExecutionPlan(runnerState));
  runnerState->dagPlan.staticValidated = true;
  FLAGCXCHECK(cacheBuiltUniRunnerDag(runnerState, key, bindings));
  FLAGCXCHECK(applyUniRunnerDagLaunchShape(runnerState));
  return flagcxSuccess;
}

flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, 2);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoLocRed, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateLocRed(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateGroupedAR(
    flagcxUniRunnerState *state, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm, int groupSize) {
  if (comm == NULL || comm->nranks <= 0 || groupSize <= 0 ||
      groupSize > comm->nranks || comm->nranks % groupSize != 0)
    return flagcxInvalidArgument;
  const int64_t nLocalSlices = flagcxParamUniRunnerLocalNSlices();
  if (nLocalSlices <= 0 || nLocalSlices > std::numeric_limits<int>::max()) {
    WARN("GroupedAR requires 0 < LOCAL_NSLICES <= INT_MAX");
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
  setUniRunnerAvgDivisor(state, op, comm->nranks);
  FLAGCXCHECK(setEffectiveUniRunnerRedSlices(state, count, groupSize));
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 1, datatype,
                                        &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;
  const size_t chunkCount = count / groupSize + (count % groupSize != 0);
  const size_t sliceCount = chunkCount / state->uniRunnerNSlices +
                           (chunkCount % state->uniRunnerNSlices != 0);
  const int nGroups = comm->nranks / groupSize;
  const size_t localScratchCount = groupSize > 1 ? sliceCount : 0;
  const size_t netScratchCount = nGroups > 1
      ? sliceCount / nGroups + (sliceCount % nGroups != 0) : 0;
  if (localScratchCount > std::numeric_limits<size_t>::max() - netScratchCount)
    return flagcxInvalidArgument;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      localScratchCount + netScratchCount, 1, datatype, &bindings.scratchBytes));
  if (count != 0 && nGroups > 1)
    FLAGCXCHECK(deviceAdaptor->streamCreate(&state->netCommStream));
  if (bindings.scratchBytes > state->groupedArScratchCapacity) {
    void *scratch = NULL;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(&scratch, bindings.scratchBytes,
                                            flagcxMemDevice, NULL));
    if (state->groupedArScratch != NULL) {
      const flagcxResult_t res = deviceAdaptor->deviceFree(
          state->groupedArScratch, flagcxMemDevice, NULL);
      if (res != flagcxSuccess) {
        (void)deviceAdaptor->deviceFree(scratch, flagcxMemDevice, NULL);
        return res;
      }
    }
    state->groupedArScratch = scratch;
    state->groupedArScratchCapacity = bindings.scratchBytes;
  }
  bindings.scratchBase = state->groupedArScratch;
  uniRunnerDagAlgorithmConfig config{};
  config.groupSize = groupSize;
  config.pipelineVersion = 2;
  config.numSlices = state->uniRunnerNSlices;
  config.numLocalSlices = static_cast<uint64_t>(nLocalSlices);
  config.numRedSlices = state->uniRunnerNRedSlices;
  config.bufferMode = sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                                          : uniRunnerDagBufferModeOutOfPlace;
  const uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoGroupedAR, config, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
  return initUniRunnerStateCached(state, key, bindings, [&]() {
    return buildUniRunnerStateGroupedAR(state, sendbuff, recvbuff, count,
                                        datatype, op, comm, groupSize,
                                        static_cast<int>(nLocalSlices));
  });
}

flagcxResult_t initUniRunnerStateGroupedAG(flagcxUniRunnerState *runnerState,
                                           const void *sendbuff, void *recvbuff,
                                           size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxComm_t comm, int groupSize) {
  runnerState->avgDivisor = 1;
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      count, static_cast<size_t>(comm->nranks), datatype,
      &bindings.outputBytes));

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.groupSize = static_cast<uint64_t>(groupSize);
  algorithmConfig.bufferMode =
      bufferMatchesElementOffset(sendbuff, recvbuff, count, comm->rank,
                                 typeSize)
          ? uniRunnerDagBufferModeInPlace
          : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key =
      makeUniRunnerDagCacheKey(uniRunnerDagAlgoGroupedAG, algorithmConfig,
                               flagcxCommOpAllGather, count, datatype,
                               flagcxRedNoOp, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff, count,
                                        datatype, comm, groupSize);
  });
}

flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  runnerState->avgDivisor = 1;
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAG, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, flagcxRedNoOp, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAG(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAR, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
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
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoSlicedAR, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateIpcAR(flagcxUniRunnerState *runnerState,
                                       const void *sendbuff, void *recvbuff,
                                       size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  flagcxResult_t result = flagcxSuccess;
  if (runnerState->uniRunnerIpcChunkSize < 16 ||
      runnerState->uniRunnerIpcChunkSize % 16 != 0) {
    WARN("UniRunner IPCCHUNKSIZE must be a positive multiple of 16 bytes");
    result = flagcxInvalidArgument;
  }
  if (result == flagcxSuccess) {
    result =
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);
  }
  size_t bytes = 0;
  if (result == flagcxSuccess) {
    result = checkedUniRunnerTypeBytes(count, 1, datatype, &bytes);
  }

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = bytes;
  bindings.outputBytes = bytes;
  bindings.localRanks = comm->localRanks;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  if (result == flagcxSuccess) {
    result = getUniRunnerIpcTopologyHash(
        comm->localRanks, comm->nranks, comm->localRankToRank,
        &algorithmConfig.topologyHash);
  }

  if (result == flagcxSuccess) {
    const uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
        uniRunnerDagAlgoIpcAR, algorithmConfig, flagcxCommOpAllReduce, count,
        datatype, op, -1, comm);
    try {
      result = initUniRunnerStateCached(runnerState, key, bindings, [&]() {
        return buildUniRunnerStateIpcAR(runnerState, sendbuff, recvbuff, count,
                                        datatype, op, comm);
      });
    } catch (...) {
      // IPC preparation is collective below: convert allocation/container
      // exceptions into an agreed result instead of letting one rank escape.
      result = flagcxSystemError;
    }
  }
  // The data-view agreement below also makes DAG-build status, data bytes and
  // IPC-node cardinality collective before any rank enters DevMem creation.
  // In particular, no local cache/build error returns before this agreement.
  FLAGCXCHECK(ensureUniRunnerIpcDataViews(
      runnerState, comm, sendbuff, recvbuff, bytes, result));
  if (runnerState->ipcReadySlots == 0)
    return flagcxSuccess;
  // Static IPC prepends abort and rank-done control slots to the logical ready
  // array. Allocate that collective storage only after static occupancy and
  // schedule validation have succeeded.
  return flagcxSuccess;
}

flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      count, static_cast<size_t>(comm->nranks), datatype,
      &bindings.inputBytes));
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.outputBytes));
  bindings.scratchBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      bufferMatchesElementOffset(recvbuff, sendbuff, count, comm->rank,
                                 typeSize)
          ? uniRunnerDagBufferModeInPlace
          : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingRS, algorithmConfig, flagcxCommOpReduceScatter,
      count, datatype, op, -1, comm);
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
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 2, datatype, &bindings.scratchBytes));

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoTreeRed, algorithmConfig, flagcxCommOpReduce, count,
      datatype, op, root, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                      scratchbuff, count, datatype, op, root,
                                      comm);
  });
}

static flagcxResult_t launchHostDagNode(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm,
                                        uniRunnerDagNode *current) {
  if (runnerState == NULL || comm == NULL || current == NULL ||
      current->nodeIdx < 0 ||
      current->nodeIdx >= runnerState->numDagNodes) {
    return flagcxInvalidArgument;
  }
  void *flag = getDagNodeFlag(runnerState, current->nodeIdx);
  flagcxStream_t currentStream =
      getDagNodeExecutionStream(runnerState, current);

  if (current->nodeType == uniRunnerDagNodeTypeP2p) {
    // Mark the node as submitted before wiring its completion dependency.
    TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWrite flag %d: PEND",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(currentStream, flag,
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
      if (runnerState->dagCompletionEvents != NULL &&
          runnerState->dagCompletionEvents[parentIdx] != NULL) {
        FLAGCXCHECK(deviceAdaptor->streamWaitEvent(
            currentStream, runnerState->dagCompletionEvents[parentIdx]));
        continue;
      }
      void *parentFlag = getDagNodeFlag(runnerState, parentIdx);
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          currentStream, parentFlag, flagcxStreamFlagDone, 0));
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
                  currentStream,
                  flagcxCollProxyChannelForEdge(comm->rank, op->peerRank, 0));
            } else if (op->type == flagcxDevicePrimRecv) {
              opResult = flagcxHeteroRecvOnChannel(
                  op->addr, op->count, op->datatype, op->peerRank, comm,
                  currentStream,
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
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(currentStream, flag,
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
    size_t copyBytes = 0;
    FLAGCXCHECK(checkedUniRunnerTypeBytes(
        current->nodeData.cpy.count, 1, current->nodeData.cpy.datatype,
        &copyBytes));
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        current->nodeData.cpy.dst, current->nodeData.cpy.src,
        copyBytes,
        flagcxMemcpyDeviceToDevice, runnerState->cpyStream, NULL));

    // Write flag to stream
    TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWrite flag %d: DONE",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->cpyStream, flag,
                                                  flagcxStreamFlagDone, 0));
  } else {
    return flagcxSystemError;
  }

  if (runnerState->dagCompletionEvents != NULL &&
      runnerState->dagCompletionEvents[current->nodeIdx] != NULL) {
    FLAGCXCHECK(deviceAdaptor->eventRecord(
        runnerState->dagCompletionEvents[current->nodeIdx], currentStream));
  }
  return flagcxSuccess;
}

static flagcxResult_t launchP2pOps(flagcxUniRunnerState *runnerState,
                                   flagcxHeteroComm_t comm) {
  uniRunnerDagNode *current =
      flagcxIntruQueueDequeue(&runnerState->p2pReadyQueue);
  if (current == NULL) {
    return flagcxInternalError;
  }
  return launchHostDagNode(runnerState, comm, current);
}

static flagcxResult_t submitStaticHostDag(flagcxUniRunnerState *runnerState,
                                          flagcxHeteroComm_t comm) {
  for (size_t topoSlot = 0; topoSlot < runnerState->dagPlan.numNodes;
       ++topoSlot) {
    const int nodeIdx = runnerState->dagPlan.topoOrder[topoSlot];
    if (nodeIdx < 0 || nodeIdx >= runnerState->numDagNodes) {
      return flagcxInternalError;
    }
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType == uniRunnerDagNodeTypeP2p ||
        node->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(launchHostDagNode(runnerState, comm, node));
    } else if (node->nodeType != uniRunnerDagNodeTypeRed &&
               node->nodeType != uniRunnerDagNodeTypeIpc) {
      return flagcxInternalError;
    }
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
  } else if (runnerState->dagNodes[nodeIdx].nodeType ==
             uniRunnerDagNodeTypeIpc) {
    flagcxIntruQueueEnqueue(&runnerState->ipcReadyQueue,
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

static flagcxResult_t allocUniRunnerIpcReadyBuffer(void **ptr, size_t bytes) {
  return deviceAdaptor->deviceMalloc(ptr, bytes, flagcxMemDevice, NULL);
}

static flagcxResult_t freeUniRunnerIpcReadyBuffer(void *ptr) {
  if (ptr == NULL)
    return flagcxSuccess;
  return deviceAdaptor->deviceFree(ptr, flagcxMemDevice, NULL);
}

struct uniRunnerResourceAgreement {
  uint64_t canReuse;
  int result;
};

struct uniRunnerInvocationAgreement {
  uint64_t signature;
  int result;
};

struct uniRunnerDataViewAgreement {
  uint64_t readySlots;
  uint64_t dataBytes;
  uint64_t canReuse;
  int result;
};

static flagcxResult_t agreeUniRunnerResourceState(
    flagcxComm_t comm, bool localCanReuse, flagcxResult_t localResult,
    bool *allCanReuse, flagcxResult_t *collectiveResult) {
  if (comm == NULL || comm->bootstrap == NULL || comm->nranks <= 0 ||
      comm->nranks > FLAGCX_MAX_LOCAL_RANKS || comm->rank < 0 ||
      comm->rank >= comm->nranks || allCanReuse == NULL ||
      collectiveResult == NULL) {
    return flagcxInvalidArgument;
  }
  *allCanReuse = false;
  *collectiveResult = flagcxSuccess;
  uniRunnerResourceAgreement exchanges[FLAGCX_MAX_LOCAL_RANKS] = {};
  exchanges[comm->rank].canReuse = localCanReuse ? uint64_t{1} : uint64_t{0};
  exchanges[comm->rank].result = static_cast<int>(localResult);
  FLAGCXCHECK(bootstrapCollAllGather(
      comm->bootstrap, exchanges, sizeof(uniRunnerResourceAgreement)));

  *allCanReuse = true;
  for (int rank = 0; rank < comm->nranks; ++rank) {
    if (exchanges[rank].result != static_cast<int>(flagcxSuccess) &&
        *collectiveResult == flagcxSuccess) {
      *collectiveResult =
          static_cast<flagcxResult_t>(exchanges[rank].result);
    }
    if (exchanges[rank].canReuse == 0) {
      *allCanReuse = false;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t agreeUniRunnerInvocationState(
    flagcxComm_t comm, uint64_t localSignature, flagcxResult_t localResult,
    flagcxResult_t *collectiveResult) {
  if (comm == NULL || comm->bootstrap == NULL || comm->nranks <= 0 ||
      comm->nranks > FLAGCX_MAX_LOCAL_RANKS || comm->rank < 0 ||
      comm->rank >= comm->nranks ||
      collectiveResult == NULL) {
    return flagcxInvalidArgument;
  }
  *collectiveResult = flagcxSuccess;
  uniRunnerInvocationAgreement exchanges[FLAGCX_MAX_LOCAL_RANKS] = {};
  exchanges[comm->rank].signature = localSignature;
  exchanges[comm->rank].result = static_cast<int>(localResult);
  FLAGCXCHECK(bootstrapCollAllGather(
      comm->bootstrap, exchanges, sizeof(uniRunnerInvocationAgreement)));

  const uint64_t expectedSignature = exchanges[0].signature;
  for (int rank = 0; rank < comm->nranks; ++rank) {
    if (exchanges[rank].result != static_cast<int>(flagcxSuccess) &&
        *collectiveResult == flagcxSuccess) {
      *collectiveResult =
          static_cast<flagcxResult_t>(exchanges[rank].result);
    }
  }
  if (*collectiveResult == flagcxSuccess) {
    for (int rank = 1; rank < comm->nranks; ++rank) {
      if (exchanges[rank].signature != expectedSignature) {
        WARN("UniRunner invocation signature mismatch: rank 0=0x%llx "
             "rank %d=0x%llx",
             static_cast<unsigned long long>(expectedSignature), rank,
             static_cast<unsigned long long>(exchanges[rank].signature));
        *collectiveResult = flagcxInvalidUsage;
        break;
      }
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t agreeUniRunnerDataViewState(
    flagcxComm_t comm, size_t localReadySlots, size_t localDataBytes,
    bool localCanReuse, flagcxResult_t localResult, bool *allCanReuse,
    flagcxResult_t *collectiveResult) {
  if (comm == NULL || comm->bootstrap == NULL || comm->nranks <= 0 ||
      comm->nranks > FLAGCX_MAX_LOCAL_RANKS || comm->rank < 0 ||
      comm->rank >= comm->nranks || allCanReuse == NULL ||
      collectiveResult == NULL) {
    return flagcxInvalidArgument;
  }
  *allCanReuse = false;
  *collectiveResult = flagcxSuccess;
  uniRunnerDataViewAgreement exchanges[FLAGCX_MAX_LOCAL_RANKS] = {};
  exchanges[comm->rank].readySlots = static_cast<uint64_t>(localReadySlots);
  exchanges[comm->rank].dataBytes = static_cast<uint64_t>(localDataBytes);
  exchanges[comm->rank].canReuse = localCanReuse ? uint64_t{1} : uint64_t{0};
  exchanges[comm->rank].result = static_cast<int>(localResult);
  FLAGCXCHECK(bootstrapCollAllGather(
      comm->bootstrap, exchanges, sizeof(uniRunnerDataViewAgreement)));

  *allCanReuse = true;
  for (int rank = 0; rank < comm->nranks; ++rank) {
    if (exchanges[rank].result != static_cast<int>(flagcxSuccess) &&
        *collectiveResult == flagcxSuccess) {
      *collectiveResult =
          static_cast<flagcxResult_t>(exchanges[rank].result);
    }
    if (exchanges[rank].canReuse == 0) {
      *allCanReuse = false;
    }
  }
  if (*collectiveResult == flagcxSuccess) {
    for (int rank = 1; rank < comm->nranks; ++rank) {
      if (exchanges[rank].readySlots != exchanges[0].readySlots ||
          exchanges[rank].dataBytes != exchanges[0].dataBytes) {
        WARN("UniRunner IPC DAG/data cardinality differs across ranks");
        *collectiveResult = flagcxInvalidUsage;
        break;
      }
    }
  }
  return flagcxSuccess;
}

static uint64_t mixUniRunnerInvocationSignature(uint64_t signature,
                                                 uint64_t value) {
  // Boost-style 64-bit hash combine. This is a consistency guard rather than a
  // cache identity; all inputs below are already range-validated integers.
  return signature ^
         (value + uint64_t{0x9e3779b97f4a7c15} + (signature << 6) +
          (signature >> 2));
}

static flagcxResult_t destroyUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState) {
  if (runnerState->ipcOwner == NULL)
    return flagcxSuccess;
  flagcxComm_t comm = runnerState->ipcOwner;
  // Input and output are always distinct DevMem handles, including in-place
  // collectives. Keeping the create/destroy count rank-invariant prevents an
  // out-of-place rank from entering an IPC bootstrap that an in-place rank
  // skipped.
  flagcxResult_t result = flagcxSuccess;
  if (runnerState->ipcInputMem != NULL) {
    result = flagcxDevMemDestroy(comm, runnerState->ipcInputMem);
  }
  if (runnerState->ipcOutputMem != NULL) {
    flagcxResult_t outputResult =
        flagcxDevMemDestroy(comm, runnerState->ipcOutputMem);
    if (result == flagcxSuccess) {
      result = outputResult;
    }
  }
  runnerState->ipcInputMem = NULL;
  runnerState->ipcOutputMem = NULL;
  runnerState->ipcInputBase = NULL;
  runnerState->ipcOutputBase = NULL;
  runnerState->ipcDataBytes = 0;
  return result;
}

static flagcxResult_t ensureUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    const void *sendbuff, void *recvbuff, size_t bytes,
    flagcxResult_t localPreparationResult) {
  const bool localCanReuse =
      localPreparationResult == flagcxSuccess &&
      runnerState->ipcReadySlots != 0 && runnerState->ipcOwner == comm &&
      runnerState->ipcInputMem != NULL &&
      runnerState->ipcOutputMem != NULL &&
      runnerState->ipcInputBase == sendbuff &&
      runnerState->ipcOutputBase == recvbuff &&
      bytes <= runnerState->ipcDataBytes;
  bool allCanReuse = false;
  flagcxResult_t collectiveResult = flagcxSuccess;
  FLAGCXCHECK(agreeUniRunnerDataViewState(
      comm, runnerState->ipcReadySlots, bytes, localCanReuse,
      localPreparationResult, &allCanReuse, &collectiveResult));
  if (collectiveResult != flagcxSuccess) {
    return collectiveResult;
  }
  if (runnerState->ipcReadySlots == 0) {
    return flagcxSuccess;
  }
  if (allCanReuse) {
    return flagcxSuccess;
  }

  flagcxResult_t res = flagcxSuccess;
  flagcxDevMem_t newInputMem = NULL;
  flagcxDevMem_t newOutputMem = NULL;
  flagcxResult_t oldDestroyResult = flagcxSuccess;

  // Every rank performs exactly two creates in the same output-then-input
  // order. Even when sendbuff == recvbuff the second handle participates in
  // the all-rank reuse consensus and retains its own IPC-table reference.
  res = flagcxDevMemCreate(comm, recvbuff, bytes, NULL, &newOutputMem);
  if (res == flagcxSuccess && newOutputMem == NULL) {
    res = flagcxSystemError;
  }
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }
  res = flagcxDevMemCreate(comm, const_cast<void *>(sendbuff), bytes, NULL,
                           &newInputMem);
  if (res == flagcxSuccess && newInputMem == NULL) {
    res = flagcxSystemError;
  }
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }
  res = newInputMem->hasWindow && newOutputMem->hasWindow
            ? flagcxSuccess
            : flagcxNotSupported;
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }

  {
    flagcxDevMem_t oldInputMem = runnerState->ipcInputMem;
    flagcxDevMem_t oldOutputMem = runnerState->ipcOutputMem;
    flagcxComm_t oldOwner = runnerState->ipcOwner;

    // Retain the prior peer bindings until both new collective creates have
    // completed. At that point every rank has left any kernel which could still
    // dereference the old views. The backing allocations remain user-owned;
    // their lifetime continues to follow the existing registered-buffer API
    // contract after these handles release their table references.
    runnerState->ipcOwner = comm;
    runnerState->ipcInputMem = newInputMem;
    runnerState->ipcOutputMem = newOutputMem;
    runnerState->ipcInputBase = sendbuff;
    runnerState->ipcOutputBase = recvbuff;
    runnerState->ipcDataBytes = bytes;
    newInputMem = NULL;
    newOutputMem = NULL;

    if (oldInputMem != NULL) {
      oldDestroyResult = flagcxDevMemDestroy(oldOwner, oldInputMem);
    }
    if (oldOutputMem != NULL) {
      flagcxResult_t outputResult =
          flagcxDevMemDestroy(oldOwner, oldOutputMem);
      if (oldDestroyResult == flagcxSuccess) {
        oldDestroyResult = outputResult;
      }
    }
  }
  FLAGCXCHECK(agreeUniRunnerResourceState(
      comm, false, oldDestroyResult, &allCanReuse, &collectiveResult));
  if (collectiveResult != flagcxSuccess) {
    return collectiveResult;
  }
  return flagcxSuccess;

fail:
  if (newInputMem != NULL) {
    (void)flagcxDevMemDestroy(comm, newInputMem);
  }
  if (newOutputMem != NULL) {
    (void)flagcxDevMemDestroy(comm, newOutputMem);
  }
  return res;
}

static flagcxResult_t ensureUniRunnerIpcReadyStorage(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    size_t requiredSlots) {
  if (requiredSlots == 0)
    return flagcxSuccess;
  if (requiredSlots >
      std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return flagcxInvalidArgument;
  }
  // requiredSlots is derived solely from the collective DAG and local-rank
  // topology and is stable for the cached IPCAR configuration. Keep this
  // runner-owned export for the communicator lifetime: flagcxDevMemDestroy
  // releases a table reference but imported CUDA IPC mappings are closed only
  // when the table is rebound or the communicator is destroyed, so an online
  // resize must not free the old exporter allocation.
  if (runnerState->ipcReadyMem != NULL &&
      runnerState->ipcReadyBuffer != NULL &&
      runnerState->ipcReadyCapacity >= requiredSlots) {
    return flagcxSuccess;
  }
  if (runnerState->ipcReadyMem != NULL ||
      runnerState->ipcReadyBuffer != NULL ||
      runnerState->ipcReadyCapacity != 0) {
    WARN("UniRunner static IPC ready storage cannot grow safely during the "
         "communicator lifetime (required=%zu capacity=%zu)",
         requiredSlots, runnerState->ipcReadyCapacity);
    return flagcxInvalidUsage;
  }
  size_t bytes = requiredSlots * sizeof(uint64_t);
  flagcxResult_t res = flagcxSuccess;
  bool allCanReuse = false;
  flagcxResult_t collectiveResult = flagcxSuccess;
  void *newBuffer = NULL;
  flagcxDevMem_t newMem = NULL;
  bool devMemCreateEntered = false;
  res = allocUniRunnerIpcReadyBuffer(&newBuffer, bytes);
  if (res == flagcxSuccess) {
    res = deviceAdaptor->deviceMemset(newBuffer, 0, bytes, flagcxMemDevice,
                                      NULL);
  }
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }
  // Creating a DevMem binding exchanges the backing allocation across all
  // ranks. The allocation remains bound until communicator teardown.
  devMemCreateEntered = true;
  res = flagcxDevMemCreate(comm, newBuffer, bytes, NULL, &newMem);
  if (res == flagcxSuccess && newMem == NULL) {
    res = flagcxSystemError;
  }
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }
  res = newMem->hasWindow ? flagcxSuccess : flagcxNotSupported;
  FLAGCXCHECKGOTO(agreeUniRunnerResourceState(
                      comm, false, res, &allCanReuse, &collectiveResult),
                  res, fail);
  if (collectiveResult != flagcxSuccess) {
    res = collectiveResult;
    goto fail;
  }

  runnerState->ipcReadyMem = newMem;
  runnerState->ipcReadyBuffer = newBuffer;
  runnerState->ipcReadyCapacity = requiredSlots;
  newMem = NULL;
  newBuffer = NULL;
  return flagcxSuccess;

fail:
  if (newMem != NULL) {
    (void)flagcxDevMemDestroy(comm, newMem);
  }
  if (newBuffer != NULL) {
    if (devMemCreateEntered) {
      // Once DevMemCreate is entered, a peer may have imported this exporter
      // even when the local call eventually reports an error or never returns
      // a handle. Never infer mapping absence from newMem == NULL.
      flagcxCommDeferFree(comm, newBuffer, flagcxMemDevice);
    } else {
      (void)freeUniRunnerIpcReadyBuffer(newBuffer);
    }
  }
  return res;
}

static flagcxResult_t prepareUniRunnerIpcParentFlags(
    flagcxUniRunnerState *runnerState) {
  if (runnerState->ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(
        runnerState->ipcParentFlagsDevice, flagcxMemDevice, NULL));
    runnerState->ipcParentFlagsDevice = NULL;
    runnerState->ipcParentFlagsCount = 0;
  }

  size_t numParentFlags = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    const uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeIpc)
      continue;
    if (node->numParents < 0 ||
        static_cast<size_t>(node->numParents) >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) -
                numParentFlags) {
      return flagcxInvalidArgument;
    }
    numParentFlags += static_cast<size_t>(node->numParents);
  }
  if (numParentFlags >
      std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return flagcxInvalidArgument;
  }

  std::vector<uint64_t> parentFlags;
  try {
    parentFlags.reserve(numParentFlags);
  } catch (...) {
    return flagcxSystemError;
  }
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeIpc)
      continue;
    node->nodeData.ipc.parentFlagsOffset =
        static_cast<uint32_t>(parentFlags.size());
    for (int parentIdx = 0; parentIdx < node->numParents; ++parentIdx) {
      parentFlags.push_back(reinterpret_cast<uint64_t>(
          getDagNodeFlag(runnerState, node->parents[parentIdx])));
    }
  }
  if (parentFlags.empty())
    return flagcxSuccess;

  size_t bytes = parentFlags.size() * sizeof(uint64_t);
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      reinterpret_cast<void **>(&runnerState->ipcParentFlagsDevice), bytes,
      flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      runnerState->ipcParentFlagsDevice, parentFlags.data(), bytes,
      flagcxMemcpyHostToDevice, NULL, NULL));
  runnerState->ipcParentFlagsCount = parentFlags.size();
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

  while (!flagcxIntruQueueEmpty(&runnerState->ipcReadyQueue)) {
    uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->ipcReadyQueue);
    uint64_t flagOut =
        reinterpret_cast<uint64_t>(getDagNodeFlag(runnerState,
                                                  current->nodeIdx));
    int idx = -1;
    FLAGCXCHECK(enqueueIpc(
        runnerState->ipcFifo->buffer, current->nodeData.ipc.srcOffsetBytes,
        current->nodeData.ipc.dstOffsetBytes, current->nodeData.ipc.bytes,
        runnerState->uniRunnerIpcChunkSize,
        current->nodeData.ipc.srcBufferType,
        current->nodeData.ipc.peerLocalRank, current->nodeData.ipc.readySlot,
        runnerState->ipcEpoch, current->nodeData.ipc.parentFlagsOffset,
        static_cast<uint32_t>(current->numParents), flagOut, &idx));
    if (idx == -1) {
      sched_yield();
      break;
    }
    flagcxIntruQueueDequeue(&runnerState->ipcReadyQueue);
    current->nodeData.ipc.triggerIdx = idx;
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  return flagcxSuccess;
}

flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  if (runnerState->dagNodes != NULL || runnerState->numDagNodes != 0 ||
      runnerState->numPendingNodes != 0 ||
      runnerState->dagPlan.topoOrder != NULL ||
      runnerState->dagPlan.ownsTopoOrder ||
      runnerState->dagPlan.staticValidated ||
      runnerState->dagPlan.numNodes != 0 ||
      runnerState->dagPlan.numHostNodes != 0 ||
      runnerState->dagPlan.numRedNodes != 0 ||
      runnerState->dagPlan.numIpcNodes != 0 ||
      runnerState->streamFlagsSize != 0 || runnerState->ipcReadySlots != 0 ||
      runnerState->fifo != NULL || runnerState->ipcFifo != NULL ||
      runnerState->ipcFifoDevicePtr != NULL ||
      runnerState->ipcCommStream != NULL || runnerState->redStream != NULL ||
      runnerState->cpyStream != NULL || runnerState->netCommStream != NULL ||
      hcomm->uniRunnerFifoBuffer != NULL ||
      runnerState->ipcParentFlagsDevice != NULL ||
      runnerState->ipcParentFlagsCount != 0 ||
      runnerState->dagCompletionEvents != NULL ||
      runnerState->numDagCompletionEvents != 0) {
    WARN("UniRunner invocation-local state was not fully cleaned before init");
    return flagcxInvalidUsage;
  }
  runnerState->dagNodes = NULL;
  runnerState->numDagNodes = 0;
  runnerState->dagPlan.topoOrder = NULL;
  runnerState->dagPlan.numNodes = 0;
  runnerState->dagPlan.numHostNodes = 0;
  runnerState->dagPlan.numRedNodes = 0;
  runnerState->dagPlan.numIpcNodes = 0;
  runnerState->dagPlan.maxParallelRedNodes = 0;
  runnerState->dagPlan.maxRedNodeCount = 0;
  runnerState->dagPlan.maxIpcNodeBytes = 0;
  runnerState->dagPlan.ownsTopoOrder = false;
  runnerState->dagPlan.staticValidated = false;
  runnerState->streamFlagsSize = 0;
  runnerState->ipcReadySlots = 0;
  runnerState->fifo = NULL;
  runnerState->ipcFifo = NULL;
  runnerState->ipcFifoDevicePtr = NULL;
  runnerState->ipcCommStream = NULL;
  runnerState->netCommStream = NULL;
  runnerState->redStream = NULL;
  runnerState->cpyStream = NULL;
  hcomm->uniRunnerFifoBuffer = NULL;

  FLAGCXCHECK(loadUniRunnerDagBuilderConfig(
      &runnerState->uniRunnerNSlices, &runnerState->uniRunnerNRedSlices,
      &runnerState->uniRunnerRedSliceSize));
  FLAGCXCHECK(loadUniRunnerLaunchConfig(
      &runnerState->uniRunnerMaxThreads, &runnerState->uniRunnerMaxBlocks,
      &runnerState->uniRunnerNIpcBlocks));
  runnerState->uniRunnerNThreads = runnerState->uniRunnerMaxThreads;
  runnerState->uniRunnerNRedBlocks = runnerState->uniRunnerMaxBlocks;
  size_t ipcChunkSize = 0;
  FLAGCXCHECK(normalizeUniRunnerIpcChunkSize(
      flagcxParamUniRunnerIpcChunkSize(), &ipcChunkSize));
  runnerState->uniRunnerIpcChunkSize = ipcChunkSize;
  runnerState->avgDivisor = 1;
  // Set device context
  FLAGCXCHECK(deviceAdaptor->setDevice(hcomm->cudaDev));

  // Both executor FIFOs are allocated only after the DAG plan and execution
  // mode are known. Static execution uses exact trigger counts; the RED-only
  // dynamic fallback initializes its legacy FIFO lazily.
  runnerState->fifo = new (std::nothrow) flagcxFifo();
  if (runnerState->fifo == NULL) {
    return flagcxSystemError;
  }
  runnerState->ipcFifo = new (std::nothrow) flagcxFifo();
  if (runnerState->ipcFifo == NULL) {
    delete runnerState->fifo;
    runnerState->fifo = NULL;
    return flagcxSystemError;
  }

  // Initialize queues
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
  runnerState->numPendingNodes = 0;

  // Create dedicated reduce and copy streams
  flagcxStream_t redStream = NULL;
  flagcxResult_t result = deviceAdaptor->streamCreate(&redStream);
  if (result != flagcxSuccess) {
    delete runnerState->ipcFifo;
    runnerState->ipcFifo = NULL;
    delete runnerState->fifo;
    runnerState->fifo = NULL;
    return result;
  }
  flagcxStream_t cpyStream = NULL;
  result = deviceAdaptor->streamCreate(&cpyStream);
  if (result != flagcxSuccess) {
    (void)deviceAdaptor->streamDestroy(redStream);
    delete runnerState->ipcFifo;
    runnerState->ipcFifo = NULL;
    delete runnerState->fifo;
    runnerState->fifo = NULL;
    return result;
  }
  runnerState->redStream = redStream;
  runnerState->cpyStream = cpyStream;
  runnerState->ipcCommStream = stream;
  return flagcxSuccess;
}

flagcxResult_t cleanupUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxStream_t ipcCommStream = hcomm->proxyState->uniRunnerState.ipcCommStream;
  flagcxStream_t redStream = hcomm->proxyState->uniRunnerState.redStream;
  flagcxStream_t cpyStream = hcomm->proxyState->uniRunnerState.cpyStream;
  flagcxStream_t netCommStream = hcomm->proxyState->uniRunnerState.netCommStream;

  // Clean up DAG scheduler
  FLAGCXCHECK(cleanupDagScheduler(&hcomm->proxyState->uniRunnerState));
  hcomm->proxyState->uniRunnerState.ipcReadySlots = 0;

  // Outstanding stream waits/writes may still touch streamFlags when
  // runUniRunner exits early on an error path, so synchronize before marking
  // the reusable flag-address queue inactive for this invocation.
  if (redStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(redStream));
  }
  if (cpyStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(cpyStream));
  }
  if (ipcCommStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(ipcCommStream));
  }

  if (netCommStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(netCommStream));
  }

  auto &state = hcomm->proxyState->uniRunnerState;
  for (size_t i = 0; i < state.numDagCompletionEvents; ++i) {
    if (state.dagCompletionEvents[i] != NULL) {
      FLAGCXCHECK(deviceAdaptor->eventDestroy(state.dagCompletionEvents[i]));
      state.dagCompletionEvents[i] = NULL;
    }
  }
  free(state.dagCompletionEvents);
  state.dagCompletionEvents = NULL;
  state.numDagCompletionEvents = 0;

  hcomm->proxyState->uniRunnerState.streamFlagsSize = 0;

  // Destroy streams
  if (redStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamDestroy(redStream));
    hcomm->proxyState->uniRunnerState.redStream = NULL;
  }
  if (cpyStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamDestroy(cpyStream));
    hcomm->proxyState->uniRunnerState.cpyStream = NULL;
  }
  if (netCommStream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamDestroy(netCommStream));
    hcomm->proxyState->uniRunnerState.netCommStream = NULL;
  }
  hcomm->proxyState->uniRunnerState.ipcCommStream = NULL;

  // Destroy fifo
  if (hcomm->proxyState->uniRunnerState.fifo != NULL) {
    FLAGCXCHECK(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
    delete hcomm->proxyState->uniRunnerState.fifo;
    hcomm->proxyState->uniRunnerState.fifo = NULL;
  }
  hcomm->uniRunnerFifoBuffer = NULL;
  if (hcomm->proxyState->uniRunnerState.ipcFifo != NULL) {
    FLAGCXCHECK(
        hcomm->proxyState->uniRunnerState.ipcFifo->flagcxIpcFifoDestroy());
    delete hcomm->proxyState->uniRunnerState.ipcFifo;
    hcomm->proxyState->uniRunnerState.ipcFifo = NULL;
  }
  hcomm->proxyState->uniRunnerState.ipcFifoDevicePtr = NULL;

  if (hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(
        hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice,
        flagcxMemDevice, NULL));
    hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice = NULL;
    hcomm->proxyState->uniRunnerState.ipcParentFlagsCount = 0;
  }

  return flagcxSuccess;
}

static flagcxResult_t initializeDynamicRedFifo(flagcxHeteroComm_t hcomm) {
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  if (runnerState->fifo == NULL || runnerState->fifo->buffer != NULL ||
      hcomm->uniRunnerFifoBuffer != NULL) {
    return flagcxInternalError;
  }
  flagcxResult_t result = runnerState->fifo->flagcxRedFifoInit();
  if (result != flagcxSuccess) {
    return result;
  }
  void *deviceBuffer = NULL;
  result = deviceAdaptor->hostGetDevicePointer(
      &deviceBuffer, static_cast<void *>(runnerState->fifo->buffer));
  if (result != flagcxSuccess || deviceBuffer == NULL) {
    if (result == flagcxSuccess) {
      result = flagcxSystemError;
    }
    flagcxResult_t destroyResult =
        runnerState->fifo->flagcxRedFifoDestroy();
    if (result == flagcxSuccess) {
      result = destroyResult;
    }
    return result;
  }
  hcomm->uniRunnerFifoBuffer = deviceBuffer;
  return flagcxSuccess;
}

static flagcxResult_t releaseUniRunnerIpcParentFlags(
    flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL || runnerState->ipcParentFlagsDevice == NULL) {
    return flagcxSuccess;
  }
  flagcxResult_t result = deviceAdaptor->deviceFree(
      runnerState->ipcParentFlagsDevice, flagcxMemDevice, NULL);
  if (result == flagcxSuccess) {
    runnerState->ipcParentFlagsDevice = NULL;
    runnerState->ipcParentFlagsCount = 0;
  }
  return result;
}

static flagcxResult_t releaseUniRunnerInvocationFifos(
    flagcxHeteroComm_t hcomm) {
  if (hcomm == NULL || hcomm->proxyState == NULL) {
    return flagcxInvalidArgument;
  }
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  flagcxResult_t result = flagcxSuccess;
  if (runnerState->fifo != NULL && runnerState->fifo->buffer != NULL) {
    flagcxResult_t redResult = runnerState->fifo->flagcxRedFifoDestroy();
    if (result == flagcxSuccess && redResult != flagcxSuccess) {
      result = redResult;
    }
  }
  hcomm->uniRunnerFifoBuffer = NULL;
  if (runnerState->ipcFifo != NULL && runnerState->ipcFifo->buffer != NULL) {
    flagcxResult_t ipcResult = runnerState->ipcFifo->flagcxIpcFifoDestroy();
    if (result == flagcxSuccess && ipcResult != flagcxSuccess) {
      result = ipcResult;
    }
  }
  runnerState->ipcFifoDevicePtr = NULL;
  return result;
}

static flagcxResult_t getUniRunnerStaticIpcMaxChunks(
    const flagcxUniRunnerState *runnerState, size_t *maxChunks) {
  if (runnerState == NULL || maxChunks == NULL ||
      runnerState->dagPlan.numIpcNodes == 0) {
    return flagcxInvalidArgument;
  }
  uint32_t chunks = 0;
  FLAGCXCHECK(checkedUniRunnerIpcChunkCount(
      runnerState->dagPlan.maxIpcNodeBytes,
      static_cast<size_t>(runnerState->uniRunnerIpcChunkSize), &chunks));
  *maxChunks = static_cast<size_t>(chunks);
  return flagcxSuccess;
}

static flagcxResult_t prepareStaticCollectiveExecution(
    flagcxComm_t comm, uniRunnerStaticExecutorSchedule *schedule,
    size_t *numRedTriggers, size_t *numIpcTriggers,
    size_t *maxExecutorBlocks, flagcxStaticIpcControl *ipcControl,
    flagcxResult_t localPreparationResult) {
  if (comm == NULL || comm->heteroComm == NULL || schedule == NULL ||
      numRedTriggers == NULL || numIpcTriggers == NULL ||
      maxExecutorBlocks == NULL || ipcControl == NULL) {
    return flagcxInvalidArgument;
  }
  *schedule = {};
  *numRedTriggers = 0;
  *numIpcTriggers = 0;
  *maxExecutorBlocks = 0;
  *ipcControl = {};

  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  const size_t ipcTasks = runnerState->dagPlan.numIpcNodes;
  const size_t redTasks = runnerState->dagPlan.numRedNodes;
  flagcxResult_t result = localPreparationResult;
  if (result == flagcxSuccess &&
      (ipcTasks == 0 || runnerState->ipcReadySlots != ipcTasks ||
      runnerState->fifo == NULL || runnerState->ipcFifo == NULL ||
      runnerState->fifo->buffer != NULL ||
      runnerState->ipcFifo->buffer != NULL ||
      hcomm->uniRunnerFifoBuffer != NULL ||
      runnerState->ipcFifoDevicePtr != NULL ||
      runnerState->ipcOwner != comm || runnerState->ipcInputMem == NULL ||
      runnerState->ipcOutputMem == NULL || runnerState->ipcDataBytes == 0 ||
      comm->localRanks <= 0 || comm->localRank < 0 ||
       comm->localRank >= comm->localRanks)) {
    result = flagcxInternalError;
  }

  size_t maxIpcParallelism = 0;
  if (result == flagcxSuccess) {
    result = getUniRunnerStaticIpcMaxChunks(runnerState, &maxIpcParallelism);
  }

#ifdef COMPILE_KERNEL_HOST
  if (result == flagcxSuccess) {
    result = flagcxGetStaticCollectiveKernelMaxExecutorBlocks(
        static_cast<size_t>(runnerState->uniRunnerNThreads),
        maxExecutorBlocks);
    if (result == flagcxSuccess) {
      *maxExecutorBlocks = std::min(
          *maxExecutorBlocks,
          static_cast<size_t>(runnerState->uniRunnerMaxBlocks));
    }
  }
#else
  if (result == flagcxSuccess) {
    result = flagcxNotSupported;
  }
#endif

  if (result == flagcxSuccess) {
    result = resolveUniRunnerStaticExecutorSchedule(
        &runnerState->dagPlan,
        static_cast<size_t>(runnerState->uniRunnerNRedBlocks),
        static_cast<size_t>(runnerState->uniRunnerNIpcBlocks),
        maxIpcParallelism, *maxExecutorBlocks, schedule);
  }
  if (result == flagcxSuccess &&
      (schedule->numRedTasks != redTasks ||
       schedule->numIpcTasks != ipcTasks || schedule->numIpcBlocks == 0 ||
       (redTasks != 0 && schedule->numRedBlocks == 0) ||
       (redTasks == 0 && schedule->numRedBlocks != 0))) {
    result = flagcxInternalError;
  }

  const size_t localRanks = comm->localRanks > 0
                                ? static_cast<size_t>(comm->localRanks)
                                : size_t{0};
  size_t controlSlots = 0;
  size_t requiredReadySlots = 0;
  const uint64_t maxControlEpoch =
      flagcxIpcControlAbortBit - uint64_t{1};
  uint64_t epoch = 0;
  uint64_t invocationSignature = 0;
  uint64_t preflightSignature = 0;
  uint64_t readyState = 0;
  if (result == flagcxSuccess && redTasks != 0) {
    result = runnerState->fifo->flagcxRedFifoInit(redTasks);
    if (result == flagcxSuccess) {
      void *redDeviceBuffer = NULL;
      result = deviceAdaptor->hostGetDevicePointer(
          &redDeviceBuffer, static_cast<void *>(runnerState->fifo->buffer));
      if (result == flagcxSuccess && redDeviceBuffer == NULL) {
        result = flagcxSystemError;
      }
      if (result == flagcxSuccess) {
        hcomm->uniRunnerFifoBuffer = redDeviceBuffer;
      }
    }
  }

  if (result == flagcxSuccess) {
    result = runnerState->ipcFifo->flagcxIpcFifoInit(ipcTasks);
    if (result == flagcxSuccess) {
      result = deviceAdaptor->hostGetDevicePointer(
          &runnerState->ipcFifoDevicePtr,
          static_cast<void *>(runnerState->ipcFifo->buffer));
      if (result == flagcxSuccess && runnerState->ipcFifoDevicePtr == NULL) {
        result = flagcxSystemError;
      }
    }
  }

  if (result == flagcxSuccess) {
    result = prepareUniRunnerIpcParentFlags(runnerState);
  }

  if (result == flagcxSuccess) {
    if (localRanks > std::numeric_limits<size_t>::max() - size_t{1}) {
      result = flagcxInvalidArgument;
    } else {
      controlSlots = size_t{1} + localRanks;
      if (runnerState->ipcReadySlots >
          std::numeric_limits<size_t>::max() - controlSlots) {
        result = flagcxInvalidArgument;
      } else {
        requiredReadySlots = controlSlots + runnerState->ipcReadySlots;
      }
    }
  }

  const bool hasReadyMem = runnerState->ipcReadyMem != NULL;
  const bool hasReadyBuffer = runnerState->ipcReadyBuffer != NULL;
  if (!hasReadyMem && !hasReadyBuffer &&
      runnerState->ipcReadyCapacity == 0) {
    readyState = 0; // Empty: every rank must enter initial creation.
  } else if (hasReadyMem && hasReadyBuffer) {
    readyState = runnerState->ipcReadyCapacity >= requiredReadySlots ? 1 : 2;
  } else {
    readyState = 3; // Partial state is never safe to reuse or rebuild.
  }
  preflightSignature = uint64_t{0xcbf29ce484222325};
  preflightSignature = mixUniRunnerInvocationSignature(
      preflightSignature, static_cast<uint64_t>(requiredReadySlots));
  preflightSignature =
      mixUniRunnerInvocationSignature(preflightSignature, readyState);
  preflightSignature = mixUniRunnerInvocationSignature(
      preflightSignature,
      static_cast<uint64_t>(runnerState->ipcReadyCapacity));

  // Everything above is rank-local, while ready DevMem creation below is a
  // bootstrap collective. Agree once before entering it so an occupancy,
  // schedule, exact-FIFO, mapping, or parent-flag allocation failure cannot
  // make one rank return while its peers block in that collective.
  {
    flagcxResult_t collectiveResult = flagcxSuccess;
    flagcxResult_t agreementResult = agreeUniRunnerInvocationState(
        comm, preflightSignature, result, &collectiveResult);
    if (agreementResult != flagcxSuccess) {
      result = agreementResult;
      goto fail;
    }
    if (collectiveResult != flagcxSuccess) {
      result = collectiveResult;
      goto fail;
    }
  }
  if (result != flagcxSuccess) {
    goto fail;
  }

  result = ensureUniRunnerIpcReadyStorage(runnerState, comm,
                                          requiredReadySlots);
  if (result != flagcxSuccess) {
    goto fail;
  }
  if (runnerState->ipcReadyMem == NULL ||
      runnerState->ipcReadyCapacity < requiredReadySlots) {
    result = flagcxInternalError;
  }

  if (result == flagcxSuccess && runnerState->ipcEpoch >= maxControlEpoch) {
    WARN("UniRunner static IPC control epoch exhausted at %llu; refusing "
         "unsafe wraparound",
         static_cast<unsigned long long>(runnerState->ipcEpoch));
    result = flagcxInvalidUsage;
  }
  if (result == flagcxSuccess) {
    epoch = runnerState->ipcEpoch + uint64_t{1};
    if (!flagcxIpcControlEpochValid(epoch)) {
      result = flagcxInternalError;
    }
  }

  if (result == flagcxSuccess) {
    ipcControl->epoch = epoch;
    ipcControl->abortSlot = 0;
    ipcControl->doneBase = 1;
    ipcControl->readyDataOffset = static_cast<uint64_t>(controlSlots);
    ipcControl->localRank = static_cast<uint32_t>(comm->localRank);
    ipcControl->localRanks = static_cast<uint32_t>(comm->localRanks);
  }

  if (result == flagcxSuccess && redTasks != 0) {
    flagcxReduceTrigger *redTriggers =
        reinterpret_cast<flagcxReduceTrigger *>(
            runnerState->fifo->buffer + flagcxFifoIdxData);
    result = populateUniRunnerStaticRedTriggers(
        runnerState->dagNodes, static_cast<size_t>(runnerState->numDagNodes),
        &runnerState->dagPlan, runnerState->streamFlags,
        runnerState->streamFlagsSize, redTriggers, redTasks,
        numRedTriggers);
  }

  if (result == flagcxSuccess) {
    uniRunnerStaticIpcTriggerConfig config = {};
    config.chunkSize =
        static_cast<size_t>(runnerState->uniRunnerIpcChunkSize);
    config.epoch = epoch;
    config.dataBytes = runnerState->ipcDataBytes;
    config.readySlots = runnerState->ipcReadySlots;
    config.parentFlagsCount = runnerState->ipcParentFlagsCount;
    config.localRanks = comm->localRanks;
    flagcxIpcTrigger *ipcTriggers = reinterpret_cast<flagcxIpcTrigger *>(
        runnerState->ipcFifo->buffer + flagcxFifoIdxData);
    size_t materializedMaxChunks = 0;
    result = populateUniRunnerStaticIpcTriggers(
        runnerState->dagNodes, static_cast<size_t>(runnerState->numDagNodes),
        &runnerState->dagPlan, runnerState->streamFlags,
        runnerState->streamFlagsSize, &config, ipcTriggers, ipcTasks,
        numIpcTriggers, &materializedMaxChunks);
    if (result == flagcxSuccess &&
        materializedMaxChunks != maxIpcParallelism) {
      result = flagcxInternalError;
    }
  }

  if (result == flagcxSuccess &&
      (*numRedTriggers != redTasks || *numIpcTriggers != ipcTasks ||
       (redTasks != 0 &&
        runnerState->fifo->buffer[flagcxFifoIdxCapacity] != redTasks) ||
       runnerState->ipcFifo->buffer[flagcxFifoIdxCapacity] != ipcTasks)) {
    result = flagcxInternalError;
  }

  invocationSignature = uint64_t{0xcbf29ce484222325};
  invocationSignature =
      mixUniRunnerInvocationSignature(invocationSignature, epoch);
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(ipcTasks));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(redTasks));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature,
      static_cast<uint64_t>(runnerState->ipcReadySlots));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(controlSlots));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(requiredReadySlots));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(schedule->numRedBlocks));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature,
      static_cast<uint64_t>(runnerState->uniRunnerNThreads));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(schedule->numIpcBlocks));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(*maxExecutorBlocks));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature,
      static_cast<uint64_t>(runnerState->ipcDataBytes));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature,
      static_cast<uint64_t>(runnerState->uniRunnerIpcChunkSize));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature,
      static_cast<uint64_t>(runnerState->ipcParentFlagsCount));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(maxIpcParallelism));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(*numRedTriggers));
  invocationSignature = mixUniRunnerInvocationSignature(
      invocationSignature, static_cast<uint64_t>(*numIpcTriggers));
  {
    flagcxResult_t collectiveResult = flagcxSuccess;
    flagcxResult_t agreementResult = agreeUniRunnerInvocationState(
        comm, invocationSignature, result, &collectiveResult);
    if (agreementResult != flagcxSuccess) {
      result = agreementResult;
      goto fail;
    }
    if (collectiveResult != flagcxSuccess) {
      result = collectiveResult;
      goto fail;
    }
  }
  runnerState->ipcEpoch = epoch;

  TRACE(FLAGCX_UNIRUNNER,
        "static RED+IPC schedule: red_tasks=%zu ipc_tasks=%zu "
        "max_ipc_chunks=%zu nthreads=%llu red_blocks=%zu ipc_blocks=%zu budget=%zu "
        "epoch=%llu",
        redTasks, ipcTasks, maxIpcParallelism,
        static_cast<unsigned long long>(runnerState->uniRunnerNThreads),
        schedule->numRedBlocks,
        schedule->numIpcBlocks, *maxExecutorBlocks,
        static_cast<unsigned long long>(epoch));
  return flagcxSuccess;

fail:
  {
    flagcxResult_t parentResult =
        releaseUniRunnerIpcParentFlags(runnerState);
    if (result == flagcxSuccess && parentResult != flagcxSuccess) {
      result = parentResult;
    }
    flagcxResult_t fifoResult = releaseUniRunnerInvocationFifos(hcomm);
    if (result == flagcxSuccess && fifoResult != flagcxSuccess) {
      result = fifoResult;
    }
  }
  return result;
}

static flagcxResult_t prepareStaticRedExecution(
    flagcxHeteroComm_t hcomm, uniRunnerStaticExecutorSchedule *schedule,
    size_t *numTriggers, size_t *maxExecutorBlocks) {
  if (hcomm == NULL || schedule == NULL || numTriggers == NULL ||
      maxExecutorBlocks == NULL) {
    return flagcxInvalidArgument;
  }
  *schedule = {};
  *numTriggers = 0;
  *maxExecutorBlocks = 0;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  if (runnerState->dagPlan.numIpcNodes != 0 || runnerState->fifo == NULL ||
      runnerState->fifo->buffer != NULL ||
      hcomm->uniRunnerFifoBuffer != NULL) {
    return flagcxInternalError;
  }

  if (runnerState->dagPlan.numRedNodes == 0) {
    return resolveUniRunnerStaticExecutorSchedule(
        &runnerState->dagPlan,
        static_cast<size_t>(runnerState->uniRunnerNRedBlocks), 0, 0, 0,
        schedule);
  }

#ifdef COMPILE_KERNEL_HOST
  *maxExecutorBlocks =
      static_cast<size_t>(runnerState->uniRunnerMaxBlocks);
  FLAGCXCHECK(resolveUniRunnerStaticExecutorSchedule(
      &runnerState->dagPlan,
      static_cast<size_t>(runnerState->uniRunnerNRedBlocks), 0, 0,
      *maxExecutorBlocks,
      schedule));
#else
  return flagcxNotSupported;
#endif

  if (schedule->numRedTasks != runnerState->dagPlan.numRedNodes ||
      schedule->numIpcTasks != 0 || schedule->numRedBlocks == 0 ||
      schedule->numIpcBlocks != 0) {
    return flagcxInternalError;
  }

  const size_t redTasks = schedule->numRedTasks;
  flagcxResult_t result = runnerState->fifo->flagcxRedFifoInit(redTasks);
  if (result != flagcxSuccess) {
    return result;
  }
  void *deviceBuffer = NULL;
  flagcxReduceTrigger *triggers = NULL;
  result = deviceAdaptor->hostGetDevicePointer(
      &deviceBuffer, static_cast<void *>(runnerState->fifo->buffer));
  if (result != flagcxSuccess || deviceBuffer == NULL) {
    if (result == flagcxSuccess) {
      result = flagcxSystemError;
    }
    goto fail;
  }
  hcomm->uniRunnerFifoBuffer = deviceBuffer;

  triggers = reinterpret_cast<flagcxReduceTrigger *>(
      runnerState->fifo->buffer + flagcxFifoIdxData);
  result = populateUniRunnerStaticRedTriggers(
      runnerState->dagNodes, static_cast<size_t>(runnerState->numDagNodes),
      &runnerState->dagPlan, runnerState->streamFlags,
      runnerState->streamFlagsSize, triggers, redTasks, numTriggers,
      schedule->numRedBlocks > 1 ? &schedule->numNetRedTasks : NULL);
  if (result != flagcxSuccess) {
    goto fail;
  }
  if (*numTriggers != redTasks ||
      runnerState->fifo->buffer[flagcxFifoIdxCapacity] != redTasks) {
    result = flagcxInternalError;
    goto fail;
  }

  if (schedule->numNetRedTasks != 0 && schedule->numNetRedTasks < redTasks) {
    const size_t localTasks = redTasks - schedule->numNetRedTasks;
    const size_t netBlocks = std::min(schedule->numNetRedTasks,
                               std::max<size_t>(1, schedule->numRedBlocks / 2));
    const size_t localBlocks = std::min(localTasks,
                                       schedule->numRedBlocks - netBlocks);
    schedule->numRedBlocks = localBlocks + netBlocks;
    schedule->numNetRedBlocks = netBlocks;
  }

  TRACE(FLAGCX_UNIRUNNER,
        "static RED schedule: tasks=%zu requested_blocks=%llu "
        "effective_blocks=%zu",
        redTasks,
        static_cast<unsigned long long>(runnerState->uniRunnerNRedBlocks),
        schedule->numRedBlocks);
  return flagcxSuccess;

fail:
  {
    flagcxResult_t destroyResult =
        runnerState->fifo->flagcxRedFifoDestroy();
    hcomm->uniRunnerFifoBuffer = NULL;
    if (result == flagcxSuccess) {
      result = destroyResult;
    }
  }
  return result;
}

static void preserveFirstUniRunnerError(flagcxResult_t candidate,
                                        flagcxResult_t *result) {
  if (*result == flagcxSuccess && candidate != flagcxSuccess) {
    *result = candidate;
  }
}

static flagcxResult_t drainUniRunnerStreams(flagcxUniRunnerState *runnerState,
                                            flagcxResult_t result) {
  preserveFirstUniRunnerError(
      deviceAdaptor->streamSynchronize(runnerState->redStream), &result);
  preserveFirstUniRunnerError(
      deviceAdaptor->streamSynchronize(runnerState->cpyStream), &result);
  preserveFirstUniRunnerError(
      deviceAdaptor->streamSynchronize(runnerState->ipcCommStream), &result);
  if (runnerState->netCommStream != NULL) {
    preserveFirstUniRunnerError(
        deviceAdaptor->streamSynchronize(runnerState->netCommStream), &result);
  }
  return result;
}

static void requestUniRunnerStaticAbort(flagcxUniRunnerState *runnerState) {
  if (runnerState->fifo != NULL && runnerState->fifo->buffer != NULL) {
    __atomic_store_n(runnerState->fifo->buffer + flagcxFifoIdxTerminate, 1,
                     __ATOMIC_RELEASE);
  }
  if (runnerState->ipcFifo != NULL &&
      runnerState->ipcFifo->buffer != NULL) {
    // The combined static kernel deliberately uses the IPC FIFO terminate word
    // as its one rank-local abort flag.
    __atomic_store_n(runnerState->ipcFifo->buffer + flagcxFifoIdxTerminate, 1,
                     __ATOMIC_RELEASE);
  }
}

enum uniRunnerStaticStreamIndex {
  uniRunnerStaticRedStream = 0,
  uniRunnerStaticCpyStream = 1,
  uniRunnerStaticIpcCommStream = 2,
  uniRunnerStaticNetCommStream = 3,
  uniRunnerStaticNumStreams = 4,
  uniRunnerStaticNoFailedStream = -1,
};

static flagcxResult_t repairUniRunnerStaticFlags(
    flagcxUniRunnerState *runnerState, flagcxStream_t stream) {
  flagcxResult_t result = flagcxSuccess;
  for (size_t flagIdx = 0; flagIdx < runnerState->streamFlagsSize; ++flagIdx) {
    preserveFirstUniRunnerError(
        deviceAdaptor->streamWriteValue64(
            stream, runnerState->streamFlags[flagIdx], flagcxStreamFlagDone,
            0),
        &result);
  }
  return result;
}

static flagcxResult_t recoverPoisonedUniRunnerRedStream(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl, flagcxResult_t result);

static flagcxResult_t pollUniRunnerStaticStreams(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl, flagcxResult_t result,
    int knownCompleteStream) {
  flagcxStream_t streams[uniRunnerStaticNumStreams] = {
      runnerState->redStream, runnerState->cpyStream,
      runnerState->ipcCommStream, runnerState->netCommStream};
  bool complete[uniRunnerStaticNumStreams] = {};
  size_t remaining = uniRunnerStaticNumStreams;
  if (knownCompleteStream >= 0 &&
      knownCompleteStream < uniRunnerStaticNumStreams) {
    complete[knownCompleteStream] = true;
    --remaining;
  }

  while (remaining != 0) {
    bool madeProgress = false;
    for (int streamIdx = 0; streamIdx < uniRunnerStaticNumStreams;
         ++streamIdx) {
      if (complete[streamIdx]) {
        continue;
      }
      flagcxResult_t queryResult =
          streamIdx == uniRunnerStaticNetCommStream && streams[streamIdx] == NULL
              ? flagcxSuccess : deviceAdaptor->streamQuery(streams[streamIdx]);
      if (queryResult == flagcxInProgress) {
        continue;
      }
      complete[streamIdx] = true;
      --remaining;
      madeProgress = true;
      preserveFirstUniRunnerError(queryResult, &result);
      if (streamIdx == uniRunnerStaticRedStream &&
          queryResult != flagcxSuccess) {
        // streamQuery returning a terminal error establishes that the local
        // executor is no longer running. Only now is an independent recovery
        // stream safe: it cannot race peer writes from the failed executor.
        return recoverPoisonedUniRunnerRedStream(
            runnerState, ipcControl, result);
      }
    }
    if (!madeProgress && remaining != 0) {
      sched_yield();
    }
  }
  return result;
}

static flagcxResult_t recoverPoisonedUniRunnerRedStream(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl, flagcxResult_t result) {
  requestUniRunnerStaticAbort(runnerState);

  // This stream is created only on the asynchronous-error path. The failed RED
  // stream is already terminal, so IPC recovery may safely publish this rank's
  // abort/done epoch and join the peer quiescence barrier before releasing any
  // host-stream DAG waits.
  flagcxStream_t recoveryStream = NULL;
  flagcxResult_t createResult =
      deviceAdaptor->streamCreate(&recoveryStream);
  if (createResult == flagcxSuccess && recoveryStream == NULL) {
    createResult = flagcxSystemError;
  }
  preserveFirstUniRunnerError(createResult, &result);
  if (createResult != flagcxSuccess || recoveryStream == NULL) {
    return result;
  }

  flagcxResult_t recoveryResult = flagcxSuccess;
#ifdef COMPILE_KERNEL_HOST
  if (ipcControl != NULL) {
    recoveryResult = flagcxLaunchStaticIpcRecoveryKernel(
        runnerState->ipcReadyMem, ipcControl, recoveryStream);
  }
#else
  if (ipcControl != NULL) {
    recoveryResult = flagcxNotSupported;
  }
#endif
  preserveFirstUniRunnerError(recoveryResult, &result);

  flagcxResult_t repairResult =
      repairUniRunnerStaticFlags(runnerState, recoveryStream);
  preserveFirstUniRunnerError(repairResult, &result);
  flagcxResult_t syncResult =
      deviceAdaptor->streamSynchronize(recoveryStream);
  preserveFirstUniRunnerError(syncResult, &result);
  preserveFirstUniRunnerError(
      deviceAdaptor->streamDestroy(recoveryStream), &result);

  if (recoveryResult != flagcxSuccess || repairResult != flagcxSuccess ||
      syncResult != flagcxSuccess) {
    // Do not poll streams which may still be blocked when recovery itself could
    // not be established. Returning the original terminal error avoids turning
    // a failed GPU/context into an unbounded host busy wait.
    return result;
  }
  return pollUniRunnerStaticStreams(
      runnerState, ipcControl, result, uniRunnerStaticRedStream);
}

static flagcxResult_t abortAndDrainStaticExecution(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl, flagcxResult_t result,
    int knownFailedStream = uniRunnerStaticNoFailedStream) {
  requestUniRunnerStaticAbort(runnerState);

  // Host submission failures and non-RED stream failures leave redStream
  // healthy. Queue repair after the executor: stream order establishes local
  // quiescence before releasing host streams blocked on unfinished DAG flags.
  preserveFirstUniRunnerError(
      repairUniRunnerStaticFlags(runnerState, runnerState->redStream),
      &result);
  return pollUniRunnerStaticStreams(
      runnerState, ipcControl, result, knownFailedStream);
}

static flagcxResult_t waitForStaticStreamCompletion(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl) {
  flagcxStream_t streams[uniRunnerStaticNumStreams] = {
      runnerState->redStream, runnerState->cpyStream,
      runnerState->ipcCommStream, runnerState->netCommStream};
  bool complete[uniRunnerStaticNumStreams] = {};
  size_t remaining = uniRunnerStaticNumStreams;
  while (remaining != 0) {
    bool madeProgress = false;
    for (int streamIdx = 0; streamIdx < uniRunnerStaticNumStreams;
         ++streamIdx) {
      if (complete[streamIdx]) {
        continue;
      }
      flagcxResult_t queryResult =
          streamIdx == uniRunnerStaticNetCommStream && streams[streamIdx] == NULL
              ? flagcxSuccess : deviceAdaptor->streamQuery(streams[streamIdx]);
      if (queryResult == flagcxSuccess) {
        complete[streamIdx] = true;
        --remaining;
        madeProgress = true;
      } else if (queryResult != flagcxInProgress) {
        if (streamIdx == uniRunnerStaticRedStream) {
          return recoverPoisonedUniRunnerRedStream(
              runnerState, ipcControl, queryResult);
        }
        return abortAndDrainStaticExecution(
            runnerState, ipcControl, queryResult, streamIdx);
      }
    }
    if (!madeProgress && remaining != 0) {
      sched_yield();
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t runDynamicDagFallback(flagcxHeteroComm_t hcomm);

// Connection setup may allocate device memory. Complete it before submitting
// cross-stream waits or launching the persistent executor for the pipeline.
static flagcxResult_t prepareUniRunnerP2pConnections(flagcxHeteroComm_t comm) {
  flagcxUniRunnerState &state = comm->proxyState->uniRunnerState;
  if (state.netCommStream == NULL) return flagcxSuccess;
  if (!comm->proxyState->initialized) FLAGCXCHECK(flagcxProxyInit(comm));
  bool pending = false;
  for (int i = 0; i < state.numDagNodes; ++i) {
    const auto &node = state.dagNodes[i];
    if (node.nodeType != uniRunnerDagNodeTypeP2p) continue;
    for (int j = 0; j < node.nodeData.p2p.numOps; ++j) {
      const auto &op = node.nodeData.p2p.ops[j];
      const bool send = op.type == flagcxDevicePrimSend;
      auto &connector = send ? comm->channels[0].peers[op.peerRank]->send[0]
                             : comm->channels[0].peers[op.peerRank]->recv[0];
      if (connector.connected) continue;
      (send ? comm->connectSend : comm->connectRecv)[op.peerRank] |= 1UL;
      connector.registered = 1;
      pending = true;
    }
  }
  if (pending) FLAGCXCHECK(flagcxTransportP2pSetup(comm, NULL, 0));
  FLAGCXCHECK(flagcxCalloc(&state.dagCompletionEvents, state.numDagNodes));
  state.numDagCompletionEvents = state.numDagNodes;
  for (int i = 0; i < state.numDagNodes; ++i) {
    const auto &node = state.dagNodes[i];
    if (node.nodeType != uniRunnerDagNodeTypeP2p) continue;
    for (int j = 0; j < node.numParents; ++j) {
      const int parentIdx = node.parents[j];
      const auto &parent = state.dagNodes[parentIdx];
      if (parent.nodeType == uniRunnerDagNodeTypeP2p &&
          parent.commDomain != node.commDomain &&
          state.dagCompletionEvents[parentIdx] == NULL) {
        FLAGCXCHECK(deviceAdaptor->eventCreate(
            &state.dagCompletionEvents[parentIdx], flagcxEventDisableTiming));
      }
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t runStaticHostRed(flagcxHeteroComm_t hcomm) {
  FLAGCXCHECK(prepareUniRunnerP2pConnections(hcomm));
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  uniRunnerStaticExecutorSchedule schedule;
  size_t numTriggers = 0;
  size_t maxExecutorBlocks = 0;
  flagcxResult_t prepareResult = prepareStaticRedExecution(
      hcomm, &schedule, &numTriggers, &maxExecutorBlocks);
  if (prepareResult != flagcxSuccess) {
#ifdef COMPILE_KERNEL_HOST
    // Capability/occupancy probing happens before the exact static FIFO is
    // allocated. Preserve the legacy executor on devices where the static
    // cooperative-progress contract is unavailable or cannot be queried.
    if ((prepareResult == flagcxNotSupported ||
         prepareResult == flagcxUnhandledDeviceError) &&
        runnerState->fifo != NULL && runnerState->fifo->buffer == NULL &&
        hcomm->uniRunnerFifoBuffer == NULL) {
      TRACE(FLAGCX_UNIRUNNER,
            "static RED unavailable (%d); using dynamic DAG fallback",
            prepareResult);
      return runDynamicDagFallback(hcomm);
    }
#endif
    return prepareResult;
  }

#ifdef COMPILE_KERNEL_HOST
  if (numTriggers != 0) {
    flagcxResult_t launchResult = flagcxLaunchStaticReduceKernel(
        hcomm->uniRunnerFifoBuffer, numTriggers,
        static_cast<size_t>(runnerState->uniRunnerNThreads),
        schedule.numRedBlocks, maxExecutorBlocks, runnerState->avgDivisor,
        runnerState->redStream, schedule.numNetRedTasks,
        schedule.numNetRedBlocks);
    if (launchResult != flagcxSuccess) {
      // A runtime launch API may report an earlier asynchronous error even if
      // this executor was accepted. Abort and queue flag repair behind it on
      // redStream so both the enqueued and definitely-unlaunched cases drain.
      return abortAndDrainStaticExecution(
          runnerState, NULL, launchResult);
    }
  }
#else
  if (numTriggers != 0) {
    return flagcxNotSupported;
  }
#endif

  flagcxResult_t result = submitStaticHostDag(runnerState, hcomm);
  if (result != flagcxSuccess) {
    result = abortAndDrainStaticExecution(runnerState, NULL, result);
  } else if (numTriggers == 0) {
    // Host-only DAGs have no persistent executor that can fail while waiting
    // for another stream. Avoid the multi-stream query loop on this latency-
    // sensitive path; every dependency-producing operation is already queued.
    result = drainUniRunnerStreams(runnerState, flagcxSuccess);
  } else {
    result = waitForStaticStreamCompletion(runnerState, NULL);
  }
  return result;
}

static flagcxResult_t recoverFailedStaticCollectiveLaunch(
    flagcxUniRunnerState *runnerState,
    const flagcxStaticIpcControl *ipcControl,
    flagcxStream_t executorStream, flagcxResult_t launchResult) {
  // Queue recovery on the same RED stream as the attempted cooperative launch.
  // If the launch actually enqueued but reported an earlier asynchronous error,
  // stream order delays recovery until the executor has stopped peer writes and
  // completed its done barrier. If it was not enqueued, recovery safely fills
  // the missing rank-local done publication. This must be the same stream used
  // for the attempted executor launch.
  requestUniRunnerStaticAbort(runnerState);
#ifdef COMPILE_KERNEL_HOST
  flagcxResult_t recoveryResult = flagcxLaunchStaticIpcRecoveryKernel(
      runnerState->ipcReadyMem, ipcControl, executorStream);
  preserveFirstUniRunnerError(recoveryResult, &launchResult);
  // Synchronize even when the recovery launch reports an error: both the
  // original executor launch and this recovery launch may have been accepted
  // while the API surfaced an earlier asynchronous failure. Stream order is
  // the only safe way to resolve that ambiguity.
  flagcxResult_t syncResult =
      deviceAdaptor->streamSynchronize(executorStream);
  preserveFirstUniRunnerError(syncResult, &launchResult);
  if (recoveryResult != flagcxSuccess || syncResult != flagcxSuccess) {
    // Resolve any remaining launch ambiguity explicitly. Host abort makes a
    // live main executor leave its polling loops; once streamQuery is terminal,
    // an independent stream can retry the idempotent recovery without racing
    // executor peer writes. A true hardware hang remains an external-abort case.
    flagcxResult_t queryResult = flagcxInProgress;
    while (queryResult == flagcxInProgress) {
      queryResult = deviceAdaptor->streamQuery(executorStream);
      if (queryResult == flagcxInProgress) {
        sched_yield();
      }
    }
    preserveFirstUniRunnerError(queryResult, &launchResult);
    return recoverPoisonedUniRunnerRedStream(
        runnerState, ipcControl, launchResult);
  }
#else
  (void)ipcControl;
  (void)executorStream;
#endif
  return launchResult;
}

static flagcxResult_t runStaticHostRedIpc(
    flagcxComm_t comm, flagcxResult_t localPreparationResult) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  uniRunnerStaticExecutorSchedule schedule = {};
  flagcxStaticIpcControl ipcControl = {};
  size_t numRedTriggers = 0;
  size_t numIpcTriggers = 0;
  size_t maxExecutorBlocks = 0;
  FLAGCXCHECK(prepareStaticCollectiveExecution(
      comm, &schedule, &numRedTriggers, &numIpcTriggers, &maxExecutorBlocks,
      &ipcControl, localPreparationResult));

#ifdef COMPILE_KERNEL_HOST
  flagcxResult_t launchResult = flagcxLaunchStaticCollectiveKernel(
      hcomm->uniRunnerFifoBuffer, numRedTriggers,
      runnerState->ipcFifoDevicePtr, numIpcTriggers,
      runnerState->ipcInputMem, runnerState->ipcOutputMem,
      runnerState->ipcReadyMem, runnerState->ipcParentFlagsDevice,
      runnerState->ipcParentFlagsCount,
      static_cast<size_t>(runnerState->uniRunnerNThreads),
      schedule.numRedBlocks, schedule.numIpcBlocks, maxExecutorBlocks,
      runnerState->avgDivisor, &ipcControl, runnerState->redStream);
  if (launchResult != flagcxSuccess) {
    // The runtime API may surface an earlier asynchronous error even if this
    // launch was accepted. Recovery therefore follows the attempted executor
    // on redStream. An independent retry is allowed only after synchronizing
    // that stream has established a terminal/quiesced state.
    return recoverFailedStaticCollectiveLaunch(
        runnerState, &ipcControl, runnerState->redStream, launchResult);
  }
#else
  return flagcxNotSupported;
#endif

  flagcxResult_t result = submitStaticHostDag(runnerState, hcomm);
  if (result != flagcxSuccess) {
    // The executor is already live. Signal its own abort word and let its
    // blocksDone epilogue establish quiescence; never launch recovery here.
    result = abortAndDrainStaticExecution(
        runnerState, &ipcControl, result);
  } else {
    result = waitForStaticStreamCompletion(runnerState, &ipcControl);
    if (result == flagcxSuccess && runnerState->ipcFifo != NULL &&
        runnerState->ipcFifo->buffer != NULL &&
        __atomic_load_n(runnerState->ipcFifo->buffer + flagcxFifoIdxTerminate,
                        __ATOMIC_ACQUIRE) != 0) {
      result = flagcxRemoteError;
    }
  }
  return result;
}

static flagcxResult_t runDynamicDagFallback(flagcxHeteroComm_t hcomm) {
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  // IPC FIFOs now carry a control prefix and are materialized statically.
  // Feeding IPC nodes through the legacy producer/dequeue path would alias its
  // ready slot zero with the abort word, so fallback is intentionally RED-only.
  if (runnerState->dagPlan.numIpcNodes != 0 ||
      runnerState->ipcFifo == NULL ||
      runnerState->ipcFifo->buffer != NULL ||
      runnerState->ipcFifoDevicePtr != NULL) {
    return flagcxNotSupported;
  }
  FLAGCXCHECK(initializeDynamicRedFifo(hcomm));

#ifdef COMPILE_KERNEL_HOST
  FLAGCXCHECK(flagcxLaunchCollectiveKernel(
      hcomm->uniRunnerFifoBuffer, NULL, NULL, NULL, NULL, NULL,
      runnerState->uniRunnerNThreads, runnerState->uniRunnerNRedBlocks,
      0, runnerState->avgDivisor,
      runnerState->redStream));
#else
  return flagcxNotSupported;
#endif

  while (true) {
    if (flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->redReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->ipcReadyQueue) &&
        runnerState->numPendingNodes == 0) {
      TRACE(
          FLAGCX_UNIRUNNER,
          "runUniRunner: dynamic RED fallback drained, terminating executor");
      __atomic_store_n(runnerState->fifo->buffer + flagcxFifoIdxTerminate, 1,
                       __ATOMIC_RELEASE);
      break;
    }
    FLAGCXCHECK(processReadyQueue(runnerState, hcomm));
  }
  return drainUniRunnerStreams(runnerState, flagcxSuccess);
}

flagcxResult_t runUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  TRACE(FLAGCX_UNIRUNNER, "runUniRunner called");
  const bool ipcIntent =
      runnerState->ipcOwner == comm && runnerState->ipcReadySlots != 0;
  flagcxResult_t localPreparationResult = flagcxSuccess;
  if (runnerState->numDagNodes < 0 ||
      (runnerState->numDagNodes == 0) != (runnerState->dagNodes == NULL)) {
    localPreparationResult = flagcxInternalError;
  }
  if (localPreparationResult == flagcxSuccess) {
    const size_t numDagNodes = static_cast<size_t>(runnerState->numDagNodes);
    const uniRunnerDagExecutionPlan &plan = runnerState->dagPlan;
    if (!plan.staticValidated || plan.numNodes != numDagNodes ||
        (numDagNodes != 0 && runnerState->dagPlan.topoOrder == NULL) ||
        plan.numHostNodes > numDagNodes ||
        plan.numRedNodes > numDagNodes - plan.numHostNodes ||
        plan.numIpcNodes != numDagNodes - plan.numHostNodes - plan.numRedNodes ||
        (plan.numRedNodes != 0 &&
         (runnerState->uniRunnerNRedBlocks == 0 ||
          runnerState->uniRunnerNThreads == 0 ||
          runnerState->uniRunnerNThreads >
              flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))) ||
        (plan.numIpcNodes != 0 &&
         (runnerState->uniRunnerNIpcBlocks == 0 ||
          runnerState->ipcReadySlots != plan.numIpcNodes))) {
      localPreparationResult = flagcxInternalError;
    }
  }
  if (localPreparationResult == flagcxSuccess) {
    localPreparationResult = prepareDagStreamFlags(runnerState);
  }
  if (ipcIntent) {
    // IPC initialization already agreed this intent across ranks. Preserve the
    // static preparation collective even when one rank fails validation or
    // stream-flag setup, so no peer enters the preflight all-gather alone.
    return runStaticHostRedIpc(comm, localPreparationResult);
  }
  if (localPreparationResult != flagcxSuccess) {
    return localPreparationResult;
  }
  return runStaticHostRed(hcomm);
}

flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState) {
  FLAGCXCHECK(cleanupDagScheduler(runnerState));
  FLAGCXCHECK(destroyStreamFlagQueue(runnerState));
  if (runnerState == NULL)
    return flagcxSuccess;

  if (runnerState->groupedArScratch != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->groupedArScratch,
                                          flagcxMemDevice, NULL));
    runnerState->groupedArScratch = NULL;
    runnerState->groupedArScratchCapacity = 0;
  }
  flagcxComm_t comm = runnerState->ipcOwner;
  FLAGCXCHECK(destroyUniRunnerIpcDataViews(runnerState));
  if (comm != NULL && runnerState->ipcReadyMem != NULL) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->ipcReadyMem));
  }
  runnerState->ipcReadyMem = NULL;
  if (runnerState->ipcReadyBuffer != NULL) {
    if (comm == NULL) {
      return flagcxInternalError;
    }
    // Peer CUDA IPC mappings are closed later by
    // flagcxCommCleanupIpcTable(). Defer the exporter allocation until the
    // communicator subsequently drains deferred frees. The communicator's
    // overflow chain preserves safety if its fixed inline slots are exhausted.
    flagcxCommDeferFree(comm, runnerState->ipcReadyBuffer, flagcxMemDevice);
  }
  runnerState->ipcReadyBuffer = NULL;
  runnerState->ipcReadyCapacity = 0;
  runnerState->ipcReadySlots = 0;
  runnerState->ipcOwner = NULL;
  if (runnerState->ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->ipcParentFlagsDevice,
                                          flagcxMemDevice, NULL));
  }
  runnerState->ipcParentFlagsDevice = NULL;
  runnerState->ipcParentFlagsCount = 0;
  return flagcxSuccess;
}
