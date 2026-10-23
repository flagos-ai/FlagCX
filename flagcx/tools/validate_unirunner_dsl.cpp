#include "uni_runner_helper.h"
#include "uni_runner_gemm.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
uint64_t fusedMaxBlocks = 128;

uint64_t integer(const Json &object, const char *name, uint64_t fallback = 0) {
  if (!object.contains(name)) return fallback;
  size_t value = 0;
  if (!uniRunnerDagSizeFromJson(object.at(name), &value))
    throw std::runtime_error(std::string("invalid integer: ") + name);
  return value;
}

void validateMaterialization(const uniRunnerDagTemplate &dag,
                             const uniRunnerDagRuntimeBindings &bindings,
                             uniRunnerDagAlgoType algo,
                             const uniRunnerDagAlgorithmConfig &config) {
  struct Instance {
    flagcxUniRunnerState state{};
    ~Instance() { cleanupUniRunnerDagTemplateInstance(&state); }
  } instance;
  auto &state = instance.state;
  const bool fused = algo == uniRunnerDagAlgoAllGatherGemm ||
                     algo == uniRunnerDagAlgoGemmReduceScatter;
  const uint64_t maxBlocks = fused ? fusedMaxBlocks : 32;
  state.uniRunnerNThreads = 256;
  state.uniRunnerMaxBlocks = maxBlocks;
  state.uniRunnerMaxThreads = 256;
  state.uniRunnerNRedSlices = config.numRedSlices;
  state.uniRunnerRedSliceSize = 1024 * 1024;
  state.redSliceWorkersEnabled = algo == uniRunnerDagAlgoGemmReduceScatter ||
      algo == uniRunnerDagAlgoSlicedAR || algo == uniRunnerDagAlgoRingRS ||
      algo == uniRunnerDagAlgoTreeRed;
  if (materializeUniRunnerDagTemplate(&state, dag, bindings) != flagcxSuccess)
    throw std::runtime_error("runtime materializer rejected graph");
  std::vector<uint64_t> counters(dag.nodes.size() * 16, 0);
  std::vector<uint64_t> flags(dag.nodes.size(), 0);
  std::vector<void *> flagAddresses;
  for (auto &flag : flags) flagAddresses.push_back(&flag);
  size_t counterOrdinal = 0;
  for (int i = 0; i < state.numDagNodes; ++i) {
    auto &node = state.dagNodes[i];
    if (node.nodeType == uniRunnerDagNodeTypeGemm) {
      auto &g = node.nodeData.gemm;
      if (resolveUniRunnerDslGemmWorkerCount(g.m, g.n, maxBlocks, fused ? static_cast<int64_t>(fusedMaxBlocks) : 0, &g.workerCount) != flagcxSuccess)
        throw std::runtime_error("GEMM worker resolution failed");
      g.completionCounter = reinterpret_cast<uint32_t *>(&counters[16 * counterOrdinal++]);
    } else if (node.nodeType == uniRunnerDagNodeTypeRed) {
      auto &r = node.nodeData.red;
      if (resolveUniRunnerRedWorkerCount(&state, r.count, &r.workerCount) != flagcxSuccess)
        throw std::runtime_error("RED worker resolution failed");
      if (r.workerCount > 1)
        r.completionCounter = reinterpret_cast<uint32_t *>(&counters[16 * counterOrdinal++]);
    }
  }
  if (compileUniRunnerDagExecutionPlan(state.dagNodes, state.numDagNodes, &state.dagPlan) != flagcxSuccess ||
      updateUniRunnerComputeRuntimeShape(state.dagNodes, state.numDagNodes, &state.dagPlan) != flagcxSuccess)
    throw std::runtime_error("runtime worker plan rejected graph");
  std::vector<flagcxReduceTrigger> triggers(state.dagPlan.numRedTasks);
  size_t numTriggers = 0, numNetTriggers = 0;
  if (populateUniRunnerStaticRedTriggers(state.dagNodes, state.numDagNodes,
          &state.dagPlan, flagAddresses.data(), flagAddresses.size(),
          triggers.data(), triggers.size(), &numTriggers, &numNetTriggers) != flagcxSuccess ||
      numTriggers != state.dagPlan.numRedTasks)
    throw std::runtime_error("persistent trigger compiler rejected graph");
  for (int i = 0; i < state.numDagNodes; ++i) {
    const auto &node = state.dagNodes[i];
    const bool gemm = node.nodeType == uniRunnerDagNodeTypeGemm;
    if (!gemm && node.nodeType != uniRunnerDagNodeTypeRed) continue;
    const uint32_t workers = gemm ? node.nodeData.gemm.workerCount : node.nodeData.red.workerCount;
    const int first = gemm ? node.nodeData.gemm.triggerIdx : node.nodeData.red.triggerIdx;
    const auto counter = reinterpret_cast<uint64_t>(gemm ? node.nodeData.gemm.completionCounter
                                                        : node.nodeData.red.completionCounter);
    if (first < 0 || size_t(first) + workers > triggers.size())
      throw std::runtime_error("worker trigger range mismatch");
    for (uint32_t worker = 0; worker < workers; ++worker) {
      const auto &trigger = triggers[first + worker];
      const uint64_t encoded = trigger.value[gemm ? 9 : 6];
      const uint64_t workerId = gemm
          ? ((encoded >> flagcxGemmTriggerOffWorkerId) & flagcxTriggerMask(flagcxGemmTriggerBitsWorkerId))
          : uint32_t(encoded);
      const uint64_t workerCount = gemm
          ? ((encoded >> flagcxGemmTriggerOffWorkerCount) & flagcxTriggerMask(flagcxGemmTriggerBitsWorkerCount))
          : encoded >> 32;
      if (workerId != worker || workerCount != workers ||
          trigger.value[gemm ? 10 : 7] != counter ||
          trigger.value[5] != reinterpret_cast<uint64_t>(flagAddresses[i]))
        throw std::runtime_error("worker completion encoding mismatch");
    }
  }
}

void validate(const Json &entry) {
  uniRunnerDagTemplate dag;
  if (!uniRunnerDeserializeDagTemplate(entry, &dag))
    throw std::runtime_error("runtime deserializer rejected DAG");
  const auto &algorithm = entry.at("algorithm");
  uniRunnerDagAlgoType algo;
  if (!uniRunnerDagAlgoTypeFromString(algorithm.at("name"), &algo))
    throw std::runtime_error("unknown algorithm");
  const auto &j = algorithm.at("config");
  uniRunnerDagAlgorithmConfig config{};
  config.matrixM = integer(j, "m");
  config.matrixN = integer(j, "n");
  config.matrixK = integer(j, "k");
  config.numKSlices = integer(j, "num_k_slices");
  config.numSlices = integer(j, "num_slices");
  config.numLocalSlices = integer(j, "num_local_slices");
  config.numRedSlices = integer(j, "num_red_slices");
  config.groupSize = integer(j, "group_size");
  config.pipelineVersion = integer(j, "pipeline_version");
  config.topologyHash = integer(j, "topology_hash");
  config.numChannels = integer(j, "num_channels");
  config.simpleBufferBytes = integer(j, "simple_buffer_bytes");
  config.nThreads = integer(j, "nthreads");
  config.chunkSteps = integer(j, "chunk_steps");
  config.sliceSteps = integer(j, "slice_steps");
  const auto mode = integer(j, "buffer_mode");
  if (mode > uniRunnerDagBufferModeInPlace)
    throw std::runtime_error("invalid buffer mode");
  config.bufferMode = static_cast<uniRunnerDagBufferMode>(mode);
  if (getUniRunnerDagAlgorithmHash(algo, config) != dag.key.algoHash)
    throw std::runtime_error("Python/C++ algorithm hash mismatch");
  uniRunnerCompiledDagTemplate compiled;
  if (compileUniRunnerDagTemplate(dag, &compiled) != flagcxSuccess)
    throw std::runtime_error("runtime DAG compiler rejected graph");
  const auto &sizes = entry.at("bindings");
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBytes = integer(sizes, "input_bytes");
  bindings.outputBytes = integer(sizes, "output_bytes");
  bindings.scratchBytes = integer(sizes, "scratch_bytes");
  bindings.weightBytes = integer(sizes, "weight_bytes");
  constexpr uintptr_t span = uintptr_t{1} << 44;
  if (bindings.inputBytes >= span || bindings.outputBytes >= span ||
      bindings.scratchBytes >= span || bindings.weightBytes >= span)
    throw std::runtime_error("synthetic binding capacity exceeded");
  // These addresses are range-check tokens only; no payload is allocated or read.
  bindings.inputBase = reinterpret_cast<const void *>(span);
  bindings.outputBase = reinterpret_cast<void *>(2 * span);
  bindings.scratchBase = reinterpret_cast<void *>(3 * span);
  bindings.weightBase = reinterpret_cast<const void *>(4 * span);
  bindings.localRanks = config.groupSize > 0 ? config.groupSize : dag.key.nranks;
  if (validateUniRunnerDagTemplateBindings(dag, bindings) != flagcxSuccess ||
      validateUniRunnerIpcDagTemplateBindings(dag, bindings.inputBytes,
          bindings.outputBytes, bindings.localRanks) != flagcxSuccess)
    throw std::runtime_error("runtime buffer binding rejected an access");
  if ((algo == uniRunnerDagAlgoAllGatherGemm || algo == uniRunnerDagAlgoGemmReduceScatter) &&
      validateUniRunnerFusedDagKey(dag.key, algo, config, bindings) != flagcxSuccess)
    throw std::runtime_error("fused API key/binding contract mismatch");
  validateMaterialization(dag, bindings, algo, config);
  uniRunnerDagTemplate restored;
  if (!uniRunnerDeserializeDagTemplate(uniRunnerSerializeDagTemplate(dag), &restored) ||
      compileUniRunnerDagTemplate(restored, &compiled) != flagcxSuccess)
    throw std::runtime_error("runtime JSON round trip failed");
}
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: validate_unirunner_dsl [--fused-max-blocks N] <rank-or-cache.json> ...\n";
    return 2;
  }
  size_t count = 0;
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::string(argv[i]) == "--fused-max-blocks") {
        if (++i == argc) throw std::runtime_error("missing block count");
        const std::string value = argv[i];
        size_t parsed = 0;
        fusedMaxBlocks = std::stoull(value, &parsed);
        if (parsed != value.size() || fusedMaxBlocks == 0 ||
            fusedMaxBlocks > static_cast<uint64_t>(std::numeric_limits<int>::max()))
          throw std::runtime_error("invalid block count");
        continue;
      }
      std::ifstream stream(argv[i]);
      if (!stream) throw std::runtime_error("cannot open input");
      Json root; stream >> root;
      if (root.contains("entries")) {
        for (const auto &entry : root.at("entries")) { validate(entry); ++count; }
      } else {
        validate(root);
        const std::string path = argv[i];
        const std::string file = path.substr(path.find_last_of('/') + 1);
        if (file.compare(0, 9, "dag_hash_") == 0) {
          const std::string expected = "dag_hash_" + root.at("hash").get<std::string>() +
              "_rank_" + std::to_string(root.at("key").at("rank").get<int>()) + ".json";
          if (file != expected) throw std::runtime_error("runtime lookup filename mismatch");
        }
        ++count;
      }
    } catch (const std::exception &error) {
      std::cerr << argv[i] << ": " << error.what() << '\n';
      return 1;
    }
  }
  if (count == 0) return 2;
  std::cout << "Validated " << count
            << " rank DAGs (fused blocks " << fusedMaxBlocks << "): runtime JSON, API keys, lookup filenames, algorithm hash, static plan, buffer bounds, host materialization, workers/counters, persistent triggers, round trip.\n";
  return 0;
}
