#include "uni_runner_helper.h"
#include <gtest/gtest.h>

namespace {
uniRunnerDagTemplate gemmTemplate() {
  uniRunnerDagTemplate dag;
  uniRunnerDagAlgorithmConfig config{};
  config.matrixM = 128;
  config.matrixN = 64;
  config.matrixK = 32;
  config.numSlices = 2;
  config.numKSlices = 1;
  config.groupSize = 1;
  config.pipelineVersion = 2;
  dag.key.algoHash = getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoAllGatherGemm, config);
  dag.key.commOp = flagcxCommOpAllGather;
  dag.key.count = 64 * 32;
  dag.key.datatype = flagcxFloat32;
  dag.key.redOp = flagcxRedNoOp;
  dag.key.rank = 0;
  dag.key.nranks = 2;
  dag.key.root = -1;
  uniRunnerDagNodeDesc node;
  node.nodeType = uniRunnerDagNodeTypeGemm;
  node.gemm.a = {uniRunnerDagBufferTypeInput, 0};
  node.gemm.b = {uniRunnerDagBufferTypeWeight, 0};
  node.gemm.c = {uniRunnerDagBufferTypeOutput, 0};
  node.gemm.m = 64;
  node.gemm.n = 64;
  node.gemm.k = 32;
  node.gemm.lda = 32;
  node.gemm.ldb = 64;
  node.gemm.ldc = 64;
  dag.nodes.push_back(node);
  return dag;
}
uniRunnerDagRuntimeBindings bindings() {
  uniRunnerDagRuntimeBindings result;
  result.inputBase = reinterpret_cast<void *>(0x1000000000ull);
  result.weightBase = reinterpret_cast<void *>(0x2000000000ull);
  result.outputBase = reinterpret_cast<void *>(0x3000000000ull);
  result.inputBytes = 64 * 32 * sizeof(float);
  result.weightBytes = 32 * 64 * sizeof(float);
  result.outputBytes = 64 * 64 * sizeof(float);
  return result;
}
}

TEST(UniRunnerDsl, GemmRoundTripKeepsOnlyStructuralFields) {
  auto dag = gemmTemplate();
  Json json = uniRunnerSerializeDagTemplate(dag);
  EXPECT_EQ(json.at("format_version"), 8);
  const auto &g = json.at("dag").at("nodes").at(0).at("gemm");
  EXPECT_EQ(g.size(), 11);
  EXPECT_FALSE(g.contains("worker_count"));
  EXPECT_FALSE(g.contains("completion_counter"));
  EXPECT_EQ(g.at("b").at("buffer"), "weight");
  uniRunnerDagTemplate restored;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &restored));
  uniRunnerCompiledDagTemplate compiled;
  ASSERT_EQ(compileUniRunnerDagTemplate(restored, &compiled), flagcxSuccess);
  EXPECT_TRUE(compiled.hasGemm);
  EXPECT_EQ(compiled.numRedNodes, 1);
  EXPECT_EQ(validateUniRunnerDagTemplateBindings(restored, bindings()), flagcxSuccess);
}

TEST(UniRunnerDsl, RejectsInvalidGemmShapeAndRuntimePayload) {
  auto json = uniRunnerSerializeDagTemplate(gemmTemplate());
  auto &g = json["dag"]["nodes"][0]["gemm"];
  g["m"] = -1;
  uniRunnerDagTemplate restored;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &restored));
  g["m"] = uint64_t{1} << 32;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &restored));
  g["m"] = 64;
  g["nthreads"] = 256;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &restored));
  g.erase("nthreads");
  g["lda"] = 31;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &restored));
  uniRunnerCompiledDagTemplate compiled;
  EXPECT_NE(compileUniRunnerDagTemplate(restored, &compiled), flagcxSuccess);
}

TEST(UniRunnerDsl, ChecksStridedMatrixBindingSpans) {
  auto dag = gemmTemplate();
  auto ranges = bindings();
  ranges.weightBytes -= 1;
  EXPECT_NE(validateUniRunnerDagTemplateBindings(dag, ranges), flagcxSuccess);
  ranges = bindings();
  dag.nodes[0].gemm.lda = 33;
  EXPECT_NE(validateUniRunnerDagTemplateBindings(dag, ranges), flagcxSuccess);
  ranges.inputBytes += 63 * sizeof(float);
  EXPECT_EQ(validateUniRunnerDagTemplateBindings(dag, ranges), flagcxSuccess);
}

TEST(UniRunnerDsl, FusedHashCoversShapeAndPipeline) {
  uniRunnerDagAlgorithmConfig config{};
  config.matrixM = 262144;
  config.matrixN = 4096;
  config.matrixK = 4096;
  config.numSlices = 4;
  config.numKSlices = 1;
  config.groupSize = 8;
  config.pipelineVersion = 2;
  const auto baseline = getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoAllGatherGemm, config);
  config.matrixK++;
  EXPECT_NE(baseline, getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoAllGatherGemm, config));
  config.matrixK--;
  config.numSlices++;
  EXPECT_NE(baseline, getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoAllGatherGemm, config));
  config.numSlices--;
  EXPECT_NE(baseline, getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGemmReduceScatter, config));
}

TEST(UniRunnerDsl, FusedKeyMatchesPublicApiBindingContract) {
  auto dag = gemmTemplate();
  uniRunnerDagAlgorithmConfig config{};
  config.matrixM = 128;
  config.matrixN = 64;
  config.matrixK = 32;
  config.numSlices = 2;
  config.numKSlices = 1;
  config.groupSize = 1;
  config.pipelineVersion = 2;
  auto ranges = bindings();
  ranges.outputBytes *= 2;
  ranges.scratchBytes = 128 * 32 * sizeof(float);
  EXPECT_EQ(validateUniRunnerFusedDagKey(dag.key, uniRunnerDagAlgoAllGatherGemm,
      config, ranges), flagcxSuccess);
  auto previousConfig = config;
  previousConfig.pipelineVersion = 1;
  auto previousKey = dag.key;
  previousKey.algoHash = getUniRunnerDagAlgorithmHash(
      uniRunnerDagAlgoAllGatherGemm, previousConfig);
  EXPECT_NE(validateUniRunnerFusedDagKey(previousKey, uniRunnerDagAlgoAllGatherGemm,
      previousConfig, ranges), flagcxSuccess);
  dag.key.redOp = flagcxSum;
  EXPECT_NE(validateUniRunnerFusedDagKey(dag.key, uniRunnerDagAlgoAllGatherGemm,
      config, ranges), flagcxSuccess);
  dag.key.redOp = flagcxRedNoOp;
  dag.key.count--;
  EXPECT_NE(validateUniRunnerFusedDagKey(dag.key, uniRunnerDagAlgoAllGatherGemm,
      config, ranges), flagcxSuccess);
}

TEST(UniRunnerDsl, LogicalRedWorkerPolicyPreservesSingleWorkerMode) {
  flagcxUniRunnerState state{};
  state.uniRunnerMaxBlocks = 32;
  state.uniRunnerRedSliceSize = 1024 * 1024;
  uint32_t workers = 0;
  ASSERT_EQ(resolveUniRunnerRedWorkerCount(&state, 16 * 1024 * 1024, &workers), flagcxSuccess);
  EXPECT_EQ(workers, 1);
  state.redSliceWorkersEnabled = true;
  ASSERT_EQ(resolveUniRunnerRedWorkerCount(&state, 16 * 1024 * 1024, &workers), flagcxSuccess);
  EXPECT_EQ(workers, 16);
  ASSERT_EQ(resolveUniRunnerRedWorkerCount(&state, 64 * 1024 * 1024, &workers), flagcxSuccess);
  EXPECT_EQ(workers, 32);
  state.uniRunnerNRedSlices = 4;
  ASSERT_EQ(resolveUniRunnerRedWorkerCount(&state, 16 * 1024 * 1024, &workers), flagcxSuccess);
  EXPECT_EQ(workers, 4);
}
