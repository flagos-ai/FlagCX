#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "uni_runner_helper.h"

namespace {

uniRunnerDagCacheKey makeCacheKey(uint64_t algoHash) {
  uniRunnerDagCacheKey key{};
  key.algoHash = algoHash;
  key.commOp = flagcxCommOpAllReduce;
  key.count = 1048576;
  key.datatype = flagcxFloat32;
  key.redOp = flagcxSum;
  key.rank = 3;
  key.nranks = 8;
  key.root = -1;
  return key;
}

uniRunnerDagTemplate makeIpcDagTemplate() {
  uniRunnerDagTemplate dagTemplate;
  dagTemplate.key = makeCacheKey(0x123456789abcdef0ull);

  uniRunnerDagNodeDesc node;
  node.nodeType = uniRunnerDagNodeTypeIpc;
  node.nodeIdx = 0;
  node.ipc.srcOffsetBytes = 128;
  node.ipc.dstOffsetBytes = 256;
  node.ipc.bytes = 4096;
  node.ipc.srcBufferType = flagcxIpcBufferOutput;
  node.ipc.peerLocalRank = 7;
  node.ipc.readySlot = 3;
  dagTemplate.nodes.push_back(node);
  return dagTemplate;
}

uniRunnerDagTemplate makeCompiledDagTemplate() {
  uniRunnerDagTemplate dagTemplate;
  dagTemplate.key = makeCacheKey(0x8877665544332211ull);
  const std::array<uniRunnerDagNodeType, 4> types = {
      uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed,
      uniRunnerDagNodeTypeCpy, uniRunnerDagNodeTypeIpc};
  uniRunnerDagP2pOpDesc p2p;
  p2p.buffer.bufferType = uniRunnerDagBufferTypeInput;
  p2p.peerRank = 4;
  p2p.count = 4;
  p2p.datatype = flagcxFloat32;
  p2p.type = flagcxDevicePrimSend;
  uniRunnerDagBufferRef input{uniRunnerDagBufferTypeInput, 0};
  uniRunnerDagBufferRef output{uniRunnerDagBufferTypeOutput, 0};
  for (size_t nodeIdx = 0; nodeIdx < types.size(); ++nodeIdx) {
    uniRunnerDagNodeDesc node;
    node.nodeIdx = static_cast<int>(nodeIdx);
    node.nodeType = types[nodeIdx];
    if (nodeIdx != 0) {
      node.parents.push_back(static_cast<int>(nodeIdx - 1));
    }
    if (nodeIdx + 1 != types.size()) {
      node.children.push_back(static_cast<int>(nodeIdx + 1));
    }
    if (node.nodeType == uniRunnerDagNodeTypeP2p) {
      node.p2pOps.push_back(p2p);
    } else if (node.nodeType == uniRunnerDagNodeTypeRed) {
      node.red.input1 = input;
      node.red.input2 = output;
      node.red.output = output;
      node.red.count = 4;
      node.red.datatype = flagcxFloat32;
      node.red.redOp = flagcxSum;
    } else if (node.nodeType == uniRunnerDagNodeTypeCpy) {
      node.cpy.src = input;
      node.cpy.dst = output;
      node.cpy.count = 4;
      node.cpy.datatype = flagcxFloat32;
    } else if (node.nodeType == uniRunnerDagNodeTypeIpc) {
      node.ipc.peerLocalRank = 4;
      node.ipc.readySlot = 0;
      node.ipc.bytes = 1048577;
    }
    dagTemplate.nodes.push_back(node);
  }
  return dagTemplate;
}

} // namespace

TEST(UniRunnerDagCache, CompilesReusableStaticTemplate) {
  const uniRunnerDagTemplate dagTemplate = makeCompiledDagTemplate();
  uniRunnerCompiledDagTemplate compiled;
  ASSERT_EQ(flagcxSuccess,
            compileUniRunnerDagTemplate(dagTemplate, &compiled));
  EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), compiled.topoOrder);
  EXPECT_EQ(4u, compiled.numNodes);
  EXPECT_EQ(2u, compiled.numHostNodes);
  EXPECT_EQ(1u, compiled.numRedNodes);
  EXPECT_EQ(1u, compiled.numIpcNodes);
  EXPECT_EQ(1u, compiled.maxParallelRedNodes);
  EXPECT_EQ(4u, compiled.maxRedNodeCount);
  EXPECT_EQ(1048577u, compiled.maxIpcNodeBytes);
  EXPECT_EQ(dagTemplate.key.algoHash,
            compiled.dagTemplate.key.algoHash);

  // Compiled-only fields remain process local and cannot alter JSON v5.
  const Json encoded = uniRunnerSerializeDagTemplate(compiled.dagTemplate);
  EXPECT_FALSE(encoded.contains("topo_order"));
  EXPECT_FALSE(encoded.contains("compiled"));
}

TEST(UniRunnerDagCache, CompilesNoIpcRedShape) {
  uniRunnerDagTemplate dagTemplate = makeCompiledDagTemplate();
  dagTemplate.nodes.pop_back();
  dagTemplate.nodes.back().children.clear();

  uniRunnerCompiledDagTemplate compiled;
  ASSERT_EQ(flagcxSuccess,
            compileUniRunnerDagTemplate(dagTemplate, &compiled));
  EXPECT_EQ((std::vector<int>{0, 1, 2}), compiled.topoOrder);
  EXPECT_EQ(1u, compiled.numRedNodes);
  EXPECT_EQ(0u, compiled.numIpcNodes);
  EXPECT_EQ(1u, compiled.maxParallelRedNodes);
  EXPECT_EQ(4u, compiled.maxRedNodeCount);
  EXPECT_EQ(0u, compiled.maxIpcNodeBytes);
}

TEST(UniRunnerDagCache, CompiledTemplateRejectsInvalidStaticTopology) {
  uniRunnerDagTemplate malformed = makeCompiledDagTemplate();
  uniRunnerCompiledDagTemplate compiled;

  malformed.nodes[1].nodeIdx = 0;
  EXPECT_EQ(flagcxInternalError,
            compileUniRunnerDagTemplate(malformed, &compiled));
  EXPECT_TRUE(compiled.topoOrder.empty());

  malformed = makeCompiledDagTemplate();
  malformed.nodes[0].children.clear();
  EXPECT_EQ(flagcxInternalError,
            compileUniRunnerDagTemplate(malformed, &compiled));

  malformed = makeCompiledDagTemplate();
  malformed.nodes[0].parents.push_back(3);
  malformed.nodes[3].children.push_back(0);
  EXPECT_EQ(flagcxInvalidArgument,
            compileUniRunnerDagTemplate(malformed, &compiled));

  malformed = makeCompiledDagTemplate();
  malformed.nodes[1].red.redOp = flagcxRedNoOp;
  EXPECT_EQ(flagcxInvalidArgument,
            compileUniRunnerDagTemplate(malformed, &compiled));
  EXPECT_TRUE(compiled.topoOrder.empty());
  EXPECT_EQ(0u, compiled.numNodes);
}

TEST(UniRunnerDagCache, BindingRangeRequiresCompleteAccess) {
  std::array<unsigned char, 32> storage{};
  int64_t offsetBytes = -1;

  EXPECT_TRUE(uniRunnerDagBindingRangeContains(
      storage.data() + 8, storage.data(), 16, 8, &offsetBytes));
  EXPECT_EQ(8, offsetBytes);
  EXPECT_FALSE(uniRunnerDagBindingRangeContains(
      storage.data() + 8, storage.data(), 16, 9, &offsetBytes));

  // Adjacent input/output allocations may have exactly this address layout.
  // A non-empty output access must not be captured as one-past the input.
  EXPECT_FALSE(uniRunnerDagBindingRangeContains(
      storage.data() + 16, storage.data(), 16, 1, &offsetBytes));
  EXPECT_TRUE(uniRunnerDagBindingRangeContains(
      storage.data() + 16, storage.data() + 16, 16, 1, &offsetBytes));
  EXPECT_EQ(0, offsetBytes);

  // Preserve one-past-end representation for a genuinely empty slice.
  EXPECT_TRUE(uniRunnerDagBindingRangeContains(
      storage.data() + 16, storage.data(), 16, 0, &offsetBytes));
  EXPECT_EQ(16, offsetBytes);
}

TEST(UniRunnerDagCache, AlgorithmHashCoversAlgorithmTypeAndBuilderInputs) {
  uniRunnerDagAlgorithmConfig config{};
  config.numSlices = 2;
  config.numRedSlices = 8;
  config.groupSize = 4;
  config.topologyHash = 0x1020304050607080ull;
  config.bufferMode = uniRunnerDagBufferModeOutOfPlace;

  const uint64_t expected =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, config);
  // Golden FNV-1a value protects the persisted identity from accidental
  // dependence on process or standard-library hashing behavior.
  EXPECT_EQ(6300673621123503851ull, expected);
  EXPECT_EQ(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, config));

  uniRunnerDagAlgorithmConfig changed = config;
  changed.numSlices++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));
  changed = config;
  changed.numRedSlices++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));
  changed = config;
  changed.bufferMode = uniRunnerDagBufferModeInPlace;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));

  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoRingAR, config));
  // SlicedAR and RingRS hash the same builder fields, so this comparison
  // specifically verifies that the algorithm type itself is part of the hash.
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoRingRS, config));
}

TEST(UniRunnerDagCache, AlgorithmHashIgnoresUnrelatedBuilderSlots) {
  uniRunnerDagAlgorithmConfig sliced{};
  sliced.numSlices = 2;
  sliced.numRedSlices = 8;
  const uint64_t slicedHash =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, sliced);
  sliced.groupSize = 4;
  sliced.topologyHash = 0x1234;
  EXPECT_EQ(slicedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, sliced));

  uniRunnerDagAlgorithmConfig grouped{};
  grouped.groupSize = 4;
  const uint64_t groupedHash =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped);
  grouped.numSlices = 2;
  grouped.numRedSlices = 8;
  grouped.topologyHash = 0x1234;
  EXPECT_EQ(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));

  grouped = {};
  grouped.groupSize = 8;
  EXPECT_NE(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));
  grouped.groupSize = 4;
  grouped.bufferMode = uniRunnerDagBufferModeInPlace;
  EXPECT_NE(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));
}

TEST(UniRunnerDagCache, IpcAlgorithmHashIncludesLocalTopology) {
  uniRunnerDagAlgorithmConfig config{};
  config.numSlices = 2;
  config.numRedSlices = 8;
  config.topologyHash = 0x1234;
  const uint64_t expected =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoIpcAR, config);
  config.topologyHash++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoIpcAR, config));
}

TEST(UniRunnerDagCache, IpcTopologyHashIsStableOrderedAndValidated) {
  const int topology[] = {2, 0, 3, 1};
  uint64_t hash = 0;
  ASSERT_EQ(flagcxSuccess,
            getUniRunnerIpcTopologyHash(4, 4, topology, &hash));
  uint64_t repeated = 0;
  ASSERT_EQ(flagcxSuccess,
            getUniRunnerIpcTopologyHash(4, 4, topology, &repeated));
  EXPECT_EQ(hash, repeated);

  const int reordered[] = {0, 2, 3, 1};
  uint64_t reorderedHash = 0;
  ASSERT_EQ(flagcxSuccess,
            getUniRunnerIpcTopologyHash(4, 4, reordered, &reorderedHash));
  EXPECT_NE(hash, reorderedHash);

  const int duplicate[] = {0, 1, 1, 3};
  EXPECT_EQ(flagcxInvalidArgument,
            getUniRunnerIpcTopologyHash(4, 4, duplicate, &repeated));
  const int outOfRange[] = {0, 1, 2, 4};
  EXPECT_EQ(flagcxInvalidArgument,
            getUniRunnerIpcTopologyHash(4, 4, outOfRange, &repeated));
  EXPECT_EQ(flagcxNotSupported,
            getUniRunnerIpcTopologyHash(2, 4, topology, &repeated));
  EXPECT_EQ(flagcxInvalidArgument,
            getUniRunnerIpcTopologyHash(4, 4, nullptr, &repeated));
  EXPECT_EQ(flagcxInvalidArgument,
            getUniRunnerIpcTopologyHash(4, 4, topology, nullptr));
}

TEST(UniRunnerDagCache, PatternHashIncludesAlgorithmHash) {
  const uniRunnerDagCacheKey first = makeCacheKey(0x123456789abcdef0ull);
  uniRunnerDagCacheKey second = first;
  second.algoHash++;
  EXPECT_NE(getUniRunnerDagPatternHash(first),
            getUniRunnerDagPatternHash(second));
}

TEST(UniRunnerDagCache, JsonKeyContainsOnlyAlgorithmHashAndCollectiveIdentity) {
  const uniRunnerDagCacheKey key = makeCacheKey(0xfedcba9876543210ull);
  const Json json = uniRunnerDagCacheKeyToJson(key);
  const std::set<std::string> expectedFields = {
      "algo_hash", "comm_op", "count", "datatype",
      "red_op",    "rank",    "nranks", "root"};
  std::set<std::string> actualFields;
  for (auto it = json.begin(); it != json.end(); ++it) {
    actualFields.insert(it.key());
  }

  EXPECT_EQ(expectedFields, actualFields);
  EXPECT_FALSE(json.contains("algo_name"));
  EXPECT_TRUE(json.at("algo_hash").is_string());
  EXPECT_EQ("18364758544493064720",
            json.at("algo_hash").get<std::string>());
  EXPECT_FALSE(json.contains("format_version"));
  EXPECT_FALSE(json.contains("num_slices"));
  EXPECT_FALSE(json.contains("num_red_slices"));
  EXPECT_FALSE(json.contains("group_size"));
}

TEST(UniRunnerDagCache, JsonRoundTripPreservesFullWidthAlgorithmHash) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);

  const Json json = uniRunnerSerializeDagTemplate(original);
  ASSERT_EQ(kUniRunnerDagCacheFormatVersion,
            json.at("format_version").get<int>());

  uniRunnerDagTemplate decoded;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &decoded));
  EXPECT_EQ(original.key.algoHash, decoded.key.algoHash);
  EXPECT_EQ(original.key.commOp, decoded.key.commOp);
  EXPECT_EQ(original.key.count, decoded.key.count);
  EXPECT_EQ(original.key.datatype, decoded.key.datatype);
  EXPECT_EQ(original.key.redOp, decoded.key.redOp);
  EXPECT_EQ(original.key.rank, decoded.key.rank);
  EXPECT_EQ(original.key.nranks, decoded.key.nranks);
  EXPECT_EQ(original.key.root, decoded.key.root);
  EXPECT_EQ(getUniRunnerDagPatternHash(original.key), decoded.hashValue);
}

TEST(UniRunnerDagCache, IpcDescriptorJsonRoundTripContainsOnlyStructure) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json json = uniRunnerSerializeDagTemplate(original);
  ASSERT_EQ(kUniRunnerDagCacheFormatVersion,
            json.at("format_version").get<int>());
  ASSERT_EQ(1u, json.at("dag").at("nodes").size());

  const Json &nodeJson = json.at("dag").at("nodes").at(0);
  ASSERT_TRUE(nodeJson.contains("ipc"));
  const Json &ipcJson = nodeJson.at("ipc");
  const std::set<std::string> expectedFields = {
      "src_offset_bytes", "dst_offset_bytes", "bytes",
      "src_buffer_type",  "peer_local_rank", "ready_slot"};
  std::set<std::string> actualFields;
  for (auto it = ipcJson.begin(); it != ipcJson.end(); ++it) {
    actualFields.insert(it.key());
  }
  EXPECT_EQ(expectedFields, actualFields);
  EXPECT_EQ("output", ipcJson.at("src_buffer_type").get<std::string>());

  for (const char *runtimeField :
       {"parent_flags_offset", "trigger_idx", "chunk_size", "epoch",
        "num_chunks", "completed_chunks", "next_chunk", "state",
        "flag_in", "flag_out", "src_ptr", "dst_ptr", "dev_mem",
        "window"}) {
    EXPECT_FALSE(ipcJson.contains(runtimeField));
    EXPECT_FALSE(nodeJson.contains(runtimeField));
  }

  uniRunnerDagTemplate decoded;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &decoded));
  ASSERT_EQ(1u, decoded.nodes.size());
  const uniRunnerDagIpcOpDesc &ipc = decoded.nodes[0].ipc;
  EXPECT_EQ(original.nodes[0].ipc.srcOffsetBytes, ipc.srcOffsetBytes);
  EXPECT_EQ(original.nodes[0].ipc.dstOffsetBytes, ipc.dstOffsetBytes);
  EXPECT_EQ(original.nodes[0].ipc.bytes, ipc.bytes);
  EXPECT_EQ(original.nodes[0].ipc.srcBufferType, ipc.srcBufferType);
  EXPECT_EQ(original.nodes[0].ipc.peerLocalRank, ipc.peerLocalRank);
  EXPECT_EQ(original.nodes[0].ipc.readySlot, ipc.readySlot);
}

TEST(UniRunnerDagCache, RejectsMalformedIpcDescriptorFields) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json valid = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  for (const char *requiredField :
       {"src_offset_bytes", "dst_offset_bytes", "bytes",
        "src_buffer_type", "peer_local_rank", "ready_slot"}) {
    Json json = valid;
    json["dag"]["nodes"][0]["ipc"].erase(requiredField);
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded))
        << requiredField;
  }

  Json json = valid;
  json["dag"]["nodes"][0]["ipc"]["epoch"] = 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["trigger_idx"] = 0;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["src_buffer_type"] = "scratch";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["src_buffer_type"] = 0;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  for (const char *sizeField :
       {"src_offset_bytes", "dst_offset_bytes", "bytes"}) {
    json = valid;
    json["dag"]["nodes"][0]["ipc"][sizeField] = -1;
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)) << sizeField;

    json = valid;
    json["dag"]["nodes"][0]["ipc"][sizeField] = "1";
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)) << sizeField;
  }
}

TEST(UniRunnerDagCache, ValidatesIpcMaterializationBindings) {
  uniRunnerDagTemplate dagTemplate = makeIpcDagTemplate();
  dagTemplate.nodes[0].ipc.srcOffsetBytes = 128;
  dagTemplate.nodes[0].ipc.dstOffsetBytes = 256;
  dagTemplate.nodes[0].ipc.bytes = 512;
  dagTemplate.nodes[0].ipc.peerLocalRank = 1;
  dagTemplate.nodes[0].ipc.readySlot = 0;
  EXPECT_EQ(flagcxSuccess, validateUniRunnerIpcDagTemplateBindings(
                              dagTemplate, 1024, 1024, 2));

  uniRunnerDagTemplate malformed = dagTemplate;
  malformed.nodes[0].ipc.srcBufferType =
      static_cast<flagcxIpcBufferType>(99);
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(malformed, 1024, 1024,
                                                    2));

  malformed = dagTemplate;
  malformed.nodes[0].ipc.peerLocalRank = 2;
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(malformed, 1024, 1024,
                                                    2));

  malformed = dagTemplate;
  malformed.nodes[0].ipc.srcOffsetBytes = 800;
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(malformed, 1024, 1024,
                                                    2));
  malformed = dagTemplate;
  malformed.nodes[0].ipc.dstOffsetBytes = 800;
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(malformed, 1024, 1024,
                                                    2));

  // A zero-byte one-past-end range is valid and cannot underflow the bounds
  // checks used by materialization.
  malformed = dagTemplate;
  malformed.nodes[0].ipc.srcOffsetBytes = 1024;
  malformed.nodes[0].ipc.dstOffsetBytes = 1024;
  malformed.nodes[0].ipc.bytes = 0;
  EXPECT_EQ(flagcxSuccess, validateUniRunnerIpcDagTemplateBindings(
                              malformed, 1024, 1024, 2));
}

TEST(UniRunnerDagCache, RejectsNonDenseOrDuplicateIpcReadySlots) {
  uniRunnerDagTemplate dagTemplate = makeIpcDagTemplate();
  dagTemplate.nodes[0].ipc.srcOffsetBytes = 0;
  dagTemplate.nodes[0].ipc.dstOffsetBytes = 0;
  dagTemplate.nodes[0].ipc.bytes = 64;
  dagTemplate.nodes[0].ipc.peerLocalRank = 0;
  dagTemplate.nodes[0].ipc.readySlot = 0;
  uniRunnerDagNodeDesc second = dagTemplate.nodes[0];
  second.nodeIdx = 1;
  second.ipc.readySlot = 1;
  dagTemplate.nodes.push_back(second);
  ASSERT_EQ(flagcxSuccess, validateUniRunnerIpcDagTemplateBindings(
                              dagTemplate, 128, 128, 2));

  dagTemplate.nodes[1].ipc.readySlot = 0;
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(dagTemplate, 128, 128,
                                                    2));
  dagTemplate.nodes[1].ipc.readySlot = 2;
  EXPECT_EQ(flagcxInvalidArgument,
            validateUniRunnerIpcDagTemplateBindings(dagTemplate, 128, 128,
                                                    2));
}

TEST(UniRunnerDagCache, RejectsIpcPeerAndReadySlotOverflow) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json valid = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  Json json = valid;
  json["dag"]["nodes"][0]["ipc"]["peer_local_rank"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["peer_local_rank"] =
      static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["ready_slot"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["ready_slot"] =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["format_version"] = 3;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagCache, RejectsOldOrMalformedAlgorithmIdentity) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  Json json = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  json["format_version"] = kUniRunnerDagCacheFormatVersion - 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["algo_hash"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["algo_hash"] = "123garbage";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  for (const char *invalidHash : {"+1", " 1", "18446744073709551616"}) {
    json = uniRunnerSerializeDagTemplate(original);
    json["key"]["algo_hash"] = invalidHash;
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
  }

  json = uniRunnerSerializeDagTemplate(original);
  json["hash"] = "0";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagCache, MalformedDocumentsReturnFalseWithoutThrowing) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  uniRunnerDagTemplate decoded;

  Json json = uniRunnerSerializeDagTemplate(original);
  json.erase("key");
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json.erase("dag");
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json["format_version"] = "3";
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json["format_version"] = 18446744073709551615ull;
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));
}

TEST(UniRunnerDagCache, RejectsUnsafeDagArraySizes) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  Json json = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  json["dag"]["num_nodes"] = 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["num_nodes"] =
      static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["num_nodes"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["nodes"] = Json::object();
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["count"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagSizing, CheckedNodeCountRejectsIntOverflow) {
  int nodeCount = -1;
  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());

  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(0, maxInt, 0, &nodeCount));
  EXPECT_EQ(0, nodeCount);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(1, maxInt, 0, &nodeCount));
  EXPECT_EQ(std::numeric_limits<int>::max(), nodeCount);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(maxInt - 1, 1, 1, &nodeCount));
  EXPECT_EQ(std::numeric_limits<int>::max(), nodeCount);

  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(2, maxInt, 0, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, maxInt, 1, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, 0, maxInt + 1, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, 1, 0, nullptr));
}

TEST(UniRunnerDagSizing, CheckedTypeBytesRejectsSizeOverflow) {
  size_t bytes = 0;
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerTypeBytes(1024, 8, flagcxInt8, &bytes));
  EXPECT_EQ(8192u, bytes);
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(std::numeric_limits<size_t>::max(), 2,
                                      flagcxInt8, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(std::numeric_limits<size_t>::max(), 1,
                                      flagcxFloat32, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(1, 1, flagcxFloat32, nullptr));

  size_t decodedSize = 0;
  EXPECT_FALSE(uniRunnerDagSizeFromJson(Json(-1), &decodedSize));
  EXPECT_TRUE(uniRunnerDagSizeFromJson(Json(0), &decodedSize));
  EXPECT_EQ(0u, decodedSize);
}
