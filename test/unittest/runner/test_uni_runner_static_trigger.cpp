#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "uni_runner_impl.h"

namespace {

void *fakePtr(uintptr_t value) { return reinterpret_cast<void *>(value); }

void setValidRedPayload(uniRunnerDagNode *node, uintptr_t base) {
  node->nodeData.red.input1 = fakePtr(base + 0x100);
  node->nodeData.red.input2 = fakePtr(base + 0x200);
  node->nodeData.red.output = fakePtr(base + 0x300);
  node->nodeData.red.count = 17;
  node->nodeData.red.nthreads = 32;
  node->nodeData.red.datatype = flagcxFloat32;
  node->nodeData.red.redOp = flagcxSum;
}

void setValidIpcPayload(uniRunnerDagNode *node, size_t srcOffsetBytes,
                        size_t dstOffsetBytes, size_t bytes,
                        flagcxIpcBufferType srcBufferType, int peerLocalRank,
                        uint32_t readySlot, uint32_t parentFlagsOffset) {
  node->nodeType = uniRunnerDagNodeTypeIpc;
  node->nodeData.ipc.srcOffsetBytes = srcOffsetBytes;
  node->nodeData.ipc.dstOffsetBytes = dstOffsetBytes;
  node->nodeData.ipc.bytes = bytes;
  node->nodeData.ipc.srcBufferType = srcBufferType;
  node->nodeData.ipc.peerLocalRank = peerLocalRank;
  node->nodeData.ipc.readySlot = readySlot;
  node->nodeData.ipc.parentFlagsOffset = parentFlagsOffset;
  node->nodeData.ipc.triggerIdx = -1;
}

struct SingleIpcMaterialization {
  uniRunnerDagNode node = {};
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  void *nodeFlag = fakePtr(0x9000);
  uniRunnerStaticIpcTriggerConfig config = {};
  flagcxIpcTrigger trigger = {};
  size_t numTriggers = 99;
  size_t maxChunksPerTrigger = 99;

  SingleIpcMaterialization() {
    node.nodeIdx = 0;
    setValidIpcPayload(&node, 0, 16, 16, flagcxIpcBufferInput, 0, 0,
                       0);
    plan.topoOrder = &topoOrder;
    plan.numNodes = 1;
    plan.numIpcNodes = 1;
    config.chunkSize = 16;
    config.epoch = 7;
    config.dataBytes = 64;
    config.readySlots = 1;
    config.parentFlagsCount = 0;
    config.localRanks = 1;
  }

  flagcxResult_t populate(size_t triggerCapacity = 1) {
    return populateUniRunnerStaticIpcTriggers(
        &node, 1, &plan, &nodeFlag, 1, &config, &trigger, triggerCapacity,
        &numTriggers, &maxChunksPerTrigger);
  }
};

} // namespace

TEST(UniRunnerStaticTrigger, ChecksHeaderPlusPayloadAllocationSize) {
  size_t bytes = 0;
  const size_t headerBytes = flagcxFifoIdxData * sizeof(uint64_t);
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(
                129, sizeof(flagcxReduceTrigger), &bytes));
  EXPECT_EQ(headerBytes + 129 * sizeof(flagcxReduceTrigger), bytes);

  const size_t maxCapacity =
      static_cast<size_t>(std::numeric_limits<int>::max());
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(
                maxCapacity, sizeof(flagcxReduceTrigger), &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(
                maxCapacity + 1, sizeof(flagcxReduceTrigger), &bytes));
  const size_t maxElementSize =
      std::numeric_limits<size_t>::max() - headerBytes;
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(1, maxElementSize, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, maxElementSize + 1, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, 0, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, 1, nullptr));
}

TEST(UniRunnerStaticTrigger, MaterializesRedNodesInFilteredTopoOrder) {
  std::vector<uniRunnerDagNode> nodes(3);
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes[i] = {};
    nodes[i].nodeIdx = static_cast<int>(i);
  }
  int parentOfZero = 1;
  int parentOfOne = 2;
  nodes[0].nodeType = uniRunnerDagNodeTypeRed;
  nodes[0].numParents = 1;
  nodes[0].parents = &parentOfZero;
  nodes[0].nodeData.red.input1 = fakePtr(0x1000);
  nodes[0].nodeData.red.input2 = fakePtr(0x2000);
  nodes[0].nodeData.red.output = fakePtr(0x3000);
  nodes[0].nodeData.red.count = 17;
  nodes[0].nodeData.red.nthreads = 32;
  nodes[0].nodeData.red.datatype = flagcxFloat32;
  nodes[0].nodeData.red.redOp = flagcxSum;
  nodes[0].nodeData.red.triggerIdx = -1;

  nodes[1].nodeType = uniRunnerDagNodeTypeRed;
  nodes[1].numParents = 1;
  nodes[1].parents = &parentOfOne;
  nodes[1].nodeData.red.input1 = fakePtr(0x4000);
  nodes[1].nodeData.red.input2 = fakePtr(0x5000);
  nodes[1].nodeData.red.output = fakePtr(0x6000);
  nodes[1].nodeData.red.count = 19;
  nodes[1].nodeData.red.nthreads = 32;
  nodes[1].nodeData.red.datatype = flagcxFloat32;
  nodes[1].nodeData.red.redOp = flagcxMax;
  nodes[1].nodeData.red.triggerIdx = -1;

  nodes[2].nodeType = uniRunnerDagNodeTypeP2p;
  const int topoOrder[] = {2, 1, 0};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = const_cast<int *>(topoOrder);
  plan.numNodes = 3;
  plan.numHostNodes = 1;
  plan.numRedNodes = 2;
  void *flags[] = {fakePtr(0x7000), fakePtr(0x8000), fakePtr(0x9000)};
  flagcxReduceTrigger triggers[2] = {};
  size_t numTriggers = 0;

  ASSERT_EQ(flagcxSuccess, populateUniRunnerStaticRedTriggers(
                               nodes.data(), nodes.size(), &plan, flags, 3,
                               triggers, 2, &numTriggers));
  ASSERT_EQ(2u, numTriggers);
  EXPECT_EQ(0, nodes[1].nodeData.red.triggerIdx);
  EXPECT_EQ(1, nodes[0].nodeData.red.triggerIdx);
  EXPECT_EQ(reinterpret_cast<uint64_t>(nodes[1].nodeData.red.input1),
            triggers[0].value[0]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(nodes[0].nodeData.red.input1),
            triggers[1].value[0]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[2]), triggers[0].value[4]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[1]), triggers[1].value[4]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[1]), triggers[0].value[5]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[0]), triggers[1].value[5]);
  EXPECT_EQ(flagcxReduceTriggerEnqueued, triggers[0].pollState());
  EXPECT_EQ(flagcxReduceTriggerEnqueued, triggers[1].pollState());
}

TEST(UniRunnerStaticTrigger, DoesNotPublishPartialMaterialization) {
  uniRunnerDagNode node = {};
  node.nodeIdx = 0;
  node.nodeType = uniRunnerDagNodeTypeRed;
  node.nodeData.red.triggerIdx = -1;
  setValidRedPayload(&node, 0x1000);
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = &topoOrder;
  plan.numNodes = 1;
  plan.numRedNodes = 1;
  void *flag = fakePtr(0x1000);
  flagcxReduceTrigger trigger = {};
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          0, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  int parents[] = {0, 0};
  node.numParents = 2;
  node.parents = parents;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  node.numParents = 0;
  node.parents = nullptr;
  node.nodeData.red.count =
      flagcxTriggerMask(flagcxReduceTriggerBitsCount) + 1;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  node.nodeData.red.count = 17;
  node.nodeData.red.nthreads = 0;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.nthreads =
      flagcxTriggerMask(flagcxReduceTriggerBitsNThreads) + 1;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.nthreads = 32;
  node.nodeData.red.datatype = static_cast<flagcxDataType_t>(flagcxNumTypes);
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.datatype = flagcxFloat32;
  node.nodeData.red.redOp = flagcxNumRedOps;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);
}

TEST(UniRunnerStaticTrigger, ValidatesAllNodesBeforeWritingAnyTrigger) {
  uniRunnerDagNode nodes[2] = {};
  for (int i = 0; i < 2; ++i) {
    nodes[i].nodeIdx = i;
    nodes[i].nodeType = uniRunnerDagNodeTypeRed;
    nodes[i].nodeData.red.triggerIdx = -7 - i;
    setValidRedPayload(&nodes[i], 0x1000 * static_cast<uintptr_t>(i + 1));
  }
  nodes[1].nodeData.red.nthreads = 0;
  int topoOrder[] = {0, 1};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = topoOrder;
  plan.numNodes = 2;
  plan.numRedNodes = 2;
  void *flags[] = {fakePtr(0x7000), fakePtr(0x8000)};
  flagcxReduceTrigger triggers[2] = {};
  triggers[0].value[0] = 0xaaaa;
  triggers[1].value[0] = 0xbbbb;
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          nodes, 2, &plan, flags, 2, triggers,
                                          2, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-7, nodes[0].nodeData.red.triggerIdx);
  EXPECT_EQ(-8, nodes[1].nodeData.red.triggerIdx);
  EXPECT_EQ(0xaaaau, triggers[0].value[0]);
  EXPECT_EQ(0xbbbbu, triggers[1].value[0]);
}

TEST(UniRunnerStaticTrigger, AcceptsHostOnlyPlanWithoutTriggerStorage) {
  uniRunnerDagNode node = {};
  node.nodeIdx = 0;
  node.nodeType = uniRunnerDagNodeTypeP2p;
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = &topoOrder;
  plan.numNodes = 1;
  plan.numHostNodes = 1;
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxSuccess, populateUniRunnerStaticRedTriggers(
                               &node, 1, &plan, nullptr, 0, nullptr, 0,
                               &numTriggers));
  EXPECT_EQ(0u, numTriggers);
}

TEST(UniRunnerStaticIpcTrigger, NormalizesConfiguredChunkSizeBeforeCasting) {
  size_t chunkSize = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            normalizeUniRunnerIpcChunkSize(-16, &chunkSize));
  EXPECT_EQ(0u, chunkSize);

  chunkSize = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            normalizeUniRunnerIpcChunkSize(0, &chunkSize));
  EXPECT_EQ(0u, chunkSize);
  chunkSize = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            normalizeUniRunnerIpcChunkSize(15, &chunkSize));
  EXPECT_EQ(0u, chunkSize);
  chunkSize = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            normalizeUniRunnerIpcChunkSize(17, &chunkSize));
  EXPECT_EQ(0u, chunkSize);

  EXPECT_EQ(flagcxSuccess, normalizeUniRunnerIpcChunkSize(16, &chunkSize));
  EXPECT_EQ(16u, chunkSize);
  EXPECT_EQ(flagcxSuccess, normalizeUniRunnerIpcChunkSize(32, &chunkSize));
  EXPECT_EQ(32u, chunkSize);
  EXPECT_EQ(flagcxInvalidArgument,
            normalizeUniRunnerIpcChunkSize(16, nullptr));

  const int64_t largestAligned =
      std::numeric_limits<int64_t>::max() - 15;
  if (static_cast<uint64_t>(largestAligned) <=
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    EXPECT_EQ(flagcxSuccess,
              normalizeUniRunnerIpcChunkSize(largestAligned, &chunkSize));
    EXPECT_EQ(static_cast<size_t>(largestAligned), chunkSize);
  } else {
    EXPECT_EQ(flagcxInvalidArgument,
              normalizeUniRunnerIpcChunkSize(largestAligned, &chunkSize));
    EXPECT_EQ(0u, chunkSize);
  }
}

TEST(UniRunnerStaticIpcTrigger, ComputesCheckedChunkCountsAtBoundaries) {
  uint32_t numChunks = 99;
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerIpcChunkCount(0, 16, &numChunks));
  EXPECT_EQ(1u, numChunks);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerIpcChunkCount(1, 16, &numChunks));
  EXPECT_EQ(1u, numChunks);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerIpcChunkCount(16, 16, &numChunks));
  EXPECT_EQ(1u, numChunks);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerIpcChunkCount(17, 16, &numChunks));
  EXPECT_EQ(2u, numChunks);

  numChunks = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerIpcChunkCount(1, 0, &numChunks));
  EXPECT_EQ(0u, numChunks);
  numChunks = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerIpcChunkCount(1, 15, &numChunks));
  EXPECT_EQ(0u, numChunks);
  numChunks = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerIpcChunkCount(1, 17, &numChunks));
  EXPECT_EQ(0u, numChunks);
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerIpcChunkCount(1, 16, nullptr));

  const size_t maxChunks =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  if (maxChunks <= std::numeric_limits<size_t>::max() / 16) {
    const size_t maxBytes = maxChunks * 16;
    EXPECT_EQ(flagcxSuccess,
              checkedUniRunnerIpcChunkCount(maxBytes, 16, &numChunks));
    EXPECT_EQ(std::numeric_limits<uint32_t>::max(), numChunks);
    if (maxBytes != std::numeric_limits<size_t>::max()) {
      numChunks = 99;
      EXPECT_EQ(flagcxInvalidArgument,
                checkedUniRunnerIpcChunkCount(maxBytes + 1, 16,
                                               &numChunks));
      EXPECT_EQ(0u, numChunks);
    }
  }

  const size_t largestAligned = std::numeric_limits<size_t>::max() - 15;
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerIpcChunkCount(
                std::numeric_limits<size_t>::max(), largestAligned,
                &numChunks));
  EXPECT_EQ(2u, numChunks);
}

TEST(UniRunnerStaticIpcTrigger, ComparesAndAdvancesWrappingEpochs) {
  const uint64_t maxEpoch = std::numeric_limits<uint64_t>::max();
  const uint64_t halfRange = uint64_t{1} << 63;

  EXPECT_FALSE(flagcxIpcEpochReached(0, 1));
  EXPECT_FALSE(flagcxIpcEpochReached(1, 0));
  EXPECT_TRUE(flagcxIpcEpochReached(7, 7));
  EXPECT_TRUE(flagcxIpcEpochReached(8, 7));
  EXPECT_FALSE(flagcxIpcEpochReached(6, 7));
  EXPECT_TRUE(flagcxIpcEpochReached(1, maxEpoch));
  EXPECT_FALSE(flagcxIpcEpochReached(maxEpoch, 1));
  EXPECT_TRUE(flagcxIpcEpochReached(halfRange, 1));
  EXPECT_FALSE(flagcxIpcEpochReached(halfRange + 1, 1));

  EXPECT_EQ(1u, flagcxNextIpcEpoch(0));
  EXPECT_EQ(2u, flagcxNextIpcEpoch(1));
  EXPECT_EQ(maxEpoch, flagcxNextIpcEpoch(maxEpoch - 1));
  EXPECT_EQ(1u, flagcxNextIpcEpoch(maxEpoch));
}

TEST(UniRunnerStaticIpcTrigger, EncodesBoundedControlEpochsAndAbortState) {
  const uint64_t maxControlEpoch = flagcxIpcControlAbortBit - 1;

  EXPECT_FALSE(flagcxIpcControlEpochValid(0));
  EXPECT_TRUE(flagcxIpcControlEpochValid(1));
  EXPECT_TRUE(flagcxIpcControlEpochValid(maxControlEpoch));
  EXPECT_FALSE(flagcxIpcControlEpochValid(flagcxIpcControlAbortBit));
  EXPECT_FALSE(flagcxIpcControlEpochValid(
      flagcxIpcControlAbortBit | uint64_t{1}));

  EXPECT_TRUE(flagcxIpcControlEpochReached(7, 7));
  EXPECT_TRUE(flagcxIpcControlEpochReached(8, 7));
  EXPECT_TRUE(flagcxIpcControlEpochReached(
      flagcxIpcControlAbortBit | uint64_t{7}, 7));
  EXPECT_TRUE(flagcxIpcControlEpochReached(
      flagcxIpcControlAbortBit | uint64_t{8}, 7));
  EXPECT_FALSE(flagcxIpcControlEpochReached(6, 7));
  EXPECT_FALSE(flagcxIpcControlEpochReached(7, 0));
  EXPECT_FALSE(flagcxIpcControlEpochReached(1, maxControlEpoch));
}

TEST(UniRunnerStaticIpcTrigger,
     MaterializesMixedPlanInFilteredTopoOrderAndReportsMaxChunks) {
  uniRunnerDagNode nodes[4] = {};
  for (int i = 0; i < 4; ++i) {
    nodes[i].nodeIdx = i;
  }
  int nodeZeroParents[] = {1};
  int nodeThreeParents[] = {1, 2};
  nodes[0].numParents = 1;
  nodes[0].parents = nodeZeroParents;
  setValidIpcPayload(&nodes[0], 8, 16, 16, flagcxIpcBufferInput, 0, 0,
                     0);
  nodes[1].nodeType = uniRunnerDagNodeTypeP2p;
  nodes[2].nodeType = uniRunnerDagNodeTypeRed;
  nodes[3].numParents = 2;
  nodes[3].parents = nodeThreeParents;
  setValidIpcPayload(&nodes[3], 64, 80, 33, flagcxIpcBufferOutput, 1, 1,
                     1);

  int topoOrder[] = {1, 2, 3, 0};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = topoOrder;
  plan.numNodes = 4;
  plan.numHostNodes = 1;
  plan.numRedNodes = 1;
  plan.numIpcNodes = 2;
  void *flags[] = {fakePtr(0x1000), fakePtr(0x2000), fakePtr(0x3000),
                   fakePtr(0x4000)};
  uniRunnerStaticIpcTriggerConfig config = {};
  config.chunkSize = 16;
  config.epoch = 42;
  config.dataBytes = 128;
  config.readySlots = 2;
  config.parentFlagsCount = 3;
  config.localRanks = 2;
  flagcxIpcTrigger triggers[2] = {};
  size_t numTriggers = 99;
  size_t maxChunksPerTrigger = 99;

  ASSERT_EQ(flagcxSuccess, populateUniRunnerStaticIpcTriggers(
                               nodes, 4, &plan, flags, 4, &config, triggers,
                               2, &numTriggers, &maxChunksPerTrigger));
  ASSERT_EQ(2u, numTriggers);
  EXPECT_EQ(3u, maxChunksPerTrigger);
  EXPECT_EQ(1, nodes[0].nodeData.ipc.triggerIdx);
  EXPECT_EQ(0, nodes[3].nodeData.ipc.triggerIdx);

  EXPECT_EQ(64u, triggers[0].srcOffsetBytes);
  EXPECT_EQ(80u, triggers[0].dstOffsetBytes);
  EXPECT_EQ(33u, triggers[0].bytes);
  EXPECT_EQ(16u, triggers[0].chunkSize);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[3]), triggers[0].flagOut);
  EXPECT_EQ(42u, triggers[0].epoch);
  EXPECT_EQ(static_cast<uint32_t>(flagcxIpcBufferOutput),
            triggers[0].srcBufferType);
  EXPECT_EQ(1u, triggers[0].peerLocalRank);
  EXPECT_EQ(1u, triggers[0].readySlot);
  EXPECT_EQ(1u, triggers[0].parentFlagsOffset);
  EXPECT_EQ(2u, triggers[0].numParentFlags);
  EXPECT_EQ(3u, triggers[0].numChunks);
  EXPECT_EQ(0u, triggers[0].completedChunks);
  EXPECT_EQ(0u, triggers[0].nextChunk);
  EXPECT_EQ(static_cast<uint32_t>(flagcxReduceTriggerEnqueued),
            triggers[0].state);

  EXPECT_EQ(8u, triggers[1].srcOffsetBytes);
  EXPECT_EQ(16u, triggers[1].dstOffsetBytes);
  EXPECT_EQ(16u, triggers[1].bytes);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[0]), triggers[1].flagOut);
  EXPECT_EQ(static_cast<uint32_t>(flagcxIpcBufferInput),
            triggers[1].srcBufferType);
  EXPECT_EQ(0u, triggers[1].peerLocalRank);
  EXPECT_EQ(0u, triggers[1].readySlot);
  EXPECT_EQ(0u, triggers[1].parentFlagsOffset);
  EXPECT_EQ(1u, triggers[1].numParentFlags);
  EXPECT_EQ(1u, triggers[1].numChunks);
}

TEST(UniRunnerStaticIpcTrigger, ZeroByteTransferStillMaterializesOneChunk) {
  SingleIpcMaterialization materialization;
  materialization.node.nodeData.ipc.srcOffsetBytes = 0;
  materialization.node.nodeData.ipc.dstOffsetBytes = 0;
  materialization.node.nodeData.ipc.bytes = 0;
  materialization.config.dataBytes = 0;

  ASSERT_EQ(flagcxSuccess, materialization.populate());
  EXPECT_EQ(1u, materialization.numTriggers);
  EXPECT_EQ(1u, materialization.maxChunksPerTrigger);
  EXPECT_EQ(0u, materialization.trigger.bytes);
  EXPECT_EQ(1u, materialization.trigger.numChunks);
}

TEST(UniRunnerStaticIpcTrigger, MaterializesUint32MaxChunksExactly) {
  const size_t maxChunks =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  if (maxChunks > std::numeric_limits<size_t>::max() / 16) {
    GTEST_SKIP() << "size_t cannot represent the UINT32_MAX chunk boundary";
  }
  const size_t maxBytes = maxChunks * 16;
  SingleIpcMaterialization materialization;
  materialization.node.nodeData.ipc.srcOffsetBytes = 0;
  materialization.node.nodeData.ipc.dstOffsetBytes = 0;
  materialization.node.nodeData.ipc.bytes = maxBytes;
  materialization.config.dataBytes = maxBytes;

  ASSERT_EQ(flagcxSuccess, materialization.populate());
  EXPECT_EQ(std::numeric_limits<uint32_t>::max(),
            materialization.trigger.numChunks);
  EXPECT_EQ(maxChunks, materialization.maxChunksPerTrigger);
}

TEST(UniRunnerStaticIpcTrigger,
     AcceptsHostOnlyPlanWithoutConfigOrTriggerStorage) {
  uniRunnerDagNode node = {};
  node.nodeIdx = 0;
  node.nodeType = uniRunnerDagNodeTypeCpy;
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = &topoOrder;
  plan.numNodes = 1;
  plan.numHostNodes = 1;
  size_t numTriggers = 99;
  size_t maxChunksPerTrigger = 99;

  EXPECT_EQ(flagcxSuccess, populateUniRunnerStaticIpcTriggers(
                               &node, 1, &plan, nullptr, 0, nullptr, nullptr,
                               0, &numTriggers, &maxChunksPerTrigger));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(0u, maxChunksPerTrigger);
}

TEST(UniRunnerStaticIpcTrigger,
     InvalidSecondNodeDoesNotPublishAnyTriggerOrDagIndex) {
  uniRunnerDagNode nodes[2] = {};
  for (int i = 0; i < 2; ++i) {
    nodes[i].nodeIdx = i;
    setValidIpcPayload(&nodes[i], 0, 16, 16, flagcxIpcBufferInput, i, i,
                       0);
    nodes[i].nodeData.ipc.triggerIdx = -7 - i;
  }
  nodes[1].nodeData.ipc.srcOffsetBytes = 49;

  int topoOrder[] = {0, 1};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = topoOrder;
  plan.numNodes = 2;
  plan.numIpcNodes = 2;
  void *flags[] = {fakePtr(0x1000), fakePtr(0x2000)};
  uniRunnerStaticIpcTriggerConfig config = {};
  config.chunkSize = 16;
  config.epoch = 7;
  config.dataBytes = 64;
  config.readySlots = 2;
  config.localRanks = 2;
  flagcxIpcTrigger triggers[2] = {};
  triggers[0].srcOffsetBytes = 0xaaaa;
  triggers[0].state = flagcxReduceTriggerComplete;
  triggers[1].srcOffsetBytes = 0xbbbb;
  triggers[1].state = flagcxReduceTriggerInprogress;
  const flagcxIpcTrigger originalTriggers[] = {triggers[0], triggers[1]};
  size_t numTriggers = 99;
  size_t maxChunksPerTrigger = 99;

  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticIpcTriggers(
                                        nodes, 2, &plan, flags, 2, &config,
                                        triggers, 2, &numTriggers,
                                        &maxChunksPerTrigger));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(0u, maxChunksPerTrigger);
  EXPECT_EQ(-7, nodes[0].nodeData.ipc.triggerIdx);
  EXPECT_EQ(-8, nodes[1].nodeData.ipc.triggerIdx);
  EXPECT_EQ(0, std::memcmp(originalTriggers, triggers, sizeof(triggers)));
}

TEST(UniRunnerStaticIpcTrigger, ValidatesSourceAndDestinationExtents) {
  auto materialize = [](size_t srcOffset, size_t dstOffset, size_t bytes,
                        size_t dataBytes) {
    SingleIpcMaterialization materialization;
    materialization.node.nodeData.ipc.srcOffsetBytes = srcOffset;
    materialization.node.nodeData.ipc.dstOffsetBytes = dstOffset;
    materialization.node.nodeData.ipc.bytes = bytes;
    materialization.config.dataBytes = dataBytes;
    return materialization.populate();
  };

  EXPECT_EQ(flagcxSuccess, materialize(48, 48, 16, 64));
  EXPECT_EQ(flagcxInvalidArgument, materialize(49, 48, 16, 64));
  EXPECT_EQ(flagcxInvalidArgument, materialize(48, 49, 16, 64));
  EXPECT_EQ(flagcxInvalidArgument, materialize(65, 0, 0, 64));
  EXPECT_EQ(flagcxInvalidArgument, materialize(0, 65, 0, 64));

  const size_t maxExtent = std::numeric_limits<size_t>::max();
  EXPECT_EQ(flagcxSuccess,
            materialize(maxExtent, maxExtent, 0, maxExtent));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(maxExtent, maxExtent, 1, maxExtent));
}

TEST(UniRunnerStaticIpcTrigger, ValidatesBufferTypeAndPeerBoundaries) {
  auto materialize = [](flagcxIpcBufferType bufferType, int peerLocalRank,
                        int localRanks) {
    SingleIpcMaterialization materialization;
    materialization.node.nodeData.ipc.srcBufferType = bufferType;
    materialization.node.nodeData.ipc.peerLocalRank = peerLocalRank;
    materialization.config.localRanks = localRanks;
    return materialization.populate();
  };

  EXPECT_EQ(flagcxSuccess,
            materialize(flagcxIpcBufferInput, 0, 3));
  EXPECT_EQ(flagcxSuccess,
            materialize(flagcxIpcBufferOutput, 2, 3));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(static_cast<flagcxIpcBufferType>(-1), 0, 3));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(static_cast<flagcxIpcBufferType>(2), 0, 3));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(flagcxIpcBufferInput, -1, 3));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(flagcxIpcBufferInput, 3, 3));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(flagcxIpcBufferInput, 0, 0));
}

TEST(UniRunnerStaticIpcTrigger, ValidatesReadySlotsAndExactCapacity) {
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxSuccess, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.node.nodeData.ipc.readySlot = 1;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.config.readySlots = 0;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.config.readySlots = 2;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate(0));
    EXPECT_EQ(0u, materialization.numTriggers);
    EXPECT_EQ(0u, materialization.maxChunksPerTrigger);
  }
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate(2));
  }

  uniRunnerDagNode nodes[2] = {};
  for (int i = 0; i < 2; ++i) {
    nodes[i].nodeIdx = i;
    setValidIpcPayload(&nodes[i], 0, 16, 16, flagcxIpcBufferInput, 0, 0,
                       0);
  }
  int topoOrder[] = {0, 1};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = topoOrder;
  plan.numNodes = 2;
  plan.numIpcNodes = 2;
  void *flags[] = {fakePtr(0x1000), fakePtr(0x2000)};
  uniRunnerStaticIpcTriggerConfig config = {};
  config.chunkSize = 16;
  config.epoch = 1;
  config.dataBytes = 64;
  config.readySlots = 2;
  config.localRanks = 1;
  flagcxIpcTrigger triggers[2] = {};
  size_t numTriggers = 99;
  size_t maxChunksPerTrigger = 99;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticIpcTriggers(
                                        nodes, 2, &plan, flags, 2, &config,
                                        triggers, 2, &numTriggers,
                                        &maxChunksPerTrigger));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(0u, maxChunksPerTrigger);
}

TEST(UniRunnerStaticIpcTrigger, ValidatesParentFlagLayoutAndReferences) {
  auto materialize = [](int numParents, bool provideParents, int parentIdx,
                        uint32_t parentFlagsOffset, size_t parentFlagsCount,
                        bool provideParentFlag) {
    uniRunnerDagNode nodes[2] = {};
    nodes[0].nodeIdx = 0;
    nodes[0].nodeType = uniRunnerDagNodeTypeP2p;
    nodes[1].nodeIdx = 1;
    int parents[] = {parentIdx, parentIdx};
    nodes[1].numParents = numParents;
    nodes[1].parents = provideParents ? parents : nullptr;
    setValidIpcPayload(&nodes[1], 0, 16, 16, flagcxIpcBufferInput, 0, 0,
                       parentFlagsOffset);
    int topoOrder[] = {0, 1};
    uniRunnerDagExecutionPlan plan = {};
    plan.topoOrder = topoOrder;
    plan.numNodes = 2;
    plan.numHostNodes = 1;
    plan.numIpcNodes = 1;
    void *flags[] = {provideParentFlag ? fakePtr(0x1000) : nullptr,
                     fakePtr(0x2000)};
    uniRunnerStaticIpcTriggerConfig config = {};
    config.chunkSize = 16;
    config.epoch = 1;
    config.dataBytes = 64;
    config.readySlots = 1;
    config.parentFlagsCount = parentFlagsCount;
    config.localRanks = 1;
    flagcxIpcTrigger trigger = {};
    size_t numTriggers = 99;
    size_t maxChunksPerTrigger = 99;
    return populateUniRunnerStaticIpcTriggers(
        nodes, 2, &plan, flags, 2, &config, &trigger, 1, &numTriggers,
        &maxChunksPerTrigger);
  };

  EXPECT_EQ(flagcxSuccess, materialize(0, false, 0, 0, 0, false));
  EXPECT_EQ(flagcxSuccess, materialize(1, true, 0, 0, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(-1, false, 0, 0, 0, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(1, false, 0, 0, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(1, true, 0, 1, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(2, true, 0, 0, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(1, true, -1, 0, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(1, true, 2, 0, 1, true));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(1, true, 0, 0, 1, false));
  EXPECT_EQ(flagcxInvalidArgument,
            materialize(0, false, 0, 0, 1, true));

  if (static_cast<size_t>(std::numeric_limits<uint32_t>::max()) <
      std::numeric_limits<size_t>::max()) {
    EXPECT_EQ(flagcxInvalidArgument,
              materialize(
                  0, false, 0, 0,
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()) +
                      1,
                  true));
  }
}

TEST(UniRunnerStaticIpcTrigger, ValidatesConfigAndStorageBoundaries) {
  {
    SingleIpcMaterialization materialization;
    materialization.config.chunkSize = 15;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.config.chunkSize = 17;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.config.epoch = 0;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    materialization.nodeFlag = nullptr;
    EXPECT_EQ(flagcxInvalidArgument, materialization.populate());
  }
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxInvalidArgument,
              populateUniRunnerStaticIpcTriggers(
                  &materialization.node, 1, &materialization.plan,
                  &materialization.nodeFlag, 0, &materialization.config,
                  &materialization.trigger, 1, &materialization.numTriggers,
                  &materialization.maxChunksPerTrigger));
  }
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxInvalidArgument,
              populateUniRunnerStaticIpcTriggers(
                  &materialization.node, 1, &materialization.plan,
                  &materialization.nodeFlag, 1, &materialization.config,
                  nullptr, 1, &materialization.numTriggers,
                  &materialization.maxChunksPerTrigger));
  }
  {
    SingleIpcMaterialization materialization;
    EXPECT_EQ(flagcxInvalidArgument,
              populateUniRunnerStaticIpcTriggers(
                  &materialization.node, 1, &materialization.plan,
                  &materialization.nodeFlag, 1, &materialization.config,
                  &materialization.trigger, 1, nullptr,
                  &materialization.maxChunksPerTrigger));
    EXPECT_EQ(flagcxInvalidArgument,
              populateUniRunnerStaticIpcTriggers(
                  &materialization.node, 1, &materialization.plan,
                  &materialization.nodeFlag, 1, &materialization.config,
                  &materialization.trigger, 1, &materialization.numTriggers,
                  nullptr));
  }
}
