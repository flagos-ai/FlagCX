#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "uni_runner_impl.h"

namespace {

class TestDag {
public:
  explicit TestDag(const std::vector<uniRunnerDagNodeType> &types)
      : nodes(types.size()), parents(types.size()), children(types.size()) {
    for (size_t i = 0; i < types.size(); ++i) {
      nodes[i] = {};
      nodes[i].nodeType = types[i];
    }
  }

  void addEdge(int parent, int child) {
    children[parent].push_back(child);
    parents[child].push_back(parent);
  }

  void bind() {
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].nodeIdx = static_cast<int>(i);
      nodes[i].numParents = static_cast<int>(parents[i].size());
      nodes[i].parents = parents[i].empty() ? nullptr : parents[i].data();
      nodes[i].pendingParents = nodes[i].numParents;
      nodes[i].numChildren = static_cast<int>(children[i].size());
      nodes[i].children =
          children[i].empty() ? nullptr : children[i].data();
    }
  }

  void setRedCount(int nodeIdx, size_t count) {
    nodes[nodeIdx].nodeData.red.count = count;
  }

  void setIpcBytes(int nodeIdx, size_t bytes) {
    nodes[nodeIdx].nodeData.ipc.bytes = bytes;
  }

  std::vector<uniRunnerDagNode> nodes;
  std::vector<std::vector<int>> parents;
  std::vector<std::vector<int>> children;
};

class PlanHolder {
public:
  ~PlanHolder() { destroyUniRunnerDagExecutionPlan(&plan); }
  uniRunnerDagExecutionPlan plan = {};
};

std::vector<int> orderOf(const uniRunnerDagExecutionPlan &plan) {
  if (plan.numNodes == 0) {
    return {};
  }
  return std::vector<int>(plan.topoOrder,
                          plan.topoOrder + plan.numNodes);
}

} // namespace

TEST(UniRunnerDagPlan, DrainsRootPhasesWithStableOrderWithinEachPhase) {
  TestDag dag({uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc,
               uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeCpy});
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{2, 3, 0, 1}), orderOf(holder.plan));
  EXPECT_EQ(2u, holder.plan.numHostNodes);
  EXPECT_EQ(1u, holder.plan.numRedNodes);
  EXPECT_EQ(1u, holder.plan.numIpcNodes);
  EXPECT_EQ(1u, holder.plan.maxParallelRedNodes);
  EXPECT_EQ(0u, holder.plan.maxRedNodeCount);
  EXPECT_EQ(0u, holder.plan.maxIpcNodeBytes);
}

TEST(UniRunnerDagPlan, ContinuesDrainingNodesUnlockedInSamePhase) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed,
               uniRunnerDagNodeTypeIpc, uniRunnerDagNodeTypeCpy,
               uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc});
  dag.addEdge(0, 3);
  dag.addEdge(1, 4);
  dag.addEdge(2, 5);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0, 3, 1, 4, 2, 5}), orderOf(holder.plan));
}

TEST(UniRunnerDagPlan, DefersNodesUnlockedInAnEarlierPhaseToNextRound) {
  TestDag dag({uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc,
               uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeP2p,
               uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc});
  dag.addEdge(0, 3);
  dag.addEdge(1, 4);
  dag.addEdge(2, 5);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{2, 0, 1, 5, 3, 4}), orderOf(holder.plan));
}

TEST(UniRunnerDagPlan, OrdersForkJoinOnlyAfterAllParents) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeP2p,
               uniRunnerDagNodeTypeCpy, uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 2);
  dag.addEdge(1, 2);
  dag.addEdge(2, 3);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), orderOf(holder.plan));
}

TEST(UniRunnerDagPlan, ComputesMaximumRedAntichainAndCount) {
  TestDag dag({uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeRed,
               uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed,
               uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 2);
  dag.addEdge(2, 3);
  dag.addEdge(1, 4);
  dag.setRedCount(0, 17);
  dag.setRedCount(1, 64);
  dag.setRedCount(3, 300);
  dag.setRedCount(4, 32);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ(2u, holder.plan.maxParallelRedNodes);
  EXPECT_EQ(300u, holder.plan.maxRedNodeCount);
}

TEST(UniRunnerDagPlan, ComputesMixedRedAndIpcShapeAcrossIpcEdges) {
  TestDag dag({uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc,
               uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeIpc});
  dag.addEdge(0, 1);
  dag.addEdge(1, 2);
  dag.setRedCount(0, 65);
  dag.setRedCount(2, 257);
  dag.setIpcBytes(1, 1048577);
  dag.setIpcBytes(3, 4096);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ(1u, holder.plan.maxParallelRedNodes);
  EXPECT_EQ(257u, holder.plan.maxRedNodeCount);
  EXPECT_EQ(1048577u, holder.plan.maxIpcNodeBytes);
}

TEST(UniRunnerDagPlan, HostOnlyDagHasNoRedShape) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeCpy});
  dag.addEdge(0, 1);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ(0u, holder.plan.maxParallelRedNodes);
  EXPECT_EQ(0u, holder.plan.maxRedNodeCount);
}

TEST(UniRunnerDagPlan, SupportsHighIndexParentsOfLowIndexChildren) {
  TestDag dag({uniRunnerDagNodeTypeRed, uniRunnerDagNodeTypeCpy,
               uniRunnerDagNodeTypeP2p});
  dag.addEdge(2, 0);
  dag.addEdge(0, 1);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{2, 0, 1}), orderOf(holder.plan));
}

TEST(UniRunnerDagPlan, PreservesChildListOrderForSamePhaseReadiness) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed,
               uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 2);
  dag.addEdge(0, 1);
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0, 2, 1}), orderOf(holder.plan));
}

TEST(UniRunnerDagPlan, DoesNotMutateRuntimeDagState) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 1);
  dag.bind();
  dag.nodes[0].pendingParents = 17;
  dag.nodes[1].pendingParents = 23;
  dag.nodes[0].next = &dag.nodes[1];
  dag.nodes[1].next = &dag.nodes[0];
  const std::vector<uniRunnerDagNode> before = dag.nodes;
  const std::vector<std::vector<int>> parentsBefore = dag.parents;
  const std::vector<std::vector<int>> childrenBefore = dag.children;
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    EXPECT_EQ(before[i].nodeIdx, dag.nodes[i].nodeIdx);
    EXPECT_EQ(before[i].nodeType, dag.nodes[i].nodeType);
    EXPECT_EQ(before[i].pendingParents, dag.nodes[i].pendingParents);
    EXPECT_EQ(before[i].next, dag.nodes[i].next);
    EXPECT_EQ(before[i].parents, dag.nodes[i].parents);
    EXPECT_EQ(before[i].children, dag.nodes[i].children);
  }
  EXPECT_EQ(parentsBefore, dag.parents);
  EXPECT_EQ(childrenBefore, dag.children);
}

TEST(UniRunnerDagPlan, RejectsWholeAndPartialCycles) {
  TestDag cycle({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  cycle.addEdge(0, 1);
  cycle.addEdge(1, 0);
  cycle.bind();
  PlanHolder holder;
  EXPECT_EQ(flagcxInvalidArgument, compileUniRunnerDagExecutionPlan(
                                          cycle.nodes.data(),
                                          cycle.nodes.size(), &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);

  TestDag partial({uniRunnerDagNodeTypeCpy, uniRunnerDagNodeTypeP2p,
                   uniRunnerDagNodeTypeRed});
  partial.addEdge(1, 2);
  partial.addEdge(2, 1);
  partial.bind();
  EXPECT_EQ(flagcxInvalidArgument, compileUniRunnerDagExecutionPlan(
                                          partial.nodes.data(),
                                          partial.nodes.size(), &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);
}

TEST(UniRunnerDagPlan, RejectsMalformedNodeMetadata) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 1);
  dag.bind();
  PlanHolder holder;

  dag.nodes[1].nodeIdx = 0;
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  dag.nodes[1].nodeIdx = 1;

  dag.children[0][0] = 2;
  dag.bind();
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  dag.children[0][0] = 1;
  dag.bind();

  dag.nodes[0].numChildren = 1;
  dag.nodes[0].children = nullptr;
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  dag.bind();

  dag.nodes[0].nodeType = static_cast<uniRunnerDagNodeType>(99);
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  dag.nodes[0].nodeType = uniRunnerDagNodeTypeP2p;

  dag.nodes[1].numParents = -1;
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  dag.bind();

  dag.parents[1][0] = -1;
  dag.bind();
  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
}

TEST(UniRunnerDagPlan, RejectsChildListThatOverDecrementsIndegree) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  dag.addEdge(0, 1);
  dag.children[0].push_back(1);
  dag.bind();
  PlanHolder holder;

  EXPECT_EQ(flagcxInternalError, compileUniRunnerDagExecutionPlan(
                                       dag.nodes.data(), dag.nodes.size(),
                                       &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
}

TEST(UniRunnerDagPlan, RecompileAndDestroyLeaveNoStalePlan) {
  TestDag first({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  first.addEdge(0, 1);
  first.bind();
  TestDag second({uniRunnerDagNodeTypeIpc});
  second.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               first.nodes.data(), first.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0, 1}), orderOf(holder.plan));
  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               second.nodes.data(), second.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0}), orderOf(holder.plan));
  EXPECT_EQ(0u, holder.plan.numHostNodes);
  EXPECT_EQ(0u, holder.plan.numRedNodes);
  EXPECT_EQ(1u, holder.plan.numIpcNodes);

  destroyUniRunnerDagExecutionPlan(&holder.plan);
  destroyUniRunnerDagExecutionPlan(&holder.plan);
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);
}

TEST(UniRunnerDagPlan, FailedOrEmptyRecompileClearsPreviousPlan) {
  TestDag valid({uniRunnerDagNodeTypeP2p});
  valid.bind();
  TestDag cycle({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeRed});
  cycle.addEdge(0, 1);
  cycle.addEdge(1, 0);
  cycle.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               valid.nodes.data(), valid.nodes.size(),
                               &holder.plan));
  ASSERT_NE(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(flagcxInvalidArgument, compileUniRunnerDagExecutionPlan(
                                          cycle.nodes.data(),
                                          cycle.nodes.size(), &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);
  EXPECT_EQ(0u, holder.plan.numHostNodes);
  EXPECT_EQ(0u, holder.plan.numRedNodes);
  EXPECT_EQ(0u, holder.plan.numIpcNodes);

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               valid.nodes.data(), valid.nodes.size(),
                               &holder.plan));
  ASSERT_NE(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(flagcxSuccess,
            compileUniRunnerDagExecutionPlan(nullptr, 0, &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);
}

TEST(UniRunnerDagPlan, HandlesEmptyAndRejectsOversizedInputBeforeDereference) {
  PlanHolder holder;
  EXPECT_EQ(flagcxSuccess,
            compileUniRunnerDagExecutionPlan(nullptr, 0, &holder.plan));
  EXPECT_EQ(nullptr, holder.plan.topoOrder);
  EXPECT_EQ(0u, holder.plan.numNodes);

  uniRunnerDagNode dummy = {};
  const size_t tooMany =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_EQ(flagcxInvalidArgument, compileUniRunnerDagExecutionPlan(
                                          &dummy, tooMany, &holder.plan));
  EXPECT_EQ(flagcxInvalidArgument,
            compileUniRunnerDagExecutionPlan(nullptr, 1, &holder.plan));
  EXPECT_EQ(flagcxInvalidArgument,
            compileUniRunnerDagExecutionPlan(&dummy, 1, nullptr));

  destroyUniRunnerDagExecutionPlan(nullptr);
}

TEST(UniRunnerDagPlan, UsesArrayOrderForRingAllGatherLikeDualRoots) {
  TestDag dag({uniRunnerDagNodeTypeP2p, uniRunnerDagNodeTypeCpy});
  dag.bind();
  PlanHolder holder;

  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ((std::vector<int>{0, 1}), orderOf(holder.plan));

  const std::vector<int> firstOrder = orderOf(holder.plan);
  ASSERT_EQ(flagcxSuccess, compileUniRunnerDagExecutionPlan(
                               dag.nodes.data(), dag.nodes.size(),
                               &holder.plan));
  EXPECT_EQ(firstOrder, orderOf(holder.plan));
}
