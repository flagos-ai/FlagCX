// Unit tests for flagcxIntruQueue — intrusive singly-linked queue.
// Source: flagcx/service/include/utils.h (inline implementation)
// Links against libflagcx.

#include <gtest/gtest.h>

#include "utils.h"

struct QueueNode {
  int val;
  QueueNode *next;
};

using TestQueue = flagcxIntruQueue<QueueNode, &QueueNode::next>;

class IntruQueueTest : public ::testing::Test {
protected:
  void SetUp() override { flagcxIntruQueueConstruct(&queue); }

  TestQueue queue;
  // Statically allocated nodes for testing (no dynamic allocation needed)
  QueueNode nodes[10];
};

TEST_F(IntruQueueTest, ConstructEmpty) {
  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
  EXPECT_EQ(flagcxIntruQueueHead(&queue), nullptr);
}

TEST_F(IntruQueueTest, EnqueueDequeue) {
  nodes[0].val = 1;
  nodes[1].val = 2;
  nodes[2].val = 3;

  flagcxIntruQueueEnqueue(&queue, &nodes[0]);
  flagcxIntruQueueEnqueue(&queue, &nodes[1]);
  flagcxIntruQueueEnqueue(&queue, &nodes[2]);

  EXPECT_FALSE(flagcxIntruQueueEmpty(&queue));

  // Dequeue in FIFO order
  QueueNode *n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 1);
  n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 2);
  n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 3);

  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
}

TEST_F(IntruQueueTest, TryDequeueEmpty) {
  QueueNode *n = flagcxIntruQueueTryDequeue(&queue);
  EXPECT_EQ(n, nullptr);
}

TEST_F(IntruQueueTest, DeleteMiddle) {
  nodes[0].val = 10;
  nodes[1].val = 20;
  nodes[2].val = 30;

  flagcxIntruQueueEnqueue(&queue, &nodes[0]);
  flagcxIntruQueueEnqueue(&queue, &nodes[1]);
  flagcxIntruQueueEnqueue(&queue, &nodes[2]);

  // Delete the middle node (val=20)
  bool found = flagcxIntruQueueDelete(&queue, &nodes[1]);
  EXPECT_TRUE(found);

  // Should have 2 remaining: 10, 30
  QueueNode *n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 10);
  n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 30);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
}

TEST_F(IntruQueueTest, DeleteHead) {
  nodes[0].val = 10;
  nodes[1].val = 20;
  nodes[2].val = 30;

  flagcxIntruQueueEnqueue(&queue, &nodes[0]);
  flagcxIntruQueueEnqueue(&queue, &nodes[1]);
  flagcxIntruQueueEnqueue(&queue, &nodes[2]);

  // Delete head (val=10)
  bool found = flagcxIntruQueueDelete(&queue, &nodes[0]);
  EXPECT_TRUE(found);

  // Remaining: 20, 30
  QueueNode *n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 20);
  n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 30);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
}

TEST_F(IntruQueueTest, DeleteTail) {
  nodes[0].val = 10;
  nodes[1].val = 20;
  nodes[2].val = 30;

  flagcxIntruQueueEnqueue(&queue, &nodes[0]);
  flagcxIntruQueueEnqueue(&queue, &nodes[1]);
  flagcxIntruQueueEnqueue(&queue, &nodes[2]);

  // Delete tail (val=30)
  bool found = flagcxIntruQueueDelete(&queue, &nodes[2]);
  EXPECT_TRUE(found);

  // Remaining: 10, 20
  QueueNode *n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 10);
  n = flagcxIntruQueueDequeue(&queue);
  EXPECT_EQ(n->val, 20);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
}
