// MPI tests for symmetric window grow.
// Requires MPI + GPUs.

#include "sym_heap.h"
#include "symmem_test.hpp"

// ---------------------------------------------------------------------------
// Basic grow: increase heap size
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, GrowBasic) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  // Register with half the buffer
  size_t halfSize = size / 2;
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, halfSize, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  if (!win->defaultBase->isVMM) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, grow not supported on IPC fallback";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Grow to full size
  flagcxResult_t res = flagcxSymWindowGrow(comm, win, devBuff, size);
  EXPECT_EQ(res, flagcxSuccess);
  EXPECT_EQ(win->defaultBase->heapSize, size);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Grow beyond max — should fail
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, GrowBeyondMax) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  if (!win->defaultBase->isVMM) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, grow not supported on IPC fallback";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Try to grow beyond maxHeapSize
  size_t tooLarge = win->defaultBase->maxHeapSize + (1ULL << 20);
  flagcxResult_t res = flagcxSymWindowGrow(comm, win, devBuff, tooLarge);
  EXPECT_NE(res, flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Grow to same size — no-op
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, GrowSameSize) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  if (!win->defaultBase->isVMM) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, grow not supported on IPC fallback";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Grow to same size — should be a no-op
  flagcxResult_t res = flagcxSymWindowGrow(comm, win, devBuff, size);
  EXPECT_EQ(res, flagcxSuccess);
  EXPECT_EQ(win->defaultBase->heapSize, size);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Grow tracks physical handles
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, GrowTracksPhysHandles) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  size_t halfSize = size / 2;
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, halfSize, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  if (!win->defaultBase->isVMM) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, grow not supported on IPC fallback";
  }

  int countBefore = win->defaultBase->growthCount;

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t res = flagcxSymWindowGrow(comm, win, devBuff, size);
  EXPECT_EQ(res, flagcxSuccess);
  EXPECT_GT(win->defaultBase->growthCount, countBefore);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}
