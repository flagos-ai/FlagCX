// MPI tests for symmetric window register / deregister.
// Requires MPI + GPUs.

#include "sym_heap.h"
#include "symmem_test.hpp"
#include <iostream>

// ---------------------------------------------------------------------------
// Register with FLAGCX_WIN_COLL_SYMMETRIC
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RegisterSymmetricWindow) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_EQ(res, flagcxSuccess);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetricDefault);
  EXPECT_NE(win->defaultBase, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  res = flagcxCommWindowDeregister(comm, win);
  EXPECT_EQ(res, flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Register with FLAGCX_WIN_DEFAULT (non-symmetric)
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RegisterDefaultFlag) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  flagcxResult_t res =
      flagcxCommWindowRegister(comm, devBuff, size, &win, FLAGCX_WIN_DEFAULT);
  // On non-homo path with default flag, win may be NULL (no sym heap created)
  EXPECT_EQ(res, flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  if (win != nullptr) {
    flagcxCommWindowDeregister(comm, win);
  }
}

// ---------------------------------------------------------------------------
// Deregister after register — basic lifecycle
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DeregisterAfterRegister) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Deregister NULL — should be safe
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DeregisterNullWindow) {
  flagcxComm_t &comm = handler->comm;
  EXPECT_EQ(flagcxCommWindowDeregister(comm, nullptr), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Verify sym window fields after registration
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, SymWindowFields) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  EXPECT_EQ(d->heapSize, size);
  EXPECT_GE(d->maxHeapSize, size);
  EXPECT_GT(d->localRanks, 0);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Register multiple windows on different buffers
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RegisterMultipleWindows) {
  flagcxComm_t &comm = handler->comm;
  flagcxWindow_t win1 = nullptr;
  flagcxWindow_t win2 = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win1,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff2, size, &win2,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);

  ASSERT_NE(win1, nullptr);
  ASSERT_NE(win2, nullptr);
  EXPECT_NE(win1, win2);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win1);
  flagcxCommWindowDeregister(comm, win2);
}
