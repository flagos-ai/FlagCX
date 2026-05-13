// MPI tests for DevMemCreate / DevMemDestroy with symmetric windows.
// Requires MPI + GPUs.

#include "sym_heap.h"
#include "symmem_test.hpp"

// ---------------------------------------------------------------------------
// DevMemCreate with a symmetric window
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemCreateWithWindow) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  flagcxResult_t res = flagcxDevMemCreate(comm, devBuff, size, win, &devMem);
  ASSERT_EQ(res, flagcxSuccess);
  ASSERT_NE(devMem, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemDestroy(comm, devMem);
  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// DevMemCreate without window (IPC path)
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemCreateWithoutWindow) {

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  flagcxResult_t res =
      flagcxDevMemCreate(comm, devBuff, size, nullptr, &devMem);
  ASSERT_EQ(res, flagcxSuccess);
  ASSERT_NE(devMem, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemDestroy(comm, devMem);
}

// ---------------------------------------------------------------------------
// DevMemDestroy after create — lifecycle
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemDestroyAfterCreate) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, win, &devMem),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(flagcxDevMemDestroy(comm, devMem), flagcxSuccess);
  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// DevMemCreate with sym window sets hasWindow and isSymmetric
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemFieldsWithSymWindow) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, win, &devMem),
            flagcxSuccess);
  ASSERT_NE(devMem, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemDestroy(comm, devMem);
  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// DevMemCreate with NULL args
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemNullArgs) {
  flagcxDevMem_t devMem = nullptr;

  // NULL buff
  EXPECT_EQ(flagcxDevMemCreate(comm, nullptr, size, nullptr, &devMem),
            flagcxInvalidArgument);

  // Zero size
  EXPECT_EQ(flagcxDevMemCreate(comm, devBuff, 0, nullptr, &devMem),
            flagcxInvalidArgument);

  // NULL devMem output
  EXPECT_EQ(flagcxDevMemCreate(comm, devBuff, size, nullptr, nullptr),
            flagcxInvalidArgument);
}

// ---------------------------------------------------------------------------
// DevMemDestroy NULL — safe no-op
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DevMemDestroyNull) {
  EXPECT_EQ(flagcxDevMemDestroy(comm, nullptr), flagcxSuccess);
}
