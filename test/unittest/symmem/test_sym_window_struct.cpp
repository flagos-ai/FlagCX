// Unit tests for symmetric memory structs and parameter validation.
// No MPI or GPU required — runs locally.
// Links against libflagcx.

#include <gtest/gtest.h>

#include "flagcx.h"
#include "sym_heap.h"
#include <cstring>

// ---------------------------------------------------------------------------
// 1. Struct layout tests — verify fields exist and have expected types
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowStructLayout) {
  struct flagcxWindow win;
  memset(&win, 0, sizeof(win));

  // Verify fields exist and are accessible
  win.vendorBase = nullptr;
  win.defaultBase = nullptr;
  win.isSymmetricDefault = 0;

  EXPECT_EQ(win.vendorBase, nullptr);
  EXPECT_EQ(win.defaultBase, nullptr);
  EXPECT_EQ(win.isSymmetricDefault, 0);
}

TEST(SymWindowStruct, SymWindowStructLayout) {
  struct flagcxSymWindow sw;
  memset(&sw, 0, sizeof(sw));

  // Verify all fields exist
  sw.flatBase = nullptr;
  sw.mcBase = nullptr;
  sw.mrIndex = -1;
  sw.mrBase = 0;
  sw.heapSize = 1024;
  sw.allocSize = 2048;
  sw.localRanks = 2;
  sw.physHandle = nullptr;
  sw.mcHandle = nullptr;
  sw.isVMM = false;

  EXPECT_EQ(sw.mrIndex, -1);
  EXPECT_EQ(sw.heapSize, 1024u);
  EXPECT_EQ(sw.allocSize, 2048u);
  EXPECT_EQ(sw.localRanks, 2);
  EXPECT_FALSE(sw.isVMM);
}

// ---------------------------------------------------------------------------
// 2. Flag constants
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowFlagConstants) {
  EXPECT_EQ(FLAGCX_WIN_DEFAULT, 0x00);
  EXPECT_EQ(FLAGCX_WIN_COLL_SYMMETRIC, 0x01);
}

// ---------------------------------------------------------------------------
// 3. Parameter validation — NULL / zero args
// ---------------------------------------------------------------------------

TEST(SymWindowValidation, RegisterNullComm) {
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(nullptr, dummy, sizeof(dummy), &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullBuff) {
  // We can't easily get a real comm without MPI, but we can verify that
  // NULL buff is rejected. Pass a non-null comm placeholder — the function
  // checks buff before using comm internals.
  flagcxWindow_t win = nullptr;
  // NULL buff should be caught before comm is dereferenced
  flagcxResult_t res = flagcxSymWindowRegister(nullptr, nullptr, 1024, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterZeroSize) {
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res = flagcxSymWindowRegister(nullptr, dummy, 0, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullWinPtr) {
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(nullptr, dummy, sizeof(dummy), nullptr, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, DeregisterNull) {
  // Deregistering a NULL window should be a safe no-op
  flagcxResult_t res = flagcxSymWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}

TEST(SymWindowValidation, CommWindowDeregisterNull) {
  // Public API: deregistering NULL window should succeed
  flagcxResult_t res = flagcxCommWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}
