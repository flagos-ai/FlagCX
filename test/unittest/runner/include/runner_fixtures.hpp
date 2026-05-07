#pragma once

// Disable MPI C++ bindings
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1

#include "flagcx.h"
#include "mpi.h"
#include <gtest/gtest.h>
#include <string>

class FlagCXTest : public ::testing::Test {
protected:
  void SetUp() override;
  void TearDown() override {}

  int rank;
  int nranks;
};

class FlagCXCollTest : public FlagCXTest {
protected:
  void SetUp() override;
  void TearDown() override;

  flagcxHandlerGroup_t handler;
  flagcxStream_t stream;
  void *sendbuff;
  void *recvbuff;
  void *hostsendbuff;
  void *hostrecvbuff;
  size_t size;
  size_t count;
};
