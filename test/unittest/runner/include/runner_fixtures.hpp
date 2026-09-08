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

  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxStream_t stream = nullptr;
  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hostsendbuff = nullptr;
  void *hostrecvbuff = nullptr;
  void *sendRegHandle = nullptr;
  void *recvRegHandle = nullptr;
  bool useRegisteredBuffers = false;
  size_t size;
  size_t count;
};
