#pragma once

// Disable MPI C++ bindings (we only use C API)
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1

#include "mpi.h"
#include <gtest/gtest.h>

// Global test environment that handles MPI_Init / MPI_Finalize.
// Register with: ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    int argc = 0;
    char **argv = nullptr;
    int mpiError = MPI_Init(&argc, &argv);
    ASSERT_FALSE(mpiError);
  }

  void TearDown() override {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
};
