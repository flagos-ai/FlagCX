// MPI test runner entry point for kernel + device API tests.
// Provides main(), FlagCXTest::SetUp().

#include "deviceapi_test.hpp"
#include "flagcx_kernel_test.hpp"

void FlagCXTest::SetUp() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
