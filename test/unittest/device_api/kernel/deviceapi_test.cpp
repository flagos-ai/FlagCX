#include "deviceapi_test.hpp"
#include <stdlib.h>

// Static member definitions
flagcxDeviceHandle_t DeviceApiTest::devHandle = nullptr;
flagcxComm_t DeviceApiTest::comm = nullptr;
flagcxStream_t DeviceApiTest::stream = nullptr;
void *DeviceApiTest::devBuff = nullptr;
size_t DeviceApiTest::size = 0;
flagcxMemAllocator_t DeviceApiTest::memAllocator = flagcxMemCCL;
flagcxDevComm_t DeviceApiTest::shmemLifetimeComm = nullptr;

void DeviceApiTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = DEVICEAPI_TEST_SIZE;

  ASSERT_EQ(flagcxDeviceHandleInit(&devHandle), flagcxSuccess);

  int numDevices;
  ASSERT_EQ(devHandle->getDeviceCount(&numDevices), flagcxSuccess);
  ASSERT_EQ(devHandle->setDevice(rank % numDevices), flagcxSuccess);

  flagcxUniqueId uniqueId;
  if (rank == 0) {
    ASSERT_EQ(flagcxGetUniqueId(&uniqueId), flagcxSuccess);
  }
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  ASSERT_EQ(flagcxCommInitRank(&comm, nranks, &uniqueId, rank), flagcxSuccess);

  ASSERT_EQ(devHandle->streamCreate(&stream), flagcxSuccess);
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  memAllocator = flagcxMemSHMEM;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &shmemLifetimeComm),
            flagcxSuccess);
#else
  memAllocator = flagcxMemCCL;
#endif
  ASSERT_EQ(flagcxMemAlloc(&devBuff, size, memAllocator), flagcxSuccess);
}

void DeviceApiTest::TearDownTestSuite() {
  if (devHandle == nullptr)
    return;

  devHandle->streamDestroy(stream);

  flagcxMemFree(devBuff, memAllocator);
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  if (shmemLifetimeComm)
    flagcxDevCommDestroy(comm, shmemLifetimeComm);
#endif
  if (comm)
    flagcxCommDestroy(comm);

  flagcxDeviceHandleFree(devHandle);

  devHandle = nullptr;
  comm = nullptr;
  stream = nullptr;
  devBuff = nullptr;
  shmemLifetimeComm = nullptr;
}

void DeviceApiTest::SetUp() {
  FlagCXTest::SetUp();
  if (comm == nullptr) {
    GTEST_SKIP() << "SetUpTestSuite failed";
  }
}
