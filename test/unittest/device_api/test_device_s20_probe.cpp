// MUSA-only diagnostic: isolates the S20 (DevSignalStandalone) primitives on a
// real 2-node DevComm. Diagnostic only; the public S16-S25 cases are untouched.
//
//   FLAGCX_MUSA_S20_PROBE=7  ResetSignal only
//                        =8  ReadSignal only
//                        =9  INTER SignalInc + WaitSignal
//                        =10 Reset -> INTER barrier -> Signal -> Wait -> Read
//                        =11 K11-shaped control (context 0, BLOCK coop)
//
// One mode per MPI job: an illegal access cannot hide the other modes.
#include <cstdio>
#include <cstdlib>
#include "check.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
#include "tools.h"

extern "C" flagcxResult_t flagcxMusaS20ProbeRun(int mode, const void *opaque,
                                                int *out,
                                                flagcxStream_t stream);

#define S20_CHECK(call)                                                        \
  do {                                                                         \
    flagcxResult_t code = (call);                                              \
    if (code != flagcxSuccess) {                                               \
      std::fprintf(stderr, "S20 PROBE setup failed: %s -> %d at line %d\n",    \
                   #call, int(code), __LINE__);                                \
      std::fflush(stderr);                                                     \
      int initialized = 0;                                                     \
      MPI_Initialized(&initialized);                                           \
      if (initialized)                                                         \
        MPI_Abort(MPI_COMM_WORLD, int(code));                                  \
      return int(code);                                                        \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char *modeText = std::getenv("FLAGCX_MUSA_S20_PROBE");
  const int mode = modeText ? std::atoi(modeText) : 0;
  if (mode < 7 || mode > 11) {
    std::fprintf(stderr, "Set FLAGCX_MUSA_S20_PROBE=7..11; diagnostic only.\n");
    return 1;
  }

  flagcxDeviceHandle_t devHandle;
  S20_CHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxComm_t comm;
  flagcxUniqueId uniqueId;

  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu = 0;
  S20_CHECK(devHandle->getDeviceCount(&nGpu));
  if (totalProcs != 8 || nGpu < 8) {
    std::fprintf(stderr, "S20 probe requires 8 ranks and 8 visible GPUs\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  S20_CHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    S20_CHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);
  S20_CHECK(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));

  // Same DevComm requirements as the real unified IR inter suite.
  constexpr int kComboCount = 18;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = kComboCount;
  reqs.interCounterCount = kComboCount;

  flagcxDevComm_t devComm = nullptr;
  S20_CHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  flagcxStream_t stream;
  S20_CHECK(devHandle->streamCreate(&stream));

  void *devCommPtr = nullptr;
  S20_CHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));

  int *devResults = nullptr;
  S20_CHECK(devHandle->deviceMalloc((void **)&devResults, 256 * sizeof(int),
                                    flagcxMemDevice, NULL));
  S20_CHECK(devHandle->deviceMemset(devResults, 0, 256 * sizeof(int),
                                    flagcxMemDevice, stream));
  S20_CHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  if (proc == 0) {
    std::printf("S20 PROBE mode=%d ranks=%d\n", mode, totalProcs);
    std::fflush(stdout);
  }

  const flagcxResult_t run =
      flagcxMusaS20ProbeRun(mode, devCommPtr, devResults, stream);
  if (run != flagcxSuccess) {
    std::fprintf(stderr, "[rank %d] S20 PROBE mode=%d RUN FAILED -> %d\n", proc,
                 mode, int(run));
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, int(run));
    return int(run);
  }

  int results[64] = {};
  S20_CHECK(devHandle->deviceMemcpy(results, devResults, sizeof(results),
                                    flagcxMemcpyDeviceToHost, nullptr));
  int local = 1;
  for (int block = 0; block < 4; ++block) {
    if (results[block * 16] != 1)
      local = 0;
    if (proc == 0)
      std::printf("  block=%d marker=%d extra=%d\n", block, results[block * 16],
                  results[block * 16 + 1]);
  }
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (proc == 0) {
    std::printf("S20 PROBE mode=%d: %s\n", mode, global ? "OK" : "MISMATCH");
    std::fflush(stdout);
  }

  S20_CHECK(devHandle->deviceFree(devResults, flagcxMemDevice, nullptr));
  S20_CHECK(flagcxDevCommDestroy(comm, devComm));
  S20_CHECK(devHandle->streamDestroy(stream));
  S20_CHECK(flagcxCommDestroy(comm));
  S20_CHECK(flagcxDeviceHandleFree(devHandle));
  MPI_Finalize();
  return global ? 0 : 1;
}
