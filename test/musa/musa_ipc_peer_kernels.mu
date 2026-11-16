/* MUSA device kernels for the standalone cross-process IPC peer test.
 *
 * This translation unit is compiled by mcc exactly like the working
 * test/kernel/musa/*.mu kernels:
 *
 *   mcc -std=c++17 --offload-arch=mp_31 -fPIC -c musa_ipc_peer_kernels.mu
 *
 * It deliberately contains NO host test logic. The host side talks to the
 * launchers below through plain C linkage so it can be compiled by g++ with
 * the same MPI/FlagCX flags as the repo's other host tests.
 *
 * Ordering contract used by the test:
 *   fill kernel -> __threadfence_system() -> musaDeviceSynchronize()
 * The threadfence makes the pattern writes visible system-wide; the explicit
 * device synchronize is the host-side completion barrier. The test then
 * exchanges the IPC handle over MPI, so the peer-open happens strictly after
 * the pattern is globally visible.
 */

#include <musa_runtime.h>

extern "C" __global__ void flagcxIpcPeerFillKernel(unsigned int *ptr,
                                                   unsigned long long n,
                                                   unsigned int seed) {
  const unsigned long long stride =
      (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
  unsigned long long i =
      (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
      (unsigned long long)threadIdx.x;
  for (; i < n; i += stride) {
    ptr[i] = seed ^ (unsigned int)(i * 16777619ULL);
  }
  /* Equivalent of cudaMemcpy-style system-scope visibility on MUSA. */
  __threadfence_system();
}

/* Returns 0 on success, a positive MUSA error code, or -1 on bad arguments.
 * The launch is followed by an explicit host-side device synchronize so the
 * caller never has to reason about stream ordering. */
extern "C" int flagcxIpcPeerFill(void *ptr, unsigned long long n,
                                 unsigned int seed) {
  if (ptr == nullptr || n == 0) {
    return -1;
  }

  int device = 0;
  musaError_t err = musaGetDevice(&device);
  if (err != musaSuccess) {
    return (int)err;
  }

  musaDeviceProp prop;
  err = musaGetDeviceProperties(&prop, device);
  if (err != musaSuccess) {
    return (int)err;
  }

  int threads = 256;
  if (prop.maxThreadsPerBlock > 0 && prop.maxThreadsPerBlock < threads) {
    threads = prop.maxThreadsPerBlock;
  }
  if (threads <= 0) {
    threads = 1;
  }

  unsigned long long need = (n + (unsigned long long)threads - 1ULL) /
                            (unsigned long long)threads;
  if (need < 1ULL) {
    need = 1ULL;
  }
  int blocks = (int)(need > 1024ULL ? 1024ULL : need);

  flagcxIpcPeerFillKernel<<<blocks, threads>>>((unsigned int *)ptr, n, seed);

  err = musaGetLastError();
  if (err != musaSuccess) {
    return (int)err;
  }

  err = musaDeviceSynchronize();
  if (err != musaSuccess) {
    return (int)err;
  }
  return 0;
}

extern "C" const char *flagcxIpcPeerErrorString(int code) {
  return musaGetErrorString((musaError_t)code);
}
