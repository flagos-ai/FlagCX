/* Minimal CUDA compatibility layer for the shared FlagCX test kernels.
 *
 * The kernel bodies in test/kernel/nvidia/ are platform-neutral SIMT code;
 * only the host-side launcher return type and stream handle are spelled with
 * CUDA names. MUSA's SDK ships no cuda_runtime.h and no cudaError_t, so the
 * MUSA wrappers map those names onto the MUSA runtime instead of forking the
 * kernel sources.
 */
#ifndef FLAGCX_TEST_MUSA_CUDA_RUNTIME_SHIM_H_
#define FLAGCX_TEST_MUSA_CUDA_RUNTIME_SHIM_H_

#include <musa_runtime.h>

typedef musaError_t cudaError_t;
typedef musaStream_t cudaStream_t;

static const cudaError_t cudaSuccess = musaSuccess;

static inline cudaError_t cudaGetLastError() { return musaGetLastError(); }
static inline cudaError_t cudaPeekAtLastError() {
  return musaPeekAtLastError();
}

#endif /* FLAGCX_TEST_MUSA_CUDA_RUNTIME_SHIM_H_ */
