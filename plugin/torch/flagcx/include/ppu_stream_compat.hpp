#pragma once

#include "flagcx.h"
#include <cstring>
// The compatibility SDK defines this macro unconditionally, while the native
// runtime may already have defined it. Keep the native spelling outside this
// include; neither SDK's internal annotation macro is part of this interface.
#pragma push_macro("__annotate__")
#undef __annotate__
#include <cuda_runtime.h>
#pragma pop_macro("__annotate__")
#include <hggc_runtime.h>
#include <type_traits>

// PPU PyTorch exposes CUDA-compatible streams. Keep that ABI conversion at
// the Torch boundary; the FlagCX adaptor and kernels use HGGC handles only.
// The PPU CI stream interoperability test checks both directions at runtime.
namespace flagcxPpuTorch {
inline cudaStream_t toTorchStream(flagcxStream_t stream) {
  static_assert(sizeof(hggcStream_t) == sizeof(cudaStream_t),
                "PPU Torch and HGGC stream handles must have the same size");
  static_assert(std::is_pointer<hggcStream_t>::value &&
                    std::is_pointer<cudaStream_t>::value,
                "PPU stream interoperability requires pointer handles");
  hggcStream_t native = nullptr;
  if (stream != nullptr)
    std::memcpy(&native, stream, sizeof(native));
  return reinterpret_cast<cudaStream_t>(native);
}
} // namespace flagcxPpuTorch
