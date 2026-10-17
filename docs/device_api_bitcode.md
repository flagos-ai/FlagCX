# Device API bitcode

FlagCX can build Device API LLVM bitcode for NVIDIA CUDA and Hygon DU. A kernel
consumer links `libflagcx_device.bc` and includes the exported headers from the
same build. Choose a GPU architecture supported by the installed toolchain.

## Build bitcode directly

Run one of these commands from the repository root:

```sh
# NVIDIA CUDA: choose the comm traits used by the host library.
make -C bindings/ir/nvidia BITCODE_LIB_ARCH=sm_90 \
  ADAPTOR_FLAG='-DUSE_NVIDIA_ADAPTOR -DFLAGCX_COMM_TRAITS_CCL' llvm_ir

# Hygon DU: DEVICE_HOME and CCL_HOME point to the installed DTK and RCCL.
make -C bindings/ir/du DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  BITCODE_LIB_ARCH=gfx906 llvm_ir
```

The result is `build/lib/libflagcx_device.bc`, with consumer headers in
`build/include/`. The DU build also keeps an architecture-specific copy at
`build/lib/du/<architecture>/libflagcx_device.bc`. Direct bitcode builds do not
build `libflagcx.so`.

For CUDA, use `-DFLAGCX_COMM_TRAITS_DEFAULT` instead of
`-DFLAGCX_COMM_TRAITS_CCL` when the host library selects the default Device API
backend. The bitcode and host library must select the same comm traits.

## Include bitcode in a Python wheel

Set `FLAGCX_BITCODE_ARCH` when installing from the repository root:

```sh
# NVIDIA CUDA; use the comm traits selected by the host library.
FLAGCX_ADAPTOR=nvidia FLAGCX_BITCODE_ARCH=sm_90 \
  FLAGCX_BITCODE_ADAPTOR_FLAGS='-DUSE_NVIDIA_ADAPTOR -DFLAGCX_COMM_TRAITS_CCL' \
  pip install . -v --no-build-isolation

# Hygon DU.
FLAGCX_ADAPTOR=du DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  FLAGCX_BITCODE_ARCH=gfx906 pip install . -v --no-build-isolation
```

The wheel includes `flagcx/lib/libflagcx_device.bc` and the matching headers
under `flagcx/include/`. The DU wheel build sets `COMPILE_KERNEL=1` when building
`libflagcx.so`, which is required for Device API net contexts. It uses
`build/du-bitcode/` so objects from a regular `COMPILE_KERNEL=0` build are not
reused. An explicit `COMPILE_KERNEL=0` conflicts with DU bitcode and stops the
wheel build.
