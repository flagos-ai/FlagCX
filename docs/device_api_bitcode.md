# Device API bitcode

FlagCX can build Device API LLVM bitcode for NVIDIA CUDA, Iluvatar CoreX,
Hygon DU, and PPU. A kernel consumer links `libflagcx_device.bc` and includes
the headers from the same build. Choose an architecture supported by the
installed SDK.

## Build from the repository root

The `device_bitcode` target builds `libflagcx.so` with `COMPILE_KERNEL=1`, then
builds bitcode with the same adaptor and CommTraits flags. Every platform and
architecture uses the same `BUILDDIR` (by default, `build/`):

```sh
make USE_NVIDIA=1 DEVICE_BITCODE_ARCH=sm_90 device_bitcode

make USE_ILUVATAR=1 DEVICE_BITCODE_ARCH=ivcore11 device_bitcode

make USE_DU=1 DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  DEVICE_BITCODE_ARCH=gfx906 device_bitcode

make USE_PPU=1 DEVICE_HOME=/usr/local/PPU_SDK \
  DEVICE_BITCODE_ARCH=ppu001 BUILDDIR="$PWD/build/ppu-bc" device_bitcode
```

`build/` contains `lib/libflagcx.so`, `lib/libflagcx_device.bc`, and consumer
headers in `include/`. Repeated builds reuse unchanged objects and bitcode;
changes to included project headers rebuild the bitcode. Each build directory
uses one platform, architecture, and adaptor configuration. Use a fresh build
directory when changing those settings. The root build selects CUDA SHMEM,
Default, or CCL according to its normal Makefile options and installed NCCL
headers; the bitcode receives that exact selection.

## PPU Device IR for FlagTree

The PPU build uses `DefaultBackend<PpuPlatform>` and compiles the common
`flagcx_device_wrapper_impl.h`, including Scalar IR, Unified IR, and the
compatibility entry points. One `libflagcx_device.bc` contains all 152 public
Device IR entry points currently listed in `bindings/ir/public_symbols.txt`,
including inter-node operations. The optimizer preserves that complete list;
the build fails if any listed function has no definition in the final IR.

The PPU example above writes `lib/libflagcx_device.bc`, `lib/libflagcx.so`,
and matching consumer headers under `build/ppu-bc/`. Readable LLVM IR is
available at `build/ppu-bc/obj/llvm_ir/libflagcx_device.ll`. Supply the BC to
FlagTree through `extern_libs={"libflagcx_device": path_to_bc}` using its PPU
backend. Linking, code generation, and device execution are validated in
FlagTree; exporting an inter-node entry point does not establish transport
support in the runtime configuration.

PPU bitcode compilation uses the SDK's native HGGC Clang frontend with
`-x hggc --hggc-device-only -emit-llvm`, rather than CUDA/NVPTX compilation.
`DEVICE_BITCODE_ARCH` is passed to Clang as `--ppu-arch` and to the native
`hgcc` host-library kernel build as `-arch`. Choose a single architecture
accepted by both drivers in the installed SDK; `ppu001` is an example.

By default, the PPU build uses `clang`, `opt`, `llvm-dis`, and `llvm-as` from
`$(DEVICE_HOME)/bin`. Set `PPU_LLVM_BIN` when these tools are installed
elsewhere, or override `PPU_CLANG`, `OPT`, `LLVM_DIS`, and `LLVM_AS` individually.
Use LLVM tools from the same SDK build as the compiler so they understand
its PPU intrinsics. These overrides also work as environment variables when
building a wheel.

## Include bitcode in a Python wheel

Set `FLAGCX_BITCODE_ARCH` when installing from the repository root:

```sh
FLAGCX_ADAPTOR=nvidia FLAGCX_BITCODE_ARCH=sm_90 \
  pip install . -v --no-build-isolation

FLAGCX_ADAPTOR=iluvatar FLAGCX_BITCODE_ARCH=ivcore11 \
  pip install . -v --no-build-isolation

FLAGCX_ADAPTOR=du DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  FLAGCX_BITCODE_ARCH=gfx906 pip install . -v --no-build-isolation

FLAGCX_ADAPTOR=ppu DEVICE_HOME=/usr/local/PPU_SDK \
  FLAGCX_BITCODE_ARCH=ppu001 pip install . -v --no-build-isolation
```

The wheel includes `flagcx/lib/libflagcx_device.bc`, `libflagcx.so`, and the
matching headers under `flagcx/include/`. The BC build selects
`COMPILE_KERNEL=1` for the host library, which is required for Device API net
contexts. An explicit `COMPILE_KERNEL=0` conflicts with this build and fails.
