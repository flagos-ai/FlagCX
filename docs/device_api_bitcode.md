# Device API bitcode

FlagCX can build Device API LLVM bitcode for NVIDIA CUDA, Iluvatar CoreX, and
Hygon DU. A kernel consumer links `libflagcx_device.bc` and includes the headers
from the same build. Choose an architecture supported by the installed SDK.

## Build from the repository root

The `device_bitcode` target builds `libflagcx.so` with `COMPILE_KERNEL=1`, then
builds bitcode with the same adaptor and CommTraits flags. Every platform and
architecture uses the same `BUILDDIR` (by default, `build/`):

```sh
make USE_NVIDIA=1 DEVICE_BITCODE_ARCH=sm_90 device_bitcode

make USE_ILUVATAR=1 DEVICE_BITCODE_ARCH=ivcore11 device_bitcode

make USE_DU=1 DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  DEVICE_BITCODE_ARCH=gfx906 device_bitcode
```

`build/` contains `lib/libflagcx.so`, `lib/libflagcx_device.bc`, and consumer
headers in `include/`. Repeated builds reuse unchanged objects and bitcode;
changes to included project headers rebuild the bitcode. Each build directory
uses one platform, architecture, and adaptor configuration. Use a fresh build
directory when changing those settings. The root build selects CUDA SHMEM,
Default, or CCL according to its normal Makefile options and installed NCCL
headers; the bitcode receives that exact selection.

## Include bitcode in a Python wheel

Set `FLAGCX_BITCODE_ARCH` when installing from the repository root:

```sh
FLAGCX_ADAPTOR=nvidia FLAGCX_BITCODE_ARCH=sm_90 \
  pip install . -v --no-build-isolation

FLAGCX_ADAPTOR=iluvatar FLAGCX_BITCODE_ARCH=ivcore11 \
  pip install . -v --no-build-isolation

FLAGCX_ADAPTOR=du DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  FLAGCX_BITCODE_ARCH=gfx906 pip install . -v --no-build-isolation
```

The wheel includes `flagcx/lib/libflagcx_device.bc`, `libflagcx.so`, and the
matching headers under `flagcx/include/`. The BC build selects
`COMPILE_KERNEL=1` for the host library, which is required for Device API net
contexts. An explicit `COMPILE_KERNEL=0` conflicts with this build and fails.
