# Device API bitcode

FlagCX can build Device API LLVM bitcode for NVIDIA CUDA, Iluvatar CoreX,
Hygon DU, MUSA, PPU, MetaX MACA, and Huawei Ascend. A kernel consumer links
`libflagcx_device.bc` and includes the headers from the same build. Choose an
architecture supported by the installed SDK.

## Build from the repository root

The `device_bitcode` target builds `libflagcx.so` with `COMPILE_KERNEL=1`, then
builds bitcode with the same adaptor and CommTraits flags. Every platform and
architecture uses the same `BUILDDIR` (by default, `build/`):

```sh
make USE_NVIDIA=1 DEVICE_BITCODE_ARCH=sm_90 device_bitcode

make USE_ILUVATAR=1 DEVICE_BITCODE_ARCH=ivcore11 device_bitcode

make USE_DU=1 DEVICE_HOME=/opt/dtk CCL_HOME=/opt/dtk \
  DEVICE_BITCODE_ARCH=gfx906 device_bitcode

make USE_MUSA=1 DEVICE_HOME=/usr/local/musa CCL_HOME=/usr/local/musa \
  DEVICE_BITCODE_ARCH=mp_31 BUILDDIR="$PWD/build/musa-bc" device_bitcode

make USE_PPU=1 DEVICE_HOME=/usr/local/PPU_SDK \
  DEVICE_BITCODE_ARCH=ppu001 BUILDDIR="$PWD/build/ppu-bc" device_bitcode

make USE_METAX=1 DEVICE_HOME=/opt/maca \
  DEVICE_BITCODE_ARCH=xcore1000 BUILDDIR="$PWD/build/metax-bc" device_bitcode

make USE_ASCEND=1 USE_SHMEM=1 COMPILE_KERNEL=1 \
  SHMEM_HOME=/usr/local/Ascend/shmem/latest \
  DEVICE_BITCODE_ARCH=Ascend910_9382 device_bitcode
```

`build/` contains `lib/libflagcx.so`, `lib/libflagcx_device.bc`, and consumer
headers in `include/`. Repeated builds reuse unchanged objects and bitcode;
changes to included project headers rebuild the bitcode. Each build directory
uses one platform, architecture, and adaptor configuration. Use a fresh build
directory when changing those settings. The root build selects CUDA SHMEM,
Default, or CCL according to its normal Makefile options and installed NCCL
headers; the bitcode receives that exact selection.

For Iluvatar, `DEVICE_BITCODE_ARCH` selects `--cuda-gpu-arch` for both the
host library's device kernels and the bitcode. Test kernels reuse the platform
compiler flags. Ordinary builds default to `ivcore11` when the variable is
unset; the root `device_bitcode` target requires an explicit architecture.
The Iluvatar bitcode currently supports only `ivcore11`.

## MUSA architecture selection

For MUSA, `DEVICE_BITCODE_ARCH` selects the same `--offload-arch` for the
device kernels embedded in `libflagcx.so` and for the Device API bitcode.
It takes precedence over `MUSA_ARCH`, including when both are passed on the
command line. Ordinary builds without `DEVICE_BITCODE_ARCH` continue to use
`MUSA_ARCH`, which defaults to `mp_31`. The root `device_bitcode` target still
requires an explicit `DEVICE_BITCODE_ARCH`.

By default the build uses `mcc`, `opt`, `llvm-dis` and `llvm-as` from
`$(DEVICE_HOME)/bin`. Set `MUSA_LLVM_BIN` for a different tool directory, or
override `MCC`, `OPT`, `LLVM_DIS` and `LLVM_AS` individually, using matching
SDK tools. As in native kernel builds, SDK device C++ wrappers precede the
host C++ headers discovered through `CXX`. The readable IR is
`build/musa-bc/obj/llvm_ir/libflagcx_device.ll`; its target CPU must match the
architecture chosen above. Removing only `lib/libflagcx_device.bc` permits
reassembly from the retained IR. Use a separate build directory for a
different architecture.

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

## MetaX Device API and Device IR

MetaX selects `DefaultBackend<MetaxPlatform>`, independently of MCCL's native
Device API support. The common implementation exports Scalar IR, Unified IR,
and compatibility entry points. All names in `bindings/ir/public_symbols.txt`
must have definitions in the final module. The build checks LLVM validity and
rejects host and foreign GPU targets.

The build uses native `mxcc -x maca -maca-device-only -emit-llvm`, with
`DEVICE_BITCODE_ARCH` passed to both bitcode generation and the host library's
kernel build. Native relocatable kernels are linked by `mxcc -maca-link`;
ordinary host objects are passed as `-maca-host-input`. The `mxcc`, `opt`,
`llvm-dis`, and `llvm-as` defaults are under `/opt/maca/mxgpu_llvm/bin`.
Override `MACA_LLVM_BIN` or the individual `MXCC`, `OPT`, `LLVM_DIS`, and
`LLVM_AS` paths for another SDK layout. Use LLVM tools from that SDK.

Synchronous `deviceMalloc` and the non-VMM GDR allocator use
`mcExtMallocWithFlags(..., mcDeviceMallocFinegrained)` for IPC kernel
publication and polling. Direct `mcMalloc` supports ordinary stream memops
in the tested SDK, but timed out in the first Unified IR intra barrier.
Use the existing Default Device API, Scalar IR, and Unified IR intra tests to
validate IPC kernel-write visibility for this allocation path. The
asynchronous pool allocator retains its stream-ordered allocation/free
contract. VMM allocations and homogeneous `mcclMemAlloc` allocations
have their own memory properties. They are not covered by the non-VMM hetero
Device API CI configuration (`FLAGCX_USE_HETERO_COMM=1`, `FLAGCX_VMM_ENABLE=0`).
Allocations supplied by a caller must support the required peer visibility.
Uncached IPC memory does not remove the adaptor's NIC DMA flush requirements.

MetaX CI temporarily excludes `DeviceAdaptorTest.StreamWaitWriteValue64`,
`DeviceAdaptorTest.StreamWaitValue64RemoteFlushCapability`, and
`RmaTest.IpcGetSameStreamCompletion`: the current MACA SDK rejects their native
stream memops on finegrained allocations. The exclusions are recorded under
`ignore_test.gtest` in `.github/configs/metax.yml`; the other adaptor/RMA and
Device API cases remain enabled. The adaptor still reports errors normally
outside CI. Its strong remote-write wait returns `flagcxNotSupported` because
an acquire/flush mapping has not been validated; this does not mean the NIC
cannot transfer GPU data.

The existing `device_api`, `device_api_host`, and `device_api_unified_ir`
MetaX CI suites build the Default backend with `COMPILE_KERNEL=1` and exercise
native API, Scalar IR and Unified IR tests. Inter tests use two logical nodes
on the same runner; actual cross-host operation and downstream FlagTree linking
and execution require validation in those environments.

## Ascend Device IR

The Ascend build uses `CommTraits<AclshmemBackend>` when `USE_SHMEM=1`
(`DefaultBackend<AscendPlatform>` otherwise) and compiles the same
`flagcx_device_wrapper_impl.h` with bisheng, the toolkit's clang 15 CCE
frontend:

```sh
bisheng -x asc --npu-soc=<soc> --cce-aicore-only -emit-llvm -c ...
```

The CCE target is a two-core machine, so one build produces two modules
instead of one:

- `lib/libflagcx_device.bc` — AI Vector (AIV) core module;
- `lib/libflagcx_device-cube.bc` — AI Cube (AIC) core module.

Both modules carry the 152 public entry points listed in
`bindings/ir/public_symbols.txt`, under the CCE core suffix: `flagcxPut.vector`
in the AIV module and `flagcxPut.cube` in the AIC module. The build fails if a
listed symbol has no suffixed definition with external linkage. It also checks
the CCE triple and each module's core marker symbol
(`__init_kfc_workspace_addr.vector` / `.cube`). Readable IR is written to
`obj/llvm_ir/ascend/libflagcx_device.ll` and `libflagcx_device-cube.ll`.

The suffix is part of CCE's device ABI, not a FlagCX choice: a bisheng-compiled
consumer references an external aicore function with the same suffix (measured
on the 910C: a call to `probeAdd` becomes `@probeAdd.vector`), so CCE-built
consumer kernels link name-for-name with these modules. Source-level C names
are unchanged. The toolkit's `llvm-link` accepts `--internalize` but does not
apply it on this install (the internal-definition count is unchanged with both
the file and list spellings), so the non-API device functions remain in the
modules; consumers include only the API declaration headers.

`DEVICE_BITCODE_ARCH` is the soc name reported by `aclrtGetSocName()`
(`Ascend910_9382` on the 910C); `bisheng` is located from `DEVICE_HOME`, then
`PATH`, and can be overridden with `BISHENG`. The same soc also targets the
host library's embedded device helper kernels; if `ASCEND_SOC` is set
explicitly as well and names a different soc, the build fails instead of
producing bitcode and helper kernels for different devices.

Consumption is an LLVM-level pipeline (for example FlagTree / Triton-Ascend on
Ascend): the bisheng driver compiles CCE from `.asc` sources and does not
accept LLVM bitcode input, so the modules are linked into the consumer's LLVM
pipeline rather than passed back to the driver. Generation and module content
are validated on real 910C hardware; upstream CI does not build Ascend
bitcode.

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

FLAGCX_ADAPTOR=metax DEVICE_HOME=/opt/maca \
  FLAGCX_BITCODE_ARCH=xcore1000 pip install . -v --no-build-isolation
```

The wheel path currently covers nvidia, iluvatar, du, musa, ppu, and metax.
Build Ascend bitcode with the make target above and package the two modules
yourself.

The wheel includes `flagcx/lib/libflagcx_device.bc`, `libflagcx.so`, and the
matching headers under `flagcx/include/`. The BC build selects
`COMPILE_KERNEL=1` for the host library, which is required for Device API net
contexts. An explicit `COMPILE_KERNEL=0` conflicts with this build and fails.

## Resource lifetime

Complete kernels launched directly by the application before destroying the
DevComm, DevMem, or Window they use, or finalizing their communicator. The same
rule applies to captured graph replays. FlagCX does not discover external
streams or track graph capture/replay.

Host communication submissions save completion events at their submission
boundaries. Communicator cleanup waits for these records and accepted proxy/RMA
work without synchronizing the whole device. DevComm, DevMem, and Window cleanup
retain their first work snapshot and accepted RMA/FIFO prefix across retries,
so later independent host work
is not added to that snapshot. Existing stream ordering and explicit dependencies
still apply.

If cleanup fails, keep the handle and retry it. Resources already released remain
cleared; the remaining event, mapping, allocation, or plugin owner is retained.
Do not submit new work through a resource whose cleanup has started.
