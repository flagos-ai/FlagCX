# Device API GDA inter tests

The XPU NetIO SDK must already be installed. `GDA_PROVIDER_HOME` points to the
prefix containing matching SDK headers, `libxpu_netio.so` and the MLX5 provider
library. This test does not build or install the SDK.

Build the library and existing inter executable with identical options:

```bash
export GDA_PROVIDER_HOME=/path/to/xpu-netio-sdk
export LD_LIBRARY_PATH="$GDA_PROVIDER_HOME/lib:$GDA_PROVIDER_HOME/lib64:$PWD/build/lib:${LD_LIBRARY_PATH:-}"

make -j8 USE_NVIDIA=1 COMPILE_KERNEL=1 USE_XPUNETIO=1 \
  GDA_PROVIDER_HOME="$GDA_PROVIDER_HOME" \
  DEVICE_COMPILER_GENCODE='-gencode=arch=compute_90,code=sm_90'

make -C test/unittest/device_api -j8 USE_NVIDIA=1 COMPILE_KERNEL=1 USE_XPUNETIO=1 \
  GDA_PROVIDER_HOME="$GDA_PROVIDER_HOME" \
  DEVICE_COMPILER_GENCODE='-gencode=arch=compute_90,code=sm_90' unit mpi

make -C test/unittest/device_api USE_NVIDIA=1 COMPILE_KERNEL=1 USE_XPUNETIO=1 \
  GDA_PROVIDER_HOME="$GDA_PROVIDER_HOME" \
  DEVICE_COMPILER_GENCODE='-gencode=arch=compute_90,code=sm_90' run-unit run-mpi-gda
```

Set `DEVICE_HOME`, `CCL_HOME` and `MPI_HOME` for non-default CUDA, NCCL and MPI
installations. The initial SDK profile targets NVIDIA SM90 and MLX5. Set
`FLAGCX_XPU_NETIO_DEVICE` to the intended active NIC when necessary; export it
through the MPI node environment as required by the local launcher.

`run-mpi-gda` runs the complete `test_device_api_inter` program. Its default
configuration uses GPU 0 and GPU 1 as two logical nodes, one MPI rank per node,
and 64 AlltoAll iterations. Override `GDA_NODE0_ENV`, `GDA_NODE1_ENV`,
`GDA_RANKS_PER_NODE` and `GDA_TEST_ARGS` for another topology or message range.
Logical nodes on one host do not constitute a physical multi-node test.

The selected DevComm must use GDA. K14 invokes the existing
`flagcxInterGdaAlltoAllKernel` through `launchKernelNetOneSidedAlltoAll`; no
standalone GDA AlltoAll program is required. Each remote chunk uses
`put(SignalInc)`, then the receiver waits for its cumulative indexed signal.
Before flush and the final World Barrier, ordinary GPU loads check the payload
against a pattern that changes each round. The host checks the final payload
as well. The test fails on a transport error, mismatch, or no aligned AlltoAll
execution.

Indexed SignalInc/Add, Put/PutValue notifications and 64-bit signal reads/waits
are exercised by the supported inter cases. These signal examples and K14 use
one CTA on the initial GDA profile. K12/K13 retain multiple CTAs and separate
Barrier channels, exercising GPU-wide QP sharing. Counter, Get and
multiple-context cases remain explicit SKIPs. GDA does not provide two-sided
send/recv semantics. Only Relaxed Barrier fences are supported by this adapter.

With `USE_XPUNETIO=1`, `run-unit` additionally runs GDA host-control tests.
The SDK-backed synthetic SQ/CQ and wait/abort tests can be run separately:

```bash
make -C test/kernel/nvidia USE_NVIDIA=1 COMPILE_KERNEL=1 USE_XPUNETIO=1 \
  GDA_PROVIDER_HOME="$GDA_PROVIDER_HOME" \
  DEVICE_COMPILER_GENCODE='-gencode=arch=compute_90,code=sm_90' gda_device_tests
```

These synthetic tests require a GPU but do not exercise NIC DMA.

For healthy Window deregistration, stop submissions, complete network work and
release its DevMem views before all ranks deregister the Window. After a
transport timeout, abort or provider error, all ranks must destroy the GDA
DevComm and retire its QPs before releasing MRs or backing allocations. A
failed teardown retains its handle for cleanup-only retries; it cannot publish
new device views. The ordinary `device_api` and `device_api_unified_ir` CI
configurations do not enable `USE_XPUNETIO` and keep their existing paths.
