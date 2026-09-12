## Build and Installation

### Obtain Source Code

```
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git submodule update --init --recursive
```

### Installation

**Option A — Pythonic Installation (pip install):**

```shell
pip install . -v --no-build-isolation
```

![flagcx_pip_install.png](images/flagcx_pip_install.png)

**Option B — C++ library (make):**

```shell
make <backend>=1 -j$(nproc)
```
where `<backend>` is one of:
- `USE_NVIDIA`: NVIDIA GPU support
- `USE_ILUVATAR_COREX`: Iluvatar Corex support
- `USE_CAMBRICON`: Cambricon support
- `USE_METAX`: MetaX support
- `USE_MUSA`: Moore Threads support
- `USE_KUNLUNXIN`: Kunlunxin support
- `USE_DU`: Hygon support
- `USE_ASCEND`: Huawei Ascend support
- `USE_AMD`: AMD support
- `USE_TSM`: TsingMicro support
- `USE_ENFLAME`: Enflame support
- `USE_SUNRISE`: Sunrise support
- `USE_PPU`: PPU support
- `USE_GLOO`: GLOO support
- `USE_MPI`: MPI support

Note that Option A also supports `<backend>=1`, allowing users to explicitly specify the backend. Otherwise, it will be selected automatically.

## Tests

### Performance Test

Performance tests are maintained in `test/perf/`, organized by API level:

- **Host API tests** (`test/perf/host_api/`) — high-level collective operations via FlagCX host API
- **Device API tests** (`test/perf/device_api/`) — low-level device kernel benchmarks via FlagCX Device API

#### Host API Performance Test

```shell
cd test/perf/host_api
make [USE_NVIDIA | USE_ILUVATAR_COREX | USE_CAMBRICON | USE_METAX | USE_MUSA | USE_KUNLUNXIN | USE_DU | USE_ASCEND | USE_AMD | USE_TSM | USE_ENFLAME | USE_SUNRISE | USE_PPU]=1
cd build/bin
mpirun --allow-run-as-root -np 8 ./perf_allreduce -b 128K -e 4G -f 2
```

#### Device API Performance Test

Device API perf tests require `-R 1` (IPC mode) or `-R 2` (window mode).
The FlagCX library must be built with `COMPILE_KERNEL=1`:

```shell
# Build FlagCX with kernel support (from project root)
make USE_NVIDIA=1 COMPILE_KERNEL=1 -j$(nproc)

cd test/perf/device_api
make USE_NVIDIA=1
cd build/bin

# Intra-node AllReduce (single node, 8 GPUs)
mpirun --allow-run-as-root -np 8 -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  ./perf_allreduce_intranode -b 1M -e 64M -f 2 -R 1

# Inter-node two-sided AlltoAll (multi-node)
mpirun --allow-run-as-root -np 16 -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  ./perf_internode_twosided -b 1M -e 64M -f 2 -R 1

# Inter-node one-sided AlltoAll (requires -R 2)
mpirun --allow-run-as-root -np 16 -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  ./perf_internode_onesided -b 1M -e 64M -f 2 -R 2
```

Note that the default MPI install path is set to `/usr/local/mpi`, you may specify the MPI path with:

```shell
make MPI_HOME=<MPI path>
```

All tests support the same set of arguments:

- Sizes to scan

  * `-b <min>` minimum size in bytes to start with. Default: 1M.
  * `-e <max>` maximum size in bytes to end at. Default: 1G.
  * `-f <increment factor>` multiplication factor between sizes. Default: 2.

- Performance

  * `-w, <warmup iterations >` number of warmup iterations (not timed). Default: 5.
  * `-n, <iterations >` number of iterations. Default: 20.

- Test operation

  * `-R, <0/1/2>` enable local buffer registration on send/recv buffers. Default: 0.
  * `-s, <OCT/DEC/HEX>` specify MPI communication split mode. Default: 0

- Utils

  * `-p, <0/1>` print buffer info. Default: 0.
  * `-h` print help message. Default: disabled.

Registration modes (`-R`):

- `-R 0`: Raw device memory (default). No explicit registration. Not supported by Device API tests.
- `-R 1`: IPC mode — `cudaMalloc` + `flagcxCommRegister`. Works on all NCCL versions.
- `-R 2`: Window mode — `flagcxMemAlloc` + `flagcxCommWindowRegister`. Requires NCCL >= 2.28.

### Device API Correctness Test

Device API correctness tests are maintained in `test/unittest/device_api/`. These tests verify the functional correctness of the Device API primitives.

| Binary | What it tests |
|---|---|
| `test_device_api` | Correctness suite for 10 one-sided Device API kernels (put, get, signal, flush, counter, etc.). |
| `test_device_ir` | IR wrapper layer correctness. |

Build and run:

```shell
# FlagCX must be built with COMPILE_KERNEL=1 (from project root)
make USE_NVIDIA=1 COMPILE_KERNEL=1 -j$(nproc)

cd test/unittest/device_api
make USE_NVIDIA=1
cd build/bin

# Device API correctness test (requires -R 1 or -R 2)
mpirun --allow-run-as-root -np 8 -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  ./test_device_api -b 1M -e 4M -f 2 -R 2

# IR wrapper correctness test
mpirun --allow-run-as-root -np 8 -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  ./test_device_ir -b 1M -e 4M -f 2 -R 2
```

#### SHMEM Device API memory ownership

When FlagCX is built with a SHMEM backend, memory passed to
`flagcxDevMemCreate`, `flagcxCommRegister`, or `flagcxCommWindowRegister` with
the `flagcxMemSHMEM` allocator must be allocated and freed through FlagCX:

```cpp
void *buffer = nullptr;
flagcxMemAlloc(&buffer, bytes, flagcxMemSHMEM);

flagcxDevMem_t devMem = nullptr;
flagcxDevMemCreate(comm, buffer, bytes, nullptr, &devMem);

flagcxDevMemDestroy(comm, devMem);
flagcxMemFree(buffer, flagcxMemSHMEM);
```

Direct `nvshmem_malloc` or `xshmem_malloc` allocations remain valid for their
native runtime APIs, but FlagCX does not have their allocation bounds or
ownership metadata. Passing such a pointer to `flagcxDevMemCreate` is rejected
with `flagcxInvalidUsage`. Existing code that combines a native SHMEM allocation
with a FlagCX Device API memory handle should migrate to the allocation flow
shown above.

### Torch API Test

Torch API tests verify FlagCX's PyTorch custom process group backend (`flagcx`) by running collective communication operations through `torch.distributed`. Test scripts are maintained in `plugin/torch/example/`.

The test script `example.py` supports the following collective operations:

`broadcast`, `reduce`, `allreduce`, `allgather`, `reducescatter`, `sendrecv`, `gather`, `scatter`, `alltoall`, `all` (default)

Use the `-o` / `--op` flag to test a specific operation (e.g., `-o allreduce`). By default, all operations are tested.

#### Single-Node Test

For single-node testing, use `run.sh`:

```shell
cd plugin/torch/example
# Edit run.sh to set environment variables for your hardware platform,
# e.g., CUDA_VISIBLE_DEVICES for NVIDIA GPUs.
./run.sh
```

To enable NCCL debug output:

```shell
./run.sh debug
```

#### Two-Node Heterogeneous Test

For a 2-node heterogeneous communication test, use `run_hetero.sh`. You need to modify and run the script on each machine separately.

On each node, edit `run_hetero.sh` to set:

- `--node_rank`: `0` on the first node, `1` on the second
- `--master_addr`: IP address of node 0
- `--nproc_per_node`: number of devices per node
- `GLOO_SOCKET_IFNAME` / `FLAGCX_SOCKET_IFNAME`: network interface for inter-node communication (e.g., `eth0` or `ibs4` for InfiniBand)
- Hardware-specific environment variables and library paths for each node's platform

Then run the script manually on each machine:

```shell
cd plugin/torch/example
./run_hetero.sh
```

#### Multi-Node Test

For multi-node testing (both homogeneous and heterogeneous), use the `run_torch_test.py` launcher. It reads a hostfile and YAML config, generates a per-node run script, copies it to each host via SCP, and launches it via SSH.

Prerequisites:
- Passwordless SSH access between all nodes
- PyYAML installed (`pip install pyyaml`)

**Step 1** — Configure the hostfile (see `example_hostfile`):

```
# <ip> slots=<n> type=<device_type>
192.168.1.1 slots=8 type=nvidia
192.168.1.2 slots=8 type=nvidia
```

Each line specifies a node's IP, the number of devices (`slots`), and the device type. All nodes must have the same `slots` count. The `type` field must match a key under `envs.device_type_specific` in the YAML config.

**Step 2** — Configure the YAML env config (see `example_torch_env.yaml`):

```yaml
cmds:
  before_start: source /root/miniconda3/bin/activate flagscale-train
test_dir: ./
log_dir: ./torch_logs
testfile: ./example.py
master_port: 8281
master_addr: 192.168.1.1
envs:
  FLAGCX_DEBUG: INFO
  FLAGCX_DEBUG_SUBSYS: ALL
  device_type_specific:
    nvidia:
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
```

Key fields:
- `cmds.before_start`: shell command to run before the test (e.g., conda environment activation)
- `test_dir`: directory on remote hosts where generated run scripts are placed
- `log_dir`: directory on remote hosts where stdout/stderr logs are written
- `testfile`: path to `example.py` on the remote hosts
- `master_port`: port for distributed rendezvous (must be in range 10000–19999)
- `master_addr`: IP of the master node (defaults to the first host in the hostfile if omitted)
- `envs`: common environment variables for all nodes
- `envs.device_type_specific`: per-device-type environment variables, keyed by the `type` field in the hostfile

**Step 3** — Launch the test:

```shell
cd plugin/torch/example
python run_torch_test.py --hostfile example_hostfile --config example_torch_env.yaml
```

To test a specific collective operation:

```shell
python run_torch_test.py --hostfile example_hostfile --config example_torch_env.yaml --extra-args "--op allreduce"
```

To validate the generated scripts without executing them:

```shell
python run_torch_test.py --hostfile example_hostfile --config example_torch_env.yaml --dry-run
```

Check log files in the configured `log_dir` on each remote host for output.

**Stopping hung processes** — If distributed training hangs, use `stop_torch_test.py` to kill `torchrun` processes across all nodes:

```shell
python stop_torch_test.py --hostfile example_hostfile
```
