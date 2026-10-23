# uniRunner DSL Design and Usage

This document describes the Python DSL in `flagcx/tools/uni_runner.py` for
building uniRunner algorithm DAGs. The DSL follows the same high-level idea as
MSCCL-style workflow builders: an algorithm is described in Python inside a
`with UniRunnerWorkflow(...)` block, and the block adds typed operation nodes
plus dependencies. The generated JSON can be written in the cache format that
the existing uniRunner runtime loads from `FLAGCX_UNIRUNNER_ALGO_PATH`.

## Goals

- Build one DAG template per rank.
- Describe `p2p`, `red`, `cpy`, and `gemm` using node kinds accepted by
  the C++ persistent executor.
- Address buffers by runtime buffer kind plus byte offset: `input`, `output`,
  `scratch`, and `weight`.
- Generate JSON matching `uni_runner_helper.h`.
- Reproduce the C++ DAG cache key hash so files can be named
  `dag_hash_<hash>_rank_<rank>.json`.
- Provide semantic validation for common collective operations and custom
  communication semantics.

## Runtime JSON Shape

The runtime accepts either a single DAG template or a cache collection. The DSL
uses the single-template file form for rank files:

```json
{
  "format_version": 8,
  "hash": "<computed pattern hash>",
  "key": {
    "algo_hash": "<computed algorithm hash>",
    "comm_op": "all_reduce",
    "count": 17,
    "datatype": 7,
    "red_op": 0,
    "rank": 0,
    "nranks": 4,
    "root": -1
  },
  "dag": {
    "num_nodes": 1,
    "nodes": []
  }
}
```

The key must match the C++ initialization path selected at runtime. The DSL
computes `algo_hash` with the same fixed-width FNV algorithm as
`getUniRunnerDagAlgorithmHash`. This identity includes the algorithm identifier
and its normalized builder configuration. The file's pattern hash then combines
the cache format, both 32-bit halves of `algo_hash`, collective operation, count,
datatype, reduction operation, rank, number of ranks, and root.

Use `algorithm_config` when defining a workflow directly. Its keys match the
normalized C++ builder fields in snake case. The built-in example builders fill
this mapping automatically:

| Algorithm | Configuration included in `algo_hash` |
| --- | --- |
| GroupedAR | `num_slices`, `num_red_slices`, `num_local_slices`, `pipeline_version`, `group_size`, `buffer_mode` |
| AllGatherGemm / GemmReduceScatter | `m`, `n`, `k`, `num_slices`, `num_k_slices`, `group_size`, `pipeline_version`, `buffer_mode` |
| Grouped AllGather | `group_size`, `buffer_mode` |
| Sliced AllReduce | `num_slices`, effective `num_red_slices`, `buffer_mode` |
| Hierarchical Sliced AllReduce | `num_slices`, effective `num_red_slices`, `group_size`, `buffer_mode` |

`buffer_mode=0` denotes out-of-place buffers and `1` denotes in-place buffers.
The supplied builders describe out-of-place algorithms. The RED slice count is
the resolved positive count, including when automatic slicing was requested.
`red_slice_size` affects the hash through that resolved count. Executor resource
settings and runtime handles are not serialized in the DAG template. Changing
an algorithm's topology or tiling produces a different file identity; switching
only executor resources does not.

The runtime loads rank files from:

```bash
FLAGCX_UNIRUNNER_ALGO_PATH=/path/to/cache
/path/to/your/flagcx/test
```

Each rank file in that directory must be named:

```text
dag_hash_<computed_hash>_rank_<rank>.json
```

Use `workflow.write_rank_files(cache_dir)` to write those files.

RED nodes in JSON only store operation parameters: inputs, output, count,
datatype, and reduction op. The runtime loader binds RED execution to
the runtime-selected thread count when the DAG is materialized, keeping the
reduce kernel launch block size and the per-node loop stride consistent. Use
`FLAGCX_UNIRUNNER_MAX_THREADS` and `FLAGCX_UNIRUNNER_MAX_BLOCKS` to bound the
current persistent executor resources.

## Basic DSL

```python
from uni_runner import UniRunnerWorkflow
from utils import Collective, DataType, RedOp

with UniRunnerWorkflow(
    "my_allreduce",
    collective=Collective.AllReduce,
    world_size=4,
    count=1024,
    datatype=DataType.float32,
    red_op=RedOp.sum,
    algo="sliced_ar",
    algorithm_config={"num_slices": 1, "num_red_slices": 1, "buffer_mode": 0},
    output_count=1024,
) as workflow:
    for rank in workflow.ranks():
        next_rank = (rank + 1) % workflow.world_size
        prev_rank = (rank - 1 + workflow.world_size) % workflow.world_size

        with workflow.rank(rank) as rb:
            p2p = rb.p2p(
                [
                    workflow.send(next_rank, workflow.input(0), 256),
                    workflow.recv(prev_rank, workflow.output(0), 256),
                ],
                name="exchange_chunk",
            )
            rb.red(
                workflow.output(0),
                workflow.input(0),
                workflow.output(0),
                256,
                red_op=RedOp.sum,
                parents=[p2p],
                name="accumulate_chunk",
            )

# This fragment illustrates node construction, not a complete AllReduce.
workflow.write_rank_files("algo_output/my_allreduce")
workflow.write_dag_json("algo_output/my_allreduce_dag.json")
```

`workflow.input(i)`, `workflow.output(i)`, and `workflow.scratch(i)` use element
offsets. The emitted JSON stores byte offsets. Use `input_bytes`,
`output_bytes`, or `scratch_bytes` only when you already have byte offsets.

Size/count helper functions in `utils.py` accept binary suffixes. `1K`, `1M`,
and `1G` are parsed as powers of 1024. Use `parse_size_count("1M")` when the
value is already an element count or byte count, and use
`count_from_size_bytes("1M", DataType.float32)` when a message size in bytes
must become a typed element count.

## Nodes

### `p2p`

A `p2p` node groups one or more device primitives. The common primitives are
`send` and `recv`.

```python
rb.p2p(
    [
        workflow.send(peer_rank, workflow.output(offset), count),
        workflow.recv(peer_rank, workflow.output(offset), count),
    ],
    parents=[parent_node_idx],
)
```

### `red`

A `red` node performs elementwise local reduction:

```python
rb.red(
    workflow.output(offset),
    workflow.input(offset),
    workflow.output(offset),
    count,
    red_op=RedOp.sum,
    parents=[p2p_node_idx],
)
```

### `cpy`

A `cpy` node performs a device-to-device copy:

```python
rb.cpy(workflow.input(0), workflow.output(rank * count), count)
```

The original pure-collective builders retain communication domain 0.
The GroupedAR and fused builders explicitly distinguish IPC and NET domains
to express the independent-stream pipeline described below. Domain selection
does not replace the runtime transport selection for each peer.

Dependencies are per-rank node indices. A node may depend on a node that is
created later; `workflow.finalize()` computes the reverse `children` edges after
all nodes are present.

## Built-In Examples

`flagcx/tools/test_uni_runner.py` defines several complete example algorithms and
also contains the Python-only tests.

```bash
python flagcx/tools/test_uni_runner.py --generate-examples
```

This writes:

- `algo_output/groupedag`: runtime cache files for grouped AllGather.
- `algo_output/slicedar`: runtime cache files for sliced AllReduce.
- `algo_output/hierarchical_slicedar`: runtime cache files for
  hierarchical sliced AllReduce.
- `groupedag_dag.json`, `slicedar_dag.json`, and
  `hierarchical_slicedar_dag.json`: readable DAG dumps.

The example builders are also importable:

```python
from test_uni_runner import (
    build_groupedag_example,
    build_hierarchical_slicedar_example,
    build_slicedar_example,
)

groupedag = build_groupedag_example()
slicedar = build_slicedar_example()
hierarchical_slicedar = build_hierarchical_slicedar_example()
```

### Hierarchical Sliced AllReduce

`build_hierarchical_slicedar` is a DSL-only hierarchical AllReduce built on the
same `p2p` / `red` / `cpy` primitives as SlicedAR:

1. Intra-node reduce-scatter inside each contiguous rank group.
2. Inter-node sliced AllReduce across ranks with the same local rank.
3. Intra-node allgather to distribute the globally reduced chunks.

The group size is the number of GPUs per node. It can be passed directly to the
Python builder or read by the builder from `UNIRUNNER_GROUPSIZE`; if neither is
set, the DSL uses `8`.
The inter-node reduce-scatter phase receives into `scratch` and reduces into
`output`, so the runtime allocates one `count`-element scratch buffer for this
loader.

```python
from test_uni_runner import build_hierarchical_slicedar

workflow = build_hierarchical_slicedar(
    world_size=16,
    count=1024,
    group_size=None,  # read UNIRUNNER_GROUPSIZE, defaulting to 8
    num_red_slices=0,
    red_slice_size="1M",
)
```

`group_size`, `num_slices`, and the effective `num_red_slices` are recorded in
the workflow algorithm configuration and contribute to `algo_hash`. They are
not repeated as individual JSON key fields. The runtime
`FLAGCX_UNIRUNNER_GROUPSIZE`, `FLAGCX_UNIRUNNER_NSLICES`, and effective
`FLAGCX_UNIRUNNER_NREDSLICES` must match the generated configuration.

The emitted runtime key uses the `hierarchical_sliced_ar` algorithm hash and
`all_reduce` collective. For the files emitted by `--generate-examples`
(`count=17`, 16 ranks, group size 8, two communication and two RED slices),
enable this path separately from the original SlicedAR loader:

```bash
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_UNIRUNNER_USE_HIERARCHICAL_SLICEDAR=1
export FLAGCX_UNIRUNNER_GROUPSIZE=8
export FLAGCX_UNIRUNNER_NSLICES=2
export FLAGCX_UNIRUNNER_NREDSLICES=2
export FLAGCX_UNIRUNNER_ALGO_PATH=algo_output/hierarchical_slicedar
```

`FLAGCX_UNIRUNNER_USE_SLICEDAR=1` still selects the original `sliced_ar` loader. The
hierarchical loader expects a matching DSL-generated cache file and does not
silently fall back to the original SlicedAR DAG.

## Semantic Checking

`semantics.py` includes symbolic correctness rules for:

- `allreduce`
- `allgather`
- `reducescatter`
- `broadcast`
- `reduce`
- `gather`
- `scatter`
- `alltoall`

The checker validates:

- all ranks are present;
- node indices are contiguous;
- parent and child edges are symmetric;
- dependencies are acyclic;
- P2P sends and receives match during symbolic execution;
- final output buffers match the selected collective semantics.

Use:

```python
workflow.semantic_check().raise_for_error()
```

For custom communication semantics, provide an explicit `CollectiveSemantics`
object:

```python
from semantics import CollectiveSemantics
from utils import Collective, RedOp

expected = [
    [(RedOp.nop, [0]), (RedOp.nop, [1])],
    [(RedOp.nop, [2]), (RedOp.nop, [3])],
]

semantic = CollectiveSemantics.custom(
    "identity_copy",
    world_size=2,
    input_count=2,
    output_count=2,
    expected_output=expected,
)
```

You can also register a procedural checker with
`register_custom_semantic(name, checker)` when expected output is easier to
compute from the simulated context than to enumerate.

## Current Runtime Integration Notes

The C++ runtime loads cache files by matching the cache key generated by the
active static initializer. Built-in examples compute algorithm hashes for
`grouped_ag` (AllGather), `sliced_ar` (the original SlicedAR AllReduce), and
`hierarchical_sliced_ar` (the DSL-defined hierarchical SlicedAR AllReduce).
GroupedAR and the two fused entry points are also supported as described below.
The exported format version and algorithm hash schema must match the runtime;
regenerate files produced by the older DSL cache format before using them.

The runtime still does not expose a generic `custom` uniRunner entry point.
The Python DSL can build arbitrary DAGs and custom semantic checks, but runtime
loading requires an existing C++ key path or a new key path added in the
uniRunner initializer.

## Test

Run the Python-only test suite:

```bash
python flagcx/tools/test_uni_runner.py
```

The test does not compile C/CUDA code. It checks JSON generation, runtime cache
file names, and symbolic correctness of the grouped AllGather, sliced
AllReduce, and hierarchical sliced AllReduce examples.

## Persistent executor contract (format 8)

The fused examples use `pipeline_version=2`; regenerate matching inputs when
changing the pipeline version. The input remains a logical DAG. No changes to the persistent device kernel or
its host executor are needed. The C++ input layer binds addresses, compiles the
existing execution plan, and reconstructs runtime state:

- `comm_domain=0` selects IPC/control; `1` selects the independent NET stream.
  Only P2P and RED support NET. COPY and GEMM use IPC/control.
- Each P2P node contains at most one send and one receive per peer.
  Use separate ordered P2P nodes for multiple blocks sent to the same peer.
- RED and GEMM each accept at most one parent. Use an empty P2P node as a join
  for multiple dependencies.
- The compiler drains ready P2P/COPY nodes, then ready RED/GEMM nodes, in FIFO
  order. Dependencies must account for this order and each stream's FIFO order.
- SlicedAR emits one logical RED per communication step; `num_red_slices`
  selects runtime RED workers. GroupedAR preserves its explicit RED segments,
  each with one worker, and its independent outer/inner slice dependencies.
- GEMM supports float32, row-major A/B/C, explicit `lda/ldb/ldc`, and
  `accumulate=0/1`. A stride is in elements; a buffer offset is in bytes.
- Threads, worker IDs/counts, completion counters, trigger/FIFO slots, CUDA
  handles, and absolute pointers are runtime state and are never serialized.

A low-level GEMM definition looks like:

```python
rb.gemm(
    workflow.input(a_offset), workflow.weight(b_offset),
    workflow.output(c_offset),
    m=rows, n=columns, k=inner,
    lda=inner, ldb=columns, ldc=columns,
    accumulate=False, parents=[data_ready], name="tile_gemm",
)
```

`weight_count` declares the weight binding capacity. Like other buffer helpers,
`workflow.weight(i)` takes an element offset; `weight_bytes(i)` takes bytes.
The C++ loader validates the full strided matrix span, not just rows times
columns. It obtains capacities from the API invocation, not from untrusted
JSON metadata. JSON `algorithm` and `bindings` fields describe the example and
support CPU validation; they do not allocate memory or override API arguments.

GEMM node workers are reconstructed from the node's M/N tile count and
`FLAGCX_UNIRUNNER_GEMM_WORKERS_PER_STEP`, bounded by
`FLAGCX_UNIRUNNER_MAX_BLOCKS`. The runtime allocates one 128-byte completion
counter slot per logical GEMM node in the loaded DAG. GemmReduceScatter's
logical RED nodes use the existing RED worker expansion; with
`NREDSLICES=0`, each node derives its worker count from `REDSLICESIZE`.

## Three 16-rank, 4-GiB definitions

`examples_4g.py` contains the complete definitions:

```python
from builders_grouped import build_groupedar
from builders_fused import build_allgather_gemm, build_gemm_reducescatter

groupedar = build_groupedar(
    world_size=16, count=(4 << 30) // 4, group_size=8,
    num_slices=4, num_local_slices=2,
    num_red_slices=0, red_slice_size="1M",
)
ag_gemm = build_allgather_gemm(
    world_size=16, m=262144, n=4096, k=4096,
    group_size=8, num_slices=4, num_k_slices=1,
)
gemm_rs = build_gemm_reducescatter(
    world_size=16, m=262144, n=4096, k=4096,
    group_size=8, num_slices=4, num_k_slices=1,
)
```

All use float32 and out-of-place buffers. Rank 0..7 occupy host 0 and rank
8..15 host 1; corresponding local ranks are NIC peers. The fused builders
currently require exactly two equally sized groups. Global M/K are used in
the DSL, while the public C++ AG API takes local M and the RS API takes local K.

| Property | GroupedAR | AllGatherGemm | GemmReduceScatter |
| --- | --- | --- | --- |
| Meaning of 4 GiB | Full input/output per rank | Gathered A | C before scatter |
| Per-rank input | 4 GiB | A[16384,4096], 256 MiB | A[262144,256], 256 MiB |
| Per-rank weights | None | B[4096,4096], 64 MiB | B[256,4096], 4 MiB |
| Per-rank output | 4 GiB | C[262144,4096], 4 GiB | C[16384,4096], 256 MiB |
| Data scratch per rank | 192 MiB | 4 GiB | 8 GiB |
| DAG nodes per rank | 1994 | 124 | 184 |
| GEMM nodes per rank | 0 | 64 | 64 |
| NET sends per rank | 512 MiB | 256 MiB | 256 MiB |

Scratch excludes the small runtime completion-counter storage.

For GroupedAR, `REDSLICESIZE=1M` means 1,048,576 **elements**, following
the existing C++ parameter. Effective R is
`ceil(count / (group_size * num_slices * REDSLICESIZE)) = 32`.
The lane is 512 MiB, outer slice 128 MiB, inner slice 64 MiB, and each
explicit RED segment is 2 MiB after the inner split. Local scratch is
128 MiB and NET scratch is 64 MiB. It retains outer IPC/NET overlap and
inner local P2P/RED overlap.

For AG, rank r computes `C_r = concat(A_0,...,A_15) @ B_r`.
For RS, rank r computes partial `A_r @ B_r`; the sum of all partial
products is scattered by rows. The RS K partition gives each rank K/16=256.
AG weights may differ by rank, as in the existing fused API.

### Analytical configuration

Both fused examples use four row slices and one K slice. Each owner block
per row slice is 4096 rows / 64 MiB. This keeps messages large and all GEMM
dimensions aligned with the existing 64x64x16 FP32 SIMT tile. The current
kernel does not use Tensor Cores; Tensor Core peak throughput is therefore
not a suitable performance estimate.

AG first exchanges corresponding lane blocks across hosts, then distributes
both hosts' blocks through an eight-rank local ring. GEMM becomes eligible
as each block arrives. RS computes blocks, performs a local paired-block
reduce-scatter, then exchanges and reduces the two hosts' partial results.
Its next row slice's GEMM waits for the preceding slice's last local RED;
the preceding NET exchange can overlap that computation.

Each fused algorithm sends 2 GiB per host over the network, balanced as
256 MiB per NIC lane. This attains the two-host cut-volume lower bound.
Each rank also sends 3.5 GiB locally. In contrast, a contiguous sixteen-rank
ring concentrates 3.75 GiB on each host's single outgoing network edge.
This volume/lane comparison does not imply the same end-to-end speedup:
GEMM, HBM traffic, and local links can dominate.

Four row slices provide overlap opportunities without multiplying the DAG
size excessively. One K slice avoids repeated C reads/writes and a serial
accumulation chain; it also avoids splitting the RS local K=256 into smaller
GEMMs. The fused executor configuration is 128 blocks, 256 threads, and
128 GEMM workers. Each example GEMM has 4096 M/N tiles. The existing residency
calculation caps blocks by occupancy and the SM-count-minus-one budget; this
setting does not override that cap. The default 32 blocks would leave only
16 local blocks for GEMM in this RS graph after IPC/NET domain partitioning.
With a 128-block budget, its four final NET RED nodes each have 16 workers,
so the static partition reserves 64 blocks for NET and 64 for local work.
The 128 logical GEMM workers execute through that local pool; they do not
mean 128 concurrent GEMM CTAs in RS. AG has no NET RED and can use the full
permitted compute pool. GroupedAR retains its 32-block setting.

These are analytically motivated throughput settings, not a measured or
provable global optimum. GEMM workers, row slices and RED workers jointly
change compute/link contention; no GPU tuning is performed in this task.

### Translate and validate without a GPU

From the container repository, after sourcing `/workspace/wyj/env.sh`:

```bash
python3 -B flagcx/tools/examples_4g.py --output-dir algo_output/dsl_v8_4g
```

The output directory must be new. This writes three subdirectories with
16 `dag_hash_*_rank_*.json` files each, plus `manifest.json` containing
buffer capacities, algorithm identities, traffic totals and environment
settings. Construction and structural validation scale with DAG nodes;
they never allocate the 4-GiB tensors.

Run the Python tests:

```bash
python3 -B flagcx/tools/test_uni_runner.py
python3 -B -m unittest discover -s flagcx/tools -p 'test*dsl*.py' -v
```

`workflow.validate_structure()` (also available as
`dsl_validation.validate_workflow`) checks graph topology, compute-parent
limits, domains, full buffer spans and ordered peer-message signatures.
The GroupedAR tests simulate small collective inputs; the fused tests
simulate small numerical GEMMs and the production phase/stream order.
The symbolic `workflow.semantic_check()` is for small supported pure
collectives; it is not the fused GEMM checker and must not be used on
the 4-GiB examples because it tracks individual elements. `validate()` retains
its original symbolic-check meaning; fused workflows return an explicit
unsupported-semantic result directing callers to `validate_structure()`.
The older hierarchical SlicedAR example is covered for Sum; its multi-stage
Avg semantics are outside these validated examples.

After the normal library build, compile the C++ CPU validator:

```bash
g++ -std=c++17 -O2 flagcx/tools/validate_unirunner_dsl.cpp \
  -Iflagcx/include -Iflagcx/adaptor/include -Iflagcx/core/include \
  -Iflagcx/runner/include -Iflagcx/service/include \
  -Ithird-party/json/single_include -Lbuild/lib \
  -Wl,-rpath,"$PWD/build/lib" -lflagcx -o build/validate_unirunner_dsl
build/validate_unirunner_dsl algo_output/dsl_v8_4g/*/dag_hash_*_rank_*.json
```

This uses the actual C++ hash, JSON decoder, plan compiler and buffer-binding
validation with synthetic addresses. It also materializes host DAG state,
resolves workers, and checks static trigger encoding using synthetic completion
counters. Fused API key fields and generated filenames are checked against
the runtime contract. It does not allocate tensor/device memory or run an operator. Passing these checks establishes input-format and
CPU model consistency, not GPU numerical correctness or measured performance.

### Runtime selection corresponding to these files

The manifest records matching settings. Common settings are:

```bash
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_UNIRUNNER_GROUPSIZE=8
export FLAGCX_UNIRUNNER_NSLICES=4
export FLAGCX_UNIRUNNER_MAX_THREADS=256
export FLAGCX_UNIRUNNER_MAX_BLOCKS=32
export FLAGCX_UNIRUNNER_NREDSLICES=0
export FLAGCX_UNIRUNNER_REDSLICESIZE=1M
```

For GroupedAR select `FLAGCX_UNIRUNNER_USE_GROUPEDAR=1`,
`FLAGCX_UNIRUNNER_LOCAL_NSLICES=2` and set
`FLAGCX_UNIRUNNER_ALGO_PATH` to the generated `groupedar` directory.

For AllGatherGemm select `FLAGCX_UNIRUNNER_FUSED_AG_GEMM_ALGO=1`; for
GemmReduceScatter select `FLAGCX_UNIRUNNER_FUSED_GEMM_RS_ALGO=1`.
Both use `FLAGCX_UNIRUNNER_GEMM_KSLICES=1`,
`FLAGCX_UNIRUNNER_MAX_BLOCKS=128`,
`FLAGCX_UNIRUNNER_GEMM_WORKERS_PER_STEP=128` and their corresponding
generated directory as `FLAGCX_UNIRUNNER_ALGO_PATH`.
The fused algorithm selector 0 retains the original built-in ring.
Selector 1 requires a matching DSL file and fails if it cannot load one.

These settings describe future operator invocation. Translation itself
needs none of these runtime environment variables. No operator is launched
by the example generator.
