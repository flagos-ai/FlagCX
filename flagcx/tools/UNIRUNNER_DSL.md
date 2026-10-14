# uniRunner DSL Design and Usage

This document describes the Python DSL in `flagcx/tools/uni_runner.py` for
building uniRunner algorithm DAGs. The DSL follows the same high-level idea as
MSCCL-style workflow builders: an algorithm is described in Python inside a
`with UniRunnerWorkflow(...)` block, and the block adds typed operation nodes
plus dependencies. The generated JSON can be written in the cache format that
the existing uniRunner runtime loads from `FLAGCX_UNIRUNNER_ALGO_PATH`.

## Goals

- Build one DAG template per rank.
- Support the uniRunner node kinds already implemented in C++: `p2p`, `red`,
  and `cpy`.
- Address buffers by runtime buffer kind plus byte offset: `input`, `output`,
  and `scratch`.
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
  "format_version": 7,
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

workflow.semantic_check().raise_for_error()
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

The supplied builders emit communication domain `0` (IPC/control stream) for
all nodes, retaining their original serialized communication order. This does
not select the physical transport: the runtime still chooses local IPC or
network transport for each peer. GroupedAR's independently selected IPC/NET
pipeline remains a separate C++ algorithm.

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
