"""CPU-only structural validation for persistent-executor DAG input files.

Validation scales with DAG nodes and transfers, never with tensor element count.
Small numerical/semantic tests remain necessary to check mathematical semantics.
"""
from collections import Counter, defaultdict, deque

try:
    from .utils import DataType, SIZE_T_MASK, _datatype_size
except ImportError:
    from utils import DataType, SIZE_T_MASK, _datatype_size

UINT32_MAX = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1


def _integer(value, name, maximum=SIZE_T_MASK, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def execution_order(nodes):
    """Match the runtime compiler's FIFO, phase-draining topological order."""
    phase_of = {"p2p": 0, "cpy": 0, "red": 1, "gemm": 1}
    ready = [deque(), deque()]
    remaining = []
    for index, node in enumerate(nodes):
        if node.node_idx != index or node.node_type not in phase_of:
            raise ValueError(f"unsupported or noncontiguous node {index}")
        remaining.append(len(node.parents))
        if not node.parents:
            ready[phase_of[node.node_type]].append(index)
    order = []
    while len(order) < len(nodes):
        before = len(order)
        for phase in range(2):
            while ready[phase]:
                index = ready[phase].popleft()
                order.append(index)
                for child in nodes[index].children:
                    remaining[child] -= 1
                    if remaining[child] == 0:
                        ready[phase_of[nodes[child].node_type]].append(child)
        if len(order) == before:
            raise ValueError("DAG contains a dependency cycle")
    return order


def buffer_capacities(workflow):
    return {
        name: _integer(getattr(workflow, f"{name}_count", 0),
                       f"{name}_count") * workflow.type_size
        for name in ("input", "output", "scratch", "weight")
    }


def _check_ref(ref, span, capacities, label, write=False, alignment=1):
    _integer(span, f"{label} span")
    if ref.buffer not in capacities:
        raise ValueError(f"{label}: unsupported buffer {ref.buffer}")
    offset = _integer(ref.offset_bytes, f"{label} offset", INT64_MAX)
    if offset % alignment:
        raise ValueError(f"{label}: misaligned buffer offset")
    capacity = capacities[ref.buffer]
    if offset > capacity or span > capacity - offset:
        raise ValueError(f"{label}: access exceeds {ref.buffer} capacity")
    if write and ref.buffer in ("input", "weight"):
        raise ValueError(f"{label}: input and weight bindings are read-only")


def _matrix_span(rows, columns, stride, label):
    _integer(rows, f"{label} rows", UINT32_MAX, 1)
    _integer(columns, f"{label} columns", UINT32_MAX, 1)
    _integer(stride, f"{label} stride", UINT32_MAX, columns)
    return ((rows - 1) * stride + columns) * 4


def validate_workflow(workflow):
    """Validate structure, buffer spans, and ordered peer message signatures.

    This is not a GPU execution or a proof of arbitrary mathematical semantics.
    No tensor storage, device memory, or GPU context is allocated.
    """
    _integer(workflow.world_size, "world_size", INT32_MAX, 1)
    _integer(workflow.count, "count")
    workflow.finalize()
    capacities = buffer_capacities(workflow)
    for kind, capacity in capacities.items():
        _integer(capacity, f"{kind} bytes", INT64_MAX)
    sends, receives = defaultdict(list), defaultdict(list)
    rank_summaries = []
    grouped = workflow.algo in ("grouped_ar", "all_gather_gemm",
                                 "gemm_reduce_scatter")
    group_size = workflow.algorithm_config.get("group_size", 0)
    if grouped and (group_size <= 0 or workflow.world_size % group_size):
        raise ValueError("group_size must divide world_size")
    for rank, template in workflow.rank_templates().items():
        nodes = template.nodes
        _integer(len(nodes), "node count", INT32_MAX)
        for index, node in enumerate(nodes):
            if node.node_idx != index:
                raise ValueError("nodes must have consecutive indices")
            if node.comm_domain not in (0, 1):
                raise ValueError("unknown communication domain")
            if node.comm_domain == 1 and node.node_type not in ("p2p", "red"):
                raise ValueError("NET domain supports P2P/RED only")
            if len(node.parents) != len(set(node.parents)):
                raise ValueError("duplicate dependency")
            for parent in node.parents:
                _integer(parent, "parent", len(nodes) - 1)
                if parent == index or index not in nodes[parent].children:
                    raise ValueError("invalid parent/child dependency")
            for child in node.children:
                _integer(child, "child", len(nodes) - 1)
                if index not in nodes[child].parents:
                    raise ValueError("invalid child/parent dependency")
            if node.node_type in ("red", "gemm") and len(node.parents) > 1:
                raise ValueError("persistent compute nodes accept at most one parent; use a P2P join")
        order = execution_order(nodes)
        sent = [0, 0]
        received = [0, 0]
        types = Counter(node.node_type for node in nodes)
        for index in order:
            node = nodes[index]
            label = f"rank {rank} node {index}"
            if node.node_type == "p2p":
                peer_ops = set()
                for op in node.p2p_ops:
                    peer_op = (op.type, op.peer_rank)
                    if peer_op in peer_ops:
                        raise ValueError("P2P groups allow at most one send and one recv per peer; split repeated peer operations into ordered nodes")
                    peer_ops.add(peer_op)
                    _integer(op.peer_rank, "peer", workflow.world_size - 1)
                    count = _integer(op.count, "P2P count")
                    size = _datatype_size(op.datatype)
                    if op.type not in ("send", "recv"):
                        raise ValueError("host P2P supports send/recv only")
                    _check_ref(op.buffer, count * size, capacities, label,
                               write=op.type == "recv", alignment=size)
                    if grouped:
                        is_local = rank // group_size == op.peer_rank // group_size
                        if is_local != (node.comm_domain == 0):
                            raise ValueError(f"{label}: peer and communication domain disagree")
                    signature = (count, op.datatype)
                    if op.type == "send":
                        sends[(rank, op.peer_rank, node.comm_domain)].append(signature)
                        sent[node.comm_domain] += count * size
                    else:
                        receives[(op.peer_rank, rank, node.comm_domain)].append(signature)
                        received[node.comm_domain] += count * size
            elif node.node_type == "cpy":
                desc = node.cpy
                count = _integer(desc.count, "copy count")
                size = _datatype_size(desc.datatype)
                _check_ref(desc.src, count * size, capacities, label, alignment=size)
                _check_ref(desc.dst, count * size, capacities, label, write=True, alignment=size)
            elif node.node_type == "red":
                desc = node.red
                count = _integer(desc.count, "RED count", UINT32_MAX)
                size = _datatype_size(desc.datatype)
                if desc.red_op.value not in range(5):
                    raise ValueError("RED needs an actual reduction operation")
                for ref in (desc.input1, desc.input2):
                    _check_ref(ref, count * size, capacities, label, alignment=size)
                _check_ref(desc.output, count * size, capacities, label,
                           write=True, alignment=size)
            else:
                desc = node.gemm
                if desc.datatype != DataType.float32.value or desc.accumulate not in (0, 1):
                    raise ValueError("GEMM supports float32 and accumulate=0/1")
                _check_ref(desc.a, _matrix_span(desc.m, desc.k, desc.lda, "A"),
                           capacities, label, alignment=4)
                _check_ref(desc.b, _matrix_span(desc.k, desc.n, desc.ldb, "B"),
                           capacities, label, alignment=4)
                _check_ref(desc.c, _matrix_span(desc.m, desc.n, desc.ldc, "C"),
                           capacities, label, write=True, alignment=4)
        rank_summaries.append({
            "rank": rank, "nodes": len(nodes), "node_types": dict(types),
            "ipc_send_bytes": sent[0], "net_send_bytes": sent[1],
            "ipc_recv_bytes": received[0], "net_recv_bytes": received[1],
        })
    if dict(sends) != dict(receives):
        for key in sends.keys() | receives.keys():
            if sends.get(key) != receives.get(key):
                raise ValueError(f"ordered send/recv signatures differ for {key}")
    return {
        "name": workflow.name, "world_size": workflow.world_size,
        "algorithm": workflow.algo, "algorithm_config": workflow.algorithm_config,
        "bindings": capacities, "ranks": rank_summaries,
        "validation": "CPU graph, bounds, compute contract, ordered peer messages",
    }
