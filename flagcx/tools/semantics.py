from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .utils import Collective, RedOp
except ImportError:  # Keep compatibility with the existing tools/*.py scripts.
    from utils import Collective, RedOp

Atom = int
DataRef = Tuple[RedOp, List[Atom]]
DataCond = List[List[DataRef]]
OptionalDataCond = List[List[Optional[DataRef]]]


@dataclass(frozen=True)
class DataValue:
    """Symbolic payload carried by the semantic simulator."""

    red_op: RedOp
    atoms: Tuple[Atom, ...]

    @staticmethod
    def empty() -> "DataValue":
        return DataValue(RedOp.nop, ())

    @staticmethod
    def single(atom: Atom) -> "DataValue":
        return DataValue(RedOp.nop, (atom,))

    @staticmethod
    def from_data_ref(ref: DataRef) -> "DataValue":
        return DataValue(ref[0], tuple(sorted(ref[1])))

    def is_empty(self) -> bool:
        return len(self.atoms) == 0

    def to_data_ref(self) -> DataRef:
        return self.red_op, list(self.atoms)


def _normal_red_op(values: Sequence[DataValue], op: RedOp) -> RedOp:
    return RedOp.nop if len(values) <= 1 else op


def reduce_values(values: Sequence[DataValue], op: RedOp) -> DataValue:
    atoms: List[Atom] = []
    non_empty_values = [value for value in values if not value.is_empty()]
    for value in non_empty_values:
        atoms.extend(value.atoms)
    if not atoms:
        return DataValue.empty()
    return DataValue(_normal_red_op(non_empty_values, op), tuple(sorted(atoms)))


def combine_values(lhs: DataValue, rhs: DataValue, op: RedOp) -> DataValue:
    return reduce_values([lhs, rhs], op)


@dataclass
class CollectiveSemantics:
    """Expected symbolic input and output state for one collective."""

    name: str
    collective: Collective
    world_size: int
    input_count: int
    output_count: int
    scratch_count: int = 0
    red_op: RedOp = RedOp.sum
    root: int = -1
    input_data: List[List[DataValue]] = field(default_factory=list)
    expected_output: List[List[Optional[DataValue]]] = field(default_factory=list)

    def __post_init__(self):
        if not self.input_data:
            self.input_data = [
                [DataValue.single(self.atom(rank, index)) for index in range(self.input_count)]
                for rank in range(self.world_size)
            ]
        if not self.expected_output:
            self.expected_output = [
                [None for _ in range(self.output_count)] for _ in range(self.world_size)
            ]

    def atom(self, rank: int, index: int) -> Atom:
        return rank * max(1, self.input_count) + index

    def expect(self, rank: int, offset: int, values: Sequence[DataValue]):
        for index, value in enumerate(values):
            self.expected_output[rank][offset + index] = value

    def expected_data_cond(self) -> OptionalDataCond:
        result: OptionalDataCond = []
        for rank_values in self.expected_output:
            result.append(
                [value.to_data_ref() if value is not None else None for value in rank_values]
            )
        return result

    @staticmethod
    def custom(
        name: str,
        world_size: int,
        input_count: int,
        output_count: int,
        expected_output: Optional[Sequence[Sequence[Optional[DataRef]]]] = None,
        input_data: Optional[Sequence[Sequence[DataRef]]] = None,
        scratch_count: int = 0,
    ) -> "CollectiveSemantics":
        semantic = CollectiveSemantics(
            name=name,
            collective=Collective.Custom,
            world_size=world_size,
            input_count=input_count,
            output_count=output_count,
            scratch_count=scratch_count,
        )
        if input_data is not None:
            semantic.input_data = [
                [DataValue.from_data_ref(ref) for ref in rank_values]
                for rank_values in input_data
            ]
        if expected_output is not None:
            semantic.expected_output = [
                [
                    DataValue.from_data_ref(ref) if ref is not None else None
                    for ref in rank_values
                ]
                for rank_values in expected_output
            ]
        return semantic


CustomSemanticChecker = Callable[["DagSemanticContext"], "SemanticCheckResult"]
_CUSTOM_CHECKERS: Dict[str, CustomSemanticChecker] = {}


def register_custom_semantic(name: str, checker: CustomSemanticChecker):
    _CUSTOM_CHECKERS[name] = checker


def get_custom_semantic(name: str) -> Optional[CustomSemanticChecker]:
    return _CUSTOM_CHECKERS.get(name)


def make_collective_semantics(
    collective: Collective,
    world_size: int,
    count: int,
    red_op: RedOp = RedOp.sum,
    root: int = 0,
) -> CollectiveSemantics:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")
    if root < 0 or root >= world_size:
        root = 0

    if collective == Collective.AllReduce:
        semantic = CollectiveSemantics(
            "allreduce",
            collective,
            world_size,
            input_count=count,
            output_count=count,
            red_op=red_op,
        )
        for rank in range(world_size):
            for index in range(count):
                semantic.expected_output[rank][index] = reduce_values(
                    [semantic.input_data[src][index] for src in range(world_size)], red_op
                )
        return semantic

    if collective == Collective.AllGather:
        semantic = CollectiveSemantics(
            "allgather",
            collective,
            world_size,
            input_count=count,
            output_count=count * world_size,
            red_op=RedOp.nop,
        )
        for rank in range(world_size):
            for src in range(world_size):
                semantic.expect(rank, src * count, semantic.input_data[src])
        return semantic

    if collective == Collective.ReduceScatter:
        semantic = CollectiveSemantics(
            "reducescatter",
            collective,
            world_size,
            input_count=count * world_size,
            output_count=count,
            red_op=red_op,
        )
        for rank in range(world_size):
            for index in range(count):
                input_index = rank * count + index
                semantic.expected_output[rank][index] = reduce_values(
                    [
                        semantic.input_data[src][input_index]
                        for src in range(world_size)
                    ],
                    red_op,
                )
        return semantic

    if collective == Collective.Broadcast:
        semantic = CollectiveSemantics(
            "broadcast",
            collective,
            world_size,
            input_count=count,
            output_count=count,
            red_op=RedOp.nop,
            root=root,
        )
        for rank in range(world_size):
            semantic.expect(rank, 0, semantic.input_data[root])
        return semantic

    if collective == Collective.Reduce:
        semantic = CollectiveSemantics(
            "reduce",
            collective,
            world_size,
            input_count=count,
            output_count=count,
            red_op=red_op,
            root=root,
        )
        for index in range(count):
            semantic.expected_output[root][index] = reduce_values(
                [semantic.input_data[src][index] for src in range(world_size)], red_op
            )
        return semantic

    if collective == Collective.Gather:
        semantic = CollectiveSemantics(
            "gather",
            collective,
            world_size,
            input_count=count,
            output_count=count * world_size,
            red_op=RedOp.nop,
            root=root,
        )
        for src in range(world_size):
            semantic.expect(root, src * count, semantic.input_data[src])
        return semantic

    if collective == Collective.Scatter:
        semantic = CollectiveSemantics(
            "scatter",
            collective,
            world_size,
            input_count=count * world_size,
            output_count=count,
            red_op=RedOp.nop,
            root=root,
        )
        for rank in range(world_size):
            semantic.expect(rank, 0, semantic.input_data[root][rank * count : (rank + 1) * count])
        return semantic

    if collective == Collective.AlltoAll:
        semantic = CollectiveSemantics(
            "alltoall",
            collective,
            world_size,
            input_count=count * world_size,
            output_count=count * world_size,
            red_op=RedOp.nop,
        )
        for dst in range(world_size):
            for src in range(world_size):
                begin = dst * count
                end = begin + count
                semantic.expect(dst, src * count, semantic.input_data[src][begin:end])
        return semantic

    raise ValueError(f"No built-in semantic rule for collective {collective}")


@dataclass
class SemanticCheckResult:
    ok: bool
    errors: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.ok = False
        self.errors.append(message)

    def merge(self, other: "SemanticCheckResult"):
        if not other.ok:
            self.ok = False
            self.errors.extend(other.errors)

    def raise_for_error(self):
        if not self.ok:
            raise AssertionError("\n".join(self.errors))


@dataclass
class DagSemanticContext:
    workflow: object
    semantic: CollectiveSemantics
    outputs: List[List[DataValue]]


class DagSemanticChecker:
    """Validates and symbolically executes a uniRunner DAG collection."""

    def __init__(self, workflow: object, semantic: CollectiveSemantics):
        self.workflow = workflow
        self.semantic = semantic
        self.type_size = int(getattr(workflow, "type_size", 1))
        self.rank_templates: Mapping[int, object] = workflow.rank_templates()

    def check(self) -> SemanticCheckResult:
        result = SemanticCheckResult(ok=True)
        result.merge(self._check_structure())
        if not result.ok:
            return result

        simulated = self._simulate(result)
        if simulated is None:
            return result

        context = DagSemanticContext(self.workflow, self.semantic, simulated)
        custom_checker = get_custom_semantic(self.semantic.name)
        if custom_checker is not None:
            result.merge(custom_checker(context))
        else:
            result.merge(self._compare_outputs(simulated))
        return result

    def _check_structure(self) -> SemanticCheckResult:
        result = SemanticCheckResult(ok=True)
        expected_ranks = set(range(self.semantic.world_size))
        if set(self.rank_templates.keys()) != expected_ranks:
            result.add_error(
                f"workflow ranks {sorted(self.rank_templates.keys())} do not match "
                f"semantic ranks {sorted(expected_ranks)}"
            )
            return result

        for rank, template in self.rank_templates.items():
            nodes = list(template.nodes)
            if [node.node_idx for node in nodes] != list(range(len(nodes))):
                result.add_error(f"rank {rank} nodes must be contiguous and ordered")
                continue

            parent_edges = set()
            for node in nodes:
                for parent in node.parents:
                    if parent < 0 or parent >= len(nodes):
                        result.add_error(
                            f"rank {rank} node {node.node_idx} has invalid parent {parent}"
                        )
                    elif parent == node.node_idx:
                        result.add_error(f"rank {rank} node {node.node_idx} depends on itself")
                    parent_edges.add((parent, node.node_idx))
                if len(node.parents) != len(set(node.parents)):
                    result.add_error(f"rank {rank} node {node.node_idx} has duplicate parents")
                if len(node.children) != len(set(node.children)):
                    result.add_error(f"rank {rank} node {node.node_idx} has duplicate children")

            child_edges = set()
            for node in nodes:
                for child in node.children:
                    if child < 0 or child >= len(nodes):
                        result.add_error(
                            f"rank {rank} node {node.node_idx} has invalid child {child}"
                        )
                    elif child == node.node_idx:
                        result.add_error(f"rank {rank} node {node.node_idx} points to itself")
                    child_edges.add((node.node_idx, child))
            if parent_edges != child_edges:
                result.add_error(f"rank {rank} parent/child dependency edges differ")
            if self._has_cycle(nodes):
                result.add_error(f"rank {rank} dependency graph contains a cycle")
        return result

    @staticmethod
    def _has_cycle(nodes: Sequence[object]) -> bool:
        state = [0 for _ in nodes]

        def visit(node_idx: int) -> bool:
            if state[node_idx] == 1:
                return True
            if state[node_idx] == 2:
                return False
            state[node_idx] = 1
            for child in nodes[node_idx].children:
                if visit(child):
                    return True
            state[node_idx] = 2
            return False

        return any(visit(index) for index in range(len(nodes)) if state[index] == 0)

    def _new_buffers(self) -> List[Dict[str, List[DataValue]]]:
        buffers: List[Dict[str, List[DataValue]]] = []
        for rank in range(self.semantic.world_size):
            buffers.append(
                {
                    "input": list(self.semantic.input_data[rank]),
                    "output": [DataValue.empty() for _ in range(self.semantic.output_count)],
                    "scratch": [DataValue.empty() for _ in range(self.semantic.scratch_count)],
                }
            )
        return buffers

    def _simulate(self, result: SemanticCheckResult) -> Optional[List[List[DataValue]]]:
        buffers = self._new_buffers()
        executed = {
            rank: [False for _ in template.nodes]
            for rank, template in self.rank_templates.items()
        }
        total_nodes = sum(len(template.nodes) for template in self.rank_templates.values())
        done_nodes = 0

        while done_nodes < total_nodes and result.ok:
            progress = False
            for rank, template in self.rank_templates.items():
                for node in template.nodes:
                    if executed[rank][node.node_idx] or node.node_type == "p2p":
                        continue
                    if not all(executed[rank][parent] for parent in node.parents):
                        continue
                    if node.node_type == "cpy":
                        self._execute_cpy(rank, node, buffers, result)
                    elif node.node_type == "red":
                        self._execute_red(rank, node, buffers, result)
                    else:
                        result.add_error(f"rank {rank} node {node.node_idx} has unknown type {node.node_type}")
                        return None
                    executed[rank][node.node_idx] = True
                    done_nodes += 1
                    progress = True

            p2p_nodes = self._ready_p2p_nodes(executed)
            executable = self._select_executable_p2p_nodes(p2p_nodes)
            if executable:
                self._execute_p2p_nodes(executable, buffers, result)
                for rank, node in executable:
                    executed[rank][node.node_idx] = True
                    done_nodes += 1
                progress = True

            if not progress:
                pending = []
                for rank, template in self.rank_templates.items():
                    for node in template.nodes:
                        if not executed[rank][node.node_idx]:
                            pending.append(f"r{rank}:n{node.node_idx}:{node.node_type}")
                result.add_error("semantic simulation made no progress; pending " + ", ".join(pending[:16]))
                break

        if not result.ok:
            return None
        return [rank_buffers["output"] for rank_buffers in buffers]

    def _ready_p2p_nodes(self, executed: Mapping[int, Sequence[bool]]) -> List[Tuple[int, object]]:
        ready = []
        for rank, template in self.rank_templates.items():
            for node in template.nodes:
                if node.node_type != "p2p" or executed[rank][node.node_idx]:
                    continue
                if all(executed[rank][parent] for parent in node.parents):
                    ready.append((rank, node))
        return ready

    @staticmethod
    def _select_executable_p2p_nodes(ready: Sequence[Tuple[int, object]]) -> List[Tuple[int, object]]:
        executable = set((rank, node.node_idx) for rank, node in ready)
        node_by_key = {(rank, node.node_idx): (rank, node) for rank, node in ready}
        changed = True
        while changed:
            changed = False
            send_counts: Dict[Tuple[int, int, int, int], int] = {}
            for rank, node_idx in executable:
                _, node = node_by_key[(rank, node_idx)]
                for op in node.p2p_ops:
                    if op.type == "send":
                        key = (rank, op.peer_rank, op.count, op.datatype)
                        send_counts[key] = send_counts.get(key, 0) + 1
            for rank, node_idx in list(executable):
                _, node = node_by_key[(rank, node_idx)]
                needed: Dict[Tuple[int, int, int, int], int] = {}
                for op in node.p2p_ops:
                    if op.type == "recv":
                        key = (op.peer_rank, rank, op.count, op.datatype)
                        needed[key] = needed.get(key, 0) + 1
                if any(send_counts.get(key, 0) < count for key, count in needed.items()):
                    executable.remove((rank, node_idx))
                    changed = True
        return [node_by_key[key] for key in sorted(executable)]

    def _execute_p2p_nodes(
        self,
        nodes: Sequence[Tuple[int, object]],
        buffers: List[Dict[str, List[DataValue]]],
        result: SemanticCheckResult,
    ):
        sends: Dict[Tuple[int, int, int, int], List[List[DataValue]]] = {}
        for rank, node in nodes:
            for op in node.p2p_ops:
                if op.type != "send":
                    continue
                values = self._read_range(buffers[rank], op.buffer, op.count, result)
                key = (rank, op.peer_rank, op.count, op.datatype)
                sends.setdefault(key, []).append(values)

        for rank, node in nodes:
            for op in node.p2p_ops:
                if op.type != "recv":
                    continue
                key = (op.peer_rank, rank, op.count, op.datatype)
                if key not in sends or not sends[key]:
                    result.add_error(
                        f"rank {rank} node {node.node_idx} recv from {op.peer_rank} has no matching send"
                    )
                    return
                self._write_range(buffers[rank], op.buffer, sends[key].pop(0), result)

    def _execute_cpy(
        self,
        rank: int,
        node: object,
        buffers: List[Dict[str, List[DataValue]]],
        result: SemanticCheckResult,
    ):
        values = self._read_range(buffers[rank], node.cpy.src, node.cpy.count, result)
        self._write_range(buffers[rank], node.cpy.dst, values, result)

    def _execute_red(
        self,
        rank: int,
        node: object,
        buffers: List[Dict[str, List[DataValue]]],
        result: SemanticCheckResult,
    ):
        lhs_values = self._read_range(buffers[rank], node.red.input1, node.red.count, result)
        rhs_values = self._read_range(buffers[rank], node.red.input2, node.red.count, result)
        out_values = [
            combine_values(lhs, rhs, node.red.red_op)
            for lhs, rhs in zip(lhs_values, rhs_values)
        ]
        self._write_range(buffers[rank], node.red.output, out_values, result)

    def _read_range(
        self,
        rank_buffers: Mapping[str, List[DataValue]],
        ref: object,
        count: int,
        result: SemanticCheckResult,
    ) -> List[DataValue]:
        kind, offset = self._resolve_ref(ref, result)
        values = rank_buffers.get(kind)
        if values is None or offset < 0 or offset + count > len(values):
            result.add_error(f"read {kind}[{offset}:{offset + count}] is out of range")
            return [DataValue.empty() for _ in range(max(0, count))]
        return list(values[offset : offset + count])

    def _write_range(
        self,
        rank_buffers: Mapping[str, List[DataValue]],
        ref: object,
        values: Sequence[DataValue],
        result: SemanticCheckResult,
    ):
        kind, offset = self._resolve_ref(ref, result)
        target = rank_buffers.get(kind)
        if target is None or offset < 0 or offset + len(values) > len(target):
            result.add_error(f"write {kind}[{offset}:{offset + len(values)}] is out of range")
            return
        target[offset : offset + len(values)] = list(values)

    def _resolve_ref(self, ref: object, result: SemanticCheckResult) -> Tuple[str, int]:
        if ref.offset_bytes % self.type_size != 0:
            result.add_error(
                f"buffer reference {ref.buffer}:{ref.offset_bytes} is not aligned to datatype size {self.type_size}"
            )
        return ref.buffer, ref.offset_bytes // self.type_size

    def _compare_outputs(self, outputs: Sequence[Sequence[DataValue]]) -> SemanticCheckResult:
        result = SemanticCheckResult(ok=True)
        expected = self.semantic.expected_output
        for rank in range(self.semantic.world_size):
            for index, expected_value in enumerate(expected[rank]):
                if expected_value is None:
                    continue
                actual = outputs[rank][index]
                if actual != expected_value:
                    result.add_error(
                        f"rank {rank} output[{index}] expected {expected_value.to_data_ref()} "
                        f"but got {actual.to_data_ref()}"
                    )
        return result


def check_dag_semantics(workflow: object, semantic: Optional[CollectiveSemantics] = None) -> SemanticCheckResult:
    if semantic is None:
        semantic = getattr(workflow, "semantic", None)
    if semantic is None:
        semantic = make_collective_semantics(
            getattr(workflow, "collective"),
            getattr(workflow, "world_size"),
            getattr(workflow, "count"),
            getattr(workflow, "red_op", RedOp.sum),
            getattr(workflow, "root", 0),
        )
    return DagSemanticChecker(workflow, semantic).check()
