from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

try:
    from .semantics import CollectiveSemantics, SemanticCheckResult, check_dag_semantics, make_collective_semantics
    from .utils import (
        COMM_OP_TO_INT,
        FORMAT_VERSION,
        AlgoType,
        BufferKind,
        Collective,
        DataType,
        P2pType,
        RedOp,
        _comm_op_name,
        _datatype_size,
        _datatype_value,
        _enum_value,
        _hash_combine,
        _algorithm_hash,
        _red_op_from_value,
        _red_op_value,
        _write_json_file,
    )
except ImportError:  # Keep compatibility with direct execution from flagcx/tools.
    from semantics import CollectiveSemantics, SemanticCheckResult, check_dag_semantics, make_collective_semantics
    from utils import (
        COMM_OP_TO_INT,
        FORMAT_VERSION,
        AlgoType,
        BufferKind,
        Collective,
        DataType,
        P2pType,
        RedOp,
        _comm_op_name,
        _datatype_size,
        _datatype_value,
        _enum_value,
        _hash_combine,
        _algorithm_hash,
        _red_op_from_value,
        _red_op_value,
        _write_json_file,
    )


@dataclass(frozen=True)
class BufferRef:
    buffer: str
    offset_bytes: int = 0

    @staticmethod
    def none() -> "BufferRef":
        return BufferRef(BufferKind.none.value, 0)

    def to_json(self) -> Dict[str, Union[str, int]]:
        return {"buffer": self.buffer, "offset_bytes": self.offset_bytes}


@dataclass
class P2pOp:
    type: str
    peer_rank: int
    buffer: BufferRef
    count: int
    datatype: int

    def to_json(self) -> Dict:
        return {
            "type": self.type,
            "peer_rank": self.peer_rank,
            "count": self.count,
            "datatype": self.datatype,
            "buffer": self.buffer.to_json(),
        }


@dataclass
class RedOpDesc:
    input1: BufferRef
    input2: BufferRef
    output: BufferRef
    count: int
    datatype: int
    red_op: RedOp

    def to_json(self) -> Dict:
        return {
            "input1": self.input1.to_json(),
            "input2": self.input2.to_json(),
            "output": self.output.to_json(),
            "count": self.count,
            "datatype": self.datatype,
            "red_op": self.red_op.value,
        }


@dataclass
class CpyOpDesc:
    src: BufferRef
    dst: BufferRef
    count: int
    datatype: int

    def to_json(self) -> Dict:
        return {
            "src": self.src.to_json(),
            "dst": self.dst.to_json(),
            "count": self.count,
            "datatype": self.datatype,
        }


@dataclass
class UniRunnerDagNode:
    node_idx: int
    node_type: str
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    p2p_ops: List[P2pOp] = field(default_factory=list)
    red: Optional[RedOpDesc] = None
    cpy: Optional[CpyOpDesc] = None
    name: str = ""

    def to_json(self) -> Dict:
        payload = {
            "node_idx": self.node_idx,
            "node_type": self.node_type,
            "comm_domain": 0,
            "parents": list(self.parents),
            "children": list(self.children),
        }
        if self.node_type == "p2p":
            payload["p2p_ops"] = [op.to_json() for op in self.p2p_ops]
        elif self.node_type == "red":
            payload["red"] = self.red.to_json()
        elif self.node_type == "cpy":
            payload["cpy"] = self.cpy.to_json()
        else:
            raise ValueError(f"Unsupported node type {self.node_type}")
        return payload


@dataclass
class UniRunnerDagCacheKey:
    algo: str
    comm_op: str
    count: int
    datatype: int
    red_op: int
    rank: int
    nranks: int
    root: int = -1
    format_version: int = FORMAT_VERSION
    algorithm_config: Mapping[str, int] = field(default_factory=dict)

    def algorithm_hash(self) -> int:
        return _algorithm_hash(self.algo, self.algorithm_config)

    def hash_value(self) -> int:
        hash_value = 0
        algo_hash = self.algorithm_hash()
        values = [
            self.format_version,
            algo_hash & 0xffffffff,
            algo_hash >> 32,
            COMM_OP_TO_INT[self.comm_op],
            self.count,
            self.datatype,
            self.red_op,
            self.rank,
            self.nranks,
            self.root + 1,
        ]
        for value in values:
            hash_value = _hash_combine(hash_value, int(value))
        return hash_value

    def file_name(self) -> str:
        return f"dag_hash_{self.hash_value()}_rank_{self.rank}.json"

    def to_json(self) -> Dict[str, Union[str, int]]:
        return {
            "algo_hash": str(self.algorithm_hash()),
            "comm_op": self.comm_op,
            "count": self.count,
            "datatype": self.datatype,
            "red_op": self.red_op,
            "rank": self.rank,
            "nranks": self.nranks,
            "root": self.root,
        }


@dataclass
class UniRunnerRankDag:
    rank: int
    nodes: List[UniRunnerDagNode] = field(default_factory=list)

    def add_node(self, node: UniRunnerDagNode):
        if node.node_idx != len(self.nodes):
            raise ValueError("node_idx must match append order")
        self.nodes.append(node)

    def finalize_dependencies(self):
        for node in self.nodes:
            node.parents = list(dict.fromkeys(node.parents))
            node.children = []
        for node in self.nodes:
            for parent in node.parents:
                if parent < 0 or parent >= len(self.nodes):
                    raise ValueError(f"rank {self.rank} node {node.node_idx} has invalid parent {parent}")
                self.nodes[parent].children.append(node.node_idx)
        for node in self.nodes:
            node.children = list(dict.fromkeys(node.children))

    def to_json_nodes(self) -> List[Dict]:
        return [node.to_json() for node in self.nodes]


class UniRunnerWorkflow:
    """Context-managed DSL for building per-rank uniRunner DAG templates."""

    def __init__(
        self,
        name: str,
        collective: Collective = Collective.Custom,
        world_size: int = 1,
        count: int = 0,
        datatype: Union[DataType, int] = DataType.float32,
        red_op: Union[RedOp, int] = RedOp.sum,
        root: int = -1,
        algo: Union[AlgoType, str] = AlgoType.dummy,
        comm_op: Optional[str] = None,
        input_count: Optional[int] = None,
        output_count: Optional[int] = None,
        scratch_count: int = 0,
        semantic: Optional[CollectiveSemantics] = None,
        algorithm_config: Optional[Mapping[str, int]] = None,
    ):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.name = name
        self.collective = collective
        self.world_size = world_size
        self.count = count
        self.datatype = _datatype_value(datatype)
        self.type_size = _datatype_size(datatype)
        self.red_op = _red_op_from_value(red_op)
        self.root = root
        self.algo = _enum_value(algo)
        self.algorithm_config = dict(algorithm_config or {})
        self.comm_op = _comm_op_name(collective, comm_op)
        self.input_count = count if input_count is None else input_count
        self.output_count = count if output_count is None else output_count
        self.scratch_count = scratch_count
        self.semantic = semantic
        self._ranks = {rank: UniRunnerRankDag(rank) for rank in range(world_size)}
        self._active_rank: Optional[int] = None
        self._finalized = False

    def __enter__(self) -> "UniRunnerWorkflow":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
        return False

    def rank(self, rank: int) -> "_RankBuilder":
        self._check_rank(rank)
        return _RankBuilder(self, rank)

    def ranks(self) -> range:
        return range(self.world_size)

    def input(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.input, offset)

    def output(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.output, offset)

    def scratch(self, offset: int = 0) -> BufferRef:
        return self.buffer(BufferKind.scratch, offset)

    def input_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.input.value, offset_bytes)

    def output_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.output.value, offset_bytes)

    def scratch_bytes(self, offset_bytes: int = 0) -> BufferRef:
        return BufferRef(BufferKind.scratch.value, offset_bytes)

    def buffer(self, kind: Union[BufferKind, str], offset: int = 0) -> BufferRef:
        kind_value = _enum_value(kind)
        return BufferRef(kind_value, offset * self.type_size)

    def send(self, peer_rank: int, buffer: BufferRef, count: int, datatype: Optional[int] = None) -> P2pOp:
        return P2pOp(P2pType.send.value, peer_rank, buffer, count, self.datatype if datatype is None else datatype)

    def recv(self, peer_rank: int, buffer: BufferRef, count: int, datatype: Optional[int] = None) -> P2pOp:
        return P2pOp(P2pType.recv.value, peer_rank, buffer, count, self.datatype if datatype is None else datatype)

    def p2p(
        self,
        rank: Optional[int] = None,
        ops: Optional[Sequence[P2pOp]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "p2p", parents, name)
        node.p2p_ops = list(ops or [])
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def red(
        self,
        rank: Optional[int] = None,
        input1: Optional[BufferRef] = None,
        input2: Optional[BufferRef] = None,
        output: Optional[BufferRef] = None,
        count: int = 0,
        red_op: Optional[Union[RedOp, int]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "red", parents, name)
        node.red = RedOpDesc(
            input1=input1 or BufferRef.none(),
            input2=input2 or BufferRef.none(),
            output=output or BufferRef.none(),
            count=count,
            datatype=self.datatype,
            red_op=self.red_op if red_op is None else _red_op_from_value(red_op),
        )
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def cpy(
        self,
        rank: Optional[int] = None,
        src: Optional[BufferRef] = None,
        dst: Optional[BufferRef] = None,
        count: int = 0,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        rank = self._resolve_rank(rank)
        node = self._new_node(rank, "cpy", parents, name)
        node.cpy = CpyOpDesc(src or BufferRef.none(), dst or BufferRef.none(), count, self.datatype)
        self._ranks[rank].add_node(node)
        self._finalized = False
        return node.node_idx

    def dep(self, rank: int, parent: int, child: int):
        self._check_rank(rank)
        nodes = self._ranks[rank].nodes
        if child < 0 or child >= len(nodes):
            raise ValueError(f"rank {rank} child node {child} does not exist")
        nodes[child].parents.append(parent)
        self._finalized = False

    def rank_templates(self) -> Mapping[int, UniRunnerRankDag]:
        self.finalize()
        return self._ranks

    def finalize(self):
        if self._finalized:
            return
        for rank_dag in self._ranks.values():
            rank_dag.finalize_dependencies()
        self._finalized = True

    def validate(self) -> SemanticCheckResult:
        return self.semantic_check()

    def semantic_check(self, semantic: Optional[CollectiveSemantics] = None) -> SemanticCheckResult:
        if semantic is None:
            semantic = self.semantic
        if semantic is None and self.collective != Collective.Custom:
            semantic = make_collective_semantics(
                self.collective,
                self.world_size,
                self.count,
                self.red_op,
                self.root if self.root >= 0 else 0,
            )
        if semantic is None:
            return SemanticCheckResult(False, ["custom workflows need an explicit semantic specification"])
        if semantic.scratch_count < self.scratch_count:
            semantic.scratch_count = self.scratch_count
        return check_dag_semantics(self, semantic)

    def cache_key(self, rank: int) -> UniRunnerDagCacheKey:
        self._check_rank(rank)
        return UniRunnerDagCacheKey(
            algo=self.algo,
            algorithm_config=self.algorithm_config,
            comm_op=self.comm_op,
            count=self.count,
            datatype=self.datatype,
            red_op=_red_op_value(self.red_op),
            rank=rank,
            nranks=self.world_size,
            root=self.root,
        )

    def runtime_entry(self, rank: int) -> Dict:
        self.finalize()
        key = self.cache_key(rank)
        nodes = self._ranks[rank].to_json_nodes()
        return {
            "format_version": FORMAT_VERSION,
            "hash": str(key.hash_value()),
            "key": key.to_json(),
            "dag": {"num_nodes": len(nodes), "nodes": nodes},
        }

    def runtime_cache_json(self, ranks: Optional[Iterable[int]] = None) -> Dict:
        selected_ranks = list(self.ranks() if ranks is None else ranks)
        return {
            "format_version": FORMAT_VERSION,
            "address_model": "buffer_kind+offset_bytes",
            "buffer_kinds": ["input", "output", "scratch"],
            "entries": [self.runtime_entry(rank) for rank in selected_ranks],
        }

    def dag_json(self) -> Dict:
        self.finalize()
        return {
            "name": self.name,
            "collective": self.collective.name,
            "world_size": self.world_size,
            "count": self.count,
            "datatype": self.datatype,
            "type_size": self.type_size,
            "algo": self.algo,
            "comm_op": self.comm_op,
            "ranks": {
                str(rank): {"nodes": rank_dag.to_json_nodes()}
                for rank, rank_dag in self._ranks.items()
            },
        }

    def write_rank_files(self, output_dir: str) -> List[str]:
        self.finalize()
        paths = []
        os.makedirs(output_dir, exist_ok=True)
        for rank in self.ranks():
            key = self.cache_key(rank)
            path = os.path.join(output_dir, key.file_name())
            _write_json_file(path, self.runtime_entry(rank))
            paths.append(path)
        return paths

    def write_runtime_cache(self, path: str, ranks: Optional[Iterable[int]] = None):
        _write_json_file(path, self.runtime_cache_json(ranks))

    def write_dag_json(self, path: str):
        _write_json_file(path, self.dag_json())

    def _new_node(
        self,
        rank: int,
        node_type: str,
        parents: Optional[Sequence[int]],
        name: str,
    ) -> UniRunnerDagNode:
        self._check_rank(rank)
        return UniRunnerDagNode(
            node_idx=len(self._ranks[rank].nodes),
            node_type=node_type,
            parents=list(parents or []),
            name=name,
        )

    def _resolve_rank(self, rank: Optional[int]) -> int:
        if rank is None:
            if self._active_rank is None:
                raise ValueError("rank must be supplied outside a workflow.rank(...) block")
            rank = self._active_rank
        self._check_rank(rank)
        return rank

    def _check_rank(self, rank: int):
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside [0, {self.world_size})")


class _RankBuilder:
    def __init__(self, workflow: UniRunnerWorkflow, rank: int):
        self.workflow = workflow
        self.rank = rank

    def __enter__(self) -> "_RankBuilder":
        if self.workflow._active_rank is not None:
            raise RuntimeError("nested rank builders are not supported")
        self.workflow._active_rank = self.rank
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workflow._active_rank = None
        return False

    def p2p(self, ops: Sequence[P2pOp], parents: Optional[Sequence[int]] = None, name: str = "") -> int:
        return self.workflow.p2p(self.rank, ops, parents, name)

    def red(
        self,
        input1: BufferRef,
        input2: BufferRef,
        output: BufferRef,
        count: int,
        red_op: Optional[Union[RedOp, int]] = None,
        parents: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> int:
        return self.workflow.red(self.rank, input1, input2, output, count, red_op, parents, name)

    def cpy(self, src: BufferRef, dst: BufferRef, count: int, parents: Optional[Sequence[int]] = None, name: str = "") -> int:
        return self.workflow.cpy(self.rank, src, dst, count, parents, name)

    def dep(self, parent: int, child: int):
        self.workflow.dep(self.rank, parent, child)


unirunnerworkflow = UniRunnerWorkflow


__all__ = [
    "BufferRef",
    "P2pOp",
    "UniRunnerDagCacheKey",
    "UniRunnerDagNode",
    "UniRunnerWorkflow",
    "unirunnerworkflow",
]
