from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

TOOLS_DIR = os.path.dirname(__file__)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "algo_output")

from semantics import CollectiveSemantics
from uni_runner import P2pOp, UniRunnerWorkflow
from utils import (
    DEFAULT_RED_SLICE_SIZE,
    FORMAT_VERSION,
    AlgoType,
    Collective,
    DataType,
    RedOp,
    _effective_red_slices,
    _rank_chunk,
    _rank_slice,
    _red_op_from_value,
    _slice_chunk,
    count_from_size_bytes,
    parse_env_size_count,
    parse_size_count,
)


DEFAULT_HIERARCHICAL_GROUP_SIZE = 8
HIERARCHICAL_GROUP_SIZE_ENV = "UNIRUNNER_GROUPSIZE"


def build_groupedag(
    name: str = "groupedag",
    world_size: int = 4,
    count: int = 8,
    group_size: int = 2,
    datatype: Union[DataType, int] = DataType.float32,
) -> UniRunnerWorkflow:
    if group_size <= 0 or world_size % group_size != 0:
        raise ValueError("group_size must divide world_size")
    with UniRunnerWorkflow(
        name,
        collective=Collective.AllGather,
        world_size=world_size,
        count=count,
        datatype=datatype,
        red_op=RedOp.nop,
        algo=AlgoType.grouped_ag,
        algorithm_config={"group_size": group_size, "buffer_mode": 0},
        input_count=count,
        output_count=count * world_size,
    ) as workflow:
        n_groups = world_size // group_size
        group_chunk_count = count * group_size

        for rank in workflow.ranks():
            group_idx = rank // group_size
            loc_rank = rank % group_size
            local_base = group_idx * group_chunk_count
            previous_p2p: Optional[int] = None

            with workflow.rank(rank) as rb:
                if world_size == 1:
                    rb.cpy(workflow.input(0), workflow.output(0), count, name="self_copy")
                    continue

                for step in range(n_groups):
                    ops: List[P2pOp] = []
                    is_last_step = step == n_groups - 1
                    for peer_index in range(group_size - 1):
                        loc_send_peer = (loc_rank + peer_index + 1) % group_size
                        loc_recv_peer = (loc_rank - peer_index - 1 + group_size) % group_size
                        send_ref = (
                            workflow.input(0)
                            if step == 0
                            else workflow.output(local_base + loc_rank * count)
                        )
                        recv_ref = workflow.output(local_base + loc_recv_peer * count)
                        ops.append(workflow.send(group_idx * group_size + loc_send_peer, send_ref, count))
                        ops.append(workflow.recv(group_idx * group_size + loc_recv_peer, recv_ref, count))

                    if not is_last_step:
                        send_group_idx = (group_idx + step + 1) % n_groups
                        recv_group_idx = (group_idx - step - 1 + n_groups) % n_groups
                        send_peer = send_group_idx * group_size + loc_rank
                        recv_peer = recv_group_idx * group_size + loc_rank
                        send_ref = (
                            workflow.input(0)
                            if step == 0
                            else workflow.output(local_base + loc_rank * count)
                        )
                        ops.append(workflow.send(send_peer, send_ref, count))
                        ops.append(workflow.recv(recv_peer, workflow.output(recv_peer * count), count))
                        local_base = recv_group_idx * group_chunk_count

                    parents = [] if previous_p2p is None else [previous_p2p]
                    previous_p2p = rb.p2p(ops, parents=parents, name=f"grouped_ag_step_{step}")

                rb.cpy(workflow.input(0), workflow.output(rank * count), count, name="local_chunk_copy")
    return workflow


def build_slicedar(
    name: str = "slicedar",
    world_size: int = 4,
    count: Union[int, str] = 16,
    datatype: Union[DataType, int] = DataType.float32,
    red_op: Union[RedOp, int] = RedOp.sum,
    num_slices: int = 2,
    num_red_slices: int = 0,
    red_slice_size: Union[int, str] = DEFAULT_RED_SLICE_SIZE,
) -> UniRunnerWorkflow:
    count = parse_size_count(count, "count")
    red_slice_size = parse_size_count(red_slice_size, "red_slice_size")
    red_op_enum = _red_op_from_value(red_op)
    effective_red_slices = _effective_red_slices(
        count, world_size, num_slices, num_red_slices, red_slice_size
    )
    with UniRunnerWorkflow(
        name,
        collective=Collective.AllReduce,
        world_size=world_size,
        count=count,
        datatype=datatype,
        red_op=red_op_enum,
        algo=AlgoType.sliced_ar,
        algorithm_config={"num_slices": num_slices,
                          "num_red_slices": effective_red_slices, "buffer_mode": 0},
        input_count=count,
        output_count=count,
    ) as workflow:
        if world_size == 1:
            for rank in workflow.ranks():
                with workflow.rank(rank) as rb:
                    rb.cpy(workflow.input(0), workflow.output(0), count, name="self_copy")
            return workflow

        nodes_per_slice = (effective_red_slices + 2) * (world_size - 1)
        num_nodes = num_slices * nodes_per_slice

        for rank in workflow.ranks():
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size
            with workflow.rank(rank) as rb:
                global_node_idx = 0

                for slice_idx in range(num_slices):
                    for step in range(world_size - 1):
                        p2p_node_idx = global_node_idx
                        tx_chunk = (rank - step + world_size) % world_size
                        rx_chunk = (rank - step - 1 + world_size) % world_size
                        tx_offset, tx_count = _rank_slice(count, world_size, num_slices, tx_chunk, slice_idx)
                        rx_offset, rx_count = _rank_slice(count, world_size, num_slices, rx_chunk, slice_idx)

                        send_ref = workflow.input(tx_offset) if step == 0 else workflow.output(tx_offset)
                        recv_ref = workflow.output(rx_offset)
                        p2p_parents: List[int] = []
                        if p2p_node_idx != 0:
                            parent_idx = p2p_node_idx - nodes_per_slice
                            if step > 0 and slice_idx == 0:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (step - 1) * (1 + effective_red_slices)
                                )
                            p2p_parents.append(parent_idx)
                            if step > 0:
                                p2p_parents.extend(
                                    p2p_node_idx - effective_red_slices + red_slice
                                    for red_slice in range(effective_red_slices)
                                )
                        rb.p2p(
                            [
                                workflow.send(next_rank, send_ref, tx_count),
                                workflow.recv(prev_rank, recv_ref, rx_count),
                            ],
                            parents=p2p_parents,
                            name=f"rs_p2p_s{slice_idx}_step{step}",
                        )
                        global_node_idx += 1

                        red_base_offset, red_base_count = rx_offset, rx_count
                        red_start_idx = global_node_idx
                        for red_slice in range(effective_red_slices):
                            sub_offset, sub_count = _slice_chunk(red_base_count, effective_red_slices, red_slice)
                            red_offset = red_base_offset + sub_offset
                            rb.red(
                                workflow.output(red_offset),
                                workflow.input(red_offset),
                                workflow.output(red_offset),
                                sub_count,
                                red_op=red_op_enum,
                                parents=[p2p_node_idx],
                                name=f"rs_red_s{slice_idx}_step{step}_r{red_slice}",
                            )
                            global_node_idx += 1
                        assert red_start_idx + effective_red_slices == global_node_idx

                    for step in range(world_size - 1):
                        p2p_node_idx = global_node_idx
                        tx_chunk = (rank - step + 1 + world_size) % world_size
                        rx_chunk = (rank - step + world_size) % world_size
                        tx_offset, tx_count = _rank_slice(count, world_size, num_slices, tx_chunk, slice_idx)
                        rx_offset, rx_count = _rank_slice(count, world_size, num_slices, rx_chunk, slice_idx)

                        p2p_parents = []
                        parent_idx = p2p_node_idx - nodes_per_slice
                        if slice_idx == 0:
                            if step == 0:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (world_size - 2) * (1 + effective_red_slices)
                                )
                            else:
                                parent_idx = (
                                    (num_slices - 1) * nodes_per_slice
                                    + (world_size - 1) * (1 + effective_red_slices)
                                    + step
                                    - 1
                                )
                        p2p_parents.append(parent_idx)
                        if step == 0:
                            p2p_parents.extend(
                                p2p_node_idx - effective_red_slices + red_slice
                                for red_slice in range(effective_red_slices)
                            )

                        rb.p2p(
                            [
                                workflow.send(next_rank, workflow.output(tx_offset), tx_count),
                                workflow.recv(prev_rank, workflow.output(rx_offset), rx_count),
                            ],
                            parents=p2p_parents,
                            name=f"ag_p2p_s{slice_idx}_step{step}",
                        )
                        global_node_idx += 1

                if global_node_idx != num_nodes:
                    raise AssertionError(f"rank {rank} built {global_node_idx} nodes, expected {num_nodes}")
    return workflow


def build_hierarchical_slicedar(
    name: str = "hierarchical_slicedar",
    world_size: int = 16,
    count: Union[int, str] = 16,
    group_size: Optional[int] = None,
    datatype: Union[DataType, int] = DataType.float32,
    red_op: Union[RedOp, int] = RedOp.sum,
    num_slices: int = 2,
    num_red_slices: int = 0,
    red_slice_size: Union[int, str] = DEFAULT_RED_SLICE_SIZE,
) -> UniRunnerWorkflow:
    count = parse_size_count(count, "count")
    red_slice_size = parse_size_count(red_slice_size, "red_slice_size")
    group_size = _resolve_hierarchical_group_size(world_size, group_size)
    n_groups = world_size // group_size
    red_op_enum = _red_op_from_value(red_op)
    effective_red_slices = _effective_red_slices(
        count, world_size, num_slices, num_red_slices, red_slice_size
    )

    with UniRunnerWorkflow(
        name,
        collective=Collective.AllReduce,
        world_size=world_size,
        count=count,
        datatype=datatype,
        red_op=red_op_enum,
        algo=AlgoType.hierarchical_sliced_ar,
        algorithm_config={"num_slices": num_slices,
                          "num_red_slices": effective_red_slices,
                          "group_size": group_size, "buffer_mode": 0},
        input_count=count,
        output_count=count,
        scratch_count=count,
    ) as workflow:
        if world_size == 1:
            with workflow.rank(0) as rb:
                rb.cpy(workflow.input(0), workflow.output(0), count, name="self_copy")
            return workflow

        for rank in workflow.ranks():
            group_idx = rank // group_size
            loc_rank = rank % group_size
            next_local_rank = group_idx * group_size + (loc_rank + 1) % group_size
            prev_local_rank = group_idx * group_size + (loc_rank - 1 + group_size) % group_size
            next_group_rank = ((group_idx + 1) % n_groups) * group_size + loc_rank
            prev_group_rank = ((group_idx - 1 + n_groups) % n_groups) * group_size + loc_rank
            owner_chunk = (loc_rank + 1) % group_size if group_size > 1 else 0
            owner_offset, owner_count = _rank_chunk(count, group_size, owner_chunk)

            with workflow.rank(rank) as rb:
                phase_tail: List[int] = []

                if group_size == 1:
                    phase_tail = [
                        rb.cpy(workflow.input(0), workflow.output(0), count, name="local_copy")
                    ]
                else:
                    phase_tail = _add_local_reduce_scatter(
                        workflow,
                        rb,
                        rank=rank,
                        loc_rank=loc_rank,
                        group_size=group_size,
                        next_rank=next_local_rank,
                        prev_rank=prev_local_rank,
                        count=count,
                        num_slices=num_slices,
                        effective_red_slices=effective_red_slices,
                        red_op=red_op_enum,
                        parents=phase_tail,
                    )

                if n_groups > 1:
                    phase_tail = _add_inter_node_allreduce_for_chunk(
                        workflow,
                        rb,
                        group_idx=group_idx,
                        n_groups=n_groups,
                        next_rank=next_group_rank,
                        prev_rank=prev_group_rank,
                        chunk_offset=owner_offset,
                        chunk_count=owner_count,
                        num_slices=num_slices,
                        effective_red_slices=effective_red_slices,
                        red_op=red_op_enum,
                        parents=phase_tail,
                    )

                if group_size > 1:
                    phase_tail = _add_local_allgather(
                        workflow,
                        rb,
                        loc_rank=loc_rank,
                        group_size=group_size,
                        next_rank=next_local_rank,
                        prev_rank=prev_local_rank,
                        count=count,
                        num_slices=num_slices,
                        parents=phase_tail,
                    )
    return workflow


def _resolve_hierarchical_group_size(world_size: int, group_size: Optional[int]) -> int:
    if group_size is None:
        group_size_text = os.environ.get(HIERARCHICAL_GROUP_SIZE_ENV)
        group_size = int(group_size_text) if group_size_text else DEFAULT_HIERARCHICAL_GROUP_SIZE
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size > world_size or world_size % group_size != 0:
        raise ValueError("group_size must divide world_size and must not exceed world_size")
    return group_size


def _add_local_reduce_scatter(
    workflow: UniRunnerWorkflow,
    rb,
    rank: int,
    loc_rank: int,
    group_size: int,
    next_rank: int,
    prev_rank: int,
    count: int,
    num_slices: int,
    effective_red_slices: int,
    red_op: RedOp,
    parents: List[int],
) -> List[int]:
    phase_tail = list(parents)
    for slice_idx in range(num_slices):
        for step in range(group_size - 1):
            tx_chunk = (loc_rank - step + group_size) % group_size
            rx_chunk = (loc_rank - step - 1 + group_size) % group_size
            tx_offset, tx_count = _rank_slice(count, group_size, num_slices, tx_chunk, slice_idx)
            rx_offset, rx_count = _rank_slice(count, group_size, num_slices, rx_chunk, slice_idx)
            send_ref = workflow.input(tx_offset) if step == 0 else workflow.output(tx_offset)
            p2p = rb.p2p(
                [
                    workflow.send(next_rank, send_ref, tx_count),
                    workflow.recv(prev_rank, workflow.output(rx_offset), rx_count),
                ],
                parents=phase_tail,
                name=f"local_rs_rank{rank}_s{slice_idx}_step{step}",
            )
            phase_tail = _add_reduce_slices(
                workflow,
                rb,
                input1_kind="output",
                input2_kind="input",
                output_kind="output",
                offset=rx_offset,
                count=rx_count,
                effective_red_slices=effective_red_slices,
                red_op=red_op,
                parents=[p2p],
                name=f"local_rs_red_rank{rank}_s{slice_idx}_step{step}",
            )
    return phase_tail


def _add_inter_node_allreduce_for_chunk(
    workflow: UniRunnerWorkflow,
    rb,
    group_idx: int,
    n_groups: int,
    next_rank: int,
    prev_rank: int,
    chunk_offset: int,
    chunk_count: int,
    num_slices: int,
    effective_red_slices: int,
    red_op: RedOp,
    parents: List[int],
) -> List[int]:
    phase_tail = list(parents)
    for slice_idx in range(num_slices):
        for step in range(n_groups - 1):
            tx_chunk = (group_idx - step + n_groups) % n_groups
            rx_chunk = (group_idx - step - 1 + n_groups) % n_groups
            tx_rel_offset, tx_count = _rank_slice(chunk_count, n_groups, num_slices, tx_chunk, slice_idx)
            rx_rel_offset, rx_count = _rank_slice(chunk_count, n_groups, num_slices, rx_chunk, slice_idx)
            tx_offset = chunk_offset + tx_rel_offset
            rx_offset = chunk_offset + rx_rel_offset
            p2p = rb.p2p(
                [
                    workflow.send(next_rank, workflow.output(tx_offset), tx_count),
                    workflow.recv(prev_rank, workflow.scratch(rx_offset), rx_count),
                ],
                parents=phase_tail,
                name=f"inter_rs_s{slice_idx}_step{step}",
            )
            phase_tail = _add_reduce_slices(
                workflow,
                rb,
                input1_kind="output",
                input2_kind="scratch",
                output_kind="output",
                offset=rx_offset,
                count=rx_count,
                effective_red_slices=effective_red_slices,
                red_op=red_op,
                parents=[p2p],
                name=f"inter_rs_red_s{slice_idx}_step{step}",
            )

        for step in range(n_groups - 1):
            tx_chunk = (group_idx - step + 1 + n_groups) % n_groups
            rx_chunk = (group_idx - step + n_groups) % n_groups
            tx_rel_offset, tx_count = _rank_slice(chunk_count, n_groups, num_slices, tx_chunk, slice_idx)
            rx_rel_offset, rx_count = _rank_slice(chunk_count, n_groups, num_slices, rx_chunk, slice_idx)
            phase_tail = [
                rb.p2p(
                    [
                        workflow.send(next_rank, workflow.output(chunk_offset + tx_rel_offset), tx_count),
                        workflow.recv(prev_rank, workflow.output(chunk_offset + rx_rel_offset), rx_count),
                    ],
                    parents=phase_tail,
                    name=f"inter_ag_s{slice_idx}_step{step}",
                )
            ]
    return phase_tail


def _add_local_allgather(
    workflow: UniRunnerWorkflow,
    rb,
    loc_rank: int,
    group_size: int,
    next_rank: int,
    prev_rank: int,
    count: int,
    num_slices: int,
    parents: List[int],
) -> List[int]:
    phase_tail = list(parents)
    for slice_idx in range(num_slices):
        for step in range(group_size - 1):
            tx_chunk = (loc_rank - step + 1 + group_size) % group_size
            rx_chunk = (loc_rank - step + group_size) % group_size
            tx_offset, tx_count = _rank_slice(count, group_size, num_slices, tx_chunk, slice_idx)
            rx_offset, rx_count = _rank_slice(count, group_size, num_slices, rx_chunk, slice_idx)
            phase_tail = [
                rb.p2p(
                    [
                        workflow.send(next_rank, workflow.output(tx_offset), tx_count),
                        workflow.recv(prev_rank, workflow.output(rx_offset), rx_count),
                    ],
                    parents=phase_tail,
                    name=f"local_ag_s{slice_idx}_step{step}",
                )
            ]
    return phase_tail


def _add_reduce_slices(
    workflow: UniRunnerWorkflow,
    rb,
    input1_kind: str,
    input2_kind: str,
    output_kind: str,
    offset: int,
    count: int,
    effective_red_slices: int,
    red_op: RedOp,
    parents: List[int],
    name: str,
) -> List[int]:
    red_nodes: List[int] = []
    for red_slice in range(effective_red_slices):
        red_rel_offset, red_count = _slice_chunk(count, effective_red_slices, red_slice)
        red_offset = offset + red_rel_offset
        red_nodes.append(
            rb.red(
                workflow.buffer(input1_kind, red_offset),
                workflow.buffer(input2_kind, red_offset),
                workflow.buffer(output_kind, red_offset),
                red_count,
                red_op=red_op,
                parents=parents,
                name=f"{name}_r{red_slice}",
            )
        )
    return red_nodes


def build_groupedag_example():
    return build_groupedag(
        name="groupedag_example",
        world_size=4,
        count=8,
        group_size=2,
        datatype=DataType.float32,
    )


def build_slicedar_example():
    return build_slicedar(
        name="slicedar_example",
        world_size=4,
        count=17,
        datatype=DataType.float32,
        red_op=RedOp.sum,
        num_slices=2,
        num_red_slices=2,
    )


def build_hierarchical_slicedar_example():
    return build_hierarchical_slicedar(
        name="hierarchical_slicedar_example",
        world_size=16,
        count=17,
        group_size=DEFAULT_HIERARCHICAL_GROUP_SIZE,
        datatype=DataType.float32,
        red_op=RedOp.sum,
        num_slices=2,
        num_red_slices=2,
    )


def write_examples(output_dir: str) -> Tuple[str, str, str]:
    groupedag_dir = os.path.join(output_dir, "groupedag")
    slicedar_dir = os.path.join(output_dir, "slicedar")
    hierarchical_slicedar_dir = os.path.join(output_dir, "hierarchical_slicedar")

    groupedag = build_groupedag_example()
    groupedag.semantic_check().raise_for_error()
    groupedag.write_rank_files(groupedag_dir)
    groupedag.write_dag_json(os.path.join(groupedag_dir, "groupedag_dag.json"))

    slicedar = build_slicedar_example()
    slicedar.semantic_check().raise_for_error()
    slicedar.write_rank_files(slicedar_dir)
    slicedar.write_dag_json(os.path.join(slicedar_dir, "slicedar_dag.json"))

    hierarchical_slicedar = build_hierarchical_slicedar_example()
    hierarchical_slicedar.semantic_check().raise_for_error()
    hierarchical_slicedar.write_rank_files(hierarchical_slicedar_dir)
    hierarchical_slicedar.write_dag_json(
        os.path.join(hierarchical_slicedar_dir, "hierarchical_slicedar_dag.json")
    )

    return groupedag_dir, slicedar_dir, hierarchical_slicedar_dir


class UniRunnerDslTest(unittest.TestCase):
    LEGACY_RUNTIME_FIELDS = {
        "group_size",
        "num_slices",
        "num_red_slices",
        "red_slice_size",
        "nthreads",
        "input_output_aliased",
        "input_scratch_aliased",
        "output_scratch_aliased",
    }

    def test_size_count_parser_accepts_binary_suffixes(self):
        self.assertEqual(parse_size_count("1K"), 1024)
        self.assertEqual(parse_size_count("1M"), 1024 * 1024)
        self.assertEqual(parse_size_count("1G"), 1024 * 1024 * 1024)
        self.assertEqual(parse_size_count("0x10"), 16)
        self.assertEqual(count_from_size_bytes("1M", DataType.float32), 262144)

        with patch.dict(os.environ, {"UNIRUNNER_TEST_SIZE": "2M"}):
            self.assertEqual(
                parse_env_size_count("UNIRUNNER_TEST_SIZE"), 2 * 1024 * 1024
            )
        self.assertEqual(parse_env_size_count("UNIRUNNER_TEST_SIZE", "4K"), 4096)

    def test_hierarchical_slicedar_red_slice_size_suffix(self):
        workflow = build_hierarchical_slicedar(
            name="hierarchical_slicedar_red_slice_size_suffix_test",
            world_size=8,
            count=count_from_size_bytes("128M", DataType.float32),
            group_size=2,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=0,
            red_slice_size="1M",
        )

        entry = workflow.runtime_entry(rank=0)
        red_nodes = [
            node for node in entry["dag"]["nodes"] if node["node_type"] == "red"
        ]
        self.assertEqual(len(red_nodes), 16)
        self.assertTrue(
            all("nthreads" not in node["red"] for node in red_nodes)
        )

    def test_groupedag_example_semantics_and_runtime_json(self):
        workflow = build_groupedag_example()

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        self.assertEqual(workflow.algo, "grouped_ag")
        self.assertEqual(entry["key"]["algo_hash"], str(workflow.cache_key(0).algorithm_hash()))
        self.assertEqual(entry["format_version"], FORMAT_VERSION)
        self.assertEqual(entry["key"]["comm_op"], "all_gather")
        self.assertEqual(entry["dag"]["num_nodes"], 3)
        self.assertEqual(entry["dag"]["nodes"][-1]["node_type"], "cpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = workflow.write_rank_files(tmpdir)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(os.path.basename(path).startswith("dag_hash_") for path in paths))
            with open(paths[0], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["hash"], entry["hash"])
            self.assertFalse(self.LEGACY_RUNTIME_FIELDS & set(payload["key"]))
            self.assertEqual(payload["dag"]["nodes"][0]["node_type"], "p2p")

    def test_slicedar_example_semantics_and_runtime_json(self):
        workflow = build_slicedar_example()

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        node_types = [node["node_type"] for node in entry["dag"]["nodes"]]
        self.assertEqual(workflow.algo, "sliced_ar")
        self.assertEqual(entry["key"]["algo_hash"], str(workflow.cache_key(0).algorithm_hash()))
        self.assertEqual(entry["format_version"], FORMAT_VERSION)
        self.assertEqual(entry["key"]["comm_op"], "all_reduce")
        self.assertFalse(self.LEGACY_RUNTIME_FIELDS & set(entry["key"]))
        self.assertIn("p2p", node_types)
        self.assertIn("red", node_types)
        self.assertEqual(
            sum(1 for node in entry["dag"]["nodes"] if node["node_type"] == "red"),
            12,
        )
        self.assertEqual(entry["dag"]["num_nodes"], 24)

    def test_hierarchical_slicedar_semantics_and_runtime_json(self):
        workflow = build_hierarchical_slicedar(
            name="hierarchical_slicedar_test",
            world_size=4,
            count=17,
            group_size=2,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=2,
        )

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        first_p2p_ops = entry["dag"]["nodes"][0]["p2p_ops"]
        node_types = [node["node_type"] for node in entry["dag"]["nodes"]]
        self.assertEqual(workflow.algo, "hierarchical_sliced_ar")
        self.assertEqual(entry["key"]["algo_hash"], str(workflow.cache_key(0).algorithm_hash()))
        self.assertEqual(entry["format_version"], FORMAT_VERSION)
        self.assertEqual(entry["key"]["comm_op"], "all_reduce")
        self.assertFalse(self.LEGACY_RUNTIME_FIELDS & set(entry["key"]))
        self.assertEqual(first_p2p_ops[0]["peer_rank"], 1)
        scratch_recvs = [
            op
            for node in entry["dag"]["nodes"]
            if node["node_type"] == "p2p"
            for op in node["p2p_ops"]
            if op["type"] == "recv" and op["buffer"]["buffer"] == "scratch"
        ]
        self.assertTrue(scratch_recvs)
        self.assertIn("p2p", node_types)
        self.assertIn("red", node_types)
        self.assertEqual(entry["dag"]["num_nodes"], 16)

    def test_hierarchical_slicedar_default_group_size_env(self):
        with patch.dict(os.environ, {HIERARCHICAL_GROUP_SIZE_ENV: "8"}):
            workflow = build_hierarchical_slicedar(
                name="hierarchical_slicedar_default_group_size_test",
                world_size=16,
                count=17,
                group_size=None,
                datatype=DataType.float32,
                red_op=RedOp.sum,
                num_slices=2,
                num_red_slices=2,
            )

        workflow.semantic_check().raise_for_error()
        entry = workflow.runtime_entry(rank=0)
        self.assertEqual(workflow.algo, "hierarchical_sliced_ar")
        self.assertEqual(entry["key"]["algo_hash"], str(workflow.cache_key(0).algorithm_hash()))
        self.assertEqual(entry["format_version"], FORMAT_VERSION)
        self.assertFalse(self.LEGACY_RUNTIME_FIELDS & set(entry["key"]))
        self.assertEqual(entry["dag"]["num_nodes"], 64)

    def test_cache_hash_tracks_normalized_dag_configuration(self):
        baseline = build_hierarchical_slicedar(
            name="hierarchical_slicedar_hash_baseline",
            world_size=16,
            count=17,
            group_size=2,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=2,
            red_slice_size=65536,
        )
        tuning_only = build_hierarchical_slicedar(
            name="hierarchical_slicedar_hash_tuning_only",
            world_size=16,
            count=17,
            group_size=4,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=4,
            num_red_slices=3,
            red_slice_size=1048576,
        )
        count_only = build_hierarchical_slicedar(
            name="hierarchical_slicedar_hash_count_only",
            world_size=16,
            count=33,
            group_size=2,
            datatype=DataType.float32,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=2,
            red_slice_size=65536,
        )
        datatype_only = build_hierarchical_slicedar(
            name="hierarchical_slicedar_hash_datatype_only",
            world_size=16,
            count=17,
            group_size=2,
            datatype=DataType.float64,
            red_op=RedOp.sum,
            num_slices=2,
            num_red_slices=2,
        )
        red_op_only = build_hierarchical_slicedar(
            name="hierarchical_slicedar_hash_red_op_only",
            world_size=16,
            count=17,
            group_size=2,
            datatype=DataType.float32,
            red_op=RedOp.max,
            num_slices=2,
            num_red_slices=2,
        )

        self.assertNotEqual(
            baseline.cache_key(0).hash_value(),
            tuning_only.cache_key(0).hash_value(),
        )
        self.assertNotEqual(
            baseline.cache_key(0).hash_value(),
            count_only.cache_key(0).hash_value(),
        )
        self.assertNotEqual(
            baseline.cache_key(0).hash_value(),
            datatype_only.cache_key(0).hash_value(),
        )
        self.assertNotEqual(
            baseline.cache_key(0).hash_value(),
            red_op_only.cache_key(0).hash_value(),
        )
        self.assertNotEqual(
            baseline.cache_key(0).hash_value(),
            baseline.cache_key(1).hash_value(),
        )

    def test_custom_semantics_for_identity_copy(self):
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

        with UniRunnerWorkflow(
            "identity_copy",
            collective=Collective.Custom,
            world_size=2,
            count=2,
            datatype=DataType.float32,
            algo=AlgoType.dummy,
            comm_op="noop",
            input_count=2,
            output_count=2,
            semantic=semantic,
        ) as workflow:
            for rank in workflow.ranks():
                with workflow.rank(rank) as rb:
                    rb.cpy(workflow.input(0), workflow.output(0), 2)

        workflow.semantic_check().raise_for_error()
        self.assertEqual(workflow.algo, "dummy")
        self.assertEqual(workflow.runtime_entry(0)["key"]["algo_hash"],
                         str(workflow.cache_key(0).algorithm_hash()))


def main():
    parser = argparse.ArgumentParser(description="Test or generate uniRunner DSL examples.")
    parser.add_argument(
        "--generate-examples",
        action="store_true",
        help="Generate groupedag, slicedar, and hierarchical_slicedar example runtime cache files instead of running tests.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated example files should be written.",
    )
    args, unittest_args = parser.parse_known_args()

    if args.generate_examples:
        groupedag_dir, slicedar_dir, hierarchical_slicedar_dir = write_examples(args.output_dir)
        print(f"groupedag runtime cache: {groupedag_dir}")
        print(f"slicedar runtime cache: {slicedar_dir}")
        print(f"hierarchical_slicedar runtime cache: {hierarchical_slicedar_dir}")
        return

    unittest.main(argv=[sys.argv[0]] + unittest_args)


if __name__ == "__main__":
    main()
