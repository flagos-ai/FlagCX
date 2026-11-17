"""Grouped allreduce DAG matching the persistent executor's two-domain pipeline."""
from __future__ import annotations
from typing import List, Tuple, Union

try:
    from .uni_runner import UniRunnerWorkflow
    from .utils import (
        AlgoType, Collective, CommDomain, DataType, RedOp,
        DEFAULT_RED_SLICE_SIZE, _datatype_size, _effective_red_slices,
        _red_op_from_value, count_from_size_bytes, parse_size_count,
    )
except ImportError:
    from uni_runner import UniRunnerWorkflow
    from utils import (
        AlgoType, Collective, CommDomain, DataType, RedOp,
        DEFAULT_RED_SLICE_SIZE, _datatype_size, _effective_red_slices,
        _red_op_from_value, count_from_size_bytes, parse_size_count,
    )

Range = Tuple[int, int]


def _partition(region: Range, parts: int, index: int) -> Range:
    offset, count = region
    base, remainder = divmod(count, parts)
    return (offset + index * base + min(index, remainder),
            base + int(index < remainder))


def _merge_dependencies(first: List[int], second: List[int]) -> List[int]:
    return list(dict.fromkeys(first + second))


def build_groupedar(
    name: str = "groupedar", world_size: int = 16,
    count: Union[int, str] = 16, group_size: int = 8,
    datatype: Union[DataType, int] = DataType.float32,
    red_op: Union[RedOp, int] = RedOp.sum,
    num_slices: int = 4, num_local_slices: int = 2,
    num_red_slices: int = 0,
    red_slice_size: Union[int, str] = DEFAULT_RED_SLICE_SIZE,
    in_place: bool = False,
) -> UniRunnerWorkflow:
    """Match the C++ GroupedAR node order, ranges and dependencies.

    ``count`` and ``red_slice_size`` are element counts, including K/M/G
    suffixes. Use ``count_from_size_bytes`` for a message size in bytes.
    A single group uses logical RED nodes split into runtime workers.
    Multiple groups retain explicit RED slices with one worker per node.
    """
    if in_place:
        raise NotImplementedError("GroupedAR DSL currently defines out-of-place buffers")
    for label, value in (("world_size", world_size), ("group_size", group_size),
                         ("num_slices", num_slices),
                         ("num_local_slices", num_local_slices)):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 0x7fffffff:
            raise ValueError(f"{label} must be an integer in [1, INT_MAX]")
    if group_size > world_size or world_size % group_size:
        raise ValueError("group_size must divide world_size")
    if (not isinstance(num_red_slices, int) or isinstance(num_red_slices, bool)
            or not 0 <= num_red_slices <= 0x7fffffff):
        raise ValueError("num_red_slices must be an integer in [0, INT_MAX]")
    count = parse_size_count(count, "count")
    red_slice_size = parse_size_count(red_slice_size, "red_slice_size")
    if count > ((1 << 64) - 1) // _datatype_size(datatype):
        raise ValueError("message bytes exceed uint64")
    if red_slice_size > (1 << 64) - 1:
        raise ValueError("red_slice_size exceeds uint64")
    requested_op = _red_op_from_value(red_op)
    if requested_op == RedOp.nop:
        raise ValueError("GroupedAR requires a reduction operator")
    n_groups = world_size // group_size
    red_slices = _effective_red_slices(
        count, group_size, num_slices, num_red_slices, red_slice_size)
    if red_slices > 0x7fffffff:
        raise ValueError("resolved RED slices exceed INT_MAX")
    logical_red_slices = 1 if n_groups == 1 else red_slices
    local_steps = (group_size - 1) * num_local_slices
    capacity = num_slices * (
        (local_steps + n_groups - 1) * (logical_red_slices + 1) + n_groups + 1) + 2
    if count and capacity > 0x7fffffff:
        raise ValueError("GroupedAR node capacity exceeds INT_MAX")
    max_chunk = (count + group_size - 1) // group_size
    max_slice = (max_chunk + num_slices - 1) // num_slices
    local_scratch = (max_chunk if n_groups == 1 else max_slice) if group_size > 1 else 0
    net_scratch = (max_slice + n_groups - 1) // n_groups if n_groups > 1 else 0

    with UniRunnerWorkflow(
        name, collective=Collective.AllReduce, world_size=world_size,
        count=count, datatype=datatype, red_op=requested_op,
        algo=AlgoType.grouped_ar, input_count=count, output_count=count,
        scratch_count=local_scratch + net_scratch,
        algorithm_config={
            "num_slices": num_slices, "num_local_slices": num_local_slices,
            "num_red_slices": red_slices,
            "pipeline_version": 3 if n_groups == 1 else 2,
            "group_size": group_size, "buffer_mode": 0,
        },
    ) as workflow:
        if count == 0:
            return workflow
        for rank in workflow.ranks():
            group, lane = divmod(rank, group_size)
            with workflow.rank(rank) as rb:
                def chunk_slice(local_lane: int, outer: int) -> Range:
                    return _partition(_partition((0, count), group_size, local_lane),
                                      num_slices, outer)

                def exchange(send_peer, tx, source, recv_peer, rx, destination,
                             parents, domain, label):
                    return [rb.p2p([
                        workflow.send(send_peer, source(tx[0]), tx[1]),
                        workflow.recv(recv_peer, destination(rx[0]), rx[1]),
                    ], parents=parents, name=label, comm_domain=domain)]

                def reduce(region, scratch_offset, parents, domain, final, label):
                    reductions = []
                    operation = (RedOp.sum if requested_op == RedOp.avg and not final
                                 else requested_op)
                    for part in range(logical_red_slices):
                        relative, amount = _partition((0, region[1]), logical_red_slices, part)
                        if amount == 0:
                            continue
                        offset = region[0] + relative
                        reductions.append(rb.red(
                            workflow.output(offset),
                            workflow.scratch(scratch_offset + relative),
                            workflow.output(offset), amount, red_op=operation,
                            parents=parents, name=f"{label}_r{part}", comm_domain=domain))
                    return reductions or list(parents)

                ready = [rb.p2p([], name="caller_stream_ready",
                                comm_domain=CommDomain.ipc)]
                own_lane = _partition((0, count), group_size, lane)
                if own_lane[1]:
                    ready = [rb.cpy(
                        workflow.input(own_lane[0]), workflow.output(own_lane[0]),
                        own_lane[1], parents=ready, name="initialize_own_lane",
                        comm_domain=CommDomain.ipc)]

                def local_reduce_scatter(first_slice, slice_span, dependencies):
                    comm_tail = list(dependencies)
                    red_tails = ([[] for _ in range(slice_span * num_local_slices)]
                                 if group_size > 1 else [])
                    for step in range(1, group_size):
                        tx_lane = (lane + step) % group_size
                        rx_lane = (lane - step) % group_size
                        for outer in range(first_slice, first_slice + slice_span):
                            own = chunk_slice(lane, outer)
                            for inner in range(num_local_slices):
                                output = _partition(own, num_local_slices, inner)
                                tx = _partition(chunk_slice(tx_lane, outer), num_local_slices, inner)
                                tail = (outer - first_slice) * num_local_slices + inner
                                scratch_offset = output[0] - (own_lane[0] if n_groups == 1 else own[0])
                                comm_tail = exchange(
                                    group * group_size + tx_lane, tx, workflow.input,
                                    group * group_size + rx_lane, (scratch_offset, output[1]),
                                    workflow.scratch,
                                    _merge_dependencies(comm_tail, red_tails[tail]),
                                    CommDomain.ipc, f"local_rs_s{outer}_step{step}_u{inner}")
                                red_tails[tail] = reduce(
                                    output, scratch_offset, comm_tail, CommDomain.ipc,
                                    n_groups == 1 and step == group_size - 1,
                                    f"local_red_s{outer}_step{step}_u{inner}")
                    done = list(comm_tail)
                    for tail in red_tails:
                        done = _merge_dependencies(done, tail)
                    if n_groups > 1:
                        done = [rb.p2p([], parents=done, name=f"local_join_s{first_slice}",
                                       comm_domain=CommDomain.ipc)]
                    return done

                def network_allreduce(outer, dependencies):
                    own = chunk_slice(lane, outer)
                    next_rank = ((group + 1) % n_groups) * group_size + lane
                    prev_rank = ((group - 1) % n_groups) * group_size + lane
                    tail = list(dependencies)
                    for step in range(n_groups - 1):
                        tx = _partition(own, n_groups, (group - step) % n_groups)
                        rx = _partition(own, n_groups, (group - step - 1) % n_groups)
                        tail = exchange(
                            next_rank, tx, workflow.output, prev_rank,
                            (local_scratch, rx[1]), workflow.scratch, tail,
                            CommDomain.net, f"net_rs_s{outer}_step{step}")
                        tail = reduce(rx, local_scratch, tail, CommDomain.net,
                                      step == n_groups - 2, f"net_red_s{outer}_step{step}")
                    for step in range(n_groups - 1):
                        tx = _partition(own, n_groups, (group - step + 1) % n_groups)
                        rx = _partition(own, n_groups, (group - step) % n_groups)
                        tail = exchange(
                            next_rank, tx, workflow.output, prev_rank, rx,
                            workflow.output, tail, CommDomain.net,
                            f"net_ag_s{outer}_step{step}")
                    return tail

                def local_allgather(outer, dependencies):
                    if group_size == 1:
                        return list(dependencies)
                    own = chunk_slice(lane, outer)
                    ops = []
                    for step in range(1, group_size):
                        tx_lane = (lane + step) % group_size
                        rx_lane = (lane - step) % group_size
                        rx = chunk_slice(rx_lane, outer)
                        ops.append(workflow.send(group * group_size + tx_lane,
                                                 workflow.output(own[0]), own[1]))
                        ops.append(workflow.recv(group * group_size + rx_lane,
                                                 workflow.output(rx[0]), rx[1]))
                    return [rb.p2p(ops, parents=dependencies,
                                   name=f"local_ag_s{outer}", comm_domain=CommDomain.ipc)]

                if n_groups == 1:
                    ipc_tail = local_reduce_scatter(0, num_slices, ready)
                    for outer in range(num_slices):
                        ipc_tail = local_allgather(outer, ipc_tail)
                else:
                    local_done = local_reduce_scatter(0, 1, ready)
                    ipc_tail = list(local_done)
                    net_tail = []
                    for outer in range(num_slices):
                        net_done = network_allreduce(
                            outer, _merge_dependencies(local_done, net_tail))
                        net_tail = net_done
                        next_local_done = []
                        if outer + 1 < num_slices:
                            next_local_done = local_reduce_scatter(outer + 1, 1, ipc_tail)
                            ipc_tail = next_local_done
                        ipc_tail = local_allgather(
                            outer, _merge_dependencies(ipc_tail, net_done))
                        local_done = next_local_done
    return workflow


def build_groupedar_4g_example() -> UniRunnerWorkflow:
    """16 ranks / 2 groups / 4 GiB per rank, with two levels of slicing."""
    return build_groupedar(
        name="groupedar_16ranks_4g", world_size=16, group_size=8,
        count=count_from_size_bytes("4G", DataType.float32),
        datatype=DataType.float32, num_slices=4, num_local_slices=2,
        num_red_slices=0, red_slice_size="1M")


__all__ = ["build_groupedar", "build_groupedar_4g_example"]
