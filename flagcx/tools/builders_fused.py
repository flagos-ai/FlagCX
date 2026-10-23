"""Topology-aware fused DAGs for two equally sized GPU hosts.

GEMM/RED nodes are logical operations; the persistent executor assigns
workers, completion counters and physical triggers at invocation time.
"""
try:
    from .uni_runner import UniRunnerWorkflow
    from .utils import AlgoType, Collective, DataType, RedOp, _slice_chunk
except ImportError:
    from uni_runner import UniRunnerWorkflow
    from utils import AlgoType, Collective, DataType, RedOp, _slice_chunk


def _workflow(name, collective, algo, world_size, m, n, k,
              group_size, num_slices, num_k_slices, rs):
    values = (world_size, m, n, k, group_size, num_slices, num_k_slices)
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in values):
        raise ValueError("dimensions, ranks and slices must be positive integers")
    if world_size != 2 * group_size:
        raise ValueError("grouped fused builders require exactly two equal groups")
    if m % world_size or (rs and k % world_size):
        raise ValueError("global M and ReduceScatter K must be divisible by world_size")
    rows, local_k = m // world_size, k // world_size if rs else k
    if num_slices > rows or num_k_slices > local_k:
        raise ValueError("slice counts must not exceed matrix dimensions")
    if max(m if rs else rows, n, local_k) > 0xffffffff:
        raise ValueError("GEMM dimensions must fit uint32")
    if rows * (n if rs else k) > 0xffffffff:
        raise ValueError("per-rank communication element count must fit uint32")
    if max(m * local_k, local_k * n, m * n * (2 if rs else 1)) * 4 > (1 << 64) - 1:
        raise ValueError("matrix byte sizes must fit uint64")
    wf = UniRunnerWorkflow(
        name=name, collective=collective, world_size=world_size,
        count=rows * (n if rs else k), datatype=DataType.float32,
        red_op=RedOp.sum if rs else RedOp.nop, algo=algo,
        input_count=m * local_k if rs else rows * k,
        output_count=rows * n if rs else m * n,
        weight_count=local_k * n, scratch_count=2 * m * n if rs else m * k,
        algorithm_config={
            "m": m, "n": n, "k": k, "num_slices": num_slices,
            "num_k_slices": num_k_slices, "group_size": group_size,
            "pipeline_version": 2, "buffer_mode": 0,
        },
    )
    return wf, rows, local_k


def _gemm_chain(wf, rb, a, c, rows, n, local_k, k_slices, parents, name):
    """Serial K accumulation; each logical node distributes M/N tiles."""
    current = list(parents)
    last = None
    for part in range(k_slices):
        off, length = _slice_chunk(local_k, k_slices, part)
        a_part = type(a)(a.buffer, a.offset_bytes + off * 4)
        last = rb.gemm(
            a=a_part, b=wf.weight(off * n), c=c,
            m=rows, n=n, k=length, lda=local_k, ldb=n, ldc=n,
            accumulate=part != 0, parents=current,
            name=f"{name}.k{part}", comm_domain=0,
        )
        current = [last]
    return last


def build_allgather_gemm(world_size, m, n, k, group_size=8,
                         num_slices=4, num_k_slices=1):
    """NIC-pair exchange, local paired-block AllGather, and per-block GEMM.

    M/K are global dimensions. Rank r owns A[M/P,K] and its B[K,N],
    and writes C[M,N]. Corresponding NIC peers occupy the same local rank
    in the two contiguous groups.
    """
    wf, rows, _ = _workflow(
        "grouped_allgather_gemm", Collective.AllGatherGemm,
        AlgoType.all_gather_gemm, world_size, m, n, k, group_size,
        num_slices, num_k_slices, False)
    for rank in wf.ranks():
        host, lane = divmod(rank, group_size)
        peer = (1 - host) * group_size + lane
        nxt = host * group_size + (lane + 1) % group_size
        prv = host * group_size + (lane - 1) % group_size
        with wf.rank(rank) as rb:
            for part in range(num_slices):
                row_off, row_count = _slice_chunk(rows, num_slices, part)
                count, off = row_count * k, row_off * k
                net = rb.p2p(
                    ops=[wf.send(peer, wf.input(off), count),
                         wf.recv(peer, wf.scratch(peer * rows * k + off), count)],
                    name=f"s{part}.net_exchange", comm_domain=1)
                _gemm_chain(wf, rb, wf.input(off),
                            wf.output((rank * rows + row_off) * n),
                            row_count, n, k, num_k_slices, [],
                            f"s{part}.owner{rank}")
                _gemm_chain(wf, rb, wf.scratch(peer * rows * k + off),
                            wf.output((peer * rows + row_off) * n),
                            row_count, n, k, num_k_slices, [net],
                            f"s{part}.owner{peer}")
                previous = net
                for step in range(group_size - 1):
                    tx, rx = (lane - step) % group_size, (lane - step - 1) % group_size
                    for source_host in (host, 1 - host):
                        tx_owner, rx_owner = source_host * group_size + tx, source_host * group_size + rx
                        send = (wf.input(off) if tx_owner == rank else
                                wf.scratch(tx_owner * rows * k + off))
                        recv = wf.scratch(rx_owner * rows * k + off)
                        # A proxy group may contain only one message per peer/direction.
                        p2p = rb.p2p(
                            ops=[wf.send(nxt, send, count), wf.recv(prv, recv, count)],
                            parents=[previous],
                            name=f"s{part}.local_ag{step}.host{source_host}", comm_domain=0)
                        _gemm_chain(
                            wf, rb, recv,
                            wf.output((rx_owner * rows + row_off) * n),
                            row_count, n, k, num_k_slices, [p2p],
                            f"s{part}.owner{rx_owner}")
                        previous = p2p
    wf.finalize()
    return wf


def build_gemm_reducescatter(world_size, m, n, k, group_size=8,
                             num_slices=4, num_k_slices=1):
    """GEMM, paired local ReduceScatter, and one exchange per NIC lane.

    K is global: rank r owns A[M,K/P], B[K/P,N], and output C[M/P,N].
    Scratch's first M*N elements hold GEMM partials; the second half holds
    local reductions. Once local reduction consumes an own-rank partial,
    its first-half region is reused for the network receive.
    """
    wf, rows, local_k = _workflow(
        "grouped_gemm_reducescatter", Collective.GemmReduceScatter,
        AlgoType.gemm_reduce_scatter, world_size, m, n, k, group_size,
        num_slices, num_k_slices, True)
    full = m * n
    for rank in wf.ranks():
        host, lane = divmod(rank, group_size)
        peer = (1 - host) * group_size + lane
        nxt = host * group_size + (lane + 1) % group_size
        prv = host * group_size + (lane - 1) % group_size
        with wf.rank(rank) as rb:
            prior_slice_gate = None
            for part in range(num_slices):
                row_off, row_count = _slice_chunk(rows, num_slices, part)
                count, off = row_count * n, row_off * n
                gemms = {}
                # Put the first ring send/receive blocks first in the static queue.
                for owner_lane in ((lane - 1 - i) % group_size for i in range(group_size)):
                    for owner_host in (host, 1 - host):
                        owner = owner_host * group_size + owner_lane
                        row = owner * rows + row_off
                        parents = [] if prior_slice_gate is None else [prior_slice_gate]
                        gemms[owner] = _gemm_chain(
                            wf, rb, wf.input(row * local_k), wf.scratch(row * n),
                            row_count, n, local_k, num_k_slices, parents,
                            f"s{part}.owner{owner}")
                previous_reds = []
                for step in range(group_size - 1):
                    tx = (lane - step - 1) % group_size
                    rx = (lane - step - 2) % group_size
                    transfers, parents = [], list(previous_reds)
                    for owner_host in (host, 1 - host):
                        tx_owner = owner_host * group_size + tx
                        rx_owner = owner_host * group_size + rx
                        tx_off, rx_off = tx_owner * rows * n + off, rx_owner * rows * n + off
                        if step == 0:
                            parents.append(gemms[tx_owner])
                        parents.append(gemms[rx_owner])
                        send = wf.scratch(tx_off + (0 if step == 0 else full))
                        recv = wf.scratch(full + rx_off)
                        transfers.append((owner_host, rx_owner, send, recv))
                    previous_reds = []
                    for owner_host, owner, send, reduced in transfers:
                        # Keep one message per peer/direction in each proxy group.
                        p2p = rb.p2p(
                            ops=[wf.send(nxt, send, count), wf.recv(prv, reduced, count)],
                            parents=parents,
                            name=f"s{part}.local_rs{step}.host{owner_host}", comm_domain=0)
                        parents = [p2p]
                        owner_off = owner * rows * n + off
                        previous_reds.append(rb.red(
                            input1=reduced, input2=wf.scratch(owner_off),
                            output=reduced, count=count, parents=[p2p],
                            name=f"s{part}.local_sum{step}.owner{owner}", comm_domain=0))
                if group_size == 1:
                    local_base = 0
                    previous_reds = [gemms[rank], gemms[peer]]
                else:
                    local_base = full
                own_off, peer_off = rank * rows * n + off, peer * rows * n + off
                recv_off = full + own_off if group_size == 1 else own_off
                net = rb.p2p(
                    ops=[wf.send(peer, wf.scratch(local_base + peer_off), count),
                         wf.recv(peer, wf.scratch(recv_off), count)],
                    parents=previous_reds, name=f"s{part}.net_exchange", comm_domain=1)
                rb.red(input1=wf.scratch(recv_off),
                       input2=wf.scratch(local_base + own_off),
                       output=wf.output(off), count=count, parents=[net],
                       name=f"s{part}.net_sum", comm_domain=1)
                # One parent fits a persistent compute trigger. This controls
                # static queue order while allowing prior NET/next GEMM overlap.
                prior_slice_gate = previous_reds[-1]
    wf.finalize()
    return wf

