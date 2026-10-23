"""CPU-only fused DAG checks, including persistent queue ordering."""
import unittest
from collections import defaultdict, deque

try:
    from .builders_fused import build_allgather_gemm, build_gemm_reducescatter
except ImportError:
    from builders_fused import build_allgather_gemm, build_gemm_reducescatter


def _phase_order(nodes):
    indegree = [len(node["parents"]) for node in nodes]
    ready = [deque(), deque()]
    phase = lambda node: 0 if node["node_type"] in ("p2p", "cpy") else 1
    for node in nodes:
        if not indegree[node["node_idx"]]:
            ready[phase(node)].append(node["node_idx"])
    result = []
    while len(result) < len(nodes):
        before = len(result)
        for queue in ready:
            while queue:
                idx = queue.popleft()
                result.append(idx)
                for child in nodes[idx]["children"]:
                    indegree[child] -= 1
                    if not indegree[child]:
                        ready[phase(nodes[child])].append(child)
        if before == len(result):
            raise AssertionError("DAG cycle")
    return result


def _simulate(workflow, inputs, weights):
    """Interpret small integer matrices in the actual phase-drain FIFO order.

    Each domain has an ordered host stream and ordered compute tasks. P2P
    groups publish their sends, then complete when all matching receives
    arrive. Matrix data is allocated only for the small numerical tests.
    """
    graphs = [workflow.runtime_entry(r)["dag"]["nodes"] for r in workflow.ranks()]
    memory = [
        {"input": list(inputs[r]), "weight": list(weights[r]),
         "output": [None] * workflow.output_count,
         "scratch": [None] * workflow.scratch_count}
        for r in workflow.ranks()
    ]
    streams = {}
    for rank, nodes in enumerate(graphs):
        for idx in _phase_order(nodes):
            node = nodes[idx]
            kind = "host" if node["node_type"] in ("p2p", "cpy") else "compute"
            streams.setdefault((rank, kind, node["comm_domain"]), deque()).append(idx)
    done, posted, messages = set(), set(), defaultdict(deque)

    def read(rank, ref, count):
        start = ref["offset_bytes"] // 4
        data = memory[rank][ref["buffer"]][start:start + count]
        if len(data) != count or any(v is None for v in data):
            raise AssertionError("out-of-bounds or uninitialized read")
        return list(data)

    def write(rank, ref, data):
        start = ref["offset_bytes"] // 4
        target = memory[rank][ref["buffer"]]
        if start < 0 or start + len(data) > len(target):
            raise AssertionError("out-of-bounds write")
        target[start:start + len(data)] = data

    def at(ref, element):
        return {"buffer": ref["buffer"], "offset_bytes": ref["offset_bytes"] + 4 * element}

    remaining = sum(map(len, graphs))
    while remaining:
        progress = False
        for (rank, kind, domain), queue in streams.items():
            if not queue:
                continue
            idx = queue[0]
            node = graphs[rank][idx]
            if any((rank, parent) not in done for parent in node["parents"]):
                continue
            node_type = node["node_type"]
            if node_type == "p2p":
                if (rank, idx) not in posted:
                    for op in node["p2p_ops"]:
                        if op["type"] == "send":
                            key = (rank, op["peer_rank"], domain)
                            messages[key].append(read(rank, op["buffer"], op["count"]))
                    posted.add((rank, idx))
                    progress = True
                needed = defaultdict(int)
                receives = [op for op in node["p2p_ops"] if op["type"] == "recv"]
                ready = True
                for op in receives:
                    key = (op["peer_rank"], rank, domain)
                    needed[key] += 1
                    if len(messages[key]) < needed[key]:
                        ready = False
                        break
                    if len(messages[key][needed[key] - 1]) != op["count"]:
                        raise AssertionError("P2P message size/order mismatch")
                if not ready:
                    continue
                for op in receives:
                    key = (op["peer_rank"], rank, domain)
                    write(rank, op["buffer"], messages[key].popleft())
            elif node_type == "gemm":
                g = node["gemm"]
                for row in range(g["m"]):
                    for col in range(g["n"]):
                        value = 0
                        for kk in range(g["k"]):
                            value += (read(rank, at(g["a"], row * g["lda"] + kk), 1)[0] *
                                      read(rank, at(g["b"], kk * g["ldb"] + col), 1)[0])
                        target = at(g["c"], row * g["ldc"] + col)
                        if g["accumulate"]:
                            value += read(rank, target, 1)[0]
                        write(rank, target, [value])
            elif node_type == "red":
                red = node["red"]
                if red["red_op"] != 0:
                    raise AssertionError("test interpreter expects sum")
                left, right = read(rank, red["input1"], red["count"]), read(rank, red["input2"], red["count"])
                write(rank, red["output"], [a + b for a, b in zip(left, right)])
            elif node_type == "cpy":
                c = node["cpy"]
                write(rank, c["dst"], read(rank, c["src"], c["count"]))
            else:
                raise AssertionError("unsupported test node")
            done.add((rank, idx))
            queue.popleft()
            remaining -= 1
            progress = True
        if not progress:
            pending = [(key, queue[0]) for key, queue in streams.items() if queue]
            raise AssertionError(f"persistent queues cannot progress: {pending}")
    if any(messages.values()):
        raise AssertionError("unmatched sends")
    return [buffers["output"] for buffers in memory]


def _a(rank, row, col):
    return 7 * (rank + 1) + 3 * row + col


def _b(rank, row, col):
    return 11 * (rank + 1) + row + 2 * col


class FusedDslTests(unittest.TestCase):
    def test_allgather_gemm_numeric_and_queues(self):
        for ranks, m, k, slices in ((4, 20, 9, 2), (2, 6, 7, 2), (4, 16, 8, 1), (16, 80, 7, 4)):
            for k_slices in (1, 2):
                with self.subTest(ranks=ranks, slices=slices, k_slices=k_slices):
                    n, rows = 5, m // ranks
                    wf = build_allgather_gemm(ranks, m, n, k, ranks // 2, slices, k_slices)
                    inputs = [[_a(r, row, col) for row in range(rows) for col in range(k)] for r in range(ranks)]
                    weights = [[_b(r, row, col) for row in range(k) for col in range(n)] for r in range(ranks)]
                    actual = _simulate(wf, inputs, weights)
                    expected = [[sum(_a(row // rows, row % rows, kk) * _b(r, kk, col)
                                     for kk in range(k))
                                 for row in range(m) for col in range(n)] for r in range(ranks)]
                    self.assertEqual(actual, expected)

    def test_gemm_reducescatter_numeric_and_queues(self):
        for ranks, m, k, slices in ((4, 20, 12, 2), (2, 6, 6, 2), (4, 16, 16, 1), (16, 80, 32, 4)):
            for k_slices in (1, 2):
                with self.subTest(ranks=ranks, slices=slices, k_slices=k_slices):
                    n, local_k, rows = 5, k // ranks, m // ranks
                    wf = build_gemm_reducescatter(ranks, m, n, k, ranks // 2, slices, k_slices)
                    inputs = [[_a(r, row, col) for row in range(m) for col in range(local_k)] for r in range(ranks)]
                    weights = [[_b(r, row, col) for row in range(local_k) for col in range(n)] for r in range(ranks)]
                    actual = _simulate(wf, inputs, weights)
                    expected = [[sum(_a(source, r * rows + row, kk) * _b(source, kk, col)
                                     for source in range(ranks) for kk in range(local_k))
                                 for row in range(rows) for col in range(n)] for r in range(ranks)]
                    self.assertEqual(actual, expected)

    def test_four_gib_graphs_without_matrix_allocation(self):
        for builder, expected_nodes, scratch_bytes in (
                (build_allgather_gemm, 124, 4 << 30),
                (build_gemm_reducescatter, 184, 8 << 30)):
            wf = builder(16, 262144, 4096, 4096)
            self.assertEqual(wf.scratch_count * 4, scratch_bytes)
            for rank in wf.ranks():
                nodes = wf.runtime_entry(rank)["dag"]["nodes"]
                self.assertEqual(len(nodes), expected_nodes)
                self.assertEqual(len(_phase_order(nodes)), expected_nodes)
                sent = sum(op["count"] * 4 for node in nodes
                           if node["node_type"] == "p2p" and node["comm_domain"] == 1
                           for op in node["p2p_ops"] if op["type"] == "send")
                self.assertEqual(sent, 256 << 20)
                for node in nodes:
                    if node["node_type"] in ("gemm", "red"):
                        self.assertLessEqual(len(node["parents"]), 1)
                    if node["node_type"] == "gemm":
                        self.assertNotIn("worker_count", node["gemm"])
                        self.assertNotIn("completion_counter", node["gemm"])

    def test_fused_api_cache_key_contract(self):
        cases = (
            (262144, 4096, 4096, 67108864, 67108864,
             "7201636012322909248", "880482809123302945"),
            (80, 5, 32, 160, 25,
             "7669934921617355633", "17531403222083669200"),
        )
        for m, n, k, ag_count, rs_count, ag_hash, rs_hash in cases:
            for builder, comm_op, count, red_op, algo_hash in (
                    (build_allgather_gemm, "all_gather", ag_count, 5, ag_hash),
                    (build_gemm_reducescatter, "reduce_scatter", rs_count, 0, rs_hash)):
                with self.subTest(builder=builder.__name__, m=m, n=n, k=k):
                    workflow = builder(16, m, n, k, 8, 4, 1)
                    self.assertEqual(workflow.algorithm_config, {
                        "m": m, "n": n, "k": k, "num_slices": 4,
                        "num_k_slices": 1, "group_size": 8,
                        "pipeline_version": 2, "buffer_mode": 0,
                    })
                    for rank in workflow.ranks():
                        self.assertEqual(workflow.runtime_entry(rank)["key"], {
                            "algo_hash": algo_hash, "comm_op": comm_op,
                            "count": count, "datatype": 7, "red_op": red_op,
                            "rank": rank, "nranks": 16, "root": -1,
                        })

    def test_proxy_group_has_one_message_per_peer_and_direction(self):
        # Proxy queues retain their head until the entire group completes.
        # Another message to that peer/direction in the group cannot progress.
        for ranks in (2, 4, 8, 16):
            for builder in (build_allgather_gemm, build_gemm_reducescatter):
                wf = builder(ranks, 8 * ranks, 16, 2 * ranks,
                             group_size=ranks // 2, num_slices=4)
                for rank in wf.ranks():
                    for node in wf.runtime_entry(rank)["dag"]["nodes"]:
                        if node["node_type"] != "p2p":
                            continue
                        keys = [(op["peer_rank"], op["type"])
                                for op in node["p2p_ops"]]
                        with self.subTest(builder=builder.__name__, ranks=ranks,
                                          rank=rank, node=node["node_idx"]):
                            self.assertEqual(len(keys), len(set(keys)))

    def test_invalid_shapes(self):
        for builder in (build_allgather_gemm, build_gemm_reducescatter):
            with self.assertRaises(ValueError):
                builder(8, 64, 16, 32, group_size=8)
            with self.assertRaises(ValueError):
                builder(4, 21, 16, 32, group_size=2)
            with self.assertRaises(ValueError):
                builder(4, 20, 16, 32, group_size=2, num_slices=6)
        with self.assertRaises(ValueError):
            build_gemm_reducescatter(4, 20, 5, 13, group_size=2)
        with self.assertRaises(ValueError):
            build_gemm_reducescatter(4, 1 << 32, 1, 4, group_size=2)


if __name__ == "__main__":
    unittest.main()

