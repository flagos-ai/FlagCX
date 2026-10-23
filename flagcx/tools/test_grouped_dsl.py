"""CPU checks of GroupedAR data semantics, message pairing and pipeline safety."""
import unittest
from collections import Counter, defaultdict

try:
    from .builders_grouped import build_groupedar, build_groupedar_4g_example
    from .utils import CommDomain, RedOp
except ImportError:
    from builders_grouped import build_groupedar, build_groupedar_4g_example
    from utils import CommDomain, RedOp


class GroupedARDslTest(unittest.TestCase):
    def test_nondivisible_and_degenerate_collectives(self):
        for world, group, count in [(4, 2, 17), (4, 1, 33), (4, 4, 33),
                                     (1, 1, 17), (4, 2, 0), (4, 2, 1)]:
            for op in (RedOp.sum, RedOp.avg):
                with self.subTest(world=world, group=group, count=count, op=op):
                    workflow = build_groupedar(
                        world_size=world, group_size=group, count=count,
                        num_slices=3, num_local_slices=2, num_red_slices=3,
                        red_op=op)
                    workflow.semantic_check().raise_for_error()

    def test_4g_ranges_reductions_and_message_pairing(self):
        workflow = build_groupedar_4g_example()
        self.assertEqual(workflow.count, 1 << 30)
        self.assertEqual(workflow.algorithm_config["num_red_slices"], 32)
        self.assertEqual(workflow.scratch_count * workflow.type_size, 192 << 20)
        capacities = {"input": workflow.input_count, "output": workflow.output_count,
                      "scratch": workflow.scratch_count}
        send_messages, recv_messages = defaultdict(list), defaultdict(list)
        for rank, dag in workflow.rank_templates().items():
            self.assertEqual(Counter(node.node_type for node in dag.nodes),
                             Counter(red=1920, p2p=73, cpy=1))
            for node in dag.nodes:
                refs = []
                if node.node_type == "p2p":
                    for op in node.p2p_ops:
                        refs.append((op.buffer, op.count))
                        if node.comm_domain == CommDomain.net:
                            self.assertEqual(rank % 8, op.peer_rank % 8)
                            self.assertNotEqual(rank // 8, op.peer_rank // 8)
                        else:
                            self.assertEqual(rank // 8, op.peer_rank // 8)
                        if op.type == "send":
                            send_messages[rank, op.peer_rank, node.comm_domain].append(
                                (op.count, op.datatype))
                        elif op.type == "recv":
                            recv_messages[op.peer_rank, rank, node.comm_domain].append(
                                (op.count, op.datatype))
                    reductions = [dag.nodes[child] for child in node.children
                                  if dag.nodes[child].node_type == "red"]
                    if reductions:
                        received = next(op for op in node.p2p_ops if op.type == "recv")
                        intervals = sorted((red.red.input2.offset_bytes,
                                            red.red.count * workflow.type_size)
                                           for red in reductions)
                        cursor = received.buffer.offset_bytes
                        for start, length in intervals:
                            self.assertEqual(start, cursor)
                            cursor += length
                        self.assertEqual(cursor, received.buffer.offset_bytes +
                                         received.count * workflow.type_size)
                        outputs = sorted((red.red.output.offset_bytes,
                                          red.red.count * workflow.type_size)
                                         for red in reductions)
                        for previous, current in zip(outputs, outputs[1:]):
                            self.assertEqual(previous[0] + previous[1], current[0])
                elif node.node_type == "red":
                    self.assertEqual(len(node.parents), 1)
                    self.assertEqual(node.red.count, 524288)
                    refs.extend((ref, node.red.count) for ref in
                                (node.red.input1, node.red.input2, node.red.output))
                elif node.node_type == "cpy":
                    refs.extend((ref, node.cpy.count) for ref in (node.cpy.src, node.cpy.dst))
                for ref, count in refs:
                    self.assertEqual(ref.offset_bytes % workflow.type_size, 0)
                    self.assertGreaterEqual(ref.offset_bytes, 0)
                    self.assertLessEqual(ref.offset_bytes + count * workflow.type_size,
                                         capacities[ref.buffer] * workflow.type_size)
                for parent in node.parents:
                    self.assertLess(parent, node.node_idx)
            self.assertEqual(len([n for n in dag.nodes if n.node_type == "p2p"
                                  and not n.p2p_ops]), 5)
        self.assertEqual(dict(send_messages), dict(recv_messages))

    def test_pipeline_dependencies_and_scratch_reuse(self):
        workflow = build_groupedar(
            world_size=4, group_size=2, count=1025,
            num_slices=4, num_local_slices=2, num_red_slices=3)
        for dag in workflow.rank_templates().values():
            named = {node.name: node for node in dag.nodes}
            ancestors = []
            for node in dag.nodes:
                mask = 0
                for parent in node.parents:
                    mask |= (1 << parent) | ancestors[parent]
                ancestors.append(mask)

            def precedes(before, after):
                return bool(ancestors[named[after].node_idx] & (1 << named[before].node_idx))

            for outer in range(4):
                self.assertTrue(precedes(f"local_join_s{outer}", f"net_rs_s{outer}_step0"))
                self.assertTrue(precedes(f"net_ag_s{outer}_step0", f"local_ag_s{outer}"))
                for inner in range(2):
                    for red in range(3):
                        self.assertTrue(precedes(
                            f"local_red_s{outer}_step1_u{inner}_r{red}",
                            f"local_join_s{outer}"))
                if outer < 3:
                    self.assertFalse(precedes(
                        f"net_rs_s{outer}_step0", f"local_rs_s{outer + 1}_step1_u0"))
                    self.assertTrue(precedes(
                        f"local_join_s{outer + 1}", f"local_ag_s{outer}"))
                if outer:
                    self.assertTrue(precedes(
                        f"net_ag_s{outer - 1}_step0", f"net_rs_s{outer}_step0"))
                self.assertFalse(precedes(
                    f"local_red_s{outer}_step1_u0_r0", f"local_rs_s{outer}_step1_u1"))

        workflow = build_groupedar(
            world_size=8, group_size=4, count=1025,
            num_slices=2, num_local_slices=2, num_red_slices=3)
        for dag in workflow.rank_templates().values():
            named = {node.name: node for node in dag.nodes}
            for outer in range(2):
                for step in (2, 3):
                    for inner in range(2):
                        exchange = named[f"local_rs_s{outer}_step{step}_u{inner}"]
                        for red in range(3):
                            previous = named[f"local_red_s{outer}_step{step - 1}_u{inner}_r{red}"]
                            self.assertIn(previous.node_idx, exchange.parents)

    def test_invalid_configuration_and_explicit_support_boundary(self):
        for kwargs in ({"group_size": 3}, {"num_slices": 0},
                       {"num_local_slices": 0}, {"num_red_slices": -1},
                       {"red_op": RedOp.nop}, {"count": 1 << 64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_groupedar(**kwargs)
        with self.assertRaises(NotImplementedError):
            build_groupedar(in_place=True)


if __name__ == "__main__":
    unittest.main()
