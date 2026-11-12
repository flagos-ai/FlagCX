"""Negative contract checks for the CPU-only DAG validator."""
import unittest

try:
    from .uni_runner import UniRunnerWorkflow
    from .utils import Collective, DataType, RedOp
    from .dsl_validation import validate_workflow
except ImportError:
    from uni_runner import UniRunnerWorkflow
    from utils import Collective, DataType, RedOp
    from dsl_validation import validate_workflow


def gemm_workflow():
    return UniRunnerWorkflow(
        "strided_gemm", Collective.AllGatherGemm, 1, 6,
        datatype=DataType.float32, red_op=RedOp.sum,
        algo="all_gather_gemm", input_count=8, output_count=4,
        weight_count=6, algorithm_config={"group_size": 1},
    )


class ValidationTests(unittest.TestCase):
    def test_strided_gemm_and_runtime_free_descriptor(self):
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            rb.gemm(wf.input(0), wf.weight(0), wf.output(0),
                    m=2, n=2, k=3, lda=5)
        result = wf.validate_structure()
        with self.assertRaisesRegex(AssertionError, "validate_structure"):
            wf.semantic_check().raise_for_error()
        self.assertEqual(result["ranks"][0]["node_types"], {"gemm": 1})
        desc = wf.runtime_entry(0)["dag"]["nodes"][0]["gemm"]
        self.assertNotIn("n_workers", desc)
        self.assertNotIn("done_counter", desc)
        wf.input_count = 7
        with self.assertRaisesRegex(ValueError, "capacity"):
            validate_workflow(wf)

    def test_invalid_stride_and_compute_fanin(self):
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            first, second = rb.p2p([]), rb.p2p([])
            idx = rb.gemm(wf.input(0), wf.weight(0), wf.output(0),
                          m=2, n=2, k=3, lda=2)
        with self.assertRaisesRegex(ValueError, "stride"):
            validate_workflow(wf)
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            first, second = rb.p2p([]), rb.p2p([])
            rb.gemm(wf.input(0), wf.weight(0), wf.output(0),
                    m=2, n=2, k=3, parents=[first, second])
        with self.assertRaisesRegex(ValueError, "at most one parent"):
            validate_workflow(wf)

    def test_ordered_peer_message_mismatch(self):
        wf = UniRunnerWorkflow(
            "exchange", Collective.AllGather, 2, 2,
            datatype=DataType.float32, algo="grouped_ag",
            input_count=2, output_count=4,
        )
        with wf.rank(0) as rb:
            rb.p2p([wf.send(1, wf.input(0), 2)])
        with wf.rank(1) as rb:
            rb.p2p([wf.recv(0, wf.output(0), 1)])
        with self.assertRaisesRegex(ValueError, "send/recv"):
            validate_workflow(wf)

    def test_net_domain_and_size_t_p2p_count(self):
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            rb.gemm(wf.input(0), wf.weight(0), wf.output(0),
                    m=2, n=2, k=3, comm_domain=1)
        with self.assertRaisesRegex(ValueError, "NET domain"):
            validate_workflow(wf)
        count = 1 << 32
        wf = UniRunnerWorkflow(
            "large_transfer", Collective.AllGather, 2, count,
            datatype=DataType.float32, algo="grouped_ag",
            input_count=count, output_count=count,
        )
        with wf.rank(0) as rb:
            rb.p2p([wf.send(1, wf.input(0), count)])
        with wf.rank(1) as rb:
            rb.p2p([wf.recv(0, wf.output(0), count)])
        self.assertEqual(validate_workflow(wf)["ranks"][0]["ipc_send_bytes"],
                         count * 4)

    def test_repeated_peer_operation_requires_separate_nodes(self):
        wf = UniRunnerWorkflow(
            "paired_blocks", Collective.AllGather, 2, 2,
            datatype=DataType.float32, algo="grouped_ag",
            input_count=2, output_count=4,
        )
        with wf.rank(0) as rb:
            rb.p2p([wf.send(1, wf.input(0), 1),
                    wf.send(1, wf.input(1), 1)])
        with wf.rank(1) as rb:
            rb.p2p([wf.recv(0, wf.output(0), 1),
                    wf.recv(0, wf.output(1), 1)])
        with self.assertRaisesRegex(ValueError, "one send and one recv per peer"):
            validate_workflow(wf)

    def test_cycle_and_read_only_binding(self):
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            first = rb.p2p([], parents=[1])
            rb.p2p([], parents=[first])
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_workflow(wf)
        wf = gemm_workflow()
        with wf.rank(0) as rb:
            rb.cpy(wf.input(0), wf.weight(0), 1)
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate_workflow(wf)


if __name__ == "__main__":
    unittest.main()
