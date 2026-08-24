import re
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
SOURCE = (PLUGIN_DIR / "flagcx/src/backend_flagcx.cpp").read_text()
HEADER = (PLUGIN_DIR / "flagcx/include/backend_flagcx.hpp").read_text()


def function_body(source, signature, next_signature):
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


class BackendSourceRegressionTest(unittest.TestCase):
    def test_ascend_flagos_does_not_reference_torch_npu_stream(self):
        get_stream = function_body(
            SOURCE,
            "flagcxStream_t flagcxBackend::getStreamByIndex",
            "std::unique_ptr<flagcxEvent>",
        )

        self.assertRegex(
            get_stream,
            re.compile(
                r"defined\(USE_ASCEND_ADAPTOR\).*"
                r"!defined\(FLAGCX_TORCH_BACKEND_FLAGOS\).*"
                r"c10_npu::getCurrentNPUStream",
                re.DOTALL,
            ),
        )
        self.assertIn("devHandle_->streamCreate", get_stream)
        self.assertRegex(
            HEADER,
            re.compile(
                r"defined\(USE_ASCEND_ADAPTOR\).*"
                r"!defined\(FLAGCX_TORCH_BACKEND_FLAGOS\).*aclrtStream",
                re.DOTALL,
            ),
        )

    def test_flattened_intermediates_are_allocated_on_flagos_comm_stream(self):
        helper = function_body(
            SOURCE,
            "at::Tensor newLikeFlatOnStream",
            "void copyTensorOnStream",
        )

        self.assertIn("#ifdef FLAGCX_TORCH_BACKEND_FLAGOS", helper)
        self.assertIn("flagcxStreamGuard guard(stream, deviceId)", helper)
        self.assertEqual(SOURCE.count("newLikeFlatOnStream("), 9)
        self.assertEqual(SOURCE.count("newLikeFlat("), 1)

    def test_flagos_intermediate_copies_use_device_memcpy(self):
        helper = function_body(
            SOURCE,
            "void copyTensorOnStream",
            "void check_device",
        )

        self.assertIn("#ifdef FLAGCX_TORCH_BACKEND_FLAGOS", helper)
        self.assertIn("devHandle->deviceMemcpy", helper)
        self.assertIn("flagcxMemcpyDeviceToDevice", helper)
        self.assertEqual(SOURCE.count("copyTensorOnStream("), 8)

    def test_enflame_gather_and_scatter_avoid_broken_eccl_primitives(self):
        gather = function_body(
            SOURCE,
            "flagcxBackend::gather",
            "c10::intrusive_ptr<Work> flagcxBackend::reduce",
        )
        scatter = function_body(
            SOURCE,
            "flagcxBackend::scatter",
            "c10::intrusive_ptr<Work> flagcxBackend::send",
        )

        for body in (gather, scatter):
            self.assertIn(
                "defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ENFLAME_ADAPTOR)",
                body,
            )
        self.assertIn("flagcxAllGather", gather)
        self.assertIn("flagcxGather", gather)
        self.assertIn("flagcxBroadcast", scatter)
        self.assertIn("flagcxScatter", scatter)


if __name__ == "__main__":
    unittest.main()
