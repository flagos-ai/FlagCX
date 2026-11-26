import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
SOURCE = (PLUGIN_DIR / "flagcx/src/backend_flagcx.cpp").read_text()
HEADER = (PLUGIN_DIR / "flagcx/include/backend_flagcx.hpp").read_text()
EVENT_HEADER = (PLUGIN_DIR / "flagcx/include/event_flagcx.hpp").read_text()
STREAM_GUARD_HEADER = (
    PLUGIN_DIR / "flagcx/include/stream_guard_flagcx.hpp"
).read_text()


def function_body(source, signature, next_signature):
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


class BackendSourceRegressionTest(unittest.TestCase):
    def test_ppu_has_independent_cuda_compatible_torch_paths(self):
        self.assertGreaterEqual(EVENT_HEADER.count("#elif USE_PPU_ADAPTOR"), 2)
        self.assertIn("class flagcxPpuEvent", EVENT_HEADER)
        self.assertGreaterEqual(
            STREAM_GUARD_HEADER.count("#elif USE_PPU_ADAPTOR"), 4
        )
        self.assertRegex(
            function_body(
                SOURCE,
                "void flagcxBackend::initComm()",
                "flagcxComm_t flagcxBackend::getOrCreatePairComm",
            ),
            re.compile(
                r"#elif USE_PPU_ADAPTOR\s+"
                r"initComm\(c10::impl::getDeviceGuardImpl"
            ),
        )
        self.assertRegex(
            HEADER,
            re.compile(r"USE_PPU_ADAPTOR\s+event_ = .*flagcxPpuEvent"),
        )
        self.assertRegex(
            HEADER,
            re.compile(r"USE_PPU_ADAPTOR\s+devName = \"cuda\";"),
        )

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

    def test_tsm_flagos_event_uses_native_tx_runtime(self):
        self.assertIn("#include <tx_runtime.h>", EVENT_HEADER)
        self.assertIn(
            "!defined(USE_TSM_ADAPTOR)", STREAM_GUARD_HEADER
        )
        tsm_event = function_body(
            EVENT_HEADER, "class flagcxTxdaEvent", "#elif USE_ENFLAME_ADAPTOR"
        )
        self.assertIn("#ifdef FLAGCX_TORCH_BACKEND_FLAGOS", tsm_event)
        self.assertIn("txEventCreate", tsm_event)
        self.assertIn("txStreamWaitEvent", tsm_event)
        self.assertIn("GetCurrentStreamForDevice", tsm_event)
        # The vendor path stays intact for FLAGCX_TORCH_BACKEND=vendor.
        self.assertIn("torch_txda::TXDAEvent", tsm_event)

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


class WorkShutdownContractTest(unittest.TestCase):
    def test_live_work_remains_waitable_across_shutdown_and_retry(self):
        """Compile real shutdown/wait methods with deterministic SDK mocks."""
        cxx = os.environ.get("CXX") or shutil.which("c++")
        if cxx is None:
            self.skipTest("C++ compiler unavailable")
        completion = function_body(
            HEADER, "struct flagcxWorkCompletion", "class flagcxWork")
        shutdown = function_body(
            SOURCE, "struct flagcxBackend::ShutdownState", "void flagcxBackend::shutdown")
        methods = function_body(
            SOURCE, "bool flagcxWork::isCompleted", "c10::intrusive_ptr<c10::ivalue::Future> flagcxWork::getFuture")
        prelude = r"""
#include "flagcx.h"
#include "cleanup_flagcx.hpp"
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <stdexcept>
#include <unordered_map>
struct FakeFuture {
  bool completed() { return true; }
  bool hasValue() { return true; }
};
static bool released;
static int waits, blocks, queries, destroys;
static flagcxResult_t destroyResult = flagcxSuccess;
static flagcxResult_t streamResult = flagcxSuccess;
static std::mutex rendezvousMutex;
static std::condition_variable rendezvous;
static bool blockDestroy, destroyEntered, allowDestroy;
#define C10D_FLAGCX_CHECK(call, unused) \
  do { if ((call) != flagcxSuccess) throw std::runtime_error("mock SDK"); } while (0)
struct flagcxEvent {
  void block(int) { assert(!released); ++blocks; }
  bool query() { assert(!released); ++queries; return true; }
};
flagcxResult_t flagcxCommDestroy(flagcxComm_t) {
  ++destroys;
  if (blockDestroy) {
    std::unique_lock<std::mutex> lock(rendezvousMutex);
    destroyEntered = true;
    rendezvous.notify_all();
    rendezvous.wait(lock, [] { return allowDestroy; });
  }
  return destroyResult;
}
flagcxResult_t flagcxDeviceHandleFree(flagcxDeviceHandle_t) {
  released = true;
  return flagcxSuccess;
}
"""
        classes = r"""
struct flagcxWork {
  std::shared_ptr<flagcxWorkCompletion> completion_;
  flagcxDeviceHandle_t devHandle_;
  flagcxStream_t stream_;
  int deviceId_ = 0;
  bool isBarrierOp_ = true;
  std::unique_ptr<flagcxEvent> event_ = std::make_unique<flagcxEvent>();
  std::shared_ptr<FakeFuture> future_ = std::make_shared<FakeFuture>();
  bool isCompleted();
  bool isSuccess() const;
  bool wait(std::chrono::milliseconds = std::chrono::milliseconds(0));
};
struct flagcxBackend {
  struct ShutdownState;
  struct inflightTensorBatch {
    flagcxEvent_t event;
    flagcxStream_t stream;
    int deviceId;
    bool completionUncertain, completed, recorded, ownsStreamWrapper;
  };
};
"""
        main = r"""
int main() {
  flagcxDeviceHandle handle = {};
  handle.setDevice = [](int) { assert(!released); return flagcxSuccess; };
  handle.getDevice = [](int *dev) { assert(!released); *dev = 0; return flagcxSuccess; };
  handle.streamSynchronize = [](flagcxStream_t) {
    assert(!released); ++waits; return flagcxSuccess;
  };
  handle.streamDestroy = [](flagcxStream_t) { return streamResult; };
  auto completion = std::make_shared<flagcxWorkCompletion>();
  flagcxWork work;
  work.completion_ = completion;
  work.devHandle_ = &handle;
  work.stream_ = (flagcxStream_t)1;
  flagcxBackend::ShutdownState owner;
  owner.completion = completion;
  owner.deviceId = 0;
  owner.handle = &handle;
  owner.comm = (flagcxComm_t)1;
  owner.streams.emplace(0, work.stream_);
  assert(work.wait() && waits == 1);
  destroyResult = flagcxRemoteError;
  assert(owner.release() == flagcxRemoteError);
  assert(!completion->completed && !released);
  assert(work.wait() && waits == 2);
  destroyResult = flagcxSuccess;
  streamResult = flagcxRemoteError;
  assert(owner.release() == flagcxRemoteError);
  assert(completion->completed && !released);
  const int savedBlocks = blocks, savedQueries = queries;
  assert(work.wait() && work.isCompleted() && work.isSuccess());
  assert(waits == 2 && blocks == savedBlocks && queries == savedQueries);
  streamResult = flagcxSuccess;
  assert(owner.release() == flagcxSuccess && released);
  assert(work.wait() && work.isCompleted());
  assert(owner.release() == flagcxSuccess);
  assert(waits == 2 && destroys == 2);
  // A concurrent wait must finish without SDK access once shutdown completes.
  released = false;
  completion->completed = false;
  owner.sdkReleased = false;
  owner.handle = &handle;
  owner.comm = (flagcxComm_t)1;
  blockDestroy = true;
  auto shutdown = std::async(std::launch::async, [&] { return owner.release(); });
  {
    std::unique_lock<std::mutex> lock(rendezvousMutex);
    rendezvous.wait(lock, [] { return destroyEntered; });
  }
  std::promise<void> waiting;
  auto wait = std::async(std::launch::async, [&] {
    waiting.set_value();
    return work.wait();
  });
  waiting.get_future().wait();
  assert(wait.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
  {
    std::lock_guard<std::mutex> lock(rendezvousMutex);
    allowDestroy = true;
  }
  rendezvous.notify_all();
  assert(shutdown.get() == flagcxSuccess && wait.get());
  assert(released && waits == 2);
}
"""
        source = prelude + completion + classes + shutdown + methods + main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test.cpp").write_text(source)
            for defines in ([], ["FLAGCX_TORCH_BACKEND_FLAGOS", "USE_ASCEND_ADAPTOR"],
                            ["FLAGCX_TORCH_BACKEND_FLAGOS", "USE_ENFLAME_ADAPTOR"]):
                with self.subTest(defines=defines):
                    command = [cxx, "-std=c++17", "-pthread",
                               f"-I{PLUGIN_DIR.parents[1] / 'flagcx/include'}",
                               f"-I{PLUGIN_DIR / 'flagcx/include'}"]
                    result = subprocess.run(
                        command + [f"-D{name}" for name in defines]
                        + [str(path / "test.cpp"), "-o", str(path / "test")],
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    result = subprocess.run([str(path / "test")],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
