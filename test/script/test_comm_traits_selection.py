"""Compile the real backend selectors with SDK-independent dependency shims."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


TRAITS = Path(__file__).resolve().parents[2] / "flagcx/adaptor/include/device_api"
CXX = shutil.which("g++") or shutil.which("clang++")


@unittest.skipUnless(CXX, "C++ compiler required")
class CommTraitsSelectionTest(unittest.TestCase):
    def test_nccl_fallback_defines_complete_default_traits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("comm_traits.h", "flagcx_device_enums.h",
                         "nvidia_comm_traits.h", "nccl_comm_traits.h"):
                shutil.copyfile(TRAITS / name, root / name)
            (root / "devnet").mkdir()
            shutil.copyfile(TRAITS / "devnet/nvidia_net_provider.h",
                            root / "devnet/nvidia_net_provider.h")
            # Exercise the real selectors without a CUDA or NCCL SDK. The
            # complete type below is available only if fallback includes it.
            (root / "platform_traits.h").write_text("""
#define FLAGCX_HOST_DEVICE_INLINE inline
struct NvidiaPlatform {};
""")
            (root / "default_comm_traits.h").write_text("""
#pragma once
template <typename P> struct CommTraits<DefaultBackend<P>> {
  struct Comm { int rank; };
};
""")
            (root / "nccl.h").write_text("""
#define NCCL_VERSION_CODE FLAGCX_TEST_NCCL_VERSION
#define NCCL_VERSION(major, minor, patch) ((major)*10000+(minor)*100+(patch))
""")
            for name in ("flagcx.h", "nvidia_adaptor.h", "sym_heap.h"):
                (root / name).write_text("#pragma once\n")
            source = root / "test.cpp"
            source.write_text("""
#define FLAGCX_DEVICE_API_VENDOR 1
#include "comm_traits.h"
#include <type_traits>
#ifdef FLAGCX_DEVICE_API_VENDOR
#error "NCCL fallback must clear the vendor selection"
#endif
static_assert(std::is_same<DeviceAPI,
    CommTraits<DefaultBackend<NvidiaPlatform>>>::value, "incorrect fallback");
static_assert(sizeof(DeviceAPI::Comm) > 0, "fallback traits must be complete");
""")
            for version, force in ((22800, False), (22900, True)):
                with self.subTest(version=version, force_default=force):
                    command = [CXX, "-std=c++17", "-fsyntax-only", str(source),
                               "-DUSE_NVIDIA_ADAPTOR", "-DFLAGCX_COMM_TRAITS_CCL",
                               f"-DFLAGCX_TEST_NCCL_VERSION={version}"]
                    if force:
                        command.append("-DFLAGCX_FORCE_DEFAULT_PATH")
                    result = subprocess.run(command, capture_output=True,
                                            text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_selection_preserves_enflame_host_and_device_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("comm_traits.h", "flagcx_device_enums.h",
                         "musa_comm_traits.h", "ascend_comm_traits.h",
                         "enflame_comm_traits.h"):
                shutil.copyfile(TRAITS / name, root / name)
            # Only the selected platform exists, just as in platform_traits.h.
            # SDK operations are irrelevant to checking the dispatch boundary.
            (root / "platform_traits.h").write_text("""
#define FLAGCX_HOST_DEVICE_INLINE inline
#if defined(USE_MUSA_ADAPTOR)
struct MusaPlatform {};
#elif defined(USE_ASCEND_ADAPTOR)
struct AscendPlatform {};
#elif defined(USE_ENFLAME_ADAPTOR)
struct EnflamePlatform {};
#else
struct DefaultPlatform {};
#endif
""")
            (root / "default_comm_traits.h").write_text("""
template <typename P> struct CommTraits<DefaultBackend<P>> {};
""")
            (root / "eccl_comm_traits.h").write_text("""
struct EcclBackend {};
using DeviceAPI = CommTraits<EcclBackend>;
""")
            cases = (
                ((), "DefaultBackend<DefaultPlatform>"),
                (("USE_MUSA_ADAPTOR", "FLAGCX_MUSA_PROXY_COMPLETION_SYNC=0"),
                 "DefaultBackend<MusaPlatform>"),
                (("USE_ASCEND_ADAPTOR",), "DefaultBackend<AscendPlatform>"),
                (("USE_ENFLAME_ADAPTOR",), "DefaultBackend<EnflamePlatform>"),
                (("USE_ENFLAME_ADAPTOR", "FLAGCX_ENFLAME_DEVICE_API"),
                 "EcclBackend"),
            )
            for defines, backend in cases:
                with self.subTest(backend=backend):
                    source = root / "test.cpp"
                    source.write_text(
                        '#include "comm_traits.h"\n#include <type_traits>\n'
                        f"static_assert(std::is_same<DeviceAPI, CommTraits<{backend}>>"
                        '::value, "incorrect backend selected");\n'
                    )
                    result = subprocess.run(
                        [CXX, "-std=c++17", "-fsyntax-only", str(source),
                         *(f"-D{define}" for define in defines)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
