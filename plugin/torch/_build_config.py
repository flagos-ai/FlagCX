"""
Shared build configuration for the flagcx torch plugin.

Used by both the root setup.py and plugin/torch/setup.py to avoid
duplicating adaptor detection, device-specific paths, and extension
class selection logic.
"""

import os
import shutil
import sys
from dataclasses import dataclass

from packaging.version import Version, parse as vparse

# ---------------------------------------------------------------------------
# Adaptor name -> C++ define flag
# ---------------------------------------------------------------------------

ADAPTOR_MAP = {
    "nvidia": "-DUSE_NVIDIA_ADAPTOR",
    "iluvatar_corex": "-DUSE_ILUVATAR_COREX_ADAPTOR",
    "cambricon": "-DUSE_CAMBRICON_ADAPTOR",
    "metax": "-DUSE_METAX_ADAPTOR",
    "musa": "-DUSE_MUSA_ADAPTOR",
    "du": "-DUSE_DU_ADAPTOR",
    "klx": "-DUSE_KUNLUNXIN_ADAPTOR",
    "ascend": "-DUSE_ASCEND_ADAPTOR",
    "amd": "-DUSE_AMD_ADAPTOR",
    "tsm": "-DUSE_TSM_ADAPTOR",
    "enflame": "-DUSE_ENFLAME_ADAPTOR",
    "sunrise": "-DUSE_SUNRISE_ADAPTOR",
    "ppu": "-DUSE_PPU_ADAPTOR",
}

# Adaptor name -> Make variable (for root setup.py make invocation)
ADAPTOR_TO_MAKE_FLAG = {
    "nvidia": "USE_NVIDIA",
    "ascend": "USE_ASCEND",
    "iluvatar_corex": "USE_ILUVATAR_COREX",
    "cambricon": "USE_CAMBRICON",
    "metax": "USE_METAX",
    "musa": "USE_MUSA",
    "klx": "USE_KUNLUNXIN",
    "du": "USE_DU",
    "amd": "USE_AMD",
    "tsm": "USE_TSM",
    "enflame": "USE_ENFLAME",
    "sunrise": "USE_SUNRISE",
    "ppu": "USE_PPU",
}

VALID_ADAPTORS = list(ADAPTOR_MAP.keys())
ADAPTOR_BY_FLAG = {flag: name for name, flag in ADAPTOR_MAP.items()}

TORCH_BACKEND_VENDOR = "vendor"
TORCH_BACKEND_FLAGOS = "flagos"
VALID_TORCH_BACKENDS = (TORCH_BACKEND_VENDOR, TORCH_BACKEND_FLAGOS)
FLAGOS_ADAPTORS = ("ascend", "enflame")


@dataclass(frozen=True)
class TorchBackendConfig:
    name: str
    python_package: str
    device_name: str
    compile_flags: tuple


def resolve_torch_backend(adaptor):
    """Resolve and validate the Torch integration for an adaptor."""
    backend = os.environ.get("FLAGCX_TORCH_BACKEND", TORCH_BACKEND_VENDOR)
    backend = backend.strip().lower() or TORCH_BACKEND_VENDOR
    if backend not in VALID_TORCH_BACKENDS:
        raise RuntimeError(
            f"Invalid FLAGCX_TORCH_BACKEND={backend!r}. "
            f"Valid values: {', '.join(VALID_TORCH_BACKENDS)}"
        )
    if backend == TORCH_BACKEND_FLAGOS and adaptor not in FLAGOS_ADAPTORS:
        raise RuntimeError(
            "FLAGCX_TORCH_BACKEND=flagos currently supports only the "
            f"ascend and enflame adaptors, not {adaptor!r}"
        )

    if backend == TORCH_BACKEND_FLAGOS:
        return TorchBackendConfig(
            name=backend,
            python_package="torch_fl",
            device_name="flagos",
            compile_flags=("-DFLAGCX_TORCH_BACKEND_FLAGOS",),
        )

    vendor_packages = {
        "ascend": ("torch_npu", "npu"),
        "enflame": ("torch_gcu", "gcu"),
    }
    python_package, device_name = vendor_packages.get(adaptor, ("", ""))
    return TorchBackendConfig(
        name=backend,
        python_package=python_package,
        device_name=device_name,
        compile_flags=(),
    )

# Platform detection: command -> adaptor name
# Order matters: nvidia-smi and rocm-smi last (some platforms are CUDA/ROCm compatible)
_PLATFORM_COMMANDS = [
    ("ixsmi", "iluvatar_corex"),
    ("cnmon", "cambricon"),
    ("mx-smi", "metax"),
    ("hy-smi", "du"),
    ("xpu-smi", "klx"),
    ("mthreads-gmi", "musa"),
    ("npu-smi", "ascend"),
    ("tsm_smi", "tsm"),
    ("efsmi", "enflame"),
    ("rocm-smi", "amd"),
    ("ppu-smi", "ppu"),
    ("nvidia-smi", "nvidia"),
    ("pt-smi", "sunrise"),
]


def _detect_platform():
    """Auto-detect hardware platform by checking for platform-specific CLI tools."""
    for cmd, adaptor_name in _PLATFORM_COMMANDS:
        if shutil.which(cmd) is not None:
            return adaptor_name
    return None


def detect_adaptor():
    """Detect the adaptor from FLAGCX_ADAPTOR env var, --adaptor CLI arg, or
    USE_* env vars. Returns the adaptor name string. Defaults to 'nvidia'."""
    # Always consume the custom option before setuptools parses sys.argv.
    # Otherwise it is reported as an unknown setuptools command option when
    # an adaptor is already selected through the environment.
    cli_adaptor = ""
    if "--adaptor" in sys.argv:
        arg_index = sys.argv.index("--adaptor")
        del sys.argv[arg_index]
        if arg_index < len(sys.argv):
            cli_adaptor = sys.argv.pop(arg_index)
        else:
            print("No adaptor provided after '--adaptor'. Using default nvidia adaptor")

    adaptor = os.environ.get("FLAGCX_ADAPTOR", "").strip() or cli_adaptor

    # Check USE_* env vars
    if not adaptor:
        for name, make_flag in ADAPTOR_TO_MAKE_FLAG.items():
            if os.environ.get(make_flag, "0") == "1":
                adaptor = name
                break

    # Auto-detect platform
    if not adaptor:
        adaptor = _detect_platform()
        if adaptor:
            print(f"[flagcx] Auto-detected platform: {adaptor}")

    # Fail with guidance if nothing detected
    if not adaptor:
        print(
            "\n[flagcx] WARNING: Failed to auto-detect hardware platform.\n"
            "Please specify the adaptor manually using one of:\n"
            "  FLAGCX_ADAPTOR=<adaptor> pip install . --no-build-isolation\n"
            "  pip install . --no-build-isolation --adaptor <adaptor>\n"
            f"Valid adaptors: {VALID_ADAPTORS}\n"
        )
        sys.exit(1)

    assert adaptor in VALID_ADAPTORS, f"Invalid adaptor: {adaptor}. Valid: {VALID_ADAPTORS}"
    return adaptor


def detect_torch_flag():
    """Detect the torch version flag for conditional compilation."""
    torch_flag = "-DTORCH_VER_LT_250"
    try:
        import torch
        torch_version = vparse(torch.__version__.split("+")[0])
        if torch_version >= Version("2.5.0"):
            print("torch version >= 2.5.0, set TORCH_VER_GE_250 flag")
            torch_flag = "-DTORCH_VER_GE_250"
    except ImportError:
        print("Warning: torch not found.")
    return torch_flag


def _get_flagos_config():
    """Return include and library paths for the torch-fl runtime."""
    install_path = os.environ.get("FLAGOS_INSTALL_PATH", "").strip()
    include_dir = os.environ.get("FLAGOS_INCLUDE_DIR", "").strip()
    library_dir = os.environ.get("FLAGOS_LIBRARY_DIR", "").strip()

    if install_path:
        include_dir = include_dir or os.path.join(install_path, "include")
        library_dir = library_dir or os.path.join(install_path, "lib")

    if not include_dir or not library_dir:
        try:
            import torch_fl
        except ImportError as error:
            raise RuntimeError(
                "FLAGCX_TORCH_BACKEND=flagos requires torch-fl or explicit "
                "FLAGOS_INCLUDE_DIR and FLAGOS_LIBRARY_DIR"
            ) from error

        torch_fl_path = os.path.dirname(os.path.abspath(torch_fl.__file__))
        if not include_dir:
            include_candidates = [
                os.path.join(torch_fl_path, "include"),
                os.path.join(os.path.dirname(torch_fl_path), "csrc", "include"),
            ]
            include_dir = next(
                (
                    path
                    for path in include_candidates
                    if os.path.isfile(os.path.join(path, "flagos.h"))
                ),
                include_candidates[0],
            )
        library_dir = library_dir or os.path.join(torch_fl_path, "lib")

    header_path = os.path.join(include_dir, "flagos.h")
    library_path = os.path.join(library_dir, "libflagos.so")
    if not os.path.isfile(header_path):
        raise RuntimeError(
            f"FLAGCX_TORCH_BACKEND=flagos requires flagos.h; not found at {header_path}. "
            "Set FLAGOS_INSTALL_PATH or FLAGOS_INCLUDE_DIR."
        )
    if not os.path.isfile(library_path):
        raise RuntimeError(
            "FLAGCX_TORCH_BACKEND=flagos requires libflagos.so; not found at "
            f"{library_path}. Set FLAGOS_INSTALL_PATH or FLAGOS_LIBRARY_DIR."
        )
    return [include_dir], [library_dir], ["flagos"]


def get_device_config(adaptor_flag, torch_backend=None):
    """Return (extra_include_dirs, extra_library_dirs, extra_libs) for the
    given adaptor define flag."""
    include_dirs = []
    library_dirs = []
    libs = []
    adaptor = ADAPTOR_BY_FLAG[adaptor_flag]
    torch_backend = torch_backend or resolve_torch_backend(adaptor)

    if adaptor_flag == "-DUSE_NVIDIA_ADAPTOR":
        include_dirs += ["/usr/local/cuda/include"]
        library_dirs += ["/usr/local/cuda/lib64"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_ILUVATAR_COREX_ADAPTOR":
        include_dirs += ["/usr/local/corex/include"]
        library_dirs += ["/usr/local/corex/lib64"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_CAMBRICON_ADAPTOR":
        import torch_mlu
        neuware_home_path = os.getenv("NEUWARE_HOME")
        torch_mlu_path = torch_mlu.__file__.split("__init__")[0]
        torch_mlu_lib_dir = os.path.join(torch_mlu_path, "csrc/lib/")
        torch_mlu_include_dir = os.path.join(torch_mlu_path, "csrc/")
        torch_mlu_include_dir2 = os.path.join(torch_mlu_path, "csrc", "include")
        include_dirs += [f"{neuware_home_path}/include", torch_mlu_include_dir, torch_mlu_include_dir2]
        library_dirs += [f"{neuware_home_path}/lib64", torch_mlu_lib_dir]
        libs += ["cnrt", "cncl", "torch_mlu"]
    elif adaptor_flag == "-DUSE_METAX_ADAPTOR":
        include_dirs += ["/opt/maca/include"]
        library_dirs += ["/opt/maca/lib64"]
        try:
            import torch
            torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
            library_dirs += [torch_lib_dir]
            libs += ["c10_cuda", "torch_cuda"]
        except ImportError:
            libs += ["c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_MUSA_ADAPTOR":
        import torch_musa
        pytorch_musa_install_path = os.path.dirname(os.path.abspath(torch_musa.__file__))
        pytorch_library_path = os.path.join(pytorch_musa_install_path, "lib")
        library_dirs += ["/usr/local/musa/lib/", pytorch_library_path]
        libs += ["musa", "musart"]
    elif adaptor_flag == "-DUSE_DU_ADAPTOR":
        cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if not cuda_home:
            cuda_home = "/usr/local/cuda"
        include_dirs += [os.path.join(cuda_home, "include")]
        library_dirs += [os.path.join(cuda_home, "lib64")]

        # Hygon's PyTorch distribution uses the HIP library names while
        # exposing the CUDA-compatible runtime headers and libraries.
        try:
            import torch
            torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
            library_dirs += [torch_lib_dir]
            if os.path.exists(os.path.join(torch_lib_dir, "libtorch_hip.so")):
                libs += ["cuda", "cudart", "c10_hip", "torch_hip"]
            else:
                libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
        except ImportError:
            libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_KUNLUNXIN_ADAPTOR":
        include_dirs += ["/opt/kunlun/include"]
        library_dirs += ["/opt/kunlun/lib"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_ASCEND_ADAPTOR":
        if torch_backend.name == TORCH_BACKEND_FLAGOS:
            cann_home = (
                os.environ.get("ASCEND_HOME_PATH")
                or os.environ.get("DEVICE_HOME")
                or "/usr/local/Ascend/ascend-toolkit/latest"
            )
            include_dirs += [os.path.join(cann_home, "include")]
            library_dirs += [os.path.join(cann_home, "lib64")]
            libs += ["ascendcl"]
            flagos_includes, flagos_libdirs, flagos_libs = _get_flagos_config()
            include_dirs += flagos_includes
            library_dirs += flagos_libdirs
            libs += flagos_libs
        else:
            import torch_npu

            pytorch_npu_install_path = os.path.dirname(
                os.path.abspath(torch_npu.__file__)
            )
            pytorch_library_path = os.path.join(pytorch_npu_install_path, "lib")
            # CANN toolkit headers must come BEFORE torch_npu bundled third_party
            # ACL headers (torch_npu 2.11.0 bundles newer ACL headers incompatible
            # with CANN 8.5.1). We also symlink torch_npu's third_party/acl/inc/acl
            # to CANN's acl/ directory (see install.sh), but adding the CANN include
            # path here is a belt-and-suspenders fix for hccl.h etc.
            cann_home = os.environ.get("ASCEND_HOME_PATH", "")
            if cann_home:
                import platform as _pf

                arch = (
                    "aarch64-linux"
                    if _pf.machine().startswith("aarch")
                    else "x86_64-linux"
                )
                cann_include = os.path.join(cann_home, arch, "include")
                if os.path.isdir(cann_include):
                    include_dirs += [cann_include]
            include_dirs += [os.path.join(pytorch_npu_install_path, "include")]
            library_dirs += [pytorch_library_path]
            libs += ["torch_npu"]
    elif adaptor_flag == "-DUSE_AMD_ADAPTOR":
        include_dirs += ["/opt/rocm/include"]
        library_dirs += ["/opt/rocm/lib"]
        libs += ["hiprtc", "c10_hip", "torch_hip"]
    elif adaptor_flag == "-DUSE_TSM_ADAPTOR":
        import torch_txda
        txda_install_path = os.path.dirname(os.path.abspath(torch_txda.__file__))
        txda_library_path = os.path.join(txda_install_path, "lib")
        include_dirs += ["/usr/local/kuiper/include", os.path.join(txda_install_path, "include")]
        library_dirs += ["/usr/local/kuiper/lib", txda_library_path]
        libs += ["torch_txda", "hpgr"]
    elif adaptor_flag == "-DUSE_ENFLAME_ADAPTOR":
        include_dirs += ["/opt/tops/include"]
        library_dirs += ["/opt/tops/lib"]
        libs += ["topsrt"]
        if torch_backend.name == TORCH_BACKEND_FLAGOS:
            flagos_includes, flagos_libdirs, flagos_libs = _get_flagos_config()
            include_dirs += flagos_includes
            library_dirs += flagos_libdirs
            libs += flagos_libs
        else:
            import torch_gcu

            pytorch_gcu_install_path = os.path.dirname(os.path.abspath(torch_gcu.__file__))
            pytorch_library_path = os.path.join(pytorch_gcu_install_path, "lib")
            include_dirs += [os.path.join(pytorch_gcu_install_path, "include")]
            library_dirs += [pytorch_library_path]
            libs += ["torch_gcu"]
    elif adaptor_flag == "-DUSE_SUNRISE_ADAPTOR":
        import torch_ptpu
        torch_ptpu_dir = os.path.dirname(os.path.abspath(torch_ptpu.__file__))
        c_so_basename = os.path.basename(torch_ptpu._C.__file__)

        tang_toolkit_dir = os.environ.get("CMAKE_TANG_TOOLKIT_DIR", "/usr/local/tangrt")
        include_dirs += [
            os.path.join(torch_ptpu_dir, "include"),
            os.path.join(tang_toolkit_dir, "include"),
        ]
        library_dirs += [
            torch_ptpu_dir,
            os.path.join(tang_toolkit_dir, "lib"),
            os.path.join(tang_toolkit_dir, "lib", "linux-x86_64"),
        ]
        libs += [f":{c_so_basename}", "tangrt_shared"]
    elif adaptor_flag == "-DUSE_PPU_ADAPTOR":
        include_dirs += ["/usr/local/cuda/include"]
        library_dirs += ["/usr/local/cuda/lib64"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]

    return include_dirs, library_dirs, libs


def get_device_rpath_dirs(adaptor_flag, library_dirs, torch_backend=None):
    """Return runtime library paths that belong in the extension artifact."""
    return list(library_dirs)


def get_ext_classes(adaptor_flag):
    """Return (CppExtension, BuildExtension) for the given adaptor, or
    (None, None) if unavailable."""
    try:
        if adaptor_flag == "-DUSE_MUSA_ADAPTOR":
            from torch_musa.utils.musa_extension import MUSAExtension as CppExtension
            from torch_musa.utils.musa_extension import BuildExtension
        else:
            from torch.utils.cpp_extension import CppExtension, BuildExtension
        return CppExtension, BuildExtension
    except ImportError:
        print("Warning: CppExtension or BuildExtension not found.")
        return None, None
