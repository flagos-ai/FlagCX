"""
Shared build configuration for the flagcx torch plugin.

Used by both the root setup.py and plugin/torch/setup.py to avoid
duplicating adaptor detection, device-specific paths, and extension
class selection logic.
"""

import os
import shutil
import sys

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
}

VALID_ADAPTORS = list(ADAPTOR_MAP.keys())

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
    ("nvidia-smi", "nvidia"),
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
    adaptor = os.environ.get("FLAGCX_ADAPTOR", "").strip()

    # Check --adaptor CLI argument (consumed from sys.argv)
    if not adaptor and "--adaptor" in sys.argv:
        arg_index = sys.argv.index("--adaptor")
        sys.argv.remove("--adaptor")
        if arg_index < len(sys.argv):
            adaptor = sys.argv[arg_index]
            sys.argv.remove(adaptor)
        else:
            print("No adaptor provided after '--adaptor'. Using default nvidia adaptor")

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


def get_device_config(adaptor_flag):
    """Return (extra_include_dirs, extra_library_dirs, extra_libs) for the
    given adaptor define flag."""
    include_dirs = []
    library_dirs = []
    libs = []

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
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_MUSA_ADAPTOR":
        import torch_musa
        pytorch_musa_install_path = os.path.dirname(os.path.abspath(torch_musa.__file__))
        pytorch_library_path = os.path.join(pytorch_musa_install_path, "lib")
        library_dirs += ["/usr/local/musa/lib/", pytorch_library_path]
        libs += ["musa", "musart"]
    elif adaptor_flag == "-DUSE_DU_ADAPTOR":
        include_dirs += ["${CUDA_PATH}/include"]
        library_dirs += ["${CUDA_PATH}/lib64"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_KUNLUNXIN_ADAPTOR":
        include_dirs += ["/opt/kunlun/include"]
        library_dirs += ["/opt/kunlun/lib"]
        libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    elif adaptor_flag == "-DUSE_ASCEND_ADAPTOR":
        import torch_npu
        pytorch_npu_install_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        pytorch_library_path = os.path.join(pytorch_npu_install_path, "lib")
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
        import torch_gcu
        pytorch_gcu_install_path = os.path.dirname(os.path.abspath(torch_gcu.__file__))
        pytorch_library_path = os.path.join(pytorch_gcu_install_path, "lib")
        include_dirs += ["/opt/tops/include", os.path.join(pytorch_gcu_install_path, "include")]
        library_dirs += ["/opt/tops/lib", pytorch_library_path]
        libs += ["topsrt", "torch_gcu"]

    return include_dirs, library_dirs, libs


def get_ext_classes(adaptor_flag):
    """Return (CppExtension, BuildExtension) for the given adaptor, or
    (None, None) if unavailable."""
    try:
        if adaptor_flag == "-DUSE_MUSA_ADAPTOR":
            from torch_musa.utils.musa_extension import MUSAExtension as CppExtension
            from torch_musa.utils.musa_extension import BuildExtension
        elif adaptor_flag == "-DUSE_ENFLAME_ADAPTOR":
            from tops_extension import TopsBuildExtension as BuildExtension
            from tops_extension.torch import TopsTorchExtension as CppExtension
        else:
            from torch.utils.cpp_extension import CppExtension, BuildExtension
        return CppExtension, BuildExtension
    except ImportError:
        print("Warning: CppExtension or BuildExtension not found.")
        return None, None
