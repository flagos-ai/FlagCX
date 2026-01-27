import os
import sys

# Disable auto load flagcx when setup
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# Modern setuptools (>=64) uses pip for 'develop' which creates isolated build envs.
# For packages depending on torch, this often fails. 
# We try to disable build isolation if not explicitly set.
if "PIP_NO_BUILD_ISOLATION" not in os.environ:
    os.environ["PIP_NO_BUILD_ISOLATION"] = "1"

from setuptools import setup, find_packages
from packaging.version import Version, parse as vparse

adaptor = os.environ.get("FLAGCX_ADAPTOR", "nvidia")
if '--adaptor' in sys.argv:
    arg_index = sys.argv.index('--adaptor')
    sys.argv.remove("--adaptor")
    if arg_index < len(sys.argv):
        adaptor = sys.argv[arg_index]
        sys.argv.remove(adaptor)
    else:
        print("No adaptor provided after '--adaptor'. Using default nvidia adaptor")

valid_adaptors = ["nvidia", "iluvatar_corex", "cambricon", "metax", "du", "klx", "ascend", "musa", "amd", "tsm"]
assert adaptor in valid_adaptors, f"Invalid adaptor: {adaptor}"
print(f"Using {adaptor} adaptor")

adaptor_map = {
    "nvidia": "-DUSE_NVIDIA_ADAPTOR",
    "iluvatar_corex": "-DUSE_ILUVATAR_COREX_ADAPTOR",
    "cambricon": "-DUSE_CAMBRICON_ADAPTOR",
    "metax": "-DUSE_METAX_ADAPTOR",
    "musa": "-DUSE_MUSA_ADAPTOR",
    "du": "-DUSE_DU_ADAPTOR",
    "klx": "-DUSE_KUNLUNXIN_ADAPTOR",
    "ascend": "-DUSE_ASCEND_ADAPTOR",
    "amd": "-DUSE_AMD_ADAPTOR",
    "tsm": "-DUSE_TSM_ADAPTOR"
}
adaptor_flag = adaptor_map[adaptor]
torch_flag = "-DTORCH_VER_LT_250"

sources = ["flagcx/src/backend_flagcx.cpp", "flagcx/src/utils_flagcx.cpp"]
include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/flagcx/include",
    f"{os.path.dirname(os.path.abspath(__file__))}/../../flagcx/include",
    f"{os.path.dirname(os.path.abspath(__file__))}/../../third-party/json/single_include",
]

library_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/../../build/lib",
]

libs = ["flagcx"]
extra_link_args = ["-Wl,-rpath,"+f"{os.path.dirname(os.path.abspath(__file__))}/../../build/lib"]

try:
    import torch
    torch_version = vparse(torch.__version__.split("+")[0])
    if torch_version >= Version("2.5.0"):
        print("torch version >= 2.5.0, set TORCH_VER_GE_250 flag")
        torch_flag = "-DTORCH_VER_GE_250"
except ImportError:
    torch = None
    print("Warning: torch not found.")

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
    neuware_home_path=os.getenv("NEUWARE_HOME")
    pytorch_home_path=os.getenv("PYTORCH_HOME")
    torch_mlu_path = torch_mlu.__file__.split("__init__")[0]
    torch_mlu_lib_dir = os.path.join(torch_mlu_path, "csrc/lib/")
    torch_mlu_include_dir = os.path.join(torch_mlu_path, "csrc/")
    include_dirs += [f"{neuware_home_path}/include", torch_mlu_include_dir]
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
    library_dirs += ['/usr/local/musa/lib/',pytorch_library_path]
    libs += ["musa","musart"]
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
    txda_install_path = torch_txda.__path__[0]
    txda_library_path = txda_install_path
    include_dirs += [os.path.join(txda_install_path, "include")]
    include_dirs += ["/usr/local/kuiper/include"]
    library_dirs += ["/usr/local/kuiper/lib",txda_install_path]
    libs += ["torch_txda", "hpgr"]
    extra_link_args += [f'-Wl,-rpath,{txda_install_path}']

try:
    if adaptor_flag == "-DUSE_MUSA_ADAPTOR":
        from torch_musa.utils.musa_extension import MUSAExtension as CppExtension
        from torch_musa.utils.musa_extension import BuildExtension
    else:
        from torch.utils.cpp_extension import CppExtension, BuildExtension
except ImportError:
    CppExtension = None
    BuildExtension = None
    print("Warning: CppExtension or BuildExtension not found.")

ext_modules = []
if CppExtension is not None:
    module = CppExtension(
        name='flagcx._C',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': [adaptor_flag, torch_flag]
        },
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libs,
    )
    ext_modules.append(module)

cmdclass = {}
if BuildExtension is not None:
    cmdclass['build_ext'] = BuildExtension

setup(
    name="flagcx",
    version="0.8.0",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    entry_points={"torch.backends": ["flagcx = flagcx:init"]},
)
