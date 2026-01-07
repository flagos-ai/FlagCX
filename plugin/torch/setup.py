import os
import sys
from setuptools import setup
# Disable auto load flagcx when setup
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from setuptools import setup, find_packages
from packaging.version import Version, parse as vparse

adaptor_flag = "-DUSE_NVIDIA_ADAPTOR"
torch_flag = "-DTORCH_VER_LT_250"
if '--adaptor' in sys.argv:
    arg_index = sys.argv.index('--adaptor')
    sys.argv.remove("--adaptor")
    if arg_index < len(sys.argv):
        assert sys.argv[arg_index] in ["nvidia", "iluvatar_corex", "cambricon", "metax", "du", "klx", "ascend", "musa", "amd"], f"Invalid adaptor: {adaptor_flag}"
        print(f"Using {sys.argv[arg_index]} adaptor")
        if sys.argv[arg_index] == "iluvatar_corex":
            adaptor_flag = "-DUSE_ILUVATAR_COREX_ADAPTOR"
        elif sys.argv[arg_index] == "cambricon":
            adaptor_flag = "-DUSE_CAMBRICON_ADAPTOR"
        elif sys.argv[arg_index] == "metax":
            adaptor_flag = "-DUSE_METAX_ADAPTOR"
        elif sys.argv[arg_index] == "musa":
            adaptor_flag = "-DUSE_MUSA_ADAPTOR"
        elif sys.argv[arg_index] == "du":
            adaptor_flag = "-DUSE_DU_ADAPTOR"
        elif sys.argv[arg_index] == "klx":
            adaptor_flag = "-DUSE_KUNLUNXIN_ADAPTOR"
        elif sys.argv[arg_index] == "ascend":
            adaptor_flag = "-DUSE_ASCEND_ADAPTOR"
        elif sys.argv[arg_index] == "amd":
            adaptor_flag = "-DUSE_AMD_ADAPTOR"
    else:
        print("No adaptor provided after '--adaptor'. Using default nvidia adaptor")
    sys.argv.remove(sys.argv[arg_index])

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

if adaptor_flag == "-DUSE_NVIDIA_ADAPTOR":
    include_dirs += ["/usr/local/cuda/include"]
    library_dirs += ["/usr/local/cuda/lib64"]
    libs += ["cuda", "cudart", "c10_cuda", "torch_cuda"]
    import torch
    torch_version = vparse(torch.__version__.split("+")[0])
    if torch_version >= Version("2.5.0"):
        print("torch version >= 2.5.0, set TORCH_VER_GE_250 flag")
        torch_flag = "-DTORCH_VER_GE_250"
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
    torch_version = vparse(torch.__version__.split("+")[0])
    if torch_version >= Version("2.5.0"):
        torch_flag = "-DTORCH_VER_GE_250"
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

if adaptor_flag == "-DUSE_MUSA_ADAPTOR":
    from torch_musa.utils.musa_extension import MUSAExtension as CppExtension
    from torch_musa.utils.musa_extension import BuildExtension
else:
    from torch.utils.cpp_extension import CppExtension, BuildExtension

module = CppExtension(
    name='flagcx._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': [adaptor_flag, torch_flag]
    },
    extra_link_args=["-Wl,-rpath,"+f"{os.path.dirname(os.path.abspath(__file__))}/../../build/lib"],
    library_dirs=library_dirs,
    libraries=libs,
)

setup(
    name="flagcx",
    version="0.7.0",
    ext_modules=[module],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    entry_points={"torch.backends": ["flagcx = flagcx:init"]},
)
