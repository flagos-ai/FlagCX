import os
import sys
import subprocess
import shutil
import multiprocessing

# Disable auto load flagcx when setup
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# Disable build isolation for torch dependency
if "PIP_NO_BUILD_ISOLATION" not in os.environ:
    os.environ["PIP_NO_BUILD_ISOLATION"] = "1"

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.join(ROOT_DIR, "plugin", "torch")

# Make the shared helper importable
sys.path.insert(0, PLUGIN_DIR)

from _build_config import (
    ADAPTOR_MAP,
    ADAPTOR_TO_MAKE_FLAG,
    detect_adaptor,
    detect_torch_flag,
    get_device_config,
    get_ext_classes,
)

# ---------------------------------------------------------------------------
# Adaptor & torch detection
# ---------------------------------------------------------------------------

adaptor = detect_adaptor()
print(f"[flagcx] Using {adaptor} adaptor")

adaptor_flag = ADAPTOR_MAP[adaptor]
adaptor_make_flag = ADAPTOR_TO_MAKE_FLAG[adaptor]
torch_flag = detect_torch_flag()

# ---------------------------------------------------------------------------
# Extension sources and include dirs
# ---------------------------------------------------------------------------

sources = [
    os.path.join("plugin", "torch", "flagcx", "src", "backend_flagcx.cpp"),
    os.path.join("plugin", "torch", "flagcx", "src", "utils_flagcx.cpp"),
]

include_dirs = [
    os.path.join(PLUGIN_DIR, "flagcx", "include"),
    os.path.join(ROOT_DIR, "flagcx", "include"),
    os.path.join(ROOT_DIR, "third-party", "json", "single_include"),
]

# Will be updated in build_ext to point at the built libflagcx.so
library_dirs = []
libs = ["flagcx"]

# Add device-specific paths
dev_includes, dev_libdirs, dev_libs = get_device_config(adaptor_flag)
include_dirs += dev_includes
library_dirs += dev_libdirs
libs += dev_libs

# ---------------------------------------------------------------------------
# Build extension classes
# ---------------------------------------------------------------------------

CppExtension, BuildExtension = get_ext_classes(adaptor_flag)

# ---------------------------------------------------------------------------
# Custom build_ext: run make first, then build torch extension
# ---------------------------------------------------------------------------

if BuildExtension is not None:
    class BuildExtWithMake(BuildExtension):
        def build_extensions(self):
            # -- Step 0: Ensure git submodules are initialized --
            submodule_marker = os.path.join(
                ROOT_DIR, "third-party", "json", "single_include"
            )
            if not os.path.isdir(submodule_marker):
                print("[flagcx] Initializing git submodules ...")
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=ROOT_DIR,
                )

            # -- Step 1: Build libflagcx.so via make --
            build_dir = os.path.join(ROOT_DIR, "build")
            lib_dir = os.path.join(build_dir, "lib")

            make_args = [f"BUILDDIR={build_dir}", f"{adaptor_make_flag}=1"]

            # Forward additional env vars to make
            env_to_make = [
                "DEVICE_HOME", "CCL_HOME", "HOST_CCL_HOME", "MPI_HOME", "UCX_HOME",
                "USE_GLOO", "USE_BOOTSTRAP", "USE_MPI", "USE_UCX", "USE_IBUC",
                "COMPILE_KERNEL",
            ]
            for var in env_to_make:
                val = os.environ.get(var, "")
                if val:
                    make_args.append(f"{var}={val}")

            nproc = str(multiprocessing.cpu_count())
            make_cmd = ["make", "-C", ROOT_DIR, "-j", nproc] + make_args
            print(f"[flagcx] Running: {' '.join(make_cmd)}")
            subprocess.check_call(make_cmd)

            src_so = os.path.join(lib_dir, "libflagcx.so")

            # -- Step 2: Update library_dirs and rpath for the extension --
            for ext in self.extensions:
                if lib_dir not in ext.library_dirs:
                    ext.library_dirs.insert(0, lib_dir)
                # Set $ORIGIN rpath so _C.so finds libflagcx.so in the same directory
                # Preserve device-specific rpaths so runtime linker can find device libs
                origin_rpath = "-Wl,-rpath,$ORIGIN"
                dev_rpaths = ["-Wl,-rpath," + d for d in dev_libdirs]
                ext.extra_link_args = [
                    arg for arg in ext.extra_link_args
                    if not arg.startswith("-Wl,-rpath,")
                ]
                ext.extra_link_args.append(origin_rpath)
                ext.extra_link_args.extend(dev_rpaths)

            # -- Step 3: Build the torch C++ extension --
            super().build_extensions()

            # -- Step 4: Copy libflagcx.so to where it's needed --
            # Into the build output dir (for wheels / regular installs)
            build_pkg_dir = os.path.join(self.build_lib, "flagcx")
            os.makedirs(build_pkg_dir, exist_ok=True)
            dst_build_so = os.path.join(build_pkg_dir, "libflagcx.so")
            print(f"[flagcx] Copying {src_so} -> {dst_build_so}")
            shutil.copy2(src_so, dst_build_so)

            # Into the source package dir (for editable installs, where
            # _C.so lives in-tree and uses $ORIGIN rpath to find it)
            src_pkg_dir = os.path.join(ROOT_DIR, "plugin", "torch", "flagcx")
            dst_src_so = os.path.join(src_pkg_dir, "libflagcx.so")
            print(f"[flagcx] Copying {src_so} -> {dst_src_so}")
            shutil.copy2(src_so, dst_src_so)
else:
    BuildExtWithMake = None

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------

ext_modules = []
if CppExtension is not None:
    module = CppExtension(
        name="flagcx._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": [adaptor_flag, torch_flag]},
        extra_link_args=[],
        library_dirs=library_dirs,
        libraries=libs,
    )
    ext_modules.append(module)

cmdclass = {}
if BuildExtWithMake is not None:
    cmdclass["build_ext"] = BuildExtWithMake

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Ensure build/ exists so egg_info can write there (via setup.cfg egg_base)
os.makedirs(os.path.join(ROOT_DIR, "build"), exist_ok=True)

setup(
    name="flagcx",
    version="0.10.0",
    description="FlagCX: A unified collective communication library",
    package_dir={"flagcx": "plugin/torch/flagcx"},
    packages=["flagcx"],
    package_data={"flagcx": ["*.so"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={"torch.backends": ["flagcx = flagcx:init"]},
)