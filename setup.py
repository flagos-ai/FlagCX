import os
import sys
import subprocess
import shutil
import glob
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
    get_device_rpath_dirs,
    get_ext_classes,
    resolve_torch_backend,
)

try:
    from _version_scheme import version_scheme
except ImportError as exc:
    # --no-build-isolation (the documented install path) skips [build-system]
    # requires, and setuptools would then silently build a 0.0.0 artifact.
    raise SystemExit(
        "setuptools_scm is required to build FlagCX from a git checkout; "
        "install it with: pip install 'setuptools_scm>=8'"
    ) from exc

# ---------------------------------------------------------------------------
# Adaptor & torch detection
# ---------------------------------------------------------------------------

adaptor = detect_adaptor()
print(f"[flagcx] Using {adaptor} adaptor")

adaptor_flag = ADAPTOR_MAP[adaptor]
adaptor_make_flag = ADAPTOR_TO_MAKE_FLAG[adaptor]
torch_flag = detect_torch_flag()
torch_backend = resolve_torch_backend(adaptor)
torch_backend_flags = list(torch_backend.compile_flags)
print(f"[flagcx] Using {torch_backend.name} torch backend")

# ---------------------------------------------------------------------------
# Extension sources and include dirs
# ---------------------------------------------------------------------------

sources = [
    os.path.join("plugin", "torch", "flagcx", "src", "backend_flagcx.cpp"),
    os.path.join("plugin", "torch", "flagcx", "src", "utils_flagcx.cpp"),
]

VENDORED_JSON_INCLUDE_DIR = os.path.join(
    ROOT_DIR, "third-party", "json", "single_include"
)
JSON_INCLUDE_DIR = os.environ.get("JSON_INCLUDE_DIR") or VENDORED_JSON_INCLUDE_DIR

include_dirs = [
    os.path.join(PLUGIN_DIR, "flagcx", "include"),
    os.path.join(ROOT_DIR, "flagcx", "include"),
    JSON_INCLUDE_DIR,
]

# Will be updated in build_ext to point at the built libflagcx.so
library_dirs = []
libs = ["flagcx"]

# Add device-specific paths
dev_includes, dev_libdirs, dev_libs = get_device_config(
    adaptor_flag, torch_backend
)
include_dirs += dev_includes
library_dirs += dev_libdirs
libs += dev_libs

# Which bindings/ir Makefile produces each adaptor's device bitcode, and under
# what variable names. nvidia's comm-traits branch (Default vs CCL) is a library
# property only knowable at build time, so its adaptor flag is a make variable
# the caller must supply; iluvatar has only the Default branch, so its Makefile
# states the flag itself and the caller supplies none. Any other adaptor has no
# bindings/ir Makefile, so requesting bitcode for it is an error, not a skip.
BITCODE_MAKEFILES = {
    "nvidia": {
        "make_dir": "bindings/ir/nvidia",
        "arch_var": "BITCODE_LIB_ARCH",
        "adaptor_flag_var": "ADAPTOR_FLAG",
        # NCCL's device headers use `typeof`, which clang refuses under the
        # Makefile's default -std=c++17.
        "cxx_std": "gnu++17",
        "extra_env": ("DEVICE_HOME", "CCL_HOME"),
    },
    "iluvatar": {
        "make_dir": "bindings/ir/iluvatar",
        "arch_var": "ILUVATAR_ARCH",
        "adaptor_flag_var": None,
        "cxx_std": None,
        # COREX_CLANG is derived from DEVICE_HOME by the Makefile; llvm-dis /
        # llvm-as / opt / llvm-nm must be on PATH (the host LLVM toolchain).
        "extra_env": ("DEVICE_HOME",),
    },
}

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
            # -- Step 0: Resolve nlohmann-json headers --
            json_header = os.path.join(JSON_INCLUDE_DIR, "nlohmann", "json.hpp")
            if (
                JSON_INCLUDE_DIR == VENDORED_JSON_INCLUDE_DIR
                and not os.path.isfile(json_header)
            ):
                print("[flagcx] Initializing git submodules ...")
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=ROOT_DIR,
                )
            if not os.path.isfile(json_header):
                raise RuntimeError(
                    f"nlohmann/json.hpp not found under JSON_INCLUDE_DIR={JSON_INCLUDE_DIR}"
                )

            # -- Step 1: Build libflagcx.so via make --
            build_dir = os.path.join(ROOT_DIR, "build")
            lib_dir = os.path.join(build_dir, "lib")

            make_args = [
                f"BUILDDIR={build_dir}",
                f"{adaptor_make_flag}=1",
                f"JSON_INCLUDE_DIR={JSON_INCLUDE_DIR}",
            ]

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

            # -- Step 1b: Build the device bitcode (opt-in) --
            # A kernel consumer of the Device API links this bitcode through
            # `extern_libs`, and it comes from a second make: bindings/ir is not
            # part of the root Makefile, because it produces clang's IR rather
            # than the library's objects. It shares BUILDDIR with the make
            # above, so build/lib and build/include end up as the one tree
            # Step 4 copies out of.
            #
            # Opt-in because it needs a clang that targets the device, which an
            # ordinary build environment has neither reason nor way to carry.
            # FLAGCX_BITCODE_ARCH names the architecture to compile at; with it
            # unset nothing here runs. Which Makefile it drives is a property of
            # the adaptor (BITCODE_MAKEFILES), not of the value: the arch for a
            # CUDA backend is an sm_*, for iluvatar an ivcore*, so the two can
            # never be confused.
            bitcode_arch = os.environ.get("FLAGCX_BITCODE_ARCH", "")
            bitcode_bc = ""
            if bitcode_arch:
                bitcode_cfg = BITCODE_MAKEFILES.get(adaptor)
                if bitcode_cfg is None:
                    raise RuntimeError(
                        "FLAGCX_BITCODE_ARCH is set but this build is for the "
                        f"{adaptor} adaptor, which has no bindings/ir Makefile, "
                        "so there is no device bitcode to compile"
                    )
                bitcode_cmd = [
                    "make",
                    "-C",
                    os.path.join(ROOT_DIR, *bitcode_cfg["make_dir"].split("/")),
                    f"BUILDDIR={build_dir}",
                    f"{bitcode_cfg['arch_var']}={bitcode_arch}",
                ]
                if bitcode_cfg["cxx_std"] is not None:
                    bitcode_cmd.append(f"BITCODE_CXX_STD={bitcode_cfg['cxx_std']}")
                if bitcode_cfg["adaptor_flag_var"] is not None:
                    # The comm-traits branch is a property of the library: the
                    # nvidia makefile picks it from the NCCL headers it finds.
                    # The bitcode's own Makefile never reads that file, so the
                    # branch reaches it only as a value — and without one it
                    # would compile against its default backend while the
                    # library shipped beside it talks to another.
                    bitcode_adaptor_flags = os.environ.get(
                        "FLAGCX_BITCODE_ADAPTOR_FLAGS", ""
                    )
                    if not bitcode_adaptor_flags:
                        raise RuntimeError(
                            "FLAGCX_BITCODE_ARCH is set but "
                            "FLAGCX_BITCODE_ADAPTOR_FLAGS is empty, so the "
                            "bitcode would be compiled for the wrong device "
                            "API backend"
                        )
                    bitcode_cmd.append(
                        f"{bitcode_cfg['adaptor_flag_var']}={bitcode_adaptor_flags}"
                    )
                for var in bitcode_cfg["extra_env"]:
                    val = os.environ.get(var, "")
                    if val:
                        bitcode_cmd.append(f"{var}={val}")
                print(f"[flagcx] Running: {' '.join(bitcode_cmd)}")
                subprocess.check_call(bitcode_cmd)
                bitcode_bc = os.path.join(lib_dir, "libflagcx_device.bc")
                if not os.path.isfile(bitcode_bc):
                    raise RuntimeError(
                        "the device bitcode make exited 0 but produced no "
                        f"{bitcode_bc}"
                    )

            # -- Step 2: Update library_dirs and rpath for the extension --
            for ext in self.extensions:
                if lib_dir not in ext.library_dirs:
                    ext.library_dirs.insert(0, lib_dir)
                # Set $ORIGIN/lib rpath so _C.so finds libflagcx.so in the
                # package's lib/ directory
                # Preserve device-specific rpaths so runtime linker can find device libs
                origin_rpath = "-Wl,-rpath,$ORIGIN/lib"
                dev_rpaths = [
                    "-Wl,-rpath," + d
                    for d in get_device_rpath_dirs(
                        adaptor_flag, dev_libdirs, torch_backend
                    )
                ]
                ext.extra_link_args = [
                    arg for arg in ext.extra_link_args
                    if not arg.startswith("-Wl,-rpath,")
                ]
                ext.extra_link_args.append(origin_rpath)
                ext.extra_link_args.extend(dev_rpaths)

            # -- Step 3: Build the torch C++ extension --
            super().build_extensions()

            # -- Step 4: Copy the built payload into the package --
            # Both trees, because `package_dir={"": "src"}` has setuptools
            # collect package_data from the source tree while a wheel is
            # assembled out of build_lib: a file in only one of them is a file
            # that ships or a file that imports, never both.
            #
            # All of it is data rather than an entry point, so nothing at import
            # time notices any of it is missing. The headers are the six the
            # root make's `all` target exports plus the wrapper the bitcode make
            # writes beside them.
            payload = [src_so]
            headers = []
            if bitcode_bc:
                payload.append(bitcode_bc)
                headers = sorted(
                    glob.glob(os.path.join(build_dir, "include", "*.h"))
                )
                if not headers:
                    raise RuntimeError(
                        f"{build_dir}/include is empty: the headers a bitcode "
                        "consumer compiles against did not get exported"
                    )

            for base in (self.build_lib, os.path.join(ROOT_DIR, "src")):
                dst_lib = os.path.join(base, "flagcx", "lib")
                os.makedirs(dst_lib, exist_ok=True)
                # Editable installs keep _C.so in-tree and reach the library
                # through its $ORIGIN/lib rpath, so the source tree is not a
                # convenience here — it is where the loader looks.
                for path in payload:
                    print(f"[flagcx] Copying {path} -> {dst_lib}")
                    shutil.copy2(path, dst_lib)
                if headers:
                    dst_inc = os.path.join(base, "flagcx", "include")
                    os.makedirs(dst_inc, exist_ok=True)
                    for path in headers:
                        print(f"[flagcx] Copying {path} -> {dst_inc}")
                        shutil.copy2(path, dst_inc)
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
        extra_compile_args={
            "cxx": [adaptor_flag, torch_flag] + torch_backend_flags
        },
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
    use_scm_version={
        "version_scheme": version_scheme,
        "local_scheme": "no-local-version",
    },
    description="FlagCX: A unified collective communication library",
    package_dir={"": "src"},
    packages=["flagcx"],
    package_data={"flagcx": ["lib/*.so", "lib/*.bc", "include/*.h"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={"torch.backends": ["flagcx = flagcx:init"]},
)
