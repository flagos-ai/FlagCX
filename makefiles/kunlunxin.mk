# makefiles/platforms/kunlunxin.mk
# KunlunXin platform configuration.

DEVICE_HOME  ?= /usr/local/xpu
DEVICE_LIB   := $(DEVICE_HOME)/so
DEVICE_INCLUDE := $(DEVICE_HOME)/include
# libbkcl.so's DT_NEEDED pulls in libxpucuda.so.1 (cu* CUDA-compat symbols)
# and libxpuml.so.1 (xpuml* device-management symbols). Modern ld does not
# auto-link a shared lib's transitive deps, so they must be listed explicitly.
DEVICE_LINK  := -lxpurt -lcudart -lxpucuda -lxpuml
# DEVICE_PLATFORM selects the test kernel subdir (test/kernel/<lowercase>): klx.
DEVICE_PLATFORM := KLX
# XPU (XTDK) clang toolchain: compiles the .xpu device sources
# ($(PLATFORM_KERNEL_DIR)/*.xpu) for the library build, and the test kernels
# under test/kernel/klx. The host .cc sources still go through g++.
XTDK_HOME ?= /workspace/my_flagcx/xtdk-llvm15-ubuntu2004_x86_64
DEVICE_COMPILER := $(XTDK_HOME)/bin/clang++
# The root kernel rule (Makefile: $(OBJDIR)/%.o: %.$(DEVICE_FILE_EXTENSION))
# passes DEVICE_COMPILE_FLAG verbatim and adds no -c of its own, so -c must be
# here or clang++ would try to LINK each .xpu into an executable. --xpu-arch
# selects P800 (xpu3); -fPIC because these objects land in libflagcx.so.
DEVICE_COMPILE_FLAG := -c --xpu-arch=xpu3 -std=c++17 -O2 -fno-builtin \
    --target=x86_64-linux-gnu -fPIC -MMD -MP
DEVICE_LINK_FLAG :=
# XTDK clang has no nvcc-style `-dlink` separate device-link step: each .xpu
# object already carries its device image (and device-links what it needs via
# -xpu-L/-xpu-l). Tell the root Makefile to skip kernel_dlink.o.
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := xpu

CCL_HOME    ?= /usr/local/xccl
CCL_LIB     := $(CCL_HOME)/so
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lbkcl
ADAPTOR_FLAG := -DUSE_KUNLUNXIN_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/kunlunxin
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))

# device_async_kernel.xpu is excluded from the library build: it does not compile
# with the XTDK toolchain (uses __global__ / mfence / XPUStream without including
# the XTDK headers, and through launch_kernel.h it pulls
# flagcx/core/include/device.h, whose static_asserts assume a 64-bit size_t while
# the xpu3 device pass has a 32-bit one).
# Dropping it is safe: `deviceAsyncKernel` is never linked in -- flagcx.cc only
# dlopen/dlsym's it when FLAGCX_DEVICE_FUNC_PATH is set, and group.cc falls back
# to the host semaphore (cpuAsyncKernel) whenever that function pointer is NULL,
# which is the default on every platform.
KLX_UNSUPPORTED_KERNEL_SRCS := $(PLATFORM_KERNEL_DIR)/device_async_kernel.$(DEVICE_FILE_EXTENSION)
KLX_SHMEM_KERNEL_SRCS := $(PLATFORM_KERNEL_DIR)/device_api_host_helpers.$(DEVICE_FILE_EXTENSION)
PLATFORM_KERNEL_SRCS := $(filter-out $(KLX_UNSUPPORTED_KERNEL_SRCS) $(KLX_SHMEM_KERNEL_SRCS),$(PLATFORM_KERNEL_SRCS))

# --- Device API backend selection ---
# USE_SHMEM is the repository-wide one-sided selector; on KunlunXin it selects
# xccl's XSHMEM and the corresponding CommTraits, just as it selects NVSHMEM
# and NvshmemBackend on NVIDIA.
# xshmem host symbols (xshmem_init/malloc/free/my_pe/n_pes) live in libbkcl
# (already linked via CCL_LINK); its headers sit under $(CCL_HOME)/include/xshmem,
# already on the include path via CCL_INCLUDE.
ifeq ($(USE_SHMEM), 1)
  SHMEM_HOME := $(CCL_HOME)
  ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_SHMEM
  PLATFORM_KERNEL_SRCS += $(KLX_SHMEM_KERNEL_SRCS)
  # Compile ONLY the xshmem host adaptor — do NOT use the shmem/*.cc wildcard
  # the way nvidia.mk does, since that would also pull in nvshmem_adaptor.cc
  # (needs nvshmem.h and would define a duplicate shmemAdaptor).
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/shmem/xshmem_adaptor.cc \
                         flagcx/adaptor/device_api/xshmem_dev_api_backend.cc
else
  # Mirror nvidia.mk: without a one-sided backend, the default device-API
  # backend must still be compiled in — flagcx_device.cc references
  # devApiBackend unconditionally, otherwise executables fail to link with
  # "undefined reference to `devApiBackend'".
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
endif
