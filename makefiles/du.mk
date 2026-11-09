# makefiles/platforms/du.mk
# DU platform configuration.

DEVICE_HOME  ?= $(HIP_PATH)
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lamdhip64
DEVICE_PLATFORM := DU
DEVICE_COMPILER := $(DEVICE_HOME)/bin/hipcc
DEVICE_COMPILE_FLAG := -std=c++17 -c -fPIC -MMD -MP -fgpu-rdc -g
DEVICE_LINK_FLAG := -fgpu-rdc -fPIC
ifneq ($(strip $(DEVICE_BITCODE_ARCH)),)
  DEVICE_COMPILE_FLAG += --offload-arch=$(DEVICE_BITCODE_ARCH)
  DEVICE_LINK_FLAG += --offload-arch=$(DEVICE_BITCODE_ARCH)
endif
DEVICE_FILE_EXTENSION := hip
DEVICE_NEEDS_DLINK := 0

CCL_HOME    ?= $(HIP_PATH)
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include/rccl
CCL_LINK    := -lrccl
ADAPTOR_FLAG := -DUSE_DU_ADAPTOR -D__HIP_PLATFORM_AMD__

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/du
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))

# --- Device API backend selection ---
# USE_SHMEM is the repository-wide one-sided selector; on DU it selects
# DUSHMEM (Sugon HIP SHMEM, DTK) and the corresponding CommTraits, just as it
# selects NVSHMEM and NvshmemBackend on NVIDIA.
# DUSHMEM headers sit under $(SHMEM_HOME)/include (added to INCLUDEDIR by the
# top-level Makefile when USE_SHMEM=1). Host entry points (dushmemx_init,
# dushmemx_malloc, ...) come from the shared libdushmem_host.
#
# Device-side symbols need the static archive, passed as a plain input file:
#   libdushmem_host.so    — host API
#   libdushmem_device.a   — HIP_SEPARABLE_COMPILATION + -fgpu-rdc, carries the
#                           device bundle (dushmemi_transfer_quiet,
#                           dushmemi_transfer_amo_nonfetch, ...)
# HIP device code cannot be resolved across a .so boundary, so the archive has
# to be linked into libflagcx.so. And the clang/dcc driver only harvests device
# bundles from plain input files: neither -ldushmem_device nor
# -Wl,--whole-archive -ldushmem_device reaches the amdgcn-link step, they both
# leave dushmemi_transfer_* undefined. Spelling the archive path out works
# because the driver then treats it as an input file. Verified on nmz17.
# (DUSHMEM_BUILD_BITCODE_LIBRARY is rejected by the DTK/HIP backend, so there
# is no .bc to link instead, and DUSHMEM_ENABLE_ALL_DEVICE_INLINING defaults to
# OFF, which is why these symbols live in the archive at all.)
# --- DUSHMEM location ---
# SHMEM_HOME defaults to /usr/local/nvshmem (top-level Makefile / test
# make.inc), which is wrong on DU. When the user did not set it explicitly,
# auto-detect the DTK-bundled DUSHMEM so a plain
#   source $DTKROOT/env.sh && make USE_DU=1 USE_SHMEM=1
# works out of the box. Required layout (used by the link line below):
#   $(SHMEM_HOME)/lib/libdushmem_device.a + libdushmem_host.so
#   $(SHMEM_HOME)/include/dushmem.h
# Known-good locations:
#   DTK-bundled:  $DTKROOT/dushmem   (lib/ and include/ are symlinks)
#   self-built:   <dushmem-hip>/build/src
ifeq ($(USE_SHMEM), 1)
  ifeq ($(filter environment command line,$(origin SHMEM_HOME)),)
    DU_SHMEM_AUTODETECT := $(firstword $(foreach d,\
        $(DTKROOT)/dushmem $(ROCM_PATH)/dushmem $(HIP_PATH)/../dushmem,\
        $(if $(wildcard $(d)/lib/libdushmem_device.a),$(abspath $(d)))))
    ifneq ($(DU_SHMEM_AUTODETECT),)
      SHMEM_HOME := $(DU_SHMEM_AUTODETECT)
      $(info du.mk: SHMEM_HOME not set - auto-detected DUSHMEM at $(SHMEM_HOME))
      $(info du.mk: NOTE the DTK-bundled DUSHMEM may lack local fixes; to link a)
      $(info du.mk: self-built dushmem-hip instead, export SHMEM_HOME=<dushmem-hip>/build/src)
    endif
  endif
  ifeq ($(wildcard $(SHMEM_HOME)/lib/libdushmem_device.a),)
    $(error USE_SHMEM=1 on DU requires DUSHMEM, but $(SHMEM_HOME)/lib/libdushmem_device.a was not found. Export SHMEM_HOME to a DUSHMEM install, e.g. $$DTKROOT/dushmem (DTK-bundled) or <dushmem-hip>/build/src (self-built))
  endif
endif

ifeq ($(USE_SHMEM), 1)
  ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_SHMEM
  # Keep archive host shadows/callbacks local to libflagcx.so, avoiding ELF
  # interposition with consumers. The archive remains a HIP device-link input;
  # each consumer initializes and registers its own device state.
  DEVICE_LINK_FLAG += -Wl,--exclude-libs,libdushmem_device.a
  DEVICE_LINK  += -L$(SHMEM_HOME)/lib -ldushmem_host \
                  $(SHMEM_HOME)/lib/libdushmem_device.a
  # Compile ONLY the dushmem host adaptor — do NOT use a shmem/*.cc wildcard,
  # since that would also pull in nvshmem_adaptor.cc (needs nvshmem.h and
  # would define a duplicate shmemAdaptor).
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/shmem/dushmem_adaptor.cc \
                         flagcx/adaptor/device_api/dushmem_dev_api_backend.cc
else
  # Without a one-sided backend, the default device-API backend must still be
  # compiled in — flagcx_device.cc references devApiBackend unconditionally,
  # otherwise executables fail to link with "undefined reference to
  # `devApiBackend'".
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
endif
