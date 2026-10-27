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
# top-level Makefile when USE_SHMEM=1). The host library is libdushmem_host;
# device-side PGAS/collective symbols resolve from the static
# libdushmem_device archive during the hipcc device link (-fgpu-rdc).
ifeq ($(USE_SHMEM), 1)
  ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_SHMEM
  DEVICE_LINK  += -L$(SHMEM_HOME)/lib -ldushmem_host -ldushmem_device
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
