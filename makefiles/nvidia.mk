# makefiles/platforms/nvidia.mk
# NVIDIA CUDA platform configuration.

DEVICE_HOME  ?= /usr/local/cuda

include $(dir $(lastword $(MAKEFILE_LIST)))nvidia_gencode.mk
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include $(DEVICE_HOME)/include/cccl
DEVICE_LINK  := -lcudart -lcuda
DEVICE_PLATFORM := CUDA
DEVICE_COMPILER := $(DEVICE_HOME)/bin/nvcc
DEVICE_COMPILE_FLAG := -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g $(DEVICE_COMPILER_GENCODE)
DEVICE_LINK_FLAG := --cudart=shared -Xcompiler -fPIC $(DEVICE_COMPILER_GENCODE)
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /usr/local/nccl/build
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_NVIDIA_ADAPTOR

ifeq ($(NVCC_GENCODE_MULTICAST_UNSUPPORTED), 1)
  ADAPTOR_FLAG += -DNVCC_GENCODE_MULTICAST_UNSUPPORTED
endif

# --- Device API backend selection ---
ifeq ($(USE_SHMEM), 1)
  ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_SHMEM
  DEVICE_LINK  += -L$(SHMEM_HOME)/lib -lnvshmem_device -lnvshmem_host
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/shmem/nvshmem_adaptor.cc \
                         flagcx/adaptor/device_api/nvshmem_dev_api_backend.cc
else ifeq ($(FORCE_DEFAULT_PATH), 1)
  ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_DEFAULT
  PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
else
  NCCL_VERSION_MAJOR := $(shell grep '\#define NCCL_MAJOR' $(CCL_INCLUDE)/nccl.h 2>/dev/null | awk '{print $$3}')
  NCCL_VERSION_MINOR := $(shell grep '\#define NCCL_MINOR' $(CCL_INCLUDE)/nccl.h 2>/dev/null | awk '{print $$3}')
  ifeq ($(NCCL_VERSION_MAJOR),)
    $(info WARNING: NCCL header not found at $(CCL_INCLUDE)/nccl.h — using DefaultBackend)
  endif
  NCCL_VERSION_OK := $(shell [ -n "$(NCCL_VERSION_MAJOR)" ] && [ "$(NCCL_VERSION_MAJOR)" -gt 2 -o \( "$(NCCL_VERSION_MAJOR)" -eq 2 -a "$(NCCL_VERSION_MINOR)" -ge 29 \) ] 2>/dev/null && echo 1 || echo 0)
  ifeq ($(NCCL_VERSION_OK), 1)
    ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_CCL
    PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/nccl_dev_api_backend.cc
  else
    ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_DEFAULT
    PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
  endif
endif

# --- Kernel sources ---
PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/nvidia
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
