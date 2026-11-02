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
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
