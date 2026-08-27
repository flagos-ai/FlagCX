# makefiles/platforms/du.mk
# DU platform configuration.

DEVICE_HOME  ?= $(CUDA_PATH)
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lcudart -lcuda
DEVICE_PLATFORM := DU
DEVICE_COMPILER := $(DEVICE_HOME)/bin/nvcc
DEVICE_COMPILE_FLAG := -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g
DEVICE_LINK_FLAG := --cudart=shared -Xcompiler -fPIC
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= $(CUDA_PATH)
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_DU_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/du
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
