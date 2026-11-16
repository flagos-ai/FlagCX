# makefiles/platforms/ppu.mk
# PPU platform configuration.

DEVICE_HOME  ?= /usr/local/cuda

DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include $(DEVICE_HOME)/include/cccl
DEVICE_LINK  := -lcudart -lcuda
# PPU exposes a CUDA-compatible device stack with its own Device API traits.
DEVICE_PLATFORM := PPU
DEVICE_COMPILER := $(DEVICE_HOME)/bin/nvcc
DEVICE_COMPILE_FLAG := -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g
DEVICE_LINK_FLAG := --cudart=shared -Xcompiler -fPIC
DEVICE_FILE_EXTENSION := cu
ifeq ($(FLAGCX_COVERAGE), 1)
DEVICE_COMPILE_FLAG += -Xcompiler --coverage
endif

CCL_HOME    ?= /usr/local/cuda
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_PPU_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/ppu
PLATFORM_KERNEL_SRCS := $(PLATFORM_KERNEL_DIR)/device_api_host_helpers.cu
# Device API backend: flagcx_device.cc dispatches through devApiBackend,
# which must be provided by a backend source (see nvidia.mk).
ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_DEFAULT
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
