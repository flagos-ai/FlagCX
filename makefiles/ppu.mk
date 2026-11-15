# makefiles/platforms/ppu.mk
# PPU platform configuration.

DEVICE_HOME  ?= /usr/local/cuda

DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lcudart -lcuda
# PPU exposes a CUDA-compatible device stack; test/make.inc maps CUDA to the
# existing NVIDIA test-kernel directory.
DEVICE_PLATFORM := CUDA
DEVICE_COMPILER := $(DEVICE_HOME)/bin/nvcc
DEVICE_COMPILE_FLAG := -std=c++17 -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g
DEVICE_LINK_FLAG := --cudart=shared -Xcompiler -fPIC
DEVICE_FILE_EXTENSION := cu
ifeq ($(FLAGCX_COVERAGE), 1)
DEVICE_COMPILE_FLAG += -Xcompiler --coverage
endif

CCL_HOME    ?= /usr/local/nccl/build
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_PPU_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/ppu
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
# Device API backend: flagcx_device.cc dispatches through devApiBackend,
# which must be provided by a backend source (see nvidia.mk).
ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_DEFAULT
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
