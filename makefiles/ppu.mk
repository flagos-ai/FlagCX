# makefiles/platforms/ppu.mk
# PPU platform configuration.

DEVICE_HOME  ?= /usr/local/cuda

DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lcudart -lcuda
# PPU exposes a CUDA-compatible device stack; tests key kernel/IR dirs off
# this (test/make.inc requires it non-empty). No PPU kernel sources exist,
# so PLATFORM_KERNEL_* stay empty below.
DEVICE_PLATFORM := CUDA
# The library has no PPU device sources, but visibility conformance links a
# minimal CUDA-compatible consumer kernel into the RMA test binary.
DEVICE_COMPILER := $(DEVICE_HOME)/bin/nvcc
DEVICE_COMPILE_FLAG := -c --cudart=shared -Xcompiler -fPIC -MMD -MP -g
DEVICE_LINK_FLAG := --cudart=shared -Xcompiler -fPIC
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /usr/local/nccl/build
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_PPU_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
# Device API backend: flagcx_device.cc dispatches through devApiBackend,
# which must be provided by a backend source (see nvidia.mk).
ADAPTOR_FLAG += -DFLAGCX_COMM_TRAITS_DEFAULT
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
