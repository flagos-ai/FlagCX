# makefiles/platforms/musa.mk
# MUSA platform configuration.

DEVICE_HOME  ?= /usr/local/musa
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lmusart -lmusa
DEVICE_PLATFORM := MUSA

# mcc is the MUSA clang driver. Two properties differ from nvcc/hipcc and are
# load-bearing here:
#   1. The main Makefile's device rule passes no -c, so the platform flag has to
#      carry it.
#   2. There is no separate device-link step: `mcc -dlink` is rejected by the
#      driver (it falls back to a host link and fails on a missing main and on
#      musaLaunchKernel). A single .mu object already carries a complete device
#      image, so DEVICE_NEEDS_DLINK is 0 below.
#
# --offload-arch must match the real device: MTT S5000 reports major=3 minor=1,
# i.e. mp_31. mp_32 compiles but every launch then fails with
# musaErrorInvalidDeviceFunction.
MUSA_ARCH ?= mp_31
DEVICE_COMPILER := $(DEVICE_HOME)/bin/mcc
DEVICE_COMPILER_GENCODE := --offload-arch=$(MUSA_ARCH)
DEVICE_COMPILE_FLAG := -std=c++17 -c -fPIC -MMD -MP -g \
    $(DEVICE_COMPILER_GENCODE)
DEVICE_LINK_FLAG :=
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := mu

CCL_HOME    ?= /usr/local/musa
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lmccl -lmusa
ADAPTOR_FLAG := -DUSE_MUSA_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/musa
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
