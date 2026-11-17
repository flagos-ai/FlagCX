# makefiles/platforms/musa.mk
# MUSA platform configuration (MTT S5000 / MUSA 4.3 / mcc 5.1.0).
#
# Main-library kernel objects use compiler-managed image registration.
# The public test kernel objects explicitly link their RDC device images.

DEVICE_HOME  ?= /usr/local/musa
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
# Host-linked test kernel objects call the MUSA runtime directly.
DEVICE_LINK  := -lmusart -lmusa
DEVICE_PLATFORM := MUSA
MUSA_ARCH ?= mp_31
DEVICE_COMPILER := $(DEVICE_HOME)/bin/mcc
DEVICE_COMPILE_FLAG := -std=c++17 -O2 -c -mtgpu \
    --musa-path=$(DEVICE_HOME) --offload-arch=$(MUSA_ARCH) \
    -fgpu-rdc -fPIC -MMD -MP -g -DFLAGCX_MUSA_DEVICE_COMPILE=1
DEVICE_LINK_FLAG := -mtgpu --musa-path=$(DEVICE_HOME) \
    --offload-arch=$(MUSA_ARCH) -fgpu-rdc -fPIC
# MUSA uses image embedding rather than an nvcc-style separate dlink object.
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := mu

CCL_HOME    ?= /usr/local/musa
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lmccl -lmusa
ADAPTOR_FLAG := -DUSE_MUSA_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT

# Main-library DevNet construction and persistent collective launcher.
PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/musa
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
# device_api/ is not globbed by the Makefile; list the DevApiBackend explicitly.
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
