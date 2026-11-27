# makefiles/platforms/metax.mk
# MetaX platform configuration.

DEVICE_HOME  ?= /opt/maca
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include $(DEVICE_HOME)/include/mcr
DEVICE_LINK  := -lmcruntime -lmccompiler
DEVICE_PLATFORM := MACA
DEVICE_COMPILER := $(DEVICE_HOME)/mxgpu_llvm/bin/mxcc
METAX_ARCH ?= $(if $(strip $(DEVICE_BITCODE_ARCH)),$(DEVICE_BITCODE_ARCH),xcore1000)
ifneq ($(strip $(DEVICE_BITCODE_ARCH)),)
ifneq ($(METAX_ARCH),$(DEVICE_BITCODE_ARCH))
$(error METAX_ARCH must match DEVICE_BITCODE_ARCH)
endif
endif
DEVICE_COMPILE_FLAG := -c -x maca --maca-path=$(DEVICE_HOME) -offload-arch=$(METAX_ARCH) -fgpu-rdc -fPIC -MMD -MP -g -std=c++17 -DFLAGCX_METAX_DEVICE_COMPILE=1
DEVICE_LINK_FLAG := --maca-path=$(DEVICE_HOME) -offload-arch=$(METAX_ARCH) -fgpu-rdc -maca-link -fPIC
# mxcc performs device linking during the final -maca-link invocation.
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /opt/maca
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include/mccl
CCL_LINK    := -lmccl
ADAPTOR_FLAG := -DUSE_METAX_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/metax
PLATFORM_KERNEL_SRCS := $(PLATFORM_KERNEL_DIR)/device_api_host_helpers.cu \
                        $(PLATFORM_KERNEL_DIR)/flagcx_reduce_kernel_device.cu
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
