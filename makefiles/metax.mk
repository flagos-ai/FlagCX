# makefiles/platforms/metax.mk
# MetaX platform configuration.

DEVICE_HOME  ?= /opt/maca
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include $(DEVICE_HOME)/include/mcr
DEVICE_LINK  := -lmcruntime -lmccompiler
DEVICE_PLATFORM := MACA
DEVICE_COMPILER := $(DEVICE_HOME)/mxgpu_llvm/bin/mxcc
DEVICE_COMPILE_FLAG := -c --maca-path=$(DEVICE_HOME) -offload-arch=xcore1000 -fgpu-rdc -fPIC -MMD -MP -g
DEVICE_LINK_FLAG := --maca-path=$(DEVICE_HOME) -offload-arch=xcore1000 -fgpu-rdc -maca-link -fPIC
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /opt/maca
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include/mccl
CCL_LINK    := -lmccl
ADAPTOR_FLAG := -DUSE_METAX_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
