# Native T-Head PPU SDK and PCCL configuration.
DEVICE_HOME ?= /usr/local/PPU_SDK
DEVICE_LIB := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include $(DEVICE_HOME)/include/cccl
DEVICE_LINK := -lhggc_wrapper -lhg_wrapper
DEVICE_PLATFORM := PPU
DEVICE_COMPILER := $(DEVICE_HOME)/bin/hgcc
DEVICE_COMPILE_FLAG := -x hg -std=c++17 -c -Xcompiler -fPIC -MMD -MP -rdc=true -g
DEVICE_LINK_FLAG := -Xcompiler -fPIC
DEVICE_FILE_EXTENSION := cu
ifeq ($(FLAGCX_COVERAGE), 1)
DEVICE_COMPILE_FLAG += -Xcompiler --coverage
endif

CCL_HOME ?= /usr/local/PPU_SDK
CCL_LIB := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
# PCCL is opened locally by the adaptor. A DT_NEEDED dependency would expose
# its private U2MM pointer symbols to BAREX's function relocations.
CCL_LINK :=
ADAPTOR_FLAG := -DUSE_PPU_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT

PLATFORM_KERNEL_DIR := flagcx/adaptor/kernel/ppu
PLATFORM_KERNEL_SRCS := $(PLATFORM_KERNEL_DIR)/device_api_host_helpers.cu \
                        $(PLATFORM_KERNEL_DIR)/flagcx_reduce_kernel_device.cu
PLATFORM_EXTRA_SRCS := flagcx/adaptor/device_api/default_dev_api_backend.cc
