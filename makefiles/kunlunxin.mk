# makefiles/platforms/kunlunxin.mk
# KunlunXin platform configuration.

DEVICE_HOME  ?= /usr/local/xpu
DEVICE_LIB   := $(DEVICE_HOME)/so
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lxpurt -lcudart
DEVICE_PLATFORM :=
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION := xpu

CCL_HOME    ?= /usr/local/xccl
CCL_LIB     := $(CCL_HOME)/so
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lbkcl
ADAPTOR_FLAG := -DUSE_KUNLUNXIN_ADAPTOR

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/kunlunxin
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
PLATFORM_EXTRA_SRCS  :=
