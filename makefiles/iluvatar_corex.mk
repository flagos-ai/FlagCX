# makefiles/platforms/iluvatar_corex.mk
# Iluvatar CoreX platform configuration.

DEVICE_HOME  ?= /usr/local/corex
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lcudart -lcuda
DEVICE_PLATFORM :=
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION :=

CCL_HOME    ?= /usr/local/corex
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_ILUVATAR_COREX_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  :=
