# makefiles/platforms/enflame.mk
# Enflame platform configuration.

DEVICE_HOME  ?= /opt/tops/
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -ltopsrt
DEVICE_PLATFORM :=
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION :=

CCL_HOME    ?= /usr
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -leccl
ADAPTOR_FLAG := -DUSE_ENFLAME_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
