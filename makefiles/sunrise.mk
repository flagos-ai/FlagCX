# makefiles/platforms/sunrise.mk
# Sunrise platform configuration.

DEVICE_HOME  ?= /usr/local/tangrt
DEVICE_LIB   := $(DEVICE_HOME)/targets/linux-x86_64/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -ltangrt_shared
DEVICE_PLATFORM :=
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION :=

CCL_HOME    ?= /usr/local/pccl
CCL_LIB     := $(CCL_HOME)/lib/linux-x86_64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lpccl
ADAPTOR_FLAG := -DUSE_SUNRISE_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  :=
