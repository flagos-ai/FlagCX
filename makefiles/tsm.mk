# makefiles/platforms/tsm.mk
# TSM platform configuration.

DEVICE_HOME  ?= /usr/local/kuiper
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lhpgr
DEVICE_PLATFORM :=
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION :=

CCL_HOME    ?= /usr/local/kuiper
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -ltccl
ADAPTOR_FLAG := -DUSE_TSM_ADAPTOR

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
# device_api/ is not globbed by the Makefile and -shared without --no-undefined
# hides that: flagcx_device.cc's devApiBackend would stay unresolved.
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
