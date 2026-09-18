# makefiles/platforms/iluvatar.mk
# Iluvatar CoreX platform configuration.

# Keep the repository-wide COMPILE_KERNEL default. Consumers that need native
# collective kernels or device-side Net construction request
# COMPILE_KERNEL=1 explicitly.

DEVICE_HOME  ?= /usr/local/corex
DEVICE_LIB   := $(DEVICE_HOME)/lib
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lcudart -lcuda
DEVICE_PLATFORM := ILUVATAR
ILUVATAR_ARCH ?= ivcore11
COREX_CLANG ?= $(DEVICE_HOME)/bin/clang
DEVICE_COMPILER := $(COREX_CLANG)
DEVICE_COMPILE_FLAG := -c -x ivcore --cuda-path=$(DEVICE_HOME) \
    --cuda-gpu-arch=$(ILUVATAR_ARCH) -std=c++17 -O2 -fPIC -MMD -MP \
    -DFLAGCX_ILUVATAR_DEVICE_COMPILE=1
DEVICE_LINK_FLAG :=
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /usr/local/corex
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_ILUVATAR_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/iluvatar
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
# device_api/ is not globbed by the Makefile and -shared without --no-undefined
# hides that: flagcx_device.cc's devApiBackend would stay unresolved.
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
