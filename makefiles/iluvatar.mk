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
COREX_CLANG ?= $(DEVICE_HOME)/bin/clang
DEVICE_COMPILER := $(COREX_CLANG)
DEVICE_COMPILE_FLAG := -c -x ivcore --cuda-path=$(DEVICE_HOME) \
    --cuda-gpu-arch=$(if $(strip $(DEVICE_BITCODE_ARCH)),$(DEVICE_BITCODE_ARCH),ivcore11) \
    -std=c++17 -O2 -fPIC -MMD -MP \
    -DFLAGCX_ILUVATAR_DEVICE_COMPILE=1
DEVICE_LINK_FLAG :=
DEVICE_NEEDS_DLINK := 0
DEVICE_FILE_EXTENSION := cu

CCL_HOME    ?= /usr/local/corex
CCL_LIB     := $(CCL_HOME)/lib
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lnccl
ADAPTOR_FLAG := -DUSE_ILUVATAR_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT

# CoreX's CUDA version number alone does not establish which driver API
# declarations its SDK ships. Compile-check the complete VMM surface without
# executing a device call. Missing runtime symbols are handled by the adaptor.
ILUVATAR_VMM_CHECK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ILUVATAR_VMM_CHECK_COMPILER = $(if $(HOST_COMPILER),$(HOST_COMPILER),$(CXX))
ILUVATAR_VMM_HEADER_SUPPORT := $(shell bash "$(ILUVATAR_VMM_CHECK_DIR)/check_iluvatar_vmm.sh" \
    check "$(DEVICE_HOME)" -- $(ILUVATAR_VMM_CHECK_COMPILER) \
    $(CPPFLAGS) $(HOST_CXX_STANDARD) $(HOST_CXXFLAGS))
ADAPTOR_FLAG += -DFLAGCX_ILUVATAR_HAS_VMM=$(ILUVATAR_VMM_HEADER_SUPPORT)

PLATFORM_KERNEL_DIR  := flagcx/adaptor/kernel/iluvatar
PLATFORM_KERNEL_SRCS := $(wildcard $(PLATFORM_KERNEL_DIR)/*.$(DEVICE_FILE_EXTENSION))
# device_api/ is not globbed by the Makefile and -shared without --no-undefined
# hides that: flagcx_device.cc's devApiBackend would stay unresolved.
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc
