# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM bitcode for the Hygon Device API DefaultBackend.

EMIT_LLVM_IR ?= 1

FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(FLAGCX_ROOT)/build
DEVICE_HOME ?= $(if $(HIP_PATH),$(HIP_PATH),/opt/dtk)
CCL_HOME ?= $(DEVICE_HOME)
BITCODE_LIB_ARCH ?= native
BITCODE_CXX_STD ?= gnu++17

OBJDIR := $(BUILDDIR)/obj/llvm_ir/du/$(BITCODE_LIB_ARCH)
ARCH_LIBDIR := $(BUILDDIR)/lib/du/$(BITCODE_LIB_ARCH)
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

LLVM_SRC := ../flagcx_device_wrapper_impl.h
LLVM_COMPAT_IMPL := ../flagcx_device_compat_ir_impl.h
SYMBOLS_FILE := $(abspath ../public_symbols.txt)

UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
OPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libflagcx_device.ll
ARCH_BC := $(ARCH_LIBDIR)/libflagcx_device.bc
FINAL_BC := $(LIBDIR)/libflagcx_device.bc

IR_PLATFORM_HEADERS := du_platform_traits.h du_comm_traits.h
include ../device_ir_headers.mk

# Keep the optimizer/assembler on the same DTK LLVM build as hipcc. Each
# executable can be overridden when the image installs the tools elsewhere.
DTK_LLVM_BIN ?= $(if $(wildcard $(DEVICE_HOME)/llvm/bin/llvm-dis),$(DEVICE_HOME)/llvm/bin,$(DEVICE_HOME)/bin)
HIP_CLANG ?= $(DEVICE_HOME)/bin/hipcc
OPT ?= $(DTK_LLVM_BIN)/opt
LLVM_DIS ?= $(DTK_LLVM_BIN)/llvm-dis
LLVM_AS ?= $(DTK_LLVM_BIN)/llvm-as

FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include
HIP_SDK_INCLUDES := \
	-I$(DEVICE_HOME)/include \
	-I$(CCL_HOME)/include/rccl

ADAPTOR_FLAG ?= -DUSE_DU_ADAPTOR -D__HIP_PLATFORM_AMD__
COMMON_HIP_FLAGS := \
	-std=$(BITCODE_CXX_STD) -x hip --offload-device-only \
	--offload-arch=$(BITCODE_LIB_ARCH) -fgpu-rdc \
	-D__clang_llvm_bitcode_lib__ $(ADAPTOR_FLAG)
HIP_FLAGS := -c -emit-llvm -O1 $(COMMON_HIP_FLAGS) \
	$(FLAGCX_INCLUDES) $(HIP_SDK_INCLUDES)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)

all: llvm_ir

ifneq ($(EMIT_LLVM_IR),0)

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL) $(SYMBOLS_FILE)
	@mkdir -p $(OBJDIR)
	$(HIP_CLANG) $(HIP_FLAGS) $(LLVM_SRC) -o $@

# Drop compiler retention lists before internalization. Unlike the NVIDIA
# route, AMDGPU module flags do not need the nvvm-reflect-ftz cleanup.
$(STRIPPED_LL): $(UNOPTIMIZED_BC)
	$(LLVM_DIS) $< -o $@.tmp
	sed -e '/@llvm\.used/d' -e '/@llvm\.compiler\.used/d' $@.tmp > $@
	rm -f $@.tmp

$(STRIPPED_BC): $(STRIPPED_LL)
	$(LLVM_AS) $< -o $@

$(OPTIMIZED_BC): $(STRIPPED_BC)
	$(OPT) $(OPT_PASSES) $< -o $@

$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	$(LLVM_DIS) $< -o $@

$(ARCH_BC): $(LLVM_IR_FILE)
	@mkdir -p $(ARCH_LIBDIR)
	$(LLVM_AS) $< -o $@

$(FINAL_BC): $(ARCH_BC) FORCE
	@mkdir -p $(LIBDIR)
	cp -f $< $@

check_public_symbols: $(LLVM_IR_FILE)
	@missing=0; \
	while IFS= read -r symbol; do \
		case "$$symbol" in ''|'#'*) continue ;; esac; \
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol[(]" $(LLVM_IR_FILE); then \
			echo "Missing public Device IR symbol: $$symbol"; missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

check_ir: $(ARCH_BC)
	$(OPT) --passes=verify $(ARCH_BC) -disable-output
	@grep -Eq '^target triple = "amdgcn-amd-amdhsa' $(LLVM_IR_FILE) || { \
		echo "Hygon Device IR does not target AMDGPU"; exit 1; \
	}

llvm_ir: $(FINAL_BC) headers check_public_symbols check_ir
	@echo "Hygon Device API bitcode ready: $(ARCH_BC)"
	@echo "Compatibility copy: $(FINAL_BC)"

else

llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(ARCH_LIBDIR)
	rm -f $(FINAL_BC) $(INCLUDEDIR)/flagcx_device_wrapper.h \
		$(addprefix $(INCLUDEDIR)/,$(IR_PLATFORM_HEADERS))

FORCE:

.PHONY: all build llvm_ir check_public_symbols check_ir clean FORCE
