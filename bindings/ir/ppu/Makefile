# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM bitcode for the complete PPU Device API DefaultBackend.

EMIT_LLVM_IR ?= 1

FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(FLAGCX_ROOT)/build
DEVICE_HOME ?= /usr/local/PPU_SDK
CCL_HOME ?= $(DEVICE_HOME)
BITCODE_LIB_ARCH ?= ppu001
BITCODE_CXX_STD ?= c++17

OBJDIR := $(BUILDDIR)/obj/llvm_ir
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

# The common wrapper includes Scalar IR, Unified IR and compatibility entries.
LLVM_SRC := ../flagcx_device_wrapper_impl.h
LLVM_COMPAT_IMPL := ../flagcx_device_compat_ir_impl.h
SYMBOLS_FILE := $(abspath ../public_symbols.txt)

UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
OPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libflagcx_device.ll
FINAL_BC := $(LIBDIR)/libflagcx_device.bc

IR_PLATFORM_HEADERS := ppu_platform_traits.h ppu_comm_traits.h
include ../device_ir_headers.mk

# Use the SDK's HGGC Clang frontend and matching LLVM tools. Override the
# directory or individual tools when the SDK installs them elsewhere.
PPU_LLVM_BIN ?= $(DEVICE_HOME)/bin
PPU_CLANG ?= $(PPU_LLVM_BIN)/clang
OPT ?= $(PPU_LLVM_BIN)/opt
LLVM_DIS ?= $(PPU_LLVM_BIN)/llvm-dis
LLVM_AS ?= $(PPU_LLVM_BIN)/llvm-as

FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include
PPU_SDK_INCLUDES := \
	-I$(DEVICE_HOME)/include \
	-I$(DEVICE_HOME)/include/cccl \
	-I$(CCL_HOME)/include

ADAPTOR_FLAG ?= -DUSE_PPU_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT
COMMON_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) -x hggc --hggc-device-only \
	--ppu-arch=$(BITCODE_LIB_ARCH) -fgpu-rdc \
	-D__clang_llvm_bitcode_lib__ $(ADAPTOR_FLAG)
CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS) \
	$(FLAGCX_INCLUDES) $(PPU_SDK_INCLUDES)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)
IR_INPUT_BC := $(UNOPTIMIZED_BC)

all: llvm_ir

ifneq ($(EMIT_LLVM_IR),0)

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL)
	@mkdir -p $(OBJDIR)
	$(PPU_CLANG) $(CLANG_FLAGS) $(IR_DEPFLAGS) $(LLVM_SRC) -o $@

# Match the other platforms: release compiler retention lists before
# internalizing, while preserving every entry in the common public API list.
$(STRIPPED_LL): $(UNOPTIMIZED_BC)
	$(LLVM_DIS) $< -o $@.tmp
	sed -e '/@llvm\.used/d' -e '/@llvm\.compiler\.used/d' $@.tmp > $@
	rm -f $@.tmp

$(STRIPPED_BC): $(STRIPPED_LL)
	$(LLVM_AS) $< -o $@

$(OPTIMIZED_BC): $(STRIPPED_BC) $(SYMBOLS_FILE)
	$(OPT) $(OPT_PASSES) $< -o $@

$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	$(LLVM_DIS) $< -o $@

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	$(LLVM_AS) $< -o $@

check_public_symbols: $(LLVM_IR_FILE)
	@missing=0; \
	while IFS= read -r symbol; do \
		case "$$symbol" in ''|'#'*) continue ;; esac; \
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol[(]" $(LLVM_IR_FILE); then \
			echo "Missing public Device IR symbol: $$symbol"; missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

check_ir: $(FINAL_BC)
	$(OPT) --passes=verify $(FINAL_BC) -disable-output
	@grep -Eq '^target triple = "alippu' $(LLVM_IR_FILE) || { \
		echo "PPU Device IR does not target ALIPPU"; exit 1; \
	}

llvm_ir: $(FINAL_BC) headers check_public_symbols check_ir
	@echo "PPU Device API bitcode ready: $(FINAL_BC)"

else

llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR)
	rm -f $(FINAL_BC) $(INCLUDEDIR)/flagcx_device_wrapper.h \
		$(addprefix $(INCLUDEDIR)/,$(IR_PLATFORM_HEADERS))

.PHONY: all build llvm_ir check_public_symbols check_ir clean
include ../device_ir_dependencies.mk
