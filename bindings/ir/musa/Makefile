# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for the complete FlagCX Device API on MUSA (MTT S5000).
#
# mcc is a clang-14 based driver: the CUDA-style device-only switches
# (--cuda-device-only / --offload-arch) and -emit-llvm produce a bitcode
# module per architecture. The emitted module targets "mtgpu-mt-musa" and
# carries the mtgpu_kernel calling convention.
#
# Usage:
#   make USE_MUSA=1 DEVICE_BITCODE_ARCH=mp_31 device_bitcode
#
# Produces:
#   $(BUILDDIR)/lib/libflagcx_device.bc         — final bitcode for Triton
#   $(BUILDDIR)/include/flagcx_device_wrapper.h — consumer header

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR),0)

LLVM_SRC := ../flagcx_device_wrapper_impl.h
LLVM_COMPAT_IMPL := ../flagcx_device_compat_ir_impl.h
SYMBOLS_FILE := $(abspath ../public_symbols.txt)

FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(FLAGCX_ROOT)/build
MUSA_ARCH ?= mp_31
SUPPORTED_ARCHES := mp_21 mp_22 mp_31 mp_32

ifeq ($(filter $(MUSA_ARCH),$(SUPPORTED_ARCHES)),)
  $(error Unsupported MUSA target '$(MUSA_ARCH)'; supported: $(SUPPORTED_ARCHES))
endif

OBJDIR := $(BUILDDIR)/obj/llvm_ir
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
OPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libflagcx_device.ll
FINAL_BC := $(LIBDIR)/libflagcx_device.bc

IR_PLATFORM_HEADERS := musa_platform_traits.h musa_comm_traits.h musa_proxy_sync.h
include ../device_ir_headers.mk

DEVICE_HOME ?= /usr/local/musa
MUSA_LLVM_BIN ?= $(DEVICE_HOME)/bin
MCC ?= $(MUSA_LLVM_BIN)/mcc
OPT ?= $(MUSA_LLVM_BIN)/opt
LLVM_DIS ?= $(MUSA_LLVM_BIN)/llvm-dis
LLVM_AS ?= $(MUSA_LLVM_BIN)/llvm-as

# Match native kernel compilation: some mcc installations do not discover
# the host C++ headers. Device wrappers must precede those host headers.
MUSA_HOST_CXX_INCLUDES ?= $(shell $(CXX) -E -x c++ -v /dev/null 2>&1 >/dev/null | \
	sed -n 's|^ \(/[^ ]*include/[^ ]*c++[^ ]*\)$$|-isystem \1|p')
MUSA_CXX_WRAPPER_DIR ?= $(shell $(MCC) -print-resource-dir 2>/dev/null)/include/cuda_wrappers

# Absolute include roots: mcc runs inside OBJDIR so that its generated file
# names stay out of the source tree.
FLAGCX_INCLUDES := \
	-I$(abspath ..) \
	-I$(abspath .) \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include \
	-I$(DEVICE_HOME)/include

ADAPTOR_FLAG ?= -DUSE_MUSA_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT
COMMON_CLANG_FLAGS := \
	-std=c++17 -x musa \
	--musa-path=$(DEVICE_HOME) --cuda-device-only \
	--offload-arch=$(MUSA_ARCH) \
	-isystem $(MUSA_CXX_WRAPPER_DIR) $(MUSA_HOST_CXX_INCLUDES) \
	$(FLAGCX_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	$(ADAPTOR_FLAG) \
	-DFLAGCX_MUSA_DEVICE_COMPILE=1 \
	-DNDEBUG

CLANG_FLAGS := -c -emit-llvm -O2 -Xclang -disable-llvm-optzns $(COMMON_CLANG_FLAGS)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)
IR_INPUT_BC := $(UNOPTIMIZED_BC)

# mcc names its output after the source and target: the single-architecture
# invocation still writes <stem>-musa-mtgpu-mt-musa-$(MUSA_ARCH).bc, so the
# module is produced in OBJDIR and then moved onto the pipeline name.
MUSA_SRC_ABS := $(abspath $(LLVM_SRC))
MUSA_RAW_BC := $(OBJDIR)/flagcx_device_wrapper_impl-musa-mtgpu-mt-musa-$(MUSA_ARCH).bc

all: llvm_ir

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL)
	@mkdir -p $(OBJDIR)
	@echo "[FlagCX][MUSA] Generating unoptimized Device IR"
	cd $(OBJDIR) && $(MCC) $(CLANG_FLAGS) $(IR_DEPFLAGS) $(MUSA_SRC_ABS)
	@mv -f $(MUSA_RAW_BC) $@

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
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol[(]" $<; then \
			echo "Missing public Device IR symbol: $$symbol"; missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

check_ir: $(FINAL_BC) $(LLVM_IR_FILE)
	$(OPT) --passes=verify $(FINAL_BC) -disable-output
	@grep -Eq '^target triple = "mtgpu' $(LLVM_IR_FILE) || { \
		echo "MUSA Device IR does not target mtgpu"; exit 1; \
	}
	@if grep -Eq 'PlatformCoop|VTable' $(LLVM_IR_FILE); then \
		echo "MUSA Device IR contains PlatformCoop/vtable state"; exit 1; \
	fi

llvm_ir: $(FINAL_BC) headers check_public_symbols check_ir
	@echo "[FlagCX][MUSA] Device IR ready: $(FINAL_BC)"

include ../device_ir_dependencies.mk

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
