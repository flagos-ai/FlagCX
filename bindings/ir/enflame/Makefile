# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for FlagCX Device API — Enflame GCU/TOPS platform.
#
# Usage (normally driven by the top-level `device_bitcode` target):
#   make USE_ENFLAME=1 DEVICE_BITCODE_ARCH=gcu400 \
#        DEVICE_HOME=/opt/tops CCL_HOME=/usr device_bitcode
#
# Produces:
#   $(BUILDDIR)/lib/libflagcx_device.bc        — final bitcode for Triton linking
#   $(BUILDDIR)/include/flagcx_device_wrapper.h — consumer header
#
# Differences from the NVIDIA route:
#   * topscc replaces nvcc/clang. Its `-dbc` flag is the device-only bitcode
#     driver (`--cuda-device-only -c -emit-llvm`); passing -c/-emit-llvm by
#     hand leaves the host pass in the pipeline, where `__global__` kernels
#     need topsLaunchKernel declarations.
#   * topscc rejects -O1/-O2 ("not supported yet"); only -O0/-O3 are accepted.
#   * The TOPS SDK ships opt but neither llvm-as nor llvm-dis, so both the
#     bitcode→text and text→bitcode steps go through `opt -S` / `opt`.
#   * There is no nvvm-reflect-ftz module flag to clean from the final IR.

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR),0)

# Source (one level up — shared wrapper)
LLVM_SRC := ../flagcx_device_wrapper_impl.h
LLVM_COMPAT_IMPL := ../flagcx_device_compat_ir_impl.h
FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/llvm_ir
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

# Output files
UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
OPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libflagcx_device.ll
FINAL_BC := $(LIBDIR)/libflagcx_device.bc

# Export only the headers used by the selected Device API backend.
ADAPTOR_FLAG ?= -DUSE_ENFLAME_ADAPTOR -DFLAGCX_COMM_TRAITS_CCL
IR_PLATFORM_HEADERS := enflame_platform_traits.h enflame_comm_traits.h \
	eccl_comm_traits.h
include ../device_ir_headers.mk

# TOPS SDK. Device code is LLVM-based (clang 18 frontend, gcu-enflame-tops
# triple); the SDK's opt is the only LLVM tool that can read and write the
# bitcode, so it is used for every conversion step.
DEVICE_HOME ?= /opt/tops
CCL_HOME ?= /usr
BITCODE_LIB_ARCH ?= gcu400
BITCODE_CXX_STD ?= c++17
TOPSCC ?= $(DEVICE_HOME)/bin/topscc
OPT ?= $(DEVICE_HOME)/bin/opt

# Include paths
#   -I..  : shared wrapper headers (flagcx_device_wrapper.h, ...)
#   -I.   : platform-specific headers
FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include

TOPS_INCLUDES := \
	-I$(DEVICE_HOME)/include

# ECCL installs eccl.h and eccl_device/ directly under $(CCL_HOME)/include.
CCL_INCLUDES := \
	-I$(CCL_HOME)/include

# topscc flags. -dbc expands to the device-only bitcode pipeline, so no
# -c/-emit-llvm here. -O3 is the only optimization level topscc accepts.
TOPSCC_FLAGS := -dbc -O3 -std=$(BITCODE_CXX_STD) -arch $(BITCODE_LIB_ARCH) \
	-D__clang_llvm_bitcode_lib__ \
	-Wno-deprecated-declarations \
	$(ADAPTOR_FLAG) \
	$(FLAGCX_INCLUDES) $(TOPS_INCLUDES) $(CCL_INCLUDES)

# Optimization passes — use file-based API list to avoid shell quoting issues
SYMBOLS_FILE := $(abspath ../public_symbols.txt)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)

IR_INPUT_BC := $(UNOPTIMIZED_BC)

# ---- Build rules ----

# Default target
all: llvm_ir

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL)
	@mkdir -p $(OBJDIR)
	@echo "[FlagCX][Enflame] Generating unoptimized Device IR"
	$(TOPSCC) $(TOPSCC_FLAGS) $(IR_DEPFLAGS) $(LLVM_SRC) -o $@

# Strip @llvm.used/@llvm.compiler.used so internalize can remove internal
# symbols. topscc has no llvm-dis, so `opt -S` produces the text form.
$(STRIPPED_LL): $(UNOPTIMIZED_BC)
	@echo "[FlagCX][Enflame] Stripping @llvm.used metadata"
	$(OPT) -S $< -o $@.tmp
	sed -e '/@llvm\.used/d' -e '/@llvm\.compiler\.used/d' $@.tmp > $@
	@rm -f $@.tmp

$(STRIPPED_BC): $(STRIPPED_LL)
	$(OPT) $< -o $@

$(OPTIMIZED_BC): $(STRIPPED_BC) $(SYMBOLS_FILE)
	@echo "[FlagCX][Enflame] Optimizing LLVM bitcode"
	$(OPT) $(OPT_PASSES) $< -o $@

# No nvvm-reflect-ftz (NVPTX-only) cleanup is needed on GCU.
$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	$(OPT) -S $< -o $@

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "[FlagCX][Enflame] Generating final bitcode"
	$(OPT) $< -o $@

check_public_symbols: $(LLVM_IR_FILE)
	@echo "[FlagCX][Enflame] Checking public Device IR symbols"
	@missing=0; \
	while IFS= read -r symbol; do \
		case "$$symbol" in ''|'#'*) continue ;; esac; \
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol\\(" $(LLVM_IR_FILE); then \
			echo "Missing public Device IR symbol: $$symbol"; \
			missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

check_ir: $(FINAL_BC)
	$(OPT) --passes=verify $(FINAL_BC) -disable-output
	@grep -Eq '^target triple = "gcu-enflame-tops' $(LLVM_IR_FILE) || { \
		echo "Enflame Device IR does not target GCU"; exit 1; \
	}

llvm_ir: $(FINAL_BC) headers check_public_symbols check_ir
	@echo "[FlagCX][Enflame] LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard:    $(BITCODE_CXX_STD)"
	@echo "  GCU Architecture: $(BITCODE_LIB_ARCH)"
	@echo "  Adaptor:         $(ADAPTOR_FLAG)"
	@echo "  Unoptimized:     $(UNOPTIMIZED_BC)"
	@echo "  Optimized:       $(OPTIMIZED_BC)"
	@echo "  LLVM IR:         $(LLVM_IR_FILE)"
	@echo "  Final BC:        $(FINAL_BC)"
	@echo "  Public Headers:  $(INCLUDEDIR)"

include ../device_ir_dependencies.mk

else
llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"
endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC)
	rm -f $(INCLUDEDIR)/flagcx_device_wrapper.h \
		$(addprefix $(INCLUDEDIR)/,$(IR_PLATFORM_HEADERS))

.PHONY: all llvm_ir check_public_symbols check_ir build clean
