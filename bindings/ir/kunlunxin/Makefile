# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for the FlagCX Device API — KunlunXin/XPU (P800) platform.
#
# Usage:
#   make EMIT_LLVM_IR=1 \
#        XTDK_HOME=/path/to/xtdk-llvm15 \
#        CCL_HOME=/path/to/xccl \
#        DEVICE_HOME=/path/to/xre
#
# The three SDK roots are taken from the environment / command line (see the
# ?= assignments below); they are NOT hard-coded to any one machine.
#
# Produces:
#   $(BUILDDIR)/lib/libflagcx_device.bc    — final bitcode for Triton/FlagTree linking
#   $(BUILDDIR)/include/flagcx_device_wrapper.h — consumer header
#
# Notes:
#   - The XTDK clang does not emit a plain bitcode file from -c -emit-llvm; it
#     emits a __CLANG_OFFLOAD_BUNDLE__ containing the xpu device module and the
#     host stub. clang-offload-bundler extracts the device module.
#   - KunlunXin has no cluster barrier outside XSHMEM, so the Device IR is
#     always built against the XSHMEM CommTraits (FLAGCX_COMM_TRAITS_SHMEM).
#     The external xshmem/xpu-printf symbols stay unresolved and are provided
#     by the consumer (libxshmem_device) at final link time, mirroring how the
#     NVIDIA bitcode leaves its CCL/NCCL references unresolved.

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR), 0)

# Source (one level up — shared wrapper)
LLVM_SRC := ../flagcx_device_wrapper_impl.h
SYMBOLS_FILE := $(abspath ../public_symbols.txt)

FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(FLAGCX_ROOT)/build

XTDK_HOME   ?= /usr/local/xtdk
CCL_HOME    ?= /usr/local/xccl
DEVICE_HOME ?= /usr/local/xpu

# XPU architecture. xpu3 selects the P800 device pass.
XPU_ARCH ?= xpu3

XPU_CLANG ?= $(XTDK_HOME)/bin/clang++
BUNDLER   ?= $(XTDK_HOME)/bin/clang-offload-bundler
OPT       ?= $(XTDK_HOME)/bin/opt
LLVM_DIS  ?= $(XTDK_HOME)/bin/llvm-dis
LLVM_AS   ?= $(XTDK_HOME)/bin/llvm-as

OBJDIR := $(BUILDDIR)/obj/llvm_ir/kunlunxin/$(XPU_ARCH)
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

# Output files
BUNDLE_BC       := $(OBJDIR)/libflagcx_device.bundle.bc
UNOPTIMIZED_BC  := $(OBJDIR)/libflagcx_device.bc.unoptimized
OPTIMIZED_BC    := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE    := $(OBJDIR)/libflagcx_device.ll
FINAL_BC        := $(LIBDIR)/libflagcx_device.bc
WRAPPER_HEADER  := $(INCLUDEDIR)/flagcx_device_wrapper.h

# Offload bundle triples emitted by the XTDK clang for this target.
XPU_TRIPLE  := xpu-$(XPU_ARCH)-baidu-none-gnu
HOST_TRIPLE := host-x86_64-unknown-linux-gnu

# Include paths
FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include \
	-I$(CCL_HOME)/include \
	-I$(DEVICE_HOME)/include

# Adaptor flag. XSHMEM is the only backend with a usable cluster barrier on
# KunlunXin; the DefaultBackend maps FLAGCX_DEVICE_SYNC_THREADS() to a trap.
ADAPTOR_FLAG ?= -DUSE_KUNLUNXIN_ADAPTOR -DFLAGCX_COMM_TRAITS_SHMEM

COMMON_CLANG_FLAGS := \
	-std=c++17 -x xpu \
	--xpu-arch=$(XPU_ARCH) \
	-fno-builtin --target=x86_64-linux-gnu \
	$(FLAGCX_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	$(ADAPTOR_FLAG) \
	-Wno-deprecated-declarations \
	-DNDEBUG

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

# Optimization passes — keep only the public Device IR symbols.
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)

# ---- Build rules ----

all: llvm_ir

$(WRAPPER_HEADER): ../flagcx_device_wrapper.h
	@mkdir -p $(INCLUDEDIR)
	cp -f $< $@

$(BUNDLE_BC): $(LLVM_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Generating KunlunXin Device IR bundle..."
	$(XPU_CLANG) $(CLANG_FLAGS) $(LLVM_SRC) -o $@

# Extract the xpu device module from the offload bundle.
$(UNOPTIMIZED_BC): $(BUNDLE_BC)
	@echo "Extracting xpu device module from offload bundle..."
	$(BUNDLER) -type=bc \
		-targets=$(HOST_TRIPLE),$(XPU_TRIPLE) \
		-input=$< \
		-output=$(OBJDIR)/libflagcx_device.bc.host \
		-output=$@ \
		-unbundle

$(OPTIMIZED_BC): $(UNOPTIMIZED_BC)
	@echo "Optimizing KunlunXin Device IR..."
	$(OPT) $(OPT_PASSES) $< -o $@

$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	$(LLVM_DIS) $< -o $@

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "Generating final bitcode..."
	$(LLVM_AS) $< -o $@

check_public_symbols: $(LLVM_IR_FILE)
	@echo "Checking public Device IR symbols..."
	@missing=0; \
	while IFS= read -r symbol; do \
		case "$$symbol" in ''|'#'*) continue ;; esac; \
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol\\(" $(LLVM_IR_FILE); then \
			echo "Missing public Device IR symbol: $$symbol"; \
			missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) $(WRAPPER_HEADER) check_public_symbols
	@echo "LLVM IR and bitcode generated successfully:"
	@echo "  XPU Architecture: $(XPU_ARCH)"
	@echo "  Adaptor:          $(ADAPTOR_FLAG)"
	@echo "  Optimized:        $(OPTIMIZED_BC)"
	@echo "  LLVM IR:          $(LLVM_IR_FILE)"
	@echo "  Final BC:         $(FINAL_BC)"
	@echo "  Wrapper Header:   $(WRAPPER_HEADER)"

else
llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"
endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(WRAPPER_HEADER)

.PHONY: all llvm_ir check_public_symbols build clean

