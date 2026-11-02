# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for FlagCX Device API — NVIDIA CUDA platform.
#
# Usage:
#   make EMIT_LLVM_IR=1 USE_NVIDIA=1 \
#        DEVICE_HOME=/usr/local/cuda CCL_HOME=/usr/local/nccl/build
#
# Produces:
#   $(BUILDDIR)/lib/libflagcx_device.bc    — final bitcode for Triton linking
#   $(BUILDDIR)/include/flagcx_device_wrapper.h — consumer header

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR), 0)

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
OPTIMIZED_BC   := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE   := $(OBJDIR)/libflagcx_device.ll
FINAL_BC       := $(LIBDIR)/libflagcx_device.bc

# Export only the headers used by the selected Device API backend.
ADAPTOR_FLAG ?= -DUSE_NVIDIA_ADAPTOR
IR_PLATFORM_HEADERS := nvidia_platform_traits.h nvidia_comm_traits.h
ifneq ($(findstring -DFLAGCX_COMM_TRAITS_CCL,$(ADAPTOR_FLAG)),)
IR_PLATFORM_HEADERS += nccl_comm_traits.h
endif
ifneq ($(findstring -DFLAGCX_COMM_TRAITS_SHMEM,$(ADAPTOR_FLAG)),)
IR_PLATFORM_HEADERS += nvshmem_comm_traits.h
endif
include ../device_ir_headers.mk

# GPU architecture
DEVICE_HOME ?= /usr/local/cuda
CCL_HOME    ?= /usr/local/nccl/build

# Detect CUDA version for arch selection
CUDA_MAJOR ?= $(shell $(DEVICE_HOME)/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)
ifeq ($(CUDA_MAJOR),)
  CUDA_MAJOR := 12
endif
CUDA_MINOR ?= 0

ifeq ($(shell test "$(CUDA_MAJOR)" -lt 12; echo $$?),0)
  BITCODE_LIB_ARCH ?= sm_70
else
  BITCODE_LIB_ARCH ?= sm_90
endif

BITCODE_CXX_STD ?= c++17

# Tools
CLANG    ?= clang
OPT      ?= opt
LLVM_DIS ?= llvm-dis
LLVM_AS  ?= llvm-as

# Include paths
#   -I..  : shared wrapper headers (flagcx_device_wrapper.h, flagcx_device_wrapper_impl.h)
#   -I.   : platform-specific headers
FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include

CUDA_INCLUDES := \
	-I$(DEVICE_HOME)/include \
	-I$(DEVICE_HOME)/include/cccl

CCL_INCLUDES := \
	-I$(CCL_HOME)/include

ifneq ($(findstring -DFLAGCX_COMM_TRAITS_SHMEM,$(ADAPTOR_FLAG)),)
SHMEM_HOME ?= /usr/local/nvshmem
CCL_INCLUDES += -I$(SHMEM_HOME)/include
endif

# Clang flags for LLVM IR generation
COMMON_CLANG_FLAGS := \
	-std=$(BITCODE_CXX_STD) -x cuda \
	--cuda-path=$(DEVICE_HOME) --cuda-device-only \
	--cuda-gpu-arch=$(BITCODE_LIB_ARCH) \
	-D__clang_llvm_bitcode_lib__ \
	-DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) \
	$(ADAPTOR_FLAG)

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS) \
	$(FLAGCX_INCLUDES) $(CUDA_INCLUDES) $(CCL_INCLUDES)

# Optimization passes — use file-based API list to avoid shell quoting issues
SYMBOLS_FILE := $(abspath ../public_symbols.txt)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)

# Intermediate files
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
IR_INPUT_BC := $(UNOPTIMIZED_BC)

# ---- Build rules ----

# Default target
all: llvm_ir

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL)
	@mkdir -p $(OBJDIR)
	@echo "Generating unoptimized LLVM bitcode..."
	$(CLANG) $(CLANG_FLAGS) $(IR_DEPFLAGS) $(LLVM_SRC) -o $@

# Strip @llvm.used/@llvm.compiler.used so internalize can remove internal symbols
$(STRIPPED_LL): $(UNOPTIMIZED_BC)
	@echo "Stripping @llvm.used metadata..."
	$(LLVM_DIS) $< -o $@.tmp
	sed -e '/@llvm\.used/d' -e '/@llvm\.compiler\.used/d' $@.tmp > $@
	@rm -f $@.tmp

$(STRIPPED_BC): $(STRIPPED_LL)
	$(LLVM_AS) $< -o $@

$(OPTIMIZED_BC): $(STRIPPED_BC) $(SYMBOLS_FILE)
	@echo "Optimizing LLVM bitcode..."
	$(OPT) $(OPT_PASSES) $< -o $@

$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	@echo "Generating LLVM IR..."
	$(LLVM_DIS) $< -o $@.tmp
	@echo "Cleaning LLVM IR (removing nvvm-reflect-ftz)..."
	@myVar="$$(cat $@.tmp | grep -E '!([0-9]+) = !\{[^"]*"nvvm-reflect-ftz"' | cut -d ' ' -f 1)"; \
	if [ -n "$$myVar" ]; then \
	  awk '!/nvvm-reflect-ftz/' $@.tmp | sed "/^!llvm\.module\.flags = /s/$$myVar, //" > $@; \
	else \
	  cp $@.tmp $@; \
	fi
	@rm -f $@.tmp

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "Generating final bitcode from cleaned LLVM IR..."
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

check_ir: $(FINAL_BC)
	$(OPT) --passes=verify $(FINAL_BC) -disable-output
	@grep -Eq '^target triple = "nvptx64-nvidia-cuda' $(LLVM_IR_FILE) || { \
		echo "NVIDIA Device IR does not target NVPTX"; exit 1; \
	}

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) headers check_public_symbols check_ir
	@echo "LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard:      $(BITCODE_CXX_STD)"
	@echo "  GPU Architecture:  $(BITCODE_LIB_ARCH)"
	@echo "  Adaptor:           $(ADAPTOR_FLAG)"
	@echo "  Unoptimized:       $(UNOPTIMIZED_BC)"
	@echo "  Optimized:         $(OPTIMIZED_BC)"
	@echo "  LLVM IR:           $(LLVM_IR_FILE)"
	@echo "  Final BC:          $(FINAL_BC)"
	@echo "  Public Headers:    $(INCLUDEDIR)"

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
