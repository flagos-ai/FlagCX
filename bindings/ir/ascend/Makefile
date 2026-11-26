# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for FlagCX Device API — Huawei Ascend platform.
#
# Usage (the root Makefile passes the platform selection through):
#   make USE_ASCEND=1 USE_SHMEM=1 COMPILE_KERNEL=1 \
#        DEVICE_BITCODE_ARCH=Ascend910_9382 \
#        SHMEM_HOME=/usr/local/Ascend/shmem/latest device_bitcode
#
# bisheng (the toolkit's clang 15 CCE frontend) compiles the shared wrapper
# header for the device pass. With --cce-aicore-only it emits one module per
# core type and no host module; the module names carry the triple and the soc:
#   <stem>-cce-<triple>-<soc>-vec.bc    (AI Vector core)
#   <stem>-cce-<triple>-<soc>-cube.bc   (AI Cube core)
# The driver rejects -o when it generates multiple files, so the modules are
# collected from a staging directory instead.
#
# CCE gives every device function a per-core suffix, so the modules define
# <name>.vector / <name>.cube rather than the bare C name. A bisheng-compiled
# consumer references external aicore functions with the same suffix (measured
# on the 910C: a call to probeAdd becomes @probeAdd.vector), so the two sides
# link name-for-name at the LLVM level. The toolkit's llvm-link accepts
# --internalize but does not apply it on this install (one internal definition
# before and after, with both the file and list spellings), so the modules are
# shipped directly and helper definitions stay external.

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR), 0)

LLVM_SRC := ../flagcx_device_wrapper_impl.h
FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/llvm_ir/ascend
STAGE_DIR := $(OBJDIR)/stage
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

FINAL_BC      := $(LIBDIR)/libflagcx_device.bc
FINAL_CUBE_BC := $(LIBDIR)/libflagcx_device-cube.bc
FINAL_LL      := $(OBJDIR)/libflagcx_device.ll
FINAL_CUBE_LL := $(OBJDIR)/libflagcx_device-cube.ll
COMPILE_STAMP := $(OBJDIR)/.compiled

# Export only the headers used by the selected Device API backend.
ADAPTOR_FLAG ?= -DUSE_ASCEND_ADAPTOR
IR_PLATFORM_HEADERS := ascend_platform_traits.h ascend_comm_traits.h
ifneq ($(findstring -DFLAGCX_COMM_TRAITS_SHMEM,$(ADAPTOR_FLAG)),)
IR_PLATFORM_HEADERS += aclshmem_comm_traits.h aclshmem_state_layout.h
endif
include ../device_ir_headers.mk

# Toolchain. The container ships bisheng on PATH; the wildcard spellings cover
# the historical toolkit layouts.
DEVICE_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
SHMEM_HOME  ?= /usr/local/Ascend/shmem/latest
ASCEND_SOC  ?= Ascend910_9382

BISHENG ?= $(firstword $(wildcard \
	$(DEVICE_HOME)/aarch64-linux/bin/bisheng \
	$(DEVICE_HOME)/x86_64-linux/bin/bisheng \
	$(DEVICE_HOME)/compiler/bin/bisheng) \
	$(shell command -v bisheng 2>/dev/null))
ifeq ($(strip $(BISHENG)),)
$(error bisheng not found: set BISHENG=/path/to/bisheng or put it on PATH)
endif

# Absolute on purpose: the compile runs with CWD in the staging directory.
FLAGCX_INCLUDES := \
	-I$(abspath ..) \
	-I$(abspath .) \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include

# Ascend C headers resolve through bisheng's builtin search path on the
# verified install; these guarded spellings cover installs with a different
# toolkit layout.
ASCENDC_INCLUDE := $(firstword \
	$(wildcard $(DEVICE_HOME)/aarch64-linux/include/ascendc) \
	$(wildcard $(DEVICE_HOME)/include/ascendc) \
	$(wildcard $(DEVICE_HOME)/compiler/include/ascendc))
DEVICE_INCLUDES := \
	$(addprefix -I,$(wildcard $(DEVICE_HOME)/include) $(ASCENDC_INCLUDE))

SHMEM_INCLUDES :=
ifneq ($(findstring -DFLAGCX_COMM_TRAITS_SHMEM,$(ADAPTOR_FLAG)),)
SHMEM_INCLUDES += -I$(SHMEM_HOME)/include -I$(SHMEM_HOME)/include/device \
	-I$(SHMEM_HOME)/src/device -I$(SHMEM_HOME)/src
endif

# -O2 matches the native Ascend device build (makefiles/ascend.mk). -NDEBUG is
# load-bearing on the aicore pass: assert() calls __assert_fail, which has no
# declaration there (see test/kernel/ascend/Makefile).
CLANG_FLAGS := -x asc --npu-soc=$(ASCEND_SOC) --cce-aicore-only \
	--cce-auto-infer-kernel-type=false -std=c++17 -O2 -c -emit-llvm \
	-D__clang_llvm_bitcode_lib__ -DNDEBUG -DFLAGCX_ASCEND_DEVICE_COMPILE=1 \
	$(ADAPTOR_FLAG) $(FLAGCX_INCLUDES) $(DEVICE_INCLUDES) $(SHMEM_INCLUDES)

SYMBOLS_FILE := $(abspath ../public_symbols.txt)

# ---- Build rules ----

all: llvm_ir

# One bisheng invocation emits both core modules; the stamp records it.
$(COMPILE_STAMP): Makefile $(LLVM_SRC) $(SYMBOLS_FILE) \
		$(wildcard $(FLAGCX_ROOT)/flagcx/adaptor/include/*.h) \
		$(wildcard $(FLAGCX_ROOT)/flagcx/adaptor/include/device_api/*.h) \
		$(wildcard $(FLAGCX_ROOT)/flagcx/include/*.h) \
		$(wildcard $(FLAGCX_ROOT)/flagcx/core/include/*.h) \
		$(wildcard ../*.h)
	@mkdir -p $(STAGE_DIR) $(LIBDIR)
	@rm -rf $(STAGE_DIR)
	@mkdir -p $(STAGE_DIR)
	@echo "Generating LLVM bitcode (bisheng, $(ASCEND_SOC))..."
	cd $(STAGE_DIR) && $(BISHENG) $(CLANG_FLAGS) $(abspath $(LLVM_SRC))
	cp $(STAGE_DIR)/*-vec.bc $(FINAL_BC)
	cp $(STAGE_DIR)/*-cube.bc $(FINAL_CUBE_BC)
	@touch $@

$(FINAL_BC) $(FINAL_CUBE_BC): $(COMPILE_STAMP)

$(FINAL_LL): $(FINAL_BC)
	$(BISHENG) -S -emit-llvm $< -o $@

$(FINAL_CUBE_LL): $(FINAL_CUBE_BC)
	$(BISHENG) -S -emit-llvm $< -o $@

check_public_symbols: $(FINAL_LL) $(FINAL_CUBE_LL)
	@echo "Checking public Device IR symbols..."
	@missing=0; \
	for core in vector cube; do \
		if [ $$core = vector ]; then ir=$(FINAL_LL); else ir=$(FINAL_CUBE_LL); fi; \
		while IFS= read -r symbol; do \
			case "$$symbol" in ''|'#'*) continue ;; esac; \
			if ! grep -Eq "^define .*@$$symbol\.$$core\\(" $$ir; then \
				echo "Missing public Device IR symbol in $$ir: $$symbol.$$core"; \
				missing=1; \
			elif grep -Eq "^define [^@]*internal[^@]*@$$symbol\.$$core\\(" $$ir; then \
				echo "Public Device IR symbol not exported in $$ir: $$symbol.$$core"; \
				missing=1; \
			fi; \
		done < $(SYMBOLS_FILE); \
	done; \
	test $$missing -eq 0

check_ir: $(FINAL_LL) $(FINAL_CUBE_LL)
	@echo "Checking Ascend Device IR modules..."
	@grep -Eq '^target triple = "aarch64-unknown-linux-gnu"' $(FINAL_LL) || { \
		echo "Ascend Device IR is not a CCE module (unexpected triple)"; exit 1; }
	@grep -q '__init_kfc_workspace_addr\.vector' $(FINAL_LL) || { \
		echo "Ascend Device IR vec module lacks the vector-core marker"; exit 1; }
	@grep -q '__init_kfc_workspace_addr\.cube' $(FINAL_CUBE_LL) || { \
		echo "Ascend Device IR cube module lacks the cube-core marker"; exit 1; }

llvm_ir: $(FINAL_BC) $(FINAL_CUBE_BC) headers check_public_symbols check_ir
	@echo "LLVM IR and bitcode generated successfully:"
	@echo "  SoC:               $(ASCEND_SOC)"
	@echo "  Adaptor:           $(ADAPTOR_FLAG)"
	@echo "  Final BC (AIV):    $(FINAL_BC)"
	@echo "  Final BC (AIC):    $(FINAL_CUBE_BC)"
	@echo "  Public Headers:    $(INCLUDEDIR)"

else
llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"
endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(FINAL_CUBE_BC)
	rm -f $(INCLUDEDIR)/flagcx_device_wrapper.h \
		$(addprefix $(INCLUDEDIR)/,$(IR_PLATFORM_HEADERS))

.PHONY: all llvm_ir check_public_symbols check_ir build clean
