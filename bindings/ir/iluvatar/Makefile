# Copyright (c) 2026 BAAI. All rights reserved.
#
# LLVM IR generation for the complete FlagCX Device API on Iluvatar/CoreX.

EMIT_LLVM_IR ?= 1

ifneq ($(EMIT_LLVM_IR),0)

LLVM_SRC := ../flagcx_device_wrapper_impl.h
LLVM_COMPAT_IMPL := ../flagcx_device_compat_ir_impl.h
SYMBOLS_FILE := $(abspath ../public_symbols.txt)

FLAGCX_ROOT := $(abspath ../../..)
BUILDDIR ?= $(FLAGCX_ROOT)/build
ILUVATAR_ARCH ?= ivcore11
SUPPORTED_ARCHES := ivcore11

ifeq ($(filter $(ILUVATAR_ARCH),$(SUPPORTED_ARCHES)),)
  $(error Unsupported Iluvatar target '$(ILUVATAR_ARCH)'; supported: $(SUPPORTED_ARCHES))
endif

OBJDIR := $(BUILDDIR)/obj/llvm_ir/iluvatar/$(ILUVATAR_ARCH)
ARCH_LIBDIR := $(BUILDDIR)/lib/iluvatar/$(ILUVATAR_ARCH)
LIBDIR := $(BUILDDIR)/lib
INCLUDEDIR := $(BUILDDIR)/include

UNOPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.unoptimized
STRIPPED_LL := $(OBJDIR)/libflagcx_device.stripped.ll
STRIPPED_BC := $(OBJDIR)/libflagcx_device.stripped.bc
OPTIMIZED_BC := $(OBJDIR)/libflagcx_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libflagcx_device.ll
ARCH_BC := $(ARCH_LIBDIR)/libflagcx_device.bc
FINAL_BC := $(LIBDIR)/libflagcx_device.bc
WRAPPER_HEADER := $(INCLUDEDIR)/flagcx_device_wrapper.h

DEVICE_HOME ?= /usr/local/corex
COREX_CLANG ?= $(DEVICE_HOME)/bin/clang
OPT ?= opt
LLVM_DIS ?= llvm-dis
LLVM_AS ?= llvm-as
LLVM_NM ?= llvm-nm

FLAGCX_INCLUDES := \
	-I.. \
	-I. \
	-I$(FLAGCX_ROOT)/flagcx/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include \
	-I$(FLAGCX_ROOT)/flagcx/adaptor/include/device_api \
	-I$(FLAGCX_ROOT)/flagcx/core/include \
	-I$(FLAGCX_ROOT)/flagcx/runner/include \
	-I$(DEVICE_HOME)/include

COMMON_CLANG_FLAGS := \
	-std=c++17 -x ivcore \
	--cuda-path=$(DEVICE_HOME) --cuda-device-only \
	--cuda-gpu-arch=$(ILUVATAR_ARCH) \
	$(FLAGCX_INCLUDES) \
	-D__clang_llvm_bitcode_lib__ \
	-DUSE_ILUVATAR_ADAPTOR \
	-DFLAGCX_ILUVATAR_DEVICE_COMPILE=1 \
	-DFLAGCX_COMM_TRAITS_DEFAULT \
	-DNDEBUG

CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)
OPT_PASSES := --passes=internalize,inline,globaldce \
	-internalize-public-api-file=$(SYMBOLS_FILE)

all: llvm_ir

$(WRAPPER_HEADER): ../flagcx_device_wrapper.h
	@mkdir -p $(INCLUDEDIR)
	cp -f $< $@

$(UNOPTIMIZED_BC): $(LLVM_SRC) $(LLVM_COMPAT_IMPL)
	@mkdir -p $(OBJDIR)
	@echo "[FlagCX][Iluvatar] Generating unoptimized Device IR"
	$(COREX_CLANG) $(CLANG_FLAGS) $(LLVM_SRC) -o $@

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

$(FINAL_BC): $(ARCH_BC)
	@mkdir -p $(LIBDIR)
	cp -f $< $@

check_public_symbols: $(LLVM_IR_FILE)
	@missing=0; \
	while IFS= read -r symbol; do \
		case "$$symbol" in ''|'#'*) continue ;; esac; \
		if ! grep -Eq "^[[:space:]]*define .*@$$symbol[(]" $<; then \
			echo "Missing public Device IR symbol: $$symbol"; missing=1; \
		fi; \
	done < $(SYMBOLS_FILE); \
	test $$missing -eq 0

check_ir: $(OPTIMIZED_BC) $(LLVM_IR_FILE)
	$(OPT) --passes=verify $(OPTIMIZED_BC) -disable-output
	@if grep -Eq 'PlatformCoop|VTable' $(LLVM_IR_FILE); then \
		echo "Iluvatar Device IR contains PlatformCoop/vtable state"; exit 1; \
	fi
	@if grep -Eq 'call[[:space:]].*%[A-Za-z0-9_.-]+[(]' $(LLVM_IR_FILE); then \
		echo "Iluvatar Device IR contains an indirect device call"; exit 1; \
	fi
	@if ! grep -Eq 'define .*@flagcxCoopAnyInitLanes[(].*i64' $(LLVM_IR_FILE); then \
		echo "Iluvatar lane-mask ABI is not i64"; exit 1; \
	fi
	@if ! grep -Eq 'define .*@flagcxDevCommGetRank[(].*addrspace[(]1[)]' $(LLVM_IR_FILE); then \
		echo "Iluvatar comm pointer ABI is not addrspace(1)"; exit 1; \
	fi
	@if grep -Eq '(atomicrmw|cmpxchg).*i64' $(LLVM_IR_FILE); then \
		echo "Iluvatar Device IR contains unsupported 64-bit RMW"; exit 1; \
	fi
	@if $(LLVM_NM) --undefined-only $(OPTIMIZED_BC) | grep -Ev '^[[:space:]]*$$' | grep -q .; then \
		echo "Iluvatar Device IR contains unresolved external symbols"; \
		$(LLVM_NM) --undefined-only $(OPTIMIZED_BC); exit 1; \
	fi

llvm_ir: $(FINAL_BC) $(ARCH_BC) $(WRAPPER_HEADER) check_public_symbols check_ir
	@echo "[FlagCX][Iluvatar] Device IR ready: $(ARCH_BC)"
	@echo "[FlagCX][Iluvatar] Compatibility copy: $(FINAL_BC)"

else

llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(ARCH_LIBDIR)
	rm -f $(FINAL_BC) $(WRAPPER_HEADER)

.PHONY: all build llvm_ir check_public_symbols check_ir clean
