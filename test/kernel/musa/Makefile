# test/kernel/musa/Makefile
# MUSA device kernel compilation for test binaries.

include ../../make.inc

# Propagate FORCE_DEFAULT_PATH
ifeq ($(FORCE_DEFAULT_PATH), 1)
ADAPTOR_DEFINE += -DFLAGCX_FORCE_DEFAULT_PATH
endif

PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

# Every kernel object includes the platform traits and the shared device
# utilities, but mcc emits no usable header dependency file here. Without these
# explicit prerequisites a traits change silently keeps the stale .o and the
# test suite then fails on kernels that were never rebuilt.
TRAITS_DEPS := \
    $(PROJECT_ROOT)/flagcx/adaptor/include/musa_adaptor.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_utils.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/platform_traits.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/comm_traits.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/musa_platform_traits.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/musa_comm_traits.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/default_comm_traits.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/default_completion.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/completion_word.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/flagcx_device_core.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/flagcx_device.h \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/fifo_producer_gate.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_unified_ir.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_unified_ir_impl.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_compat_ir_impl.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_scalar_ir.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_scalar_ir_impl.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_wrapper.h \
    $(PROJECT_ROOT)/bindings/ir/flagcx_device_wrapper_impl.h

# The shared NVIDIA kernel sources are reused through thin .mu wrappers. MUSA
# has no cuda_runtime.h, so this directory also provides a compatibility shim
# that the wrappers put on the include path.
ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    -I$(CURDIR) \
    $(PLATFORM_INCLUDES) \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE) \
    $(SHMEM_INCLUDE)

.PHONY: all clean device_api.o device_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.mu cuda_runtime.h \
	../nvidia/device_api.cu $(TRAITS_DEPS)
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.mu (MUSA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -fPIC -c -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.mu cuda_runtime.h \
	../nvidia/device_ir.cu $(TRAITS_DEPS)
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.mu (MUSA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -fPIC -c -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

clean:
	@rm -rf $(BUILDDIR)
