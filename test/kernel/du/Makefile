# test/kernel/du/Makefile
# DU platform kernel compilation for test binaries.

include ../../make.inc

# Propagate FORCE_DEFAULT_PATH
ifeq ($(FORCE_DEFAULT_PATH), 1)
ADAPTOR_DEFINE += -DFLAGCX_FORCE_DEFAULT_PATH
endif

PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    $(PLATFORM_INCLUDES) \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE)

NVCC_RDC_FLAG := -c

.PHONY: all clean device_api.o device_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.cu (DU)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.cu (DU)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# DU does not use SHMEM device-link targets
dlink.o dlink_api.o dlink_ir.o:
	@echo "DU: no device-link needed (no SHMEM support)"

clean:
	@rm -rf $(BUILDDIR)
