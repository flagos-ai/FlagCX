# MUSA test kernels: device compile -> device link -> host image embedding.
# Final objects implement the public test/kernel/include launcher ABI.
include ../../make.inc

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
	-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
	-I$(PROJECT_ROOT)/bindings/ir \
	-I$(PROJECT_ROOT)/test/kernel/include \
	$(CCL_INCLUDE) $(DEVICE_INCLUDE)

MUSA_TOOLCHAIN_ROOT ?= $(abspath $(dir $(realpath $(DEVICE_COMPILER)))/..)
MUSA_DEVICE_LD ?= $(MUSA_TOOLCHAIN_ROOT)/bin/lld
# This is an argument emitted by mcc, not a mandatory installed file.
MUSA_LTO_PLUGIN ?= $(MUSA_TOOLCHAIN_ROOT)/lib/LLVMgold.so
MUSA_RESOURCE_DIR ?= $(shell $(DEVICE_COMPILER) -print-resource-dir)
MUSA_TRAP_HANDLER ?= $(MUSA_RESOURCE_DIR)/lib/linux/trap_handler.o

MUSA_KERNEL_FLAGS := $(DEVICE_COMPILE_FLAG) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)
KERNEL_NAMES := device_api device_ir
KERNEL_OBJECTS := $(addprefix $(OBJDIR)/,$(addsuffix .o,$(KERNEL_NAMES)))
DEVICE_OBJECTS := $(addprefix $(OBJDIR)/,$(addsuffix .device.o,$(KERNEL_NAMES)))
DEVICE_IMAGES := $(addprefix $(OBJDIR)/,$(addsuffix .mubin,$(KERNEL_NAMES)))

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o

all: $(KERNEL_OBJECTS)

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

# Keep intermediate images; do not rebuild them on every incremental make.
.SECONDARY: $(DEVICE_OBJECTS) $(DEVICE_IMAGES)

$(DEVICE_OBJECTS): $(OBJDIR)/%.device.o: %.cu Makefile $(PROJECT_ROOT)/makefiles/musa.mk
	@mkdir -p $(OBJDIR)
	$(DEVICE_COMPILER) $(MUSA_KERNEL_FLAGS) $(ALL_INCLUDES) \
		--cuda-device-only -Wno-deprecated-declarations -o $@ $<

$(DEVICE_IMAGES): $(OBJDIR)/%.mubin: $(OBJDIR)/%.device.o $(MUSA_TRAP_HANDLER) Makefile
	$(MUSA_DEVICE_LD) -flavor gnu --no-undefined \
		-plugin $(MUSA_LTO_PLUGIN) -plugin-opt=mcpu=$(MUSA_ARCH) -plugin-opt=O2 \
		$< $(MUSA_TRAP_HANDLER) -o $@

$(KERNEL_OBJECTS): $(OBJDIR)/%.o: %.cu $(OBJDIR)/%.mubin Makefile $(PROJECT_ROOT)/makefiles/musa.mk
	$(DEVICE_COMPILER) $(MUSA_KERNEL_FLAGS) $(ALL_INCLUDES) \
		--cuda-host-only -Wno-deprecated-declarations \
		-Xclang -fcuda-include-gpubinary -Xclang $(OBJDIR)/$*.mubin \
		-o $@ $<

# Each final host object embeds its explicitly linked device image.
# The public test driver therefore needs no additional nvcc-style dlink object.
dlink.o dlink_api.o dlink_ir.o: all
	@echo "MUSA device images are linked and embedded in device_api.o/device_ir.o"

clean:
	$(RM) $(KERNEL_OBJECTS) $(DEVICE_OBJECTS) $(DEVICE_IMAGES) \
		$(KERNEL_OBJECTS:.o=.d) $(DEVICE_OBJECTS:.o=.d)

-include $(KERNEL_OBJECTS:.o=.d) $(DEVICE_OBJECTS:.o=.d)
