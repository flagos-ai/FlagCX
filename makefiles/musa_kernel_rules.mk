# Included by the root Makefile after DEVOBJ and build-config are resolved.
# MUSA main-library kernels use the same split pipeline as test/kernel/musa.
MUSA_TOOLCHAIN_ROOT ?= $(abspath $(dir $(realpath $(DEVICE_COMPILER)))/..)
MUSA_DEVICE_LD ?= $(MUSA_TOOLCHAIN_ROOT)/bin/lld
MUSA_LTO_PLUGIN ?= $(MUSA_TOOLCHAIN_ROOT)/lib/LLVMgold.so
MUSA_RESOURCE_DIR ?= $(shell $(DEVICE_COMPILER) -print-resource-dir)
MUSA_TRAP_HANDLER ?= $(MUSA_RESOURCE_DIR)/lib/linux/trap_handler.o
MUSA_DEVICE_OBJECTS := $(DEVOBJ:.o=.musa_device.o)
MUSA_DEVICE_IMAGES := $(DEVOBJ:.o=.mubin)
MUSA_KERNEL_INCLUDES := $(foreach dir,$(INCLUDEDIR),-I$(dir)) \
	-I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) \
	-I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(SHCA_INCLUDE_FLAG)
MUSA_KERNEL_DEFINES := $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) \
	$(NET_ADAPTOR_FLAG) $(COMPILE_KERNEL_FLAG)

.SECONDARY: $(MUSA_DEVICE_OBJECTS) $(MUSA_DEVICE_IMAGES)
.DELETE_ON_ERROR:

$(MUSA_DEVICE_OBJECTS): $(OBJDIR)/%.musa_device.o: %.cu Makefile makefiles/musa.mk makefiles/musa_kernel_rules.mk $(BUILD_CONFIG_FILE)
	@mkdir -p $(dir $@)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) $(MUSA_KERNEL_INCLUDES) \
		$(MUSA_KERNEL_DEFINES) --cuda-device-only -o $@ $<

$(MUSA_DEVICE_IMAGES): $(OBJDIR)/%.mubin: $(OBJDIR)/%.musa_device.o $(MUSA_TRAP_HANDLER) makefiles/musa_kernel_rules.mk
	$(MUSA_DEVICE_LD) -flavor gnu --no-undefined \
		-plugin $(MUSA_LTO_PLUGIN) -plugin-opt=mcpu=$(MUSA_ARCH) -plugin-opt=O2 \
		$< $(MUSA_TRAP_HANDLER) -o $@

$(DEVOBJ): $(OBJDIR)/%.o: %.cu $(OBJDIR)/%.mubin makefiles/musa_kernel_rules.mk
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) $(MUSA_KERNEL_INCLUDES) \
		$(MUSA_KERNEL_DEFINES) --cuda-host-only \
		-Xclang -fcuda-include-gpubinary -Xclang $(OBJDIR)/$*.mubin -o $@ $<

ifneq ($(BUILD_CONFIG_CHANGED),)
$(MUSA_DEVICE_OBJECTS): build_config_force
endif

-include $(MUSA_DEVICE_OBJECTS:.o=.d)
