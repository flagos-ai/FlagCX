# Let mcc compile, link and register each self-contained library kernel image.
# Do not embed a combined image into an RDC owner plus a second host-only TU:
# that registers the same weak image through multiple module constructors.
MUSA_KERNEL_INCLUDES := $(foreach dir,$(INCLUDEDIR),-I$(dir)) \
        -I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) \
        -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(SHCA_INCLUDE_FLAG)
MUSA_KERNEL_DEFINES := $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) \
        $(NET_ADAPTOR_FLAG) $(COMPILE_KERNEL_FLAG)

$(DEVOBJ): $(OBJDIR)/%.o: %.cu Makefile makefiles/musa.mk makefiles/musa_kernel_rules.mk $(BUILD_CONFIG_FILE)
	@mkdir -p $(dir $@)
	$(DEVICE_COMPILER) $(filter-out -fgpu-rdc,$(DEVICE_COMPILE_FLAG)) \
	        $(MUSA_KERNEL_INCLUDES) $(MUSA_KERNEL_DEFINES) -o $@ $<
