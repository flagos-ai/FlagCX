# Ascend (CANN) Device API kernel compilation.
#
# Flags mirror makefiles/ascend.mk's DEVICE_COMPILE_FLAG, which was validated on
# Ascend 910C (CANN 9.1.0-beta.3):
#   bisheng -x asc --npu-soc=Ascend910_9382 --cce-auto-infer-kernel-type=false
#       -c f.asc -o f.o                                   -> rc=0
# --npu-soc takes the runtime-reported soc name (aclrtGetSocName()); the
# --npu-arch and --cce-soc-version spellings are rejected by this toolchain.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
	-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
	-I$(PROJECT_ROOT)/bindings/ir \
	-I$(PROJECT_ROOT)/test/kernel/include \
	$(CCL_INCLUDE) \
	$(DEVICE_INCLUDE)

ASCEND_TEST_FLAGS := -x asc --npu-soc=$(ASCEND_SOC) \
	--cce-auto-infer-kernel-type=false -std=c++17 -O2 -c -fPIC \
	-DFLAGCX_ASCEND_DEVICE_COMPILE=1

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o dlink.o \
	dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o

device_api.o: $(OBJDIR)/device_api.o

$(OBJDIR)/device_api.o: device_api.asc
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.asc (Ascend/CANN)"
	@$(DEVICE_COMPILER) $(ASCEND_TEST_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# Not implemented on Ascend yet, and requested only by suites that this
# platform does not run. Fail loudly instead of producing an empty object.
device_ir.o gdr_visibility.o:
	@echo "Ascend: $@ is not implemented (Device IR needs bindings/ir/ascend;" \
		"GDR visibility belongs to the peer-memory stage)" && false

# bisheng embeds the device image in each object (.aicore_binary section), so
# there is no separate device-link product.
dlink.o dlink_api.o dlink_ir.o:
	@echo "Ascend/CANN embeds the device image in each object; no dlink object is needed"

clean:
	@rm -rf $(BUILDDIR)
