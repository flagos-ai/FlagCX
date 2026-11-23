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
	$(DEVICE_INCLUDE) \
	$(SHMEM_INCLUDE)

# -DNDEBUG is load-bearing on this pass, not hygiene: probe r141 measured
# that assert() in the instantiated shared templates (default_comm_traits.h
# :1525/:1621, the barrier index checks) lowers to a __assert_fail call that
# has no matching declaration on the bisheng aicore pass - four
# "no matching function for call to '__assert_fail'" errors, one per assert
# site per compilation pass, with everything else about the include chain
# clean. Other vendor passes bind their libc's assert normally, so the flag
# lives in this file only, and the checks these asserts encode stay enforced
# by the host side of every test that reads the results buffers back.
ASCEND_TEST_FLAGS := -x asc --npu-soc=$(ASCEND_SOC) \
	--cce-auto-infer-kernel-type=false -std=c++17 -O2 -c -fPIC \
	-DFLAGCX_ASCEND_DEVICE_COMPILE=1 -DNDEBUG

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o dlink.o \
	dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.asc
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.asc (Ascend/CANN)"
	@$(DEVICE_COMPILER) $(ASCEND_TEST_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# Device IR launchers for test_device_ir_unified_intra (S16-S18; the file
# itself records why S19-S25 stay undefined). Same flags as the device_api
# object: the bisheng pass links the device image inside -c, so an
# undefined trap reference fails HERE, not at test link.
$(OBJDIR)/device_ir.o: device_ir.asc
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.asc (Ascend/CANN)"
	@$(DEVICE_COMPILER) $(ASCEND_TEST_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# Not implemented on Ascend yet, and requested only by suites that this
# platform does not run. Fail loudly instead of producing an empty object.
gdr_visibility.o:
	@echo "Ascend: $@ is not implemented (GDR visibility belongs to the" \
		"peer-memory stage)" && false

# bisheng embeds the device image in each object (.aicore_binary section), so
# there is no separate device-link product.
dlink.o dlink_api.o dlink_ir.o:
	@echo "Ascend/CANN embeds the device image in each object; no dlink object is needed"

clean:
	@rm -rf $(BUILDDIR)
