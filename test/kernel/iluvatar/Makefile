# Iluvatar/CoreX Device API kernel compilation.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
	-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
	-I$(PROJECT_ROOT)/bindings/ir \
	-I$(PROJECT_ROOT)/test/kernel/include \
	$(CCL_INCLUDE) \
	$(DEVICE_INCLUDE)

COREX_TEST_FLAGS := -x ivcore --cuda-path=$(DEVICE_HOME) \
	--cuda-gpu-arch=$(ILUVATAR_ARCH) -std=c++17 -O2 -c -fPIC \
	-DFLAGCX_ILUVATAR_DEVICE_COMPILE=1

.PHONY: all clean device_api.o device_ir.o dlink.o dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o

$(OBJDIR)/device_api.o: device_api.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.cu (Iluvatar/CoreX)"
	@$(DEVICE_COMPILER) $(COREX_TEST_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.cu (Iluvatar/CoreX)"
	@$(DEVICE_COMPILER) $(COREX_TEST_FLAGS) -Wno-deprecated-declarations \
		-o $@ $< $(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

dlink.o dlink_api.o dlink_ir.o:
	@echo "Iluvatar/CoreX embeds the device image in each object; no dlink object is needed"

clean:
	@rm -rf $(BUILDDIR)
