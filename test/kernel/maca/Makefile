# MetaX/MACA Device API, Device IR and GDR visibility kernels.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
	-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
	-I$(PROJECT_ROOT)/bindings/ir \
	-I$(PROJECT_ROOT)/test/kernel/include \
	$(CCL_INCLUDE) \
	$(DEVICE_INCLUDE)

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o
gdr_visibility.o: $(OBJDIR)/gdr_visibility.o

$(OBJDIR)/device_api.o: device_api.cu
	@mkdir -p $(OBJDIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu
	@mkdir -p $(OBJDIR)
	$(DEVICE_COMPILER) $(DEVICE_COMPILE_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/gdr_visibility.o: gdr_visibility.cu \
	../include/gdr_visibility_test.h \
	../include/gdr_visibility_test_kernel.cuh
	@mkdir -p $(OBJDIR)
	@echo "Compiling gdr_visibility.cu (MetaX/MACA)"
	@$(DEVICE_COMPILER) -std=c++17 $(filter-out -fgpu-rdc,$(DEVICE_COMPILE_FLAG)) -o $@ $< \
		$(ALL_INCLUDES) $(ADAPTOR_DEFINE)

clean:
	@rm -rf $(BUILDDIR)

-include $(OBJDIR)/*.d
