# PPU CUDA-compatible Device API and RMA visibility test kernels.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE)

PPU_NVCC_FLAGS := -std=c++17 -c --cudart=shared -Xcompiler -fPIC
PPU_TRAITS_HEADERS := $(wildcard $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/*.h) \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_utils.h

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o device_traits

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o
gdr_visibility.o: $(OBJDIR)/gdr_visibility.o

$(OBJDIR)/device_api.o: device_api.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(PPU_NVCC_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(PPU_NVCC_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/gdr_visibility.o: gdr_visibility.cu \
    ../include/gdr_visibility_test.h \
    ../include/gdr_visibility_test_kernel.cuh
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(PPU_NVCC_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		$(ADAPTOR_DEFINE)

$(BINDIR)/test_device_api_traits: device_api_traits.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(BINDIR)
	@$(DEVICE_COMPILER) -std=c++17 --cudart=shared -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

device_traits: $(BINDIR)/test_device_api_traits
	@$(BINDIR)/test_device_api_traits

clean:
	@rm -rf $(BUILDDIR)
