# HGGC Device API and RMA visibility test kernels.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE)

HGGC_COMPILE_FLAGS := -x hg -std=c++17 -c -Xcompiler -fPIC
PPU_TRAITS_HEADERS := $(wildcard $(PROJECT_ROOT)/flagcx/adaptor/include/device_api/*.h) \
    $(PROJECT_ROOT)/flagcx/adaptor/include/device_utils.h

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o device_traits stream_compat

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o
gdr_visibility.o: $(OBJDIR)/gdr_visibility.o

$(OBJDIR)/device_api.o: device_api.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(HGGC_COMPILE_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(HGGC_COMPILE_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		-DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/gdr_visibility.o: gdr_visibility.cu \
    ../include/gdr_visibility_test.h \
    ../include/gdr_visibility_test_kernel.cuh
	@mkdir -p $(OBJDIR)
	@$(DEVICE_COMPILER) $(HGGC_COMPILE_FLAGS) -o $@ $< $(ALL_INCLUDES) \
		$(ADAPTOR_DEFINE)

$(BINDIR)/test_device_api_traits: device_api_traits.cu $(PPU_TRAITS_HEADERS)
	@mkdir -p $(BINDIR)
	@$(DEVICE_COMPILER) -x hg -std=c++17 -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE) -L$(DEVICE_LIB) $(DEVICE_LINK)

device_traits: $(BINDIR)/test_device_api_traits
	@status=0; \
	for memory in device host; do \
	  for bits in 32 64; do \
	    for op in load store add sub cas exchange max or and; do \
	      $(BINDIR)/test_device_api_traits --atomic $$memory $$bits $$op || status=1; \
	    done; \
	  done; \
	done; \
	$(BINDIR)/test_device_api_traits || status=1; \
	exit $$status

# Compatibility APIs are limited to the Torch boundary. No CUDA libraries are
# linked into libflagcx or the native Device API kernels.
$(BINDIR)/test_ppu_stream_compat: stream_compat.cc \
    $(PROJECT_ROOT)/plugin/torch/flagcx/include/ppu_stream_compat.hpp \
    $(PROJECT_ROOT)/flagcx/adaptor/include/ppu_adaptor.h
	@mkdir -p $(BINDIR)
	@$(CXX) $(CXXFLAGS) -o $@ $< $(ALL_INCLUDES) $(ADAPTOR_DEFINE) \
		-I$(PROJECT_ROOT)/plugin/torch/flagcx/include \
		-I$(DEVICE_HOME)/CUDA_SDK/include -L$(DEVICE_HOME)/CUDA_SDK/lib64 \
		-L$(DEVICE_LIB) $(DEVICE_LINK) -lcudart -lcuda

stream_compat: $(BINDIR)/test_ppu_stream_compat
	@$(BINDIR)/test_ppu_stream_compat

clean:
	@rm -rf $(BUILDDIR)
