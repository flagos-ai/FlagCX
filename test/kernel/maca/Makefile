# MetaX/MACA kernel compilation for GDR visibility conformance tests.

include ../../make.inc

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
	-I$(PROJECT_ROOT)/test/kernel/include \
	$(CCL_INCLUDE) \
	$(DEVICE_INCLUDE)

.PHONY: all clean gdr_visibility.o

all: $(OBJDIR)/gdr_visibility.o

gdr_visibility.o: $(OBJDIR)/gdr_visibility.o

$(OBJDIR)/gdr_visibility.o: gdr_visibility.cu \
	../include/gdr_visibility_test.h \
	../include/gdr_visibility_test_kernel.cuh
	@mkdir -p $(OBJDIR)
	@echo "Compiling gdr_visibility.cu (MetaX/MACA)"
	@$(DEVICE_COMPILER) -std=c++17 $(filter-out -fgpu-rdc,$(DEVICE_COMPILE_FLAG)) -o $@ $< \
		$(ALL_INCLUDES) $(ADAPTOR_DEFINE)

clean:
	@rm -rf $(BUILDDIR)
