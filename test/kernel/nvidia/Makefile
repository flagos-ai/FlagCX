# test/kernel/nvidia/Makefile
# NVIDIA CUDA kernel compilation for test binaries.

include ../../make.inc

# Propagate FORCE_DEFAULT_PATH
ifeq ($(FORCE_DEFAULT_PATH), 1)
ADAPTOR_DEFINE += -DFLAGCX_FORCE_DEFAULT_PATH
endif

PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

ALL_INCLUDES := $(FLAGCX_INCLUDES) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/bindings/ir \
    -I$(PROJECT_ROOT)/test/kernel/include \
    $(PLATFORM_INCLUDES) \
    $(CCL_INCLUDE) \
    $(DEVICE_INCLUDE) \
    $(SHMEM_INCLUDE)

# NVSHMEM requires relocatable device code (-dc) + device-link step
ifeq ($(USE_SHMEM), 1)
NVCC_RDC_FLAG := -dc
else
NVCC_RDC_FLAG := -c
endif

.PHONY: all clean device_api.o device_ir.o gdr_visibility.o nvshmem_adaptor_kernel.o dlink.o dlink_api.o dlink_ir.o device_traits

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o
gdr_visibility.o: $(OBJDIR)/gdr_visibility.o
nvshmem_adaptor_kernel.o: $(OBJDIR)/nvshmem_adaptor_kernel.o

# Standalone single-GPU coverage for the NVIDIA default PlatformTraits API.
$(BINDIR)/test_device_api_traits: device_api_traits.cu
	@mkdir -p $(BINDIR)
	@$(DEVICE_COMPILER) -std=c++11 -rdc=true $(DEVICE_COMPILER_GENCODE) -o $@ $< \
		$(ALL_INCLUDES) -DUSE_NVIDIA_ADAPTOR -DFLAGCX_FORCE_DEFAULT_PATH \
		-DFLAGCX_COMM_TRAITS_DEFAULT -DCOMPILE_KERNEL

device_traits: $(BINDIR)/test_device_api_traits
	@$(BINDIR)/test_device_api_traits

# SDK-backed synthetic SQ/CQ and wait/abort tests; these use a GPU, not NIC DMA.
ifeq ($(USE_XPUNETIO),1)
$(BINDIR)/test_net_gda_%: net_gda_%.cu
	@mkdir -p $(BINDIR)
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -o $@ $< \
	    -MMD -MP -MF $@.d -MT $@ \
	    $(ALL_INCLUDES) $(ADAPTOR_DEFINE) -DCOMPILE_KERNEL $(DEVICE_LINK)

.PHONY: gda_device_tests
gda_device_tests: $(BINDIR)/test_net_gda_put_signal $(BINDIR)/test_net_gda_wait_control
	@$(BINDIR)/test_net_gda_put_signal
	@$(BINDIR)/test_net_gda_wait_control
endif

$(OBJDIR)/gdr_visibility.o: gdr_visibility.cu \
	../include/gdr_visibility_test.h \
	../include/gdr_visibility_test_kernel.cuh
	@mkdir -p $(OBJDIR)
	@echo "Compiling gdr_visibility.cu (CUDA-compatible)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -c -o $@ $< \
		$(ALL_INCLUDES) $(ADAPTOR_DEFINE)

$(OBJDIR)/device_api.o: device_api.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_api.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

$(OBJDIR)/device_ir.o: device_ir.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling device_ir.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE)

# The same objects serve Proxy, NCCL and GDA builds. Invalidate them when
# compiler flags, SDK paths or the selected backend change.
DEVICE_CONFIG_STAMP := $(OBJDIR)/.device_config
.PHONY: device_config_force
$(DEVICE_CONFIG_STAMP): device_config_force
	@mkdir -p $(OBJDIR)
	@printf '%s\n' '$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) $(ALL_INCLUDES) $(ADAPTOR_DEFINE)' | cmp -s - $@ || \
	  printf '%s\n' '$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) $(NVCC_RDC_FLAG) $(ALL_INCLUDES) $(ADAPTOR_DEFINE)' > $@

$(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o: $(DEVICE_CONFIG_STAMP)

$(OBJDIR)/nvshmem_adaptor_kernel.o: nvshmem_adaptor_kernel.cu
	@mkdir -p $(OBJDIR)
	@echo "Compiling nvshmem_adaptor_kernel.cu (CUDA)"
	@$(DEVICE_COMPILER) -std=c++17 $(DEVICE_COMPILER_GENCODE) -dc -o $@ $< \
		$(ALL_INCLUDES) -DCOMPILE_KERNEL $(ADAPTOR_DEFINE) \
		-DFLAGCX_COMM_TRAITS_SHMEM -DUSE_NVIDIA_ADAPTOR

# Device-link targets (NVSHMEM only)
ifeq ($(USE_SHMEM), 1)
SHMEM_DEVICE_LIB := $(SHMEM_HOME)/lib/libnvshmem_device.a

$(OBJDIR)/dlink_api.o: $(OBJDIR)/device_api.o
	@echo "Device-linking device_api (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

$(OBJDIR)/dlink_ir.o: $(OBJDIR)/device_ir.o
	@echo "Device-linking device_ir (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

$(OBJDIR)/dlink.o: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o
	@echo "Device-linking all (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

$(OBJDIR)/dlink_shmem_test.o: $(OBJDIR)/nvshmem_adaptor_kernel.o
	@echo "Device-linking nvshmem_adaptor_kernel (NVSHMEM)"
	@$(DEVICE_COMPILER) $(DEVICE_COMPILER_GENCODE) -dlink -o $@ $^ \
		$(SHMEM_DEVICE_LIB) -lcudart -lcudadevrt

dlink.o: $(OBJDIR)/dlink.o
dlink_api.o: $(OBJDIR)/dlink_api.o
dlink_ir.o: $(OBJDIR)/dlink_ir.o
dlink_shmem_test.o: $(OBJDIR)/dlink_shmem_test.o

all: $(OBJDIR)/dlink.o
endif

clean:
	@rm -rf $(BUILDDIR)

ifeq ($(USE_XPUNETIO),1)
$(BINDIR)/test_net_gda_put_signal $(BINDIR)/test_net_gda_wait_control: $(DEVICE_CONFIG_STAMP)
-include $(wildcard $(BINDIR)/test_net_gda_*.d)
endif
