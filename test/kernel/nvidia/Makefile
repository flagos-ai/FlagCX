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

.PHONY: all clean device_api.o device_ir.o nvshmem_adaptor_kernel.o dlink.o dlink_api.o dlink_ir.o

all: $(OBJDIR)/device_api.o $(OBJDIR)/device_ir.o

device_api.o: $(OBJDIR)/device_api.o
device_ir.o: $(OBJDIR)/device_ir.o
nvshmem_adaptor_kernel.o: $(OBJDIR)/nvshmem_adaptor_kernel.o

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
