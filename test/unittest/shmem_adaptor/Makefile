# test/unittest/shmem_adaptor/Makefile
# NVSHMEM adaptor integration tests.
# Build: make USE_NVIDIA=1 USE_SHMEM=1 SHMEM_HOME=/path/to/nvshmem
# Run:   nvshmrun -np 2 build/bin/test_nvshmem_adaptor

include ../../make.inc

# Kernel objects (from test/kernel/nvidia/)
KERNEL_OBJ := $(KERNEL_OBJ_DIR)/nvshmem_adaptor_kernel.o
DLINK_OBJ  := $(KERNEL_OBJ_DIR)/dlink_shmem_test.o

# Local objects
SYNC_OBJ := $(OBJDIR)/nvshmem_state_sync.o

ALL_INCLUDES := $(FLAGCX_INCLUDES) $(KERNEL_INCLUDE) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api \
    -I$(PROJECT_ROOT)/flagcx/adaptor/shmem \
    $(SHMEM_INCLUDE) $(DEVICE_INCLUDE)

ALL_LINK := -L$(SHMEM_HOME)/lib -lnvshmem_host \
    $(DEVICE_LINK) -lcudart -lcuda

TARGET := $(BINDIR)/test_nvshmem_adaptor

.PHONY: all clean

all: $(TARGET)

# Build kernel object via dispatcher
$(KERNEL_OBJ) $(DLINK_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) nvshmem_adaptor_kernel.o dlink_shmem_test.o \
		USE_SHMEM=$(USE_SHMEM) SHMEM_HOME=$(SHMEM_HOME)

$(SYNC_OBJ): nvshmem_state_sync.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling nvshmem_state_sync.cc"
	@$(CXX) -std=c++17 -c -o $@ $< $(SHMEM_INCLUDE) $(ALL_INCLUDES)

$(TARGET): test_nvshmem_adaptor.cpp $(KERNEL_OBJ) $(DLINK_OBJ) $(SYNC_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(KERNEL_OBJ) $(DLINK_OBJ) $(SYNC_OBJ) \
		$(ALL_INCLUDES) $(ALL_LINK) \
		-L$(PROJECT_ROOT)/build/lib -Wl,-rpath,$(PROJECT_ROOT)/build/lib -lflagcx

clean:
	@rm -rf $(BUILDDIR)
