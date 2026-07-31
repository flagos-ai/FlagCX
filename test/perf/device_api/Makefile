include ../../make.inc

# Each test_*.cpp becomes its own binary: $(BINDIR)/perf_<name>
SRCS := $(wildcard test_*.cpp)
BINS := $(SRCS:test_%.cpp=$(BINDIR)/perf_%)

# Shared source files
TOOLS_SRC   := $(PROJECT_ROOT)/test/tools.cc
COMMON_SRCS := $(TOOLS_SRC)

# Shared device kernel object (built in test/kernel/[platform]/)
DEVICE_API_OBJ := $(KERNEL_OBJ_DIR)/device_api.o

# Device-link object (only needed for NVSHMEM)
ifeq ($(USE_SHMEM), 1)
DLINK_OBJ := $(KERNEL_OBJ_DIR)/dlink_api.o
SHMEM_ADAPTOR_DIR := $(PROJECT_ROOT)/test/unittest/shmem_adaptor
SHMEM_SYNC_OBJ := $(OBJDIR)/nvshmem_state_sync.o
else
DLINK_OBJ :=
SHMEM_SYNC_OBJ :=
endif

# Includes
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(PERF_INCLUDE) $(MPI_INCLUDE) \
    -I$(PROJECT_ROOT)/flagcx/core -I$(PROJECT_ROOT)/flagcx/service \
    $(KERNEL_INCLUDE) $(DEVICE_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

CXXFLAGS += -DCOMPILE_KERNEL

.PHONY: all clean

all: $(BINS)

# Build kernel objects via dispatcher
KERNEL_TARGETS := device_api.o
ifeq ($(USE_SHMEM), 1)
KERNEL_TARGETS += dlink_api.o
endif

$(DEVICE_API_OBJ) $(DLINK_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_TARGETS)

# Build nvshmem_state_sync.o locally from shmem_adaptor source
ifeq ($(USE_SHMEM), 1)
$(SHMEM_SYNC_OBJ): $(SHMEM_ADAPTOR_DIR)/nvshmem_state_sync.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling nvshmem_state_sync.cc"
	@$(CXX) -std=c++17 -c -o $@ $< \
		$(SHMEM_INCLUDE) $(FLAGCX_INCLUDES) \
		-I$(PROJECT_ROOT)/flagcx/adaptor/include/device_api $(DEVICE_INCLUDE)
endif

$(BINDIR)/perf_%: test_%.cpp $(COMMON_SRCS) $(DEVICE_API_OBJ) $(DLINK_OBJ) $(SHMEM_SYNC_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Compiling $@"
	@$(CXX) $(CXXFLAGS) $(ADAPTOR_DEFINE) -o $@ $< $(COMMON_SRCS) $(DEVICE_API_OBJ) $(SHMEM_SYNC_OBJ) $(DLINK_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)
