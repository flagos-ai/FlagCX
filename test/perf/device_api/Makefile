include ../../make.inc

# Each test_*.cpp becomes its own binary: $(BINDIR)/perf_<name>
SRCS := $(wildcard test_*.cpp)
BINS := $(SRCS:test_%.cpp=$(BINDIR)/perf_%)

# Shared source files
TOOLS_SRC   := $(PROJECT_ROOT)/test/tools.cc
COMMON_SRCS := $(TOOLS_SRC)

# Shared device kernel object (built in test/kernel/[platform]/)
DEVICE_API_OBJ := $(KERNEL_OBJ_DIR)/device_api.o

# Device-link object (only needed for NVSHMEM). KLX selects xccl xshmem with the
# same USE_SHMEM switch, but its device code is linked into device_api.o and it
# needs no nvshmem_state_sync.o (device state is passed as a kernel argument).
ifeq ($(USE_SHMEM)$(KERNEL_PLATFORM), 1klx)
DLINK_OBJ :=
SHMEM_SYNC_OBJ :=
else ifeq ($(USE_SHMEM), 1)
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
    $(KERNEL_INCLUDE) $(DEVICE_INCLUDE) $(SHMEM_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# Kunlunxin: device_api.o (native xshmem kernel) references xshmem host symbols
# that live in libbkcl, and the XRE runtime libs. libflagcx.so imports (not
# exports) these, so the executable must link them explicitly. Add rpaths so the
# binary finds libbkcl / libxpurt* at runtime.
ifeq ($(USE_KUNLUNXIN), 1)
ALL_LINK += -L$(CCL_LIB) -Wl,-rpath,$(CCL_LIB) $(CCL_LINK) \
            -Wl,-rpath,$(DEVICE_LIB)
endif

CXXFLAGS += -DCOMPILE_KERNEL

.PHONY: all clean

all: $(BINS)

# Build kernel objects via dispatcher
KERNEL_TARGETS := device_api.o
ifneq ($(DLINK_OBJ),)
KERNEL_TARGETS += dlink_api.o
endif

$(DEVICE_API_OBJ) $(DLINK_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_TARGETS)

# Build nvshmem_state_sync.o only for NVSHMEM. XSHMEM passes its device state
# through the DevComm value and has no consumer-binary constant to synchronize.
ifneq ($(SHMEM_SYNC_OBJ),)
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
	@$(CXX) $(CXXFLAGS) $(HOST_ADAPTOR_DEFINE) -o $@ $< $(COMMON_SRCS) $(DEVICE_API_OBJ) $(SHMEM_SYNC_OBJ) $(DLINK_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)
