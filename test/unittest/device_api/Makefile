include ../../make.inc

# Shared device kernel objects (built in test/kernel/[platform]/)
DEVICE_API_OBJ := $(KERNEL_OBJ_DIR)/device_api.o
DEVICE_IR_OBJ  := $(KERNEL_OBJ_DIR)/device_ir.o

# Device-link objects (only needed for NVSHMEM)
ifeq ($(USE_SHMEM), 1)
DLINK_API_OBJ := $(KERNEL_OBJ_DIR)/dlink_api.o
DLINK_IR_OBJ  := $(KERNEL_OBJ_DIR)/dlink_ir.o
else
DLINK_API_OBJ :=
DLINK_IR_OBJ  :=
endif

# Shared tools object
TOOLS_SRC := $(PROJECT_ROOT)/test/tools.cc
TOOLS_OBJ := $(OBJDIR)/tools.o

# IR wrapper headers
IR_INCLUDE := -I$(PROJECT_ROOT)/bindings/ir
PLATFORM_INCLUDES := -I$(PLATFORM_IR_DIR)

# All includes
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(IR_INCLUDE) $(PLATFORM_INCLUDES) \
    $(KERNEL_INCLUDE) $(PERF_INCLUDE) $(MPI_INCLUDE) $(DEVICE_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# Kernel flag
CXXFLAGS += -DCOMPILE_KERNEL

# Test targets
MPI_TARGETS := $(BINDIR)/test_device_ir_intra \
               $(BINDIR)/test_device_ir_inter \
               $(BINDIR)/test_device_api

.PHONY: all clean run-unit run-mpi run-mpi-intra run-mpi-inter run

all: $(MPI_TARGETS)

# Build device kernel objects via dispatcher
KERNEL_API_TARGETS := device_api.o
KERNEL_IR_TARGETS  := device_ir.o
ifeq ($(USE_SHMEM), 1)
KERNEL_API_TARGETS += dlink_api.o
KERNEL_IR_TARGETS  += dlink_ir.o
endif

$(DEVICE_API_OBJ) $(DLINK_API_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_API_TARGETS)

$(DEVICE_IR_OBJ) $(DLINK_IR_OBJ):
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_IR_TARGETS)

$(TOOLS_OBJ): $(TOOLS_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(ALL_INCLUDES)

$(BINDIR)/test_device_ir_intra: test_device_ir_intra.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir_inter: test_device_ir_inter.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_api: test_device_api.cpp $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)

# No pure unit tests here — these all require MPI + GPU
run-unit:
	@echo "device_api: no unit tests (all require MPI + GPU)"

# Env vars for inter-node testing on single node
HETERO_ENV := -x FLAGCX_USE_HETERO_COMM=1 \
              -x FLAGCX_MEM_ENABLE=1 \
              -x FLAGCX_VMM_ENABLE=0 \
              -x FLAGCX_P2P_DISABLE=1 \
              -x FLAGCX_DEBUG=Version \
              -x FLAGCX_DEBUG_SUBSYS= \
              -x LD_LIBRARY_PATH

# Intra-node tests: safe to run on a single node
run-mpi-intra: $(BINDIR)/test_device_ir_intra $(BINDIR)/test_device_api
	@echo "Running intra-node tests..."
	mpirun -np 8 --allow-run-as-root $(BINDIR)/test_device_ir_intra
	mpirun -np 8 --allow-run-as-root $(HETERO_ENV) $(BINDIR)/test_device_api -b 1M -e 4M -f 2 -R 2

# Inter-node tests: require FLAGCX_USE_HETERO_COMM=1 or actual multi-node
run-mpi-inter: $(BINDIR)/test_device_ir_inter
	@echo "Running inter-node Device IR tests..."
	mpirun -np 8 --allow-run-as-root $(HETERO_ENV) $(BINDIR)/test_device_ir_inter

run-mpi: run-mpi-intra run-mpi-inter

run: run-mpi
