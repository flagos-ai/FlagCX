include ../../make.inc

# MPI launcher configuration. Open MPI remains the default used by CI.
# Use `MPI_FLAVOR=mpich` for MPICH/Hydra (`--genv`, no root-only option).
MPI_FLAVOR ?= openmpi
MPIRUN ?= mpirun
ifeq ($(MPI_FLAVOR),openmpi)
MPI_ENV_FLAG := -x
MPI_ROOT_ARGS ?= --allow-run-as-root
else ifeq ($(MPI_FLAVOR),mpich)
MPI_ENV_FLAG := --genv
MPI_ROOT_ARGS ?=
else
$(error Unsupported MPI_FLAVOR '$(MPI_FLAVOR)'; use openmpi or mpich)
endif

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
UNIT_SRC := test_dev_comm_cleanup.cpp
UNIT_OBJ := $(OBJDIR)/test_dev_comm_cleanup.o
UNIT_TARGET := $(BINDIR)/test_dev_comm_cleanup
UNIT_CXXFLAGS := $(filter-out -DCOMPILE_KERNEL,$(CXXFLAGS))
UNIT_INCLUDES := $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) \
                 -I$(PROJECT_ROOT)/flagcx/adaptor/device_api

MPI_TARGETS := $(BINDIR)/test_device_ir_intra \
               $(BINDIR)/test_device_ir_inter \
               $(BINDIR)/test_device_ir_unified_intra \
               $(BINDIR)/test_device_ir_unified_inter \
               $(BINDIR)/test_device_api_intra \
               $(BINDIR)/test_device_api_inter

.PHONY: all unit mpi clean run-unit run-mpi run-mpi-intra run-mpi-inter \
        run-mpi-unified-intra run-mpi-unified-inter run

all: unit mpi

unit: $(UNIT_TARGET)

mpi: $(MPI_TARGETS)

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

$(UNIT_OBJ): $(UNIT_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(UNIT_CXXFLAGS) $(UNIT_INCLUDES)

$(UNIT_TARGET): $(UNIT_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK)

$(BINDIR)/test_device_ir_intra: test_device_ir_intra.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir_inter: test_device_ir_inter.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir_unified_intra: test_device_ir_unified_intra.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_ir_unified_inter: test_device_ir_unified_inter.cpp $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_api_intra: test_device_api_intra.cpp $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

$(BINDIR)/test_device_api_inter: test_device_api_inter.cpp $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)

run-unit: unit
	@$(UNIT_TARGET)

# Env vars for inter-node testing on single node
HETERO_ENV := $(MPI_ENV_FLAG) FLAGCX_USE_HETERO_COMM=1 \
              $(MPI_ENV_FLAG) FLAGCX_MEM_ENABLE=1 \
              $(MPI_ENV_FLAG) FLAGCX_VMM_ENABLE=0 \
              $(MPI_ENV_FLAG) FLAGCX_P2P_DISABLE=1 \
              $(MPI_ENV_FLAG) FLAGCX_DEBUG=Version \
              $(MPI_ENV_FLAG) FLAGCX_DEBUG_SUBSYS= \
              $(MPI_ENV_FLAG) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)

# Unified IR keeps P2P enabled. The inter target creates two logical nodes with
# MPI MPMD app contexts instead of globally disabling P2P or using a wrapper.
UNIFIED_COMMON_ENV := $(MPI_ENV_FLAG) FLAGCX_USE_HETERO_COMM=1 \
                      $(MPI_ENV_FLAG) FLAGCX_VMM_ENABLE=0 \
                      $(MPI_ENV_FLAG) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
INTER_NODE0_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node0 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node0
INTER_NODE1_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node1 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node1
UNIFIED_TEST_ARGS := -b 1K -e 16M -f 2

# Intra-node tests: safe to run on a single node
run-mpi-intra: $(BINDIR)/test_device_ir_intra $(BINDIR)/test_device_api_intra
	@echo "Running intra-node tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(BINDIR)/test_device_ir_intra
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(BINDIR)/test_device_api_intra

# Inter-node tests: require FLAGCX_USE_HETERO_COMM=1 or actual multi-node
run-mpi-inter: $(BINDIR)/test_device_ir_inter $(BINDIR)/test_device_api_inter
	@echo "Running inter-node Device IR tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(HETERO_ENV) $(INTER_NODE0_ENV) $(BINDIR)/test_device_ir_inter \
		: -n 4 $(HETERO_ENV) $(INTER_NODE1_ENV) $(BINDIR)/test_device_ir_inter
	@echo "Running inter-node Device API tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(HETERO_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2 \
		: -n 4 $(HETERO_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2

# Unified Intra suite: INTRA + WORLD teams (single-node)
run-mpi-unified-intra: $(BINDIR)/test_device_ir_unified_intra
	@echo "Running unified intra suite tests (INTRA + WORLD, 8-combo)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(UNIFIED_COMMON_ENV) \
		$(BINDIR)/test_device_ir_unified_intra $(UNIFIED_TEST_ARGS)

# Unified Inter suite: INTER + WORLD teams (two logical 4-rank nodes)
run-mpi-unified-inter: $(BINDIR)/test_device_ir_unified_inter
	@echo "Running unified inter suite tests (INTER + WORLD, 8-combo)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(UNIFIED_COMMON_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS) \
		: -n 4 $(UNIFIED_COMMON_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS)

run-mpi: run-mpi-intra run-mpi-inter run-mpi-unified-intra run-mpi-unified-inter

run: run-unit run-mpi
