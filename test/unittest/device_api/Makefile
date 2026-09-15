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

# Device-link objects (only needed for NVSHMEM). KLX selects xccl xshmem with the
# same USE_SHMEM switch, but its device code is linked into device_api.o.
ifeq ($(USE_SHMEM)$(KERNEL_PLATFORM), 1klx)
DLINK_API_OBJ :=
DLINK_IR_OBJ  :=
else ifeq ($(USE_SHMEM), 1)
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

# Kunlunxin: device_api.o (xshmem Device API kernel) references xshmem host symbols
# that live in libbkcl, and the XRE runtime libs. libflagcx.so imports (not
# exports) these, so the executable must link them explicitly. Add rpaths so the
# binary finds libbkcl / libxpurt* at runtime.
UNIT_RPATH :=
ifeq ($(USE_KUNLUNXIN), 1)
ALL_LINK += -L$(CCL_LIB) -Wl,-rpath,$(CCL_LIB) $(CCL_LINK) \
            -Wl,-rpath,$(DEVICE_LIB)
# The host-only lifecycle test links no kernel object, but it still loads
# libflagcx.so, which pulls in the XRE runtime at process start.
UNIT_RPATH := -Wl,-rpath,$(DEVICE_LIB) -Wl,-rpath,$(CCL_LIB)
endif

# Kernel flag
CXXFLAGS += -DCOMPILE_KERNEL

# Host-only lifecycle/provenance tests.
UNIT_SRC := test_dev_comm_cleanup.cpp
UNIT_OBJ := $(OBJDIR)/test_dev_comm_cleanup.o
UNIT_TARGET := $(BINDIR)/test_dev_comm_cleanup
# This host-only test must not select device CommTraits: KLX's SHMEM traits are
# compiled by xpu-clang, not g++. Pass only the allocator choice needed by the
# provenance assertions.
UNIT_CXXFLAGS := $(filter-out -DCOMPILE_KERNEL,$(CXXFLAGS))
UNIT_INCLUDES := $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) \
                 -I$(PROJECT_ROOT)/flagcx/adaptor/device_api

# Device API tests link test/kernel/<platform>/device_api.o. IR tests need a
# device_ir.<ext> kernel. KLX exercises the backend through the two unified IR
# suites; its legacy Device API launchers intentionally remain incomplete.
# NVIDIA/DU retain both regular and unified suites.
IR_TARGETS  := $(BINDIR)/test_device_ir_intra \
               $(BINDIR)/test_device_ir_inter \
               $(BINDIR)/test_device_ir_unified_intra \
               $(BINDIR)/test_device_ir_unified_inter
API_TARGETS := $(BINDIR)/test_device_api_intra \
               $(BINDIR)/test_device_api_inter

SKIPPED_TARGETS :=
ifeq ($(KERNEL_PLATFORM), klx)
  KLX_IR_TARGETS  := $(BINDIR)/test_device_ir_unified_intra \
                     $(BINDIR)/test_device_ir_unified_inter
  SKIPPED_TARGETS := $(notdir $(filter-out $(KLX_IR_TARGETS),\
                       $(IR_TARGETS) $(API_TARGETS)))
  IR_TARGETS      := $(KLX_IR_TARGETS)
  API_TARGETS     :=
  # S19 (DevGet) is compiled out of both unified suites: reading a peer's
  # memory traps on P800, so there is no implementation to exercise.
  # Cross-node data puts additionally require a full XPU cluster.  The unified
  # inter suite therefore excludes only its two THREAD put slots; WARP and
  # BLOCK remain required cases.
  CXXFLAGS += -DFLAGCX_TEST_NO_REMOTE_READ \
              -DFLAGCX_TEST_NO_REMOTE_THREAD_PUT
endif

MPI_TARGETS := $(IR_TARGETS) $(API_TARGETS)

.PHONY: all unit mpi clean run-unit run-mpi run-mpi-intra run-mpi-inter \
        run-mpi-unified-intra run-mpi-unified-inter run

all: unit mpi
ifneq ($(SKIPPED_TARGETS),)
	@echo "Skipped unsupported targets on $(KERNEL_PLATFORM): $(SKIPPED_TARGETS)"
endif

unit: $(UNIT_TARGET)

mpi: $(MPI_TARGETS)

# Build device kernel objects via dispatcher
KERNEL_API_TARGETS := device_api.o
KERNEL_IR_TARGETS  := device_ir.o
ifneq ($(DLINK_API_OBJ),)
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
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(DEVICE_LINK) $(UNIT_RPATH) $(GTEST_LINK)

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
              $(MPI_ENV_FLAG) FLAGCX_DEBUG=TRACE \
              $(MPI_ENV_FLAG) FLAGCX_DEBUG_SUBSYS=ALL \
              $(MPI_ENV_FLAG) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)

# Unified IR keeps P2P enabled. Inter-node runs create two logical nodes with
# MPI MPMD app contexts and distinct host IDs.
UNIFIED_COMMON_ENV := $(MPI_ENV_FLAG) FLAGCX_USE_HETERO_COMM=1 \
                      $(MPI_ENV_FLAG) FLAGCX_VMM_ENABLE=0 \
                      $(MPI_ENV_FLAG) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
INTER_NODE0_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node0 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node0
INTER_NODE1_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node1 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node1
UNIFIED_TEST_ARGS := -b 1K -e 16M -f 2

# Legacy intra-node tests. Platforms only run targets selected in MPI_TARGETS;
# this prevents a stale executable from bypassing a platform capability filter.
run-mpi-intra: $(filter $(BINDIR)/test_device_ir_intra,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_intra,$(MPI_TARGETS))

ifneq ($(filter $(BINDIR)/test_device_ir_intra,$(MPI_TARGETS)),)
	@echo "Running intra-node tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(BINDIR)/test_device_ir_intra
endif
ifneq ($(filter $(BINDIR)/test_device_api_intra,$(MPI_TARGETS)),)
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(BINDIR)/test_device_api_intra
endif
ifeq ($(filter $(BINDIR)/test_device_ir_intra $(BINDIR)/test_device_api_intra,$(MPI_TARGETS)),)
	@echo "Skipped legacy intra Device API/IR tests on $(KERNEL_PLATFORM)"
endif

# Inter-node tests use two logical four-rank nodes. This exercises inter-node
# discovery even on a single physical host.
run-mpi-inter: $(filter $(BINDIR)/test_device_ir_inter,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_inter,$(MPI_TARGETS))
ifneq ($(filter $(BINDIR)/test_device_ir_inter,$(MPI_TARGETS)),)
	@echo "Running inter-node Device IR tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(HETERO_ENV) $(INTER_NODE0_ENV) $(BINDIR)/test_device_ir_inter \
		: -n 4 $(HETERO_ENV) $(INTER_NODE1_ENV) $(BINDIR)/test_device_ir_inter
endif
ifneq ($(filter $(BINDIR)/test_device_api_inter,$(MPI_TARGETS)),)
	@echo "Running inter-node Device API tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(HETERO_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2 \
		: -n 4 $(HETERO_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2
endif
ifeq ($(filter $(BINDIR)/test_device_ir_inter $(BINDIR)/test_device_api_inter,$(MPI_TARGETS)),)
	@echo "Skipped legacy inter Device API/IR tests on $(KERNEL_PLATFORM)"
endif

ifneq ($(filter $(BINDIR)/test_device_ir_unified_intra,$(MPI_TARGETS)),)
run-mpi-unified-intra: $(BINDIR)/test_device_ir_unified_intra
	@echo "Running unified intra suite tests (INTRA + WORLD, full coop matrices)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n 8 $(UNIFIED_COMMON_ENV) \
		$(BINDIR)/test_device_ir_unified_intra $(UNIFIED_TEST_ARGS)
else
run-mpi-unified-intra:
	@echo "Skipped unified intra IR test on $(KERNEL_PLATFORM)"
endif

ifneq ($(filter $(BINDIR)/test_device_ir_unified_inter,$(MPI_TARGETS)),)
run-mpi-unified-inter: $(BINDIR)/test_device_ir_unified_inter
	@echo "Running unified inter suite tests (INTER + WORLD, full coop matrices)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n 4 $(UNIFIED_COMMON_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS) \
		: -n 4 $(UNIFIED_COMMON_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS)
else
run-mpi-unified-inter:
	@echo "Skipped unified inter IR test on $(KERNEL_PLATFORM)"
endif

run-mpi: run-mpi-intra run-mpi-inter run-mpi-unified-intra \
         run-mpi-unified-inter

run: run-unit run-mpi
