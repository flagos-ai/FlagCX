include ../../make.inc

# The host-only unit target must use the host compiler even when the Device
# API MPI targets use HIPCC for their device objects.
HOST_CXX := $(CXX)
ifeq ($(USE_DU), 1)
CXX := $(DEVICE_COMPILER)
CXXFLAGS += -fgpu-rdc
endif

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
# same USE_SHMEM switch, but its device code is linked into device_api.o; on
# Ascend bisheng embeds the device image inside -c (makefiles/ascend.mk:179),
# so there is no device-link product either.
ifneq ($(filter 1klx 1ascend,$(USE_SHMEM)$(KERNEL_PLATFORM)),)
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
    $(KERNEL_INCLUDE) $(PERF_INCLUDE) $(MPI_INCLUDE) $(DEVICE_INCLUDE) $(CCL_INCLUDE)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)

# DU kernel objects contain absolute host relocations. Link their test
# executables without PIE so the host linker accepts those relocations.
ifeq ($(USE_DU), 1)
ALL_LINK += -no-pie
endif

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
ifeq ($(COMPILE_KERNEL),1)
CXXFLAGS += -DCOMPILE_KERNEL
endif

# Host-only lifecycle/provenance tests.
UNIT_SRC := test_dev_comm_cleanup.cpp test_comm_stream_isolation.cpp
UNIT_OBJ := $(UNIT_SRC:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/test_dev_comm_cleanup

# Provider dispatch is host-testable without a GPU or a FlagCX library link.
PROVIDER_TEST_TARGET := $(BINDIR)/test_default_net_provider
.PHONY: provider_test
$(PROVIDER_TEST_TARGET): test_default_net_provider.cpp
	@mkdir -p $(BINDIR)
	@$(HOST_CXX) -std=c++11 -o $@ $< $(FLAGCX_INCLUDES)

provider_test: $(PROVIDER_TEST_TARGET)
	@$(PROVIDER_TEST_TARGET)

# Only NVIDIA's host suite runs the NVIDIA Provider test automatically.
PROVIDER_UNIT_TARGET :=
ifeq ($(KERNEL_PLATFORM),nvidia)
PROVIDER_UNIT_TARGET := $(PROVIDER_TEST_TARGET)
endif

# This host-only test must not select device CommTraits: KLX's SHMEM traits are
# compiled by xpu-clang, not g++. Pass only the allocator choice needed by the
# provenance assertions.
UNIT_CXXFLAGS := $(filter-out -DCOMPILE_KERNEL -fgpu-rdc,$(CXXFLAGS))
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
ALL_DEVICE_TEST_TARGETS := $(IR_TARGETS) $(API_TARGETS)

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

# Enflame: use platform-specific test files with dual-buffer pattern
ifeq ($(KERNEL_PLATFORM), enflame)
  ENFLAME_IR_TARGETS  := $(BINDIR)/test_device_ir_intra_enflame \
                         $(BINDIR)/test_device_ir_inter_enflame
  ENFLAME_API_TARGETS := $(BINDIR)/test_device_api_intra_enflame \
                         $(BINDIR)/test_device_api_inter_enflame
  SKIPPED_TARGETS := $(notdir $(filter-out $(ENFLAME_IR_TARGETS) $(ENFLAME_API_TARGETS),\
                       $(IR_TARGETS) $(API_TARGETS)))
  IR_TARGETS      := $(ENFLAME_IR_TARGETS)
  API_TARGETS     := $(ENFLAME_API_TARGETS)
  ALL_DEVICE_TEST_TARGETS += $(ENFLAME_IR_TARGETS) $(ENFLAME_API_TARGETS)
  # Skip host-only test: Enflame doesn't implement CommTraits/DefaultCompletionStorage
  UNIT_TARGET :=
endif

# Ascend/CANN.  Two suites build.  test_device_api_intra keeps its launchers
# in test/kernel/ascend/device_api.asc and the switches that compile out the
# cases the selected backend cannot satisfy (peer memory and coop groups on
# both; the intra barrier only on the IPC build, since cann-shmem has the
# transport for it).  test_device_ir_unified_intra is the Device IR suite: its
# launchers live in test/kernel/ascend/device_ir.asc (S16-S25; the net half
# only under USE_SHMEM=1), which is
# build-gated by the in--c device link - an undefined trap reference fails
# the object, not the test.  The two leg-across suites (test_device_api_inter
# and test_device_ir_unified_inter) build only under USE_SHMEM=1, where the
# net transport exists; the IPC build declares their launchers absent.
# NO_REMOTE_READ compiles the S19 call sites out in the IPC build (read-back
# of peer GM is not a measured path there; the reference file itself states
# the same rule for P800); the shmem build keeps S19, whose transport is
# cann-shmem's MTE get.  NO_ATOMIC_COMPLETION compiles S20-S25 out of the
# IPC build: 2201 exposes no global-memory RMW that survives this pass (u32
# fetchAdd is refused even under a target-feature spelling, and the
# plain-u64 loads/stores bound first are the dcci pair, not an add), so the
# completion-based cases have no implementation to exercise - and their
# launchers remain declared-but-not-defined, which turns any resurrected
# call site into a link failure instead of a reportable pass.  The shmem
# backend needs no RMW (per-source tickets + a single-writer local counter;
# aclshmem_comm_traits.h), so S20-S25 are graded there.  Each switch names
# its reason at the site it guards; the point is that a skipped case never
# reaches the report as a pass.
ifeq ($(KERNEL_PLATFORM), ascend)
  ASCEND_IR_TARGETS  := $(BINDIR)/test_device_ir_unified_intra
  ASCEND_API_TARGETS := $(BINDIR)/test_device_api_intra
  # The leg-across suites build only on the cann-shmem backend: they need the
  # net transport (aclshmem MTE puts + signal tickets) that the IPC build
  # declares absent.  Must be added before SKIPPED_TARGETS is computed.
  ifeq ($(USE_SHMEM),1)
    ASCEND_API_TARGETS += $(BINDIR)/test_device_api_inter
    ASCEND_IR_TARGETS  += $(BINDIR)/test_device_ir_unified_inter
  endif
  # The skip list is the original universe minus everything this platform
  # actually builds.  Ascend keeps ONE IR suite (unified_intra) AND the
  # legacy api_intra suite, so filter out both - not just the IR target -
  # or the report falsely names api_intra as skipped while `mpi` builds it.
  SKIPPED_TARGETS := $(notdir $(filter-out $(ASCEND_IR_TARGETS) $(ASCEND_API_TARGETS),\
                       $(IR_TARGETS) $(API_TARGETS)))
  IR_TARGETS      := $(ASCEND_IR_TARGETS)
  API_TARGETS     := $(ASCEND_API_TARGETS)
  CXXFLAGS += -DFLAGCX_TEST_NO_COOP_GROUPS -DFLAGCX_TEST_NO_PEER_POINTER \
              -DFLAGCX_ASCEND_PUTVALUE_LINE_STRIDE
  # NO_CROSS_RANK_COMM skips S16-S18 outright: the measured peer-store
  # fault freeze (ascend.sh) makes cross-rank rendezvous inoperable, and
  # at world size 1 those cases degenerate to zero-iteration peer loops /
  # self copies whose PASS would be a fake certification (MC952 review
  # P1).  PUTVALUE_LINE_STRIDE keeps the S18 oracle aligned with the
  # 910C kernel's 128-byte line-isolated slots (the stride the shmem
  # kernel writes; every other platform keeps the packed 8-byte default).
  ifeq ($(USE_SHMEM),1)
    # cann-shmem supplies the cross-rank transport (MTE puts, MTE gets and
    # local polling) that the peer-VA path cannot: S16-S25 are runnable,
    # and so are K8/K9/K10 (device_api.asc: the INTRA barrier plus peer
    # reads through aclshmem_ptr, both proven on this backend).
    # Everything else this platform lacks is unchanged.
    # PUTVALUE_TAIL adds S18 sub-block C: 1/2-byte putValue destinations
    # whose last byte is the last byte of their registered allocation
    # (MC952 review P2).
    CXXFLAGS += -DFLAGCX_ASCEND_PUTVALUE_TAIL
  else
    CXXFLAGS += -DFLAGCX_TEST_NO_CROSS_RANK_COMM \
                -DFLAGCX_TEST_NO_REMOTE_READ \
                -DFLAGCX_TEST_NO_ATOMIC_COMPLETION \
                -DFLAGCX_TEST_NO_INTRA_BARRIER
  endif
endif

MPI_TARGETS := $(IR_TARGETS) $(API_TARGETS)

.PHONY: all unit mpi clean require-compile-kernel run-unit run-mpi run-mpi-intra run-mpi-inter \
        run-mpi-unified-intra run-mpi-unified-inter run

all: require-compile-kernel unit mpi

unit: $(UNIT_TARGET) $(PROVIDER_UNIT_TARGET)

require-compile-kernel:
	$(REQUIRE_COMPILE_KERNEL)

mpi: require-compile-kernel $(MPI_TARGETS)

$(ALL_DEVICE_TEST_TARGETS) $(DEVICE_API_OBJ) $(DEVICE_IR_OBJ) \
    $(DLINK_API_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ): | require-compile-kernel

run-mpi run-mpi-intra run-mpi-inter run-mpi-unified-intra \
    run-mpi-unified-inter: | require-compile-kernel

KERNEL_API_SRCS := $(KERNEL_DIR)/$(KERNEL_PLATFORM)/device_api.$(DEVICE_FILE_EXTENSION)
KERNEL_IR_SRCS  := $(KERNEL_DIR)/$(KERNEL_PLATFORM)/device_ir.$(DEVICE_FILE_EXTENSION)

KERNEL_API_TARGETS := device_api.o
KERNEL_IR_TARGETS  := device_ir.o
ifneq ($(DLINK_API_OBJ),)
KERNEL_API_TARGETS += dlink_api.o
KERNEL_IR_TARGETS  += dlink_ir.o
endif

$(DEVICE_API_OBJ) $(DLINK_API_OBJ): $(KERNEL_API_SRCS)
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_API_TARGETS)

$(DEVICE_IR_OBJ) $(DLINK_IR_OBJ): $(KERNEL_IR_SRCS)
	@$(MAKE) -C $(KERNEL_DIR) $(KERNEL_IR_TARGETS)

$(TOOLS_OBJ): $(TOOLS_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Compiling tools.cc"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(HOST_ADAPTOR_DEFINE) $(ALL_INCLUDES)

$(UNIT_OBJ): $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(HOST_CXX) $< -o $@ -c $(UNIT_CXXFLAGS) $(UNIT_INCLUDES)

$(UNIT_TARGET): $(UNIT_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(HOST_CXX) $^ -o $@ $(FLAGCX_LINK) $(DEVICE_LINK) $(UNIT_RPATH) $(GTEST_LINK)

# HIPCC treats object files as HIP source when a .cpp input is present in the
# same link invocation. Compile each test entry point first so links receive
# object files only. This also keeps the host sources out of device linking.
$(OBJDIR)/test_device_%.o: test_device_%.cpp | require-compile-kernel
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) $(HOST_ADAPTOR_DEFINE) -c -o $@ $< $(ALL_INCLUDES)

$(BINDIR)/test_device_ir_intra: $(OBJDIR)/test_device_ir_intra.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_ir_inter: $(OBJDIR)/test_device_ir_inter.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_ir_unified_intra: $(OBJDIR)/test_device_ir_unified_intra.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_ir_unified_inter: $(OBJDIR)/test_device_ir_unified_inter.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_api_intra: $(OBJDIR)/test_device_api_intra.o $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_api_inter: $(OBJDIR)/test_device_api_inter.o $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

# Enflame-specific test targets (dual-buffer pattern)
$(OBJDIR)/test_device_api_intra_enflame.o: enflame/test_device_api_intra.cpp | require-compile-kernel
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c -o $@ $< $(ALL_INCLUDES)

$(OBJDIR)/test_device_ir_intra_enflame.o: enflame/test_device_ir_intra.cpp | require-compile-kernel
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c -o $@ $< $(ALL_INCLUDES)

$(BINDIR)/test_device_api_intra_enflame: $(OBJDIR)/test_device_api_intra_enflame.o $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_ir_intra_enflame: $(OBJDIR)/test_device_ir_intra_enflame.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(OBJDIR)/test_device_api_inter_enflame.o: enflame/test_device_api_inter.cpp | require-compile-kernel
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c -o $@ $< $(ALL_INCLUDES)

$(OBJDIR)/test_device_ir_inter_enflame.o: enflame/test_device_ir_inter.cpp | require-compile-kernel
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c -o $@ $< $(ALL_INCLUDES)

$(BINDIR)/test_device_api_inter_enflame: $(OBJDIR)/test_device_api_inter_enflame.o $(DEVICE_API_OBJ) $(DLINK_API_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

$(BINDIR)/test_device_ir_inter_enflame: $(OBJDIR)/test_device_ir_inter_enflame.o $(DEVICE_IR_OBJ) $(DLINK_IR_OBJ) $(TOOLS_OBJ)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)

run-unit: unit
	@GTEST_DEATH_TEST_STYLE=threadsafe $(UNIT_TARGET)
	@$(DEVICEPTR_ARGS_TARGET)
ifneq ($(PROVIDER_UNIT_TARGET),)
	@$(PROVIDER_UNIT_TARGET)
endif

# Use the same backend selection as the library and device kernels. Default
# tests retain their IPC configuration; vendor tests require homo/SYMMETRIC.
ifneq ($(filter -DFLAGCX_COMM_TRAITS_CCL -DFLAGCX_COMM_TRAITS_SHMEM,$(ADAPTOR_DEFINE)),)
DEVICE_TEST_PATH_ENV := $(MPI_ENV_FLAG) FLAGCX_USE_HETERO_COMM=0
FLAGCX_TEST_WIN_MODE ?= symmetric
FLAGCX_VMM_ENABLE ?= 1
else
DEVICE_TEST_PATH_ENV := $(MPI_ENV_FLAG) FLAGCX_USE_HETERO_COMM=1
FLAGCX_TEST_WIN_MODE ?= asymmetric
FLAGCX_VMM_ENABLE ?= 0
endif
DEVICE_TEST_ALLOC_ENV := $(MPI_ENV_FLAG) FLAGCX_VMM_ENABLE=$(FLAGCX_VMM_ENABLE)

DEVICE_TEST_ENV := $(DEVICE_TEST_PATH_ENV) $(DEVICE_TEST_ALLOC_ENV) \
                   $(MPI_ENV_FLAG) FLAGCX_TEST_WIN_MODE=$(FLAGCX_TEST_WIN_MODE) \
                   $(MPI_ENV_FLAG) FLAGCX_MEM_ENABLE=1 \
                   $(MPI_ENV_FLAG) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
INTRA_ENV := $(DEVICE_TEST_ENV) $(MPI_ENV_FLAG) FLAGCX_P2P_DISABLE=0
INTER_ENV := $(DEVICE_TEST_ENV) \
             $(MPI_ENV_FLAG) FLAGCX_P2P_DISABLE=1 \
             $(MPI_ENV_FLAG) FLAGCX_DEBUG=TRACE \
             $(MPI_ENV_FLAG) FLAGCX_DEBUG_SUBSYS=ALL

# Unified IR keeps P2P enabled. Inter-node runs create two logical nodes with
# MPI MPMD app contexts and distinct host IDs.
UNIFIED_COMMON_ENV := $(DEVICE_TEST_ENV) $(MPI_ENV_FLAG) FLAGCX_P2P_DISABLE=0
INTER_NODE0_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node0 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node0
INTER_NODE1_ENV := $(MPI_ENV_FLAG) FLAGCX_HOSTID=node1 \
                   $(MPI_ENV_FLAG) NCCL_HOSTID=node1
UNIFIED_TEST_ARGS := -b 1K -e 16M -f 2
# Intra-node rank count (default 4).
INTRA_NP ?= 4
# Unified-intra world size.  Platforms whose device-side cross-rank
# rendezvous is not a supported hardware capability narrow this to 1
# (see set_env/ascend.sh); everywhere else keeps the historical -n 4.
UNIFIED_IR_NP ?= 4
# Unified-inter ranks per logical node (the asc 2x4 leg overrides to 4).
UNIFIED_IR_INTER_NP ?= 2

# Legacy intra-node tests. Platforms only run targets selected in MPI_TARGETS;
# this prevents a stale executable from bypassing a platform capability filter.
run-mpi-intra: $(filter $(BINDIR)/test_device_ir_intra,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_intra,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_ir_intra_enflame,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_intra_enflame,$(MPI_TARGETS))

ifneq ($(filter $(BINDIR)/test_device_ir_intra,$(MPI_TARGETS)),)
	@echo "Running intra-node tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n $(INTRA_NP) $(INTRA_ENV) $(BINDIR)/test_device_ir_intra
endif
ifneq ($(filter $(BINDIR)/test_device_api_intra,$(MPI_TARGETS)),)
	$(MPIRUN) $(MPI_ROOT_ARGS) -n $(INTRA_NP) $(INTRA_ENV) $(BINDIR)/test_device_api_intra
endif
ifneq ($(filter $(BINDIR)/test_device_ir_intra_enflame,$(MPI_TARGETS)),)
	@echo "Running Enflame intra-node IR tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n $(INTRA_NP) $(BINDIR)/test_device_ir_intra_enflame
endif
ifneq ($(filter $(BINDIR)/test_device_api_intra_enflame,$(MPI_TARGETS)),)
	@echo "Running Enflame intra-node API tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n $(INTRA_NP) $(BINDIR)/test_device_api_intra_enflame
endif
ifeq ($(filter $(BINDIR)/test_device_ir_intra $(BINDIR)/test_device_api_intra $(BINDIR)/test_device_ir_intra_enflame $(BINDIR)/test_device_api_intra_enflame,$(MPI_TARGETS)),)
	@echo "Skipped legacy intra Device API/IR tests on $(KERNEL_PLATFORM)"
endif

# Inter-node tests use two logical nodes of INTER_NP ranks each (default 2).
# This exercises inter-node discovery even on a single physical host; the
# Ascend shmem run overrides it to 4 (-n 4 : -n 4, the delivery's 2x4).
INTER_NP ?= 2

run-mpi-inter: $(filter $(BINDIR)/test_device_ir_inter,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_inter,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_ir_inter_enflame,$(MPI_TARGETS)) \
               $(filter $(BINDIR)/test_device_api_inter_enflame,$(MPI_TARGETS))
ifneq ($(filter $(BINDIR)/test_device_ir_inter,$(MPI_TARGETS)),)
	@echo "Running inter-node Device IR tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n $(INTER_NP) $(INTER_ENV) $(INTER_NODE0_ENV) $(BINDIR)/test_device_ir_inter \
		: -n $(INTER_NP) $(INTER_ENV) $(INTER_NODE1_ENV) $(BINDIR)/test_device_ir_inter
endif
ifneq ($(filter $(BINDIR)/test_device_api_inter,$(MPI_TARGETS)),)
	@echo "Running inter-node Device API tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n $(INTER_NP) $(INTER_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2 \
		: -n $(INTER_NP) $(INTER_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_api_inter -b 1M -e 4M -f 2 -R 2
endif
ifneq ($(filter $(BINDIR)/test_device_ir_inter_enflame,$(MPI_TARGETS)),)
	@echo "Running Enflame inter-node IR tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n $(INTER_NP) $(INTER_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_ir_inter_enflame -b 1M -e 4M -f 2 -R 2 \
		: -n $(INTER_NP) $(INTER_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_ir_inter_enflame -b 1M -e 4M -f 2 -R 2
endif
ifneq ($(filter $(BINDIR)/test_device_api_inter_enflame,$(MPI_TARGETS)),)
	@echo "Running Enflame inter-node API tests..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n $(INTER_NP) $(INTER_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_api_inter_enflame -b 1M -e 4M -f 2 -R 2 \
		: -n $(INTER_NP) $(INTER_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_api_inter_enflame -b 1M -e 4M -f 2 -R 2
endif
ifeq ($(filter $(BINDIR)/test_device_ir_inter $(BINDIR)/test_device_api_inter $(BINDIR)/test_device_ir_inter_enflame $(BINDIR)/test_device_api_inter_enflame,$(MPI_TARGETS)),)
	@echo "Skipped legacy inter Device API/IR tests on $(KERNEL_PLATFORM)"
endif

ifneq ($(filter $(BINDIR)/test_device_ir_unified_intra,$(MPI_TARGETS)),)
run-mpi-unified-intra: $(BINDIR)/test_device_ir_unified_intra
	@echo "Running unified intra suite tests (INTRA + WORLD, full coop matrices)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) -n $(UNIFIED_IR_NP) $(UNIFIED_COMMON_ENV) \
		$(BINDIR)/test_device_ir_unified_intra $(UNIFIED_TEST_ARGS)
else
run-mpi-unified-intra:
	@echo "Skipped unified intra IR test on $(KERNEL_PLATFORM)"
endif

ifneq ($(filter $(BINDIR)/test_device_ir_unified_inter,$(MPI_TARGETS)),)
run-mpi-unified-inter: $(BINDIR)/test_device_ir_unified_inter
	@echo "Running unified inter suite tests (INTER + WORLD, full coop matrices)..."
	$(MPIRUN) $(MPI_ROOT_ARGS) \
		-n $(UNIFIED_IR_INTER_NP) $(UNIFIED_COMMON_ENV) $(INTER_NODE0_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS) \
		: -n $(UNIFIED_IR_INTER_NP) $(UNIFIED_COMMON_ENV) $(INTER_NODE1_ENV) \
		$(BINDIR)/test_device_ir_unified_inter $(UNIFIED_TEST_ARGS)
else
run-mpi-unified-inter:
	@echo "Skipped unified inter IR test on $(KERNEL_PLATFORM)"
endif

run-mpi: run-mpi-intra run-mpi-inter run-mpi-unified-intra \
         run-mpi-unified-inter

run: require-compile-kernel run-unit run-mpi
