include ../../make.inc

MPIRUN ?= mpirun

# Local include directory for RMA test fixtures
LOCAL_INCLUDE := -I$(abspath include) \
	-I$(PROJECT_ROOT)/flagcx/adaptor/device_api \
	$(KERNEL_INCLUDE)

# [unit] tests: test_*.cpp — no MPI or accelerator execution required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/rma_unit_tests

# [mpi] tests: coll_*.cpp + rma_test.cpp + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) rma_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/rma_mpi_tests

# CUDA, MetaX, Hygon/DU, and PPU run the same immediate device-consumer
# visibility test through a tiny platform-compiled kernel object.
VISIBILITY_PLATFORM := $(filter 1,$(USE_NVIDIA) $(USE_METAX) $(USE_DU) $(USE_PPU))
ifneq ($(strip $(VISIBILITY_PLATFORM)),)
CXXFLAGS += -DFLAGCX_TEST_GDR_VISIBILITY_CONSUMER
GDR_VISIBILITY_BUILD_DIR := $(abspath $(BUILDDIR))/kernel/$(KERNEL_PLATFORM)
GDR_VISIBILITY_OBJ := $(GDR_VISIBILITY_BUILD_DIR)/obj/gdr_visibility.o

gdr-visibility-obj:
	@$(MAKE) -C $(KERNEL_DIR) gdr_visibility.o \
		BUILDDIR=$(GDR_VISIBILITY_BUILD_DIR)
endif

# Common environment for single-node heterogeneous RMA tests. Keep transport
# selection out of this list so CI can exercise IPC and RDMA independently.
RMA_PLATFORM_ENV ?=
RMA_VMM_ENABLE ?= 0
NET_FILTER ?= *
HETERO_ENV := -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=$(RMA_VMM_ENABLE) $(RMA_PLATFORM_ENV)
IPC_ENV := $(HETERO_ENV) -x FLAGCX_IB_DISABLE=1 -x FLAGCX_P2P_DISABLE=0
NET_ENV := $(HETERO_ENV) -x FLAGCX_IB_DISABLE=0 -x FLAGCX_P2P_DISABLE=1
IPC_FILTER := RmaTest.IpcResolvedPeerPointerSupportsDirectCopy:RmaTest.IpcPutWithoutNetworkMr:RmaTest.PutSignalSmall:RmaTest.PutSignalLarge:RmaTest.SignalOnlyNoData:RmaTest.MultipleSignals

.PHONY: all clean run-unit run-mpi-ipc run-mpi-net run-mpi run \
	gdr-visibility-obj

ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif

all: $(ALL_TARGETS)

$(UNIT_TARGET): $(UNIT_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK)

$(MPI_TARGET): gdr-visibility-obj $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(MPI_OBJS) $(GDR_VISIBILITY_OBJ) -o $@ $(FLAGCX_LINK) -L$(GTEST_LIB) -lgtest -lpthread $(MPI_LINK) $(DEVICE_LINK)

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/rma_test.o: rma_test.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/main_mpi.o: main_mpi.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi-ipc: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np 2 $(IPC_ENV) $(MPI_TARGET) --gtest_filter=$(IPC_FILTER)

run-mpi-net: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np 2 $(NET_ENV) -x GTEST_FILTER=$(NET_FILTER) $(MPI_TARGET)

run-mpi: run-mpi-ipc run-mpi-net

run: run-unit run-mpi
