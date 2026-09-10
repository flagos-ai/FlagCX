include ../../make.inc

MPIRUN ?= mpirun

# Local include directory for RMA test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# [unit] tests: test_*.cpp — no MPI or accelerator execution required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/rma_unit_tests

# [mpi] tests: coll_*.cpp + rma_test.cpp + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) rma_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/rma_mpi_tests

# Common environment for single-node heterogeneous RMA tests. Keep transport
# selection out of this list so CI can exercise IPC and RDMA independently.
RMA_PLATFORM_ENV ?=
HETERO_ENV := -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 $(RMA_PLATFORM_ENV)
IPC_ENV := $(HETERO_ENV) -x FLAGCX_RMA_FORCE_NET=0 -x FLAGCX_RMA_TEST_REQUIRE_IPC=1
NET_ENV := $(HETERO_ENV) -x FLAGCX_RMA_FORCE_NET=1 -x FLAGCX_RMA_TEST_REQUIRE_IPC=0
IPC_FILTER := RmaTest.PutSignalSmall:RmaTest.PutSignalLarge:RmaTest.SignalOnlyNoData:RmaTest.MultipleSignals

.PHONY: all clean run-unit run-mpi-ipc run-mpi-net run-mpi run

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

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) -L$(GTEST_LIB) -lgtest -lpthread $(MPI_LINK)

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
	@$(MPIRUN) --allow-run-as-root -np 2 $(NET_ENV) $(MPI_TARGET)

run-mpi: run-mpi-ipc run-mpi-net

run: run-unit run-mpi
