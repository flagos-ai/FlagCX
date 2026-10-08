include ../../make.inc

# Engine READ is not used by the current P2P workload. Keep its source for
# future work, but exclude READ semantics from the CI binary.
UNIT_SRCS   := $(filter-out test_p2p_engine_read.cpp test_p2p_gpu_read.cpp test_p2p_visibility_policy.cpp test_p2p_batch.cpp,$(wildcard test_*.cpp))
# IB_P2P remains the default Engine's private net adaptor. Keep its original
# direct adaptor tests in the legacy build; the shared build does not link it.
ifeq ($(USE_SHARED_P2P_ENGINE),1)
UNIT_SRCS := $(filter-out test_p2p_adaptor.cpp,$(UNIT_SRCS))
endif
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_SRCS    := coll_p2p_engine_write.cpp
MPI_OBJS    := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
GTEST_SRC_DIR := $(GTEST_SRC_ROOT)/src
GTEST_OBJS := $(OBJDIR)/gtest/gtest-all.o $(OBJDIR)/gtest/gtest_main.o
UNIT_TARGET := $(BINDIR)/p2p_unit_tests
MPI_TARGET  := $(BINDIR)/p2p_mpi_tests
MPIRUN      ?= mpirun

.PHONY: all clean run-unit run-mpi run

ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ALL_TARGETS += $(MPI_TARGET)

all: $(ALL_TARGETS)

$(UNIT_TARGET): $(UNIT_OBJS) $(GTEST_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) -lpthread

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(MPI_LINK) -lpthread

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(MPI_INCLUDE)

$(OBJDIR)/gtest/%.o: $(GTEST_SRC_DIR)/%.cc
	@mkdir -p $(dir $@)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(GTEST_INCLUDE) -I$(GTEST_SRC_ROOT)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np 2 \
		-x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 \
		-x FLAGCX_VMM_ENABLE=0 -x LD_LIBRARY_PATH $(MPI_ENV) $(MPI_TARGET)

run: run-unit run-mpi
