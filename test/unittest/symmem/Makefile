include ../../make.inc

# Local include directory for symmem test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/symmem_unit_tests

# [mpi] tests: coll_*.cpp + symmem_test.cpp + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) symmem_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/symmem_mpi_tests

# Environment variables to force hetero mode on single node
HETERO_ENV := FLAGCX_USE_HETERO_COMM=1 FLAGCX_CLUSTER_SPLIT_LIST=2 FLAGCX_MEM_ENABLE=1

.PHONY: all clean run-unit run-mpi run

# Only build targets that have sources
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
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK)

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/symmem_test.o: symmem_test.cpp
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

run-mpi: $(MPI_TARGET)
	@$(HETERO_ENV) mpirun --allow-run-as-root -np 4 $(MPI_TARGET)

run: run-unit run-mpi
