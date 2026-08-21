include ../../make.inc

# Local include directory for RMA test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# [mpi] tests: coll_*.cpp + rma_test.cpp + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) rma_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/rma_mpi_tests

# Environment variables to force hetero mode on single node
HETERO_ENV := -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0

.PHONY: all clean run-unit run-mpi run

ALL_TARGETS :=
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif

all: $(ALL_TARGETS)

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) -L$(GTEST_LIB) -lgtest -lpthread $(MPI_LINK)

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

run-unit:
	@echo "No non-MPI unit tests for rma."

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 2 $(HETERO_ENV) $(MPI_TARGET)

run: run-unit run-mpi
