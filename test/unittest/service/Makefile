include ../../make.inc

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/service_unit_tests

# IBUC is optional at the project level, so normal CUDA/MetaX/Hygon builds do
# not preprocess its implementation. Keep a syntax-only target in the common
# service suite to prevent optional-adaptor compile regressions.
IBUC_SOURCE := $(PROJECT_ROOT)/flagcx/adaptor/net/ibuc_adaptor.cc

# [mpi] tests: coll_*.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp)
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/service_mpi_tests

.PHONY: all clean compile-ibuc run-unit run-mpi run

ALL_TARGETS :=
ALL_TARGETS += compile-ibuc
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif

all: $(ALL_TARGETS)

compile-ibuc:
	@echo "Checking  $(IBUC_SOURCE) with USE_IBUC=1"
	@$(CXX) $(IBUC_SOURCE) -fsyntax-only -DUSE_IBUC $(CXXFLAGS) $(FLAGCX_INCLUDES)

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
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 8 $(MPI_TARGET)

run: run-unit run-mpi
