include ../../make.inc

ifeq ($(USE_PPU), 1)
# Compile the PCCL/BAREX interoperability case with the native PPU types.
CXXFLAGS += $(HOST_ADAPTOR_DEFINE)
FLAGCX_INCLUDES += $(DEVICE_INCLUDE) $(CCL_INCLUDE)
endif

# [unit] tests: test_*.cpp — single GPU, no MPI required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/adaptor_unit_tests

# [mpi] tests: coll_*.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp)
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/adaptor_mpi_tests
MPIRUN     ?= mpirun
MPI_NP     ?= 8
MPI_ENV    ?=
MPI_XML_DIR ?=
MPI_TEST_CMD = $(MPI_TARGET)
ifneq ($(strip $(MPI_XML_DIR)),)
MPI_TEST_CMD = bash $(PROJECT_ROOT)/.github/scripts/ci/run_gtest_rank_xml.sh $(MPI_XML_DIR) $(MPI_TARGET)
endif

.PHONY: all clean run-unit run-mpi run

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
	@$(MPIRUN) --allow-run-as-root -np $(MPI_NP) $(MPI_ENV) $(MPI_TEST_CMD)

run: run-unit run-mpi
