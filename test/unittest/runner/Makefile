include ../../make.inc

# Local include directory for runner fixtures
LOCAL_INCLUDE := -I$(abspath include)

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/runner_unit_tests

# [mpi] tests: keep UniRunner-specific GPU cases in the experimental suite.
UNIRUNNER_EXPERIMENTAL_SRCS := coll_allgather_gemm.cpp coll_gemm_reducescatter.cpp \
                           coll_uni_runner_group.cpp
SENDRECV_PROBE_SRC := coll_sendrecv_probe.cpp
MPI_SRCS   := $(filter-out $(UNIRUNNER_EXPERIMENTAL_SRCS) $(SENDRECV_PROBE_SRC),$(wildcard coll_*.cpp)) main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/runner_mpi_tests
SENDRECV_PROBE_OBJS := $(OBJDIR)/coll_sendrecv_probe.o $(OBJDIR)/main_mpi.o
SENDRECV_PROBE_TARGET := $(BINDIR)/runner_sendrecv_probe_mpi_tests
UNIRUNNER_EXPERIMENTAL_MPI_SRCS := $(UNIRUNNER_EXPERIMENTAL_SRCS) main_mpi.cpp
UNIRUNNER_EXPERIMENTAL_MPI_OBJS := $(UNIRUNNER_EXPERIMENTAL_MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIRUNNER_EXPERIMENTAL_MPI_TARGET := $(BINDIR)/runner_unirunner_experimental_mpi_tests

.PHONY: all clean run-unit run-mpi run unirunner-experimental sendrecv-probe

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

$(SENDRECV_PROBE_TARGET): $(SENDRECV_PROBE_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK)

$(UNIRUNNER_EXPERIMENTAL_MPI_TARGET): $(UNIRUNNER_EXPERIMENTAL_MPI_OBJS)
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
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/main_mpi.o: main_mpi.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 8 $(MPI_TARGET)

unirunner-experimental: $(UNIRUNNER_EXPERIMENTAL_MPI_TARGET)

sendrecv-probe: $(SENDRECV_PROBE_TARGET)

run: run-unit run-mpi
