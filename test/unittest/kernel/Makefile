include ../../make.inc

# Local include directory for kernel test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# [mpi] tests: coll_*.cpp + main_mpi.cpp + flagcx_kernel_test.cpp
MPI_SRCS   := $(wildcard coll_*.cpp) flagcx_kernel_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/kernel_mpi_tests

.PHONY: all clean run-unit run-mpi run

all: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit:
	@echo "No unit tests in kernel/"

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 8 $(MPI_TARGET)

run: run-mpi
