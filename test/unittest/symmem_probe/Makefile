include ../../make.inc

INCLUDES := $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(MPI_INCLUDE) -I$(abspath ../adaptor)
UNIT_TARGET := $(BINDIR)/symmem_probe_unit_tests
MPI_TARGET := $(BINDIR)/symmem_probe_mpi_tests
UNIT_OBJS := $(OBJDIR)/test_prerequisites.o $(OBJDIR)/test_vmm.o $(OBJDIR)/test_net.o
MPI_OBJS := $(OBJDIR)/coll_probe.o $(OBJDIR)/main_mpi.o

.PHONY: all clean
all: $(UNIT_TARGET) $(MPI_TARGET)
$(UNIT_TARGET): $(UNIT_OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK)
$(MPI_TARGET): $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_BASE_LINK) $(MPI_LINK)
$(OBJDIR)/%.o: %.cpp probe.hpp
	@mkdir -p $(OBJDIR)
	$(CXX) $< -o $@ -c $(CXXFLAGS) $(INCLUDES)
clean:
	@rm -rf $(BUILDDIR)
