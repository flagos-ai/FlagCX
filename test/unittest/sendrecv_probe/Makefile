include ../../make.inc

# Reuse the runner fixture and MPI main while keeping the probe in its own
# build directory and CI suite.
LOCAL_INCLUDE := -I$(abspath ../runner/include)
TARGET := $(BINDIR)/sendrecv_probe_mpi_tests
OBJS := $(OBJDIR)/coll_sendrecv_probe.o $(OBJDIR)/main_mpi.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK)

$(OBJDIR)/coll_sendrecv_probe.o: coll_sendrecv_probe.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

$(OBJDIR)/main_mpi.o: ../runner/main_mpi.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)
