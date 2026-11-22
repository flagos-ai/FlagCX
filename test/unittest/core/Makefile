include ../../make.inc
.DEFAULT_GOAL := all

# Local include directory for core test fixtures
LOCAL_INCLUDE := -I$(abspath include) \
    -I$(PROJECT_ROOT)/flagcx/adaptor/device_api

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/core_unit_tests
UNIT_DEPS   := $(UNIT_OBJS:.o=.d)

# Lifecycle SDK shims and their tests live together in the adaptor suite.
# Compile only the outer-Comm integration cases into the core executable.
UNIT_OBJS += $(OBJDIR)/test_ccl_lifecycle.o
$(OBJDIR)/test_ccl_lifecycle.o: ../adaptor/test_ccl_lifecycle.cpp \
    $(PROJECT_ROOT)/flagcx/adaptor/ccl/nccl_comm_lifecycle.h \
    $(PROJECT_ROOT)/flagcx/adaptor/ccl/compat_comm_lifecycle.h \
    $(PROJECT_ROOT)/flagcx/adaptor/ccl/ppu_comm_lifecycle.h
	@mkdir -p $(OBJDIR)
	@$(CXX) $< -o $@ -c $(CXXFLAGS) -DFLAGCX_TEST_CCL_LIFECYCLE_CORE \
	  $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE)

# Host-only v1 plugin verifies cleanup reference ownership through the real loader.
HANDLE_DEVICE_PLUGIN := $(abspath $(OBJDIR))/libflagcx-handle-device-test.so
HANDLE_CCL_PLUGIN := $(abspath $(OBJDIR))/libflagcx-handle-owner-test.so
$(OBJDIR)/test_device_handle_lifecycle.o: CXXFLAGS += -DFLAGCX_TEST_HANDLE_CCL_PLUGIN='"$(HANDLE_CCL_PLUGIN)"'
$(OBJDIR)/test_device_handle_lifecycle.o: CXXFLAGS += -DFLAGCX_TEST_HANDLE_DEVICE_PLUGIN='"$(HANDLE_DEVICE_PLUGIN)"'
$(OBJDIR)/test_device_handle_lifecycle.o: $(HANDLE_CCL_PLUGIN) $(HANDLE_DEVICE_PLUGIN)

$(HANDLE_DEVICE_PLUGIN): fixtures/handle_owner_device_plugin.cpp
	@mkdir -p $(dir $@)
	@$(CXX) -std=c++17 -shared -fPIC $< -o $@ $(FLAGCX_INCLUDES)

$(HANDLE_CCL_PLUGIN): fixtures/handle_owner_ccl_plugin.cpp
	@mkdir -p $(dir $@)
	@$(CXX) -std=c++17 -shared -fPIC $< -o $@ $(FLAGCX_INCLUDES)

# [mpi] tests: coll_*.cpp + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/core_mpi_tests
MPI_DEPS   := $(MPI_OBJS:.o=.d)

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
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE) -MMD -MP -MF $(@:.o=.d)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE) -MMD -MP -MF $(@:.o=.d)

$(OBJDIR)/main_mpi.o: main_mpi.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(MPI_INCLUDE) -MMD -MP -MF $(@:.o=.d)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@mpirun --allow-run-as-root -np 8 $(MPI_TARGET)

run: run-unit run-mpi

-include $(UNIT_DEPS) $(MPI_DEPS)
