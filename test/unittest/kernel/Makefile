include ../../make.inc

# Local include directories for test fixtures
LOCAL_INCLUDE := -I$(abspath include)

# device_api.cu object (compiled in test/kernel/[platform]/)
DEVICE_API_OBJ := $(KERNEL_OBJ_DIR)/device_api.o
DEVICE_API_INCLUDE := $(KERNEL_INCLUDE)

# Device-link object (only needed for NVSHMEM)
ifeq ($(USE_SHMEM), 1)
DLINK_OBJ := $(KERNEL_OBJ_DIR)/dlink_api.o
else
DLINK_OBJ :=
endif

# Device linking provided by make.inc (DEVICE_LINK, DEVICE_HOME, etc.)

# Environment variables to force hetero mode on single node
HETERO_ENV := FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 FLAGCX_P2P_DISABLE=1

# [unit] tests: test_*.cpp — no MPI, no GPU required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/kernel_unit_tests

# [mpi] tests: coll_*.cpp + fixture sources + main_mpi.cpp — requires MPI + GPUs
MPI_SRCS   := $(wildcard coll_*.cpp) flagcx_kernel_test.cpp deviceapi_test.cpp main_mpi.cpp
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/kernel_mpi_tests

.PHONY: all clean run-unit run-mpi run device-api-obj

# Only build targets that have sources
ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif

all: $(ALL_TARGETS)

# Build device_api.o (and dlink_api.o if USE_SHMEM) from test/kernel/
DEVICE_API_TARGETS := device_api.o
ifeq ($(USE_SHMEM), 1)
DEVICE_API_TARGETS += dlink_api.o
endif

device-api-obj:
	@$(MAKE) -C $(KERNEL_DIR) $(DEVICE_API_TARGETS)

$(UNIT_TARGET): $(UNIT_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK)

$(MPI_TARGET): device-api-obj $(MPI_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $(MPI_OBJS) $(DEVICE_API_OBJ) $(DLINK_OBJ) -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK) $(DEVICE_LINK)

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(LOCAL_INCLUDE)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(UNITTEST_INCLUDE) $(LOCAL_INCLUDE) $(DEVICE_API_INCLUDE) $(MPI_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@$(HETERO_ENV) mpirun --allow-run-as-root -np 4 $(MPI_TARGET)

run: run-unit run-mpi
