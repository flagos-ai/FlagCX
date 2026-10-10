include ../../make.inc

# [unit] tests: test_*.cpp — single GPU, no MPI required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/adaptor_unit_tests

# [mpi] tests: coll_*.cpp — requires MPI + GPUs
# coll_launch_kernel.cpp is excluded here on purpose: it also links the device
# test-kernel objects, so it gets its own target and the plain MPI suite keeps
# its lighter "host compiler + MPI only" build requirement.
MPI_SRCS   := $(filter-out coll_launch_kernel.cpp,$(wildcard coll_*.cpp))
MPI_OBJS   := $(MPI_SRCS:%.cpp=$(OBJDIR)/%.o)
MPI_TARGET := $(BINDIR)/adaptor_mpi_tests
MPIRUN     ?= mpirun
MPI_NP     ?= 8
MPI_ENV    ?=

# [mpi + device kernels] launchKernel verification: links the test-kernel
# objects so it can launch real kernels from test/kernel/<platform>/. Only
# meaningful where the launch bridges exist, i.e. Iluvatar/CoreX.
LAUNCH_KERNEL_SRC    := coll_launch_kernel.cpp
LAUNCH_KERNEL_OBJ    := $(OBJDIR)/$(LAUNCH_KERNEL_SRC:%.cpp=%.o)
LAUNCH_KERNEL_TARGET := $(BINDIR)/adaptor_launch_kernel_tests
# Only the Device API leg is linked. device_ir.o also compiles now, but both
# objects emit out-of-line copies of the same header-inline flagcxTeam* helpers
# (flagcx_device_core.h declares them FLAGCX_HOST_DEVICE_INLINE, i.e. neither
# static nor inline on the device pass), so linking both is a duplicate-symbol
# error. Keeping one object leaves link semantics exactly as before.
LAUNCH_KERNEL_OBJS   := $(KERNEL_OBJ_DIR)/device_api.o
# Device objects call into the CoreX libcudart entry points
# (__ixPopCallConfiguration and friends), which the host test link line does not
# otherwise pull in.
ifeq ($(KERNEL_PLATFORM),iluvatar)
LAUNCH_KERNEL_DEVICE_LINK := $(DEVICE_LINK)
else
LAUNCH_KERNEL_DEVICE_LINK :=
endif
# Default to a single rank: launchKernel correctness does not depend on the peer
# count, and one rank avoids cross-device IPC entirely.
LAUNCH_KERNEL_NP     ?= 1

.PHONY: all clean run-unit run-mpi run run-launch-kernel

ALL_TARGETS :=
ifneq ($(UNIT_SRCS),)
ALL_TARGETS += $(UNIT_TARGET)
endif
ifneq ($(MPI_SRCS),)
ALL_TARGETS += $(MPI_TARGET)
endif
ifeq ($(KERNEL_PLATFORM),iluvatar)
ALL_TARGETS += $(LAUNCH_KERNEL_TARGET)
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

# Dispatch the device test-kernel objects to the active platform directory.
# $@ is .../test/kernel/<platform>/build/obj/<name>.o and $(@F) recovers just
# "<name>.o", so one rule serves every kernel object and forwards only the one
# that was actually asked for (a literal recipe would always build both).
$(KERNEL_OBJ_DIR)/%.o:
	@$(MAKE) -C $(KERNEL_DIR) $(@F)

$(LAUNCH_KERNEL_TARGET): $(LAUNCH_KERNEL_OBJ) $(LAUNCH_KERNEL_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK) $(LAUNCH_KERNEL_DEVICE_LINK)

$(OBJDIR)/test_%.o: test_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE)

$(OBJDIR)/coll_%.o: coll_%.cpp
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $< -o $@ -c $(CXXFLAGS) $(FLAGCX_INCLUDES) $(GTEST_INCLUDE) $(FIXTURE_INCLUDE) $(MPI_INCLUDE) $(KERNEL_INCLUDE)

clean:
	@rm -rf $(BUILDDIR)

run-unit: $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np $(MPI_NP) $(MPI_ENV) $(MPI_TARGET)

run-launch-kernel: $(LAUNCH_KERNEL_TARGET)
	@$(MPIRUN) --allow-run-as-root -np $(LAUNCH_KERNEL_NP) $(MPI_ENV) \
	    $(LAUNCH_KERNEL_TARGET)

run: run-unit run-mpi
