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
# Both device legs are linked, so one binary exercises a kernel from
# device_api.o and one from device_ir.o.
#
# They clash on 39 symbols: five flagcxTeam* helpers that flagcx_device_core.h
# declares FLAGCX_HOST_DEVICE_INLINE (always_inline but neither static nor inline
# on the device pass), plus CoreX's per-module device-image hooks
# (__ix_fatbin_wrapper, __ix_module_ctor/dtor) and local labels.
# --allow-multiple-definition resolves them.
#
# That switch was verified not to be hiding a real failure: a probe linked both
# objects this way and launched one kernel from EACH object in the same process,
# and both ran -- device_ir's kernel wrote its real result vector rather than
# stale data -- so both device images really are registered. The deduplicated
# helpers are pure functions with no global state, which is why keeping one copy
# is sound. If a future duplicate carried mutable state this must be revisited;
# merging the two .cu files into one device translation unit would be the
# structural fix.
LAUNCH_KERNEL_OBJS   := $(KERNEL_OBJ_DIR)/device_api.o $(KERNEL_OBJ_DIR)/device_ir.o
LAUNCH_KERNEL_LDFLAGS := -Wl,--allow-multiple-definition
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
	@$(CXX) $^ -o $@ $(FLAGCX_LINK) $(GTEST_LINK) $(MPI_LINK) $(LAUNCH_KERNEL_DEVICE_LINK) $(LAUNCH_KERNEL_LDFLAGS)

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
