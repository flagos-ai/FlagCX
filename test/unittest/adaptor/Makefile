include ../../make.inc
.DEFAULT_GOAL := all

ifeq ($(USE_PPU), 1)
# Compile the PCCL/BAREX interoperability case with the native PPU types.
CXXFLAGS += $(HOST_ADAPTOR_DEFINE)
FLAGCX_INCLUDES += $(DEVICE_INCLUDE) $(CCL_INCLUDE)
endif

# [unit] tests: test_*.cpp — single GPU, no MPI required
UNIT_SRCS   := $(wildcard test_*.cpp)
UNIT_OBJS   := $(UNIT_SRCS:%.cpp=$(OBJDIR)/%.o)
UNIT_TARGET := $(BINDIR)/adaptor_unit_tests

FIXTURE_INCLUDE += -I$(OBJDIR)
ADAPTOR_CONTRACTS := $(OBJDIR)/adaptor-contracts/.stamp
$(ADAPTOR_CONTRACTS): $(PROJECT_ROOT)/test/script/test_ppu_adaptors.py \
    $(wildcard $(PROJECT_ROOT)/flagcx/adaptor/device/*_adaptor.cc) \
    $(wildcard $(PROJECT_ROOT)/flagcx/adaptor/ccl/*_adaptor.cc) \
    $(wildcard $(PROJECT_ROOT)/flagcx/adaptor/include/*_adaptor.h)
	@PYTHONDONTWRITEBYTECODE=1 python3 $(PROJECT_ROOT)/test/script/test_ppu_adaptors.py --generate-contracts $(dir $@)
	@touch $@

$(OBJDIR)/test_device_event.o $(OBJDIR)/test_ccl_lifecycle.o $(OBJDIR)/test_host_staging_cleanup.o: $(ADAPTOR_CONTRACTS)

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

ALL_TARGETS := compile-sunrise compile-platform-traits check-sdk-contracts
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

run-unit: compile-sunrise compile-platform-traits check-sdk-contracts $(UNIT_TARGET)
	@$(UNIT_TARGET)

run-mpi: $(MPI_TARGET)
	@$(MPIRUN) --allow-run-as-root -np $(MPI_NP) $(MPI_ENV) $(MPI_TEST_CMD)

run: run-unit run-mpi

# Compile the real guarded translation unit on every host CI platform. SDK
# declarations are intentionally isolated from the real vendor includes; this
# catches missing internal headers even on hosts without a Sunrise runtime.
compile-sunrise:
	@echo "Checking Sunrise adaptor declarations"
	@$(CXX) $(filter-out -DUSE_%_ADAPTOR -DCOMPILE_KERNEL%,$(CXXFLAGS)) \
	  -DUSE_SUNRISE_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT \
	  -Ifixtures/sunrise_sdk $(FLAGCX_INCLUDES) \
	  -fsyntax-only $(PROJECT_ROOT)/flagcx/adaptor/device/ptpu_adaptor.cc

# Host selection must bind the communication backend to the same platform as
# PlatformTraits, even without a vendor Device API SDK or a device compiler.
# Only opaque host stream aliases are stubbed, in an isolated build directory;
# this gate checks header/type composition, not SDK ABI or device intrinsics.
compile-platform-traits:
	@mkdir -p $(OBJDIR)/traits-sdk/tops
	@printf '%s\n' '#pragma once' 'typedef void *musaStream_t;' > $(OBJDIR)/traits-sdk/musa_runtime.h
	@printf '%s\n' '#pragma once' 'typedef void *topsStream_t;' > $(OBJDIR)/traits-sdk/tops/tops_runtime.h
	@set -e; for binding in ASCEND:AscendPlatform MUSA:MusaPlatform ENFLAME:EnflamePlatform; do \
	  platform=$${binding%%:*}; expected=$${binding#*:}; \
	  echo "Checking $$platform host Device API selection"; \
	  printf '%s\n' \
	    '#include "device_api/flagcx_device.h"' \
	    '#include "device_api/completion_word.h"' \
	    '#include <type_traits>' \
	    'static_assert(std::is_same<DeviceAPI, CommTraits<DefaultBackend<FLAGCX_TEST_PLATFORM>>>::value, "wrong Device API platform");' \
	    'static_assert(std::is_same<flagcxCompletionWord_t, DeviceAPI::CompletionWord>::value, "wrong FIFO control word");' \
	    'flagcxCompletionWord_t *control(uint64_t *p) { return flagcxFifoControlPtr(p, flagcxFifoIdxProduced); }' \
	    | $(CXX) $(filter-out -DUSE_%_ADAPTOR -DCOMPILE_KERNEL% -DFLAGCX_COMM_TRAITS_% -DFLAGCX_ENFLAME_DEVICE_API,$(CXXFLAGS)) \
	      -DUSE_$${platform}_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT \
	      -DFLAGCX_TEST_PLATFORM=$$expected -I$(OBJDIR)/traits-sdk $(FLAGCX_INCLUDES) \
	      -x c++ -fsyntax-only -; \
	done

HOST_MPI_TARGET := $(BINDIR)/adaptor_host_mpi_cleanup_tests
HOST_MPI_OBJ := $(OBJDIR)/host_mpi_adaptor.o
HOST_MPI_FLAGS := $(filter-out -DUSE_%_ADAPTOR -DCOMPILE_KERNEL%,$(CXXFLAGS)) -DUSE_MPI_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT
$(HOST_MPI_OBJ): $(PROJECT_ROOT)/flagcx/adaptor/ccl/mpi_adaptor.cc $(PROJECT_ROOT)/flagcx/adaptor/include/mpi_adaptor.h
	@mkdir -p $(dir $@)
	@$(CXX) $(HOST_MPI_FLAGS) $(FLAGCX_INCLUDES) $(MPI_INCLUDE) -c $< -o $@

$(HOST_MPI_TARGET): host_mpi_cleanup.cpp $(HOST_MPI_OBJ)
	@mkdir -p $(dir $@)
	@$(CXX) $(HOST_MPI_FLAGS) $(FLAGCX_INCLUDES) $(MPI_INCLUDE) $(GTEST_INCLUDE) $^ -o $@ $(FLAGCX_LINK) $(GTEST_BASE_LINK) $(MPI_LINK)

all: $(HOST_MPI_TARGET)

.PHONY: check-sdk-contracts
check-sdk-contracts:
	@CXX="$(CXX)" PYTHONDONTWRITEBYTECODE=1 python3 $(PROJECT_ROOT)/test/script/test_ppu_adaptors.py SdkCompileTest DeviceSelectionTest PpuAdaptorTest.test_hggc_device_selection_always_binds_native_context
