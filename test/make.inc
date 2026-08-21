# Shared test configuration — included by all test Makefiles.
# Provides common paths, compiler flags, GoogleTest, FlagCX library,
# MPI, and fixture include settings.

# Project root (relative to each leaf Makefile, set before including)
PROJECT_ROOT ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

# Build directories
BUILDDIR ?= $(abspath ./build)
BINDIR   := $(BUILDDIR)/bin
OBJDIR   := $(BUILDDIR)/obj

# Compiler
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wvla -Wno-unused-function -Wno-sign-compare -g
CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX

# FlagCX includes (mirrors main Makefile)
FLAGCX_INCLUDES := \
    -I$(PROJECT_ROOT)/flagcx/include \
    -I$(PROJECT_ROOT)/flagcx/adaptor/include \
    -I$(PROJECT_ROOT)/flagcx/core/include \
    -I$(PROJECT_ROOT)/flagcx/runner/include \
    -I$(PROJECT_ROOT)/flagcx/service/include \
    -I$(PROJECT_ROOT)/third-party/json/single_include

# FlagCX library
FLAGCX_LIB  := $(PROJECT_ROOT)/build/lib
FLAGCX_LINK := -L$(FLAGCX_LIB) -Wl,--no-as-needed -Wl,-rpath,$(FLAGCX_LIB) -lflagcx

# GoogleTest. Prefer the vendored copy, fall back to distro packages in the
# Docker image when submodules are not checked out.
GTEST_DIR ?= $(PROJECT_ROOT)/third-party/googletest
ifneq ($(wildcard $(GTEST_DIR)/googletest/include/gtest/gtest.h),)
GTEST_INCLUDE := -I$(GTEST_DIR)/googletest/include
GTEST_LIB     := $(GTEST_DIR)/build/lib
GTEST_SRC_ROOT := $(GTEST_DIR)/googletest
else
GTEST_DIR     := /usr/src/googletest
GTEST_INCLUDE := -I$(GTEST_DIR)/googletest/include
GTEST_LIB     := /usr/lib/x86_64-linux-gnu
GTEST_SRC_ROOT := $(GTEST_DIR)/googletest
endif
GTEST_LINK    := -L$(GTEST_LIB) -lgtest -lgtest_main -lpthread

# Shared test fixtures
FIXTURE_INCLUDE := -I$(PROJECT_ROOT)/test/fixtures

# Shared unittest base fixtures (flagcx_test.hpp, etc.)
UNITTEST_INCLUDE := -I$(PROJECT_ROOT)/test/unittest/include

# MPI (used by _mpi_tests targets)
MPI_HOME ?= /usr/local/mpi
ifeq ($(wildcard $(MPI_HOME)/include/mpi.h),)
  ifneq ($(wildcard /usr/include/x86_64-linux-gnu/mpi/mpi.h),)
    MPI_HOME := /usr
    MPI_INCLUDE := -I/usr/include/x86_64-linux-gnu/mpi
    MPI_LINK := -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
  else
    MPI_INCLUDE := -I$(MPI_HOME)/include
    MPI_LINK := -L$(MPI_HOME)/lib -lmpi
  endif
else
  MPI_INCLUDE := -I$(MPI_HOME)/include
  MPI_LINK := -L$(MPI_HOME)/lib -lmpi
endif

# CCL (vendor collective library)
CCL_HOME    ?= /usr/local
CCL_INCLUDE := -I$(CCL_HOME)/include

# Perf test common infrastructure
PERF_INCLUDE := -I$(PROJECT_ROOT)/test/include

# Vendor flags (pass through from environment)
USE_NVIDIA         ?= 0
USE_ASCEND         ?= 0
USE_ILUVATAR_COREX ?= 0
USE_CAMBRICON      ?= 0
USE_METAX          ?= 0
USE_MUSA           ?= 0
USE_KUNLUNXIN      ?= 0
USE_DU             ?= 0
USE_AMD            ?= 0
USE_ENFLAME        ?= 0
USE_PPU            ?= 0

# ---------------------------------------------------------------------------
# Device platform resolution — include platform .mk from library build.
# ---------------------------------------------------------------------------
USE_SHMEM ?= 0
SHMEM_HOME ?= /usr/local/nvshmem

ifeq ($(USE_NVIDIA), 1)
  include $(PROJECT_ROOT)/makefiles/nvidia.mk
else ifeq ($(USE_KUNLUNXIN), 1)
  include $(PROJECT_ROOT)/makefiles/kunlunxin.mk
else ifeq ($(USE_DU), 1)
  include $(PROJECT_ROOT)/makefiles/du.mk
else ifeq ($(USE_ASCEND), 1)
  include $(PROJECT_ROOT)/makefiles/ascend.mk
else ifeq ($(USE_ENFLAME), 1)
  include $(PROJECT_ROOT)/makefiles/enflame.mk
else ifeq ($(USE_METAX), 1)
  include $(PROJECT_ROOT)/makefiles/metax.mk
else ifeq ($(USE_PPU), 1)
  include $(PROJECT_ROOT)/makefiles/ppu.mk
else
  ifneq ($(MAKECMDGOALS),clean)
    $(error No platform selected. Set one of USE_NVIDIA=1, USE_DU=1, USE_ASCEND=1, USE_METAX=1, USE_ENFLAME=1, etc.)
  endif
endif

# Test-specific overrides (format for compiler flags)
DEVICE_INCLUDE := $(addprefix -I,$(DEVICE_INCLUDE))
CCL_INCLUDE    := -I$(CCL_INCLUDE)
DEVICE_LINK    := -L$(DEVICE_LIB) -L$(DEVICE_HOME)/lib64/stubs $(DEVICE_LINK)
ADAPTOR_DEFINE := $(ADAPTOR_FLAG)
ifeq ($(DEVICE_PLATFORM), CUDA)
  PLATFORM_IR_DIR := $(PROJECT_ROOT)/bindings/ir/nvidia
else
  PLATFORM_IR_DIR := $(PROJECT_ROOT)/bindings/ir/$(shell echo $(DEVICE_PLATFORM) | tr 'A-Z' 'a-z')
endif

# SHMEM include path (used when USE_SHMEM=1)
ifeq ($(USE_SHMEM), 1)
SHMEM_INCLUDE := -I$(SHMEM_HOME)/include
CXXFLAGS += -DFLAGCX_COMM_TRAITS_SHMEM
else
SHMEM_INCLUDE :=
endif

# Test kernel build directory (platform-specific device code)
KERNEL_DIR := $(PROJECT_ROOT)/test/kernel
# Only require DEVICE_PLATFORM for build targets, not for clean/help/info
ifeq ($(filter clean help info,$(MAKECMDGOALS)),)
ifeq ($(strip $(DEVICE_PLATFORM)),)
  $(error DEVICE_PLATFORM is not set by the selected makefiles/<platform>.mk; cannot determine test kernel platform)
endif
endif
KERNEL_PLATFORM := $(shell echo $(DEVICE_PLATFORM) | tr 'A-Z' 'a-z')
ifeq ($(KERNEL_PLATFORM), cuda)
  KERNEL_PLATFORM := nvidia
endif
KERNEL_OBJ_DIR := $(KERNEL_DIR)/$(KERNEL_PLATFORM)/build/obj
KERNEL_INCLUDE := -I$(KERNEL_DIR)/include

# Debug helper — use 'make print-config' to verify paths.
# NOTE: Do NOT define targets in make.inc (would override default target
# in leaf Makefiles). This is just a variable definition.
PRINT_CONFIG = \
	echo "PROJECT_ROOT:    $(PROJECT_ROOT)"; \
	echo "BUILDDIR:        $(BUILDDIR)"; \
	echo "FLAGCX_INCLUDES: $(FLAGCX_INCLUDES)"; \
	echo "FLAGCX_LIB:      $(FLAGCX_LIB)"; \
	echo "GTEST_DIR:       $(GTEST_DIR)"; \
	echo "MPI_HOME:        $(MPI_HOME)"
