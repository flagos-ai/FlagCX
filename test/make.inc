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

# GoogleTest
GTEST_DIR     := $(PROJECT_ROOT)/third-party/googletest
GTEST_INCLUDE := -I$(GTEST_DIR)/googletest/include
GTEST_LIB     := $(GTEST_DIR)/build/lib
GTEST_LINK    := -L$(GTEST_LIB) -lgtest -lgtest_main -lpthread

# Shared test fixtures
FIXTURE_INCLUDE := -I$(PROJECT_ROOT)/test/fixtures

# Shared unittest base fixtures (flagcx_test.hpp, etc.)
UNITTEST_INCLUDE := -I$(PROJECT_ROOT)/test/unittest/include

# MPI (used by _mpi_tests targets)
MPI_HOME    ?= /usr/local/mpi
MPI_INCLUDE := -I$(MPI_HOME)/include
MPI_LINK    := -L$(MPI_HOME)/lib -lmpi

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
