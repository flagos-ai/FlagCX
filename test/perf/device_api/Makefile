include ../../make.inc

# Each test_*.cpp becomes its own binary: $(BINDIR)/perf_<name>
SRCS := $(wildcard test_*.cpp)
BINS := $(SRCS:test_%.cpp=$(BINDIR)/perf_%)

# Shared source files
TOOLS_SRC   := $(PROJECT_ROOT)/test/tools.cc
COMMON_SRCS := $(TOOLS_SRC)

# Device SDK paths (vendor-specific)
DEVICE_HOME    ?=
DEVICE_LIB     =
DEVICE_INCLUDE =
DEVICE_LINK    =

ifeq ($(USE_NVIDIA), 1)
    DEVICE_HOME    = /usr/local/cuda
    DEVICE_LIB     = $(DEVICE_HOME)/lib64 -L$(DEVICE_HOME)/lib64/stubs
    DEVICE_INCLUDE = $(DEVICE_HOME)/include
    DEVICE_LINK    = -lcudart -lcuda
endif

# Includes: flagcx (all subdirs), perf headers, MPI, device SDK
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(PERF_INCLUDE) $(MPI_INCLUDE) \
    -I$(PROJECT_ROOT)/flagcx/core -I$(PROJECT_ROOT)/flagcx/service
ifneq ($(DEVICE_INCLUDE),)
    ALL_INCLUDES += -I$(DEVICE_INCLUDE)
endif

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK) $(DEVICE_LINK)
ifneq ($(DEVICE_LIB),)
    ALL_LINK += -L$(DEVICE_LIB)
endif

# Kernel tests need -DCOMPILE_KERNEL
CXXFLAGS += -DCOMPILE_KERNEL

.PHONY: all clean

all: $(BINS)

$(BINDIR)/perf_%: test_%.cpp $(COMMON_SRCS)
	@mkdir -p $(BINDIR)
	@echo "Compiling $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_SRCS) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)
