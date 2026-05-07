include ../../make.inc

# Each test_*.cpp becomes its own binary: $(BINDIR)/perf_<name>
SRCS := $(wildcard test_*.cpp)
BINS := $(SRCS:test_%.cpp=$(BINDIR)/perf_%)

# Shared source files compiled into every binary
TOOLS_SRC      := $(PROJECT_ROOT)/test/tools.cc
PERF_COMMON_SRC := $(PROJECT_ROOT)/test/perf_common.cc
COMMON_SRCS    := $(TOOLS_SRC) $(PERF_COMMON_SRC)

# Includes: flagcx (all subdirs), perf common headers, MPI, JSON
# Some tests (core_sendrecv, ipc_sendrecv, put) include internal headers
# from flagcx/core and flagcx/service directly, so we add those too.
ALL_INCLUDES := $(FLAGCX_INCLUDES) $(PERF_INCLUDE) $(MPI_INCLUDE) \
    -I$(PROJECT_ROOT)/flagcx/core -I$(PROJECT_ROOT)/flagcx/service \
    -I$(PROJECT_ROOT)

ALL_LINK := $(FLAGCX_LINK) $(MPI_LINK)

.PHONY: all clean

all: $(BINS)

$(BINDIR)/perf_%: test_%.cpp $(COMMON_SRCS)
	@mkdir -p $(BINDIR)
	@echo "Compiling $@"
	@$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_SRCS) $(ALL_INCLUDES) $(ALL_LINK)

clean:
	@rm -rf $(BUILDDIR)
