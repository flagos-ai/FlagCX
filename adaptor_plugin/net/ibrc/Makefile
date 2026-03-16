FLAGCX_HOME ?= ../../..
FLAGCX_BUILDDIR ?= $(FLAGCX_HOME)/build

# Include directories - use FlagCX source tree headers
INCLUDE_DIRS = \
    $(FLAGCX_HOME)/flagcx/include \
    $(FLAGCX_HOME)/flagcx/adaptor/include \
    $(FLAGCX_HOME)/flagcx/core/include \
    $(FLAGCX_HOME)/flagcx/service/include \
    $(FLAGCX_HOME)/third-party/json/single_include

CXXFLAGS = $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) \
           -fPIC -fvisibility=hidden -Wall -g

# Link against libflagcx.so to reuse ibvwrap, socket, param, debug, ib_common, etc.
LDFLAGS = -shared -L$(FLAGCX_BUILDDIR)/lib -lflagcx \
          -Wl,-rpath,'$$ORIGIN/../../../build/lib'

build: libflagcx-net-ibrc.so

libflagcx-net-ibrc.so: ibrc_plugin.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libflagcx-net-ibrc.so
