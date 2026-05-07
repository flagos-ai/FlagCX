CCL_HOME ?= /usr/local/nccl/build
CUDA_HOME ?= /usr/local/cuda

INCLUDE_DIRS = \
    . \
    $(CCL_HOME)/include \
    $(CUDA_HOME)/include

CXXFLAGS = $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) \
           -fPIC -fvisibility=hidden -Wall -g

LDFLAGS = -shared \
          -L$(CCL_HOME)/lib -lnccl \
          -L$(CUDA_HOME)/lib64 -lcudart \
          -Wl,-rpath,$(CCL_HOME)/lib \
          -Wl,-rpath,$(CUDA_HOME)/lib64

build: libflagcx-ccl-example.so

libflagcx-ccl-example.so: plugin.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libflagcx-ccl-example.so
