MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib
MPI_LINK = -lmpi
COMPILER = g++
EXTRA_COMPILER_FLAG = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wl,-rpath,../../build/lib -std=c++11 -g

INCLUDEDIR := $(abspath include)
LIBSRCFILES:= $(wildcard *.cc)

DEVICE_HOME ?=
DEVICE_LIB =
DEVICE_INCLUDE =
DEVICE_LINK =
DEVICE_COMPILER =
ifeq ($(USE_NVIDIA), 1)
	DEVICE_HOME = /usr/local/cuda
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
endif

all: test-kernel-demo test-kernel-put

test-kernel-demo: test_kernel_demo.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_kernel_demo test_kernel_demo.cpp $(LIBSRCFILES) -I../../flagcx/include -I../../flagcx/adaptor/include -I../../flagcx/service/include -I../../flagcx/core/include -I../../third-party/json/single_include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -I$(DEVICE_INCLUDE) -L../../build/lib -L$(MPI_LIB) -L$(DEVICE_LIB) -lflagcx $(MPI_LINK) $(DEVICE_LINK) -DCOMPILE_KERNEL

test-kernel-put: test_kernel_put.cpp
       @echo "Compiling $@"
       @$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_kernel_put test_kernel_put.cpp $(LIBSRCFILES) -I../../flagcx/include -I../../flagcx/adaptor/include -I../../flagcx/service/include -I../../flagcx/core/include -I../../third-party/json/single_include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -I$(DEVICE_INCLUDE) -L../../build/lib -L$(MPI_LIB) -L$(DEVICE_LIB) -lflagcx $(MPI_LINK) $(DEVICE_LINK) -DCOMPILE_KERNEL

clean:
	@rm -f test_kernel_demo
	@rm -f test_kernel_put

print_var:
	@echo "INCLUDEDIR: $(INCLUDEDIR)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
	@echo "MPI_INCLUDE: $(MPI_INCLUDE)"
	@echo "MPI_LIB: $(MPI_LIB)"
	@echo "COMPILER: $(COMPILER)"
	@echo "EXTRA_COMPILER_FLAG: $(EXTRA_COMPILER_FLAG)"
