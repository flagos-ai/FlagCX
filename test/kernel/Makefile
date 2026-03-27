#*************************************************************************
# Copyright (c) 2026 BAAI. All rights reserved.
#*************************************************************************

MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib
MPI_LINK = -lmpi
COMPILER = g++
EXTRA_COMPILER_FLAG = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wl,-rpath,../../build/lib -std=c++11 -g

INCLUDEDIR := $(abspath ../include)
LIBSRCFILES:= ../tools.cc

DEVICE_HOME ?=
DEVICE_LIB =
DEVICE_INCLUDE =
DEVICE_LINK =
DEVICE_COMPILER =
ifeq ($(USE_NVIDIA), 1)
	DEVICE_HOME = /usr/local/cuda
	DEVICE_LIB = $(DEVICE_HOME)/lib64 -L$(DEVICE_HOME)/lib64/stubs
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
endif

COMMON_INCLUDES = -I../../flagcx/include -I../../flagcx/adaptor/include -I../../flagcx/service/include -I../../flagcx/core/include -I../../third-party/json/single_include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -I$(DEVICE_INCLUDE)
COMMON_LIBS = -L../../build/lib -L$(MPI_LIB) -L$(DEVICE_LIB) -lflagcx $(MPI_LINK) $(DEVICE_LINK)

all: test-intranode test-internode-twosided test-internode-onesided test-device-api

test-intranode: test_intranode.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_intranode test_intranode.cpp $(LIBSRCFILES) $(COMMON_INCLUDES) $(COMMON_LIBS) -DCOMPILE_KERNEL

test-internode-twosided: test_internode_twosided.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_internode_twosided test_internode_twosided.cpp $(LIBSRCFILES) $(COMMON_INCLUDES) $(COMMON_LIBS) -DCOMPILE_KERNEL

test-internode-onesided: test_internode_onesided.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_internode_onesided test_internode_onesided.cpp $(LIBSRCFILES) $(COMMON_INCLUDES) $(COMMON_LIBS) -DCOMPILE_KERNEL

test-device-api: test_device_api.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_device_api test_device_api.cpp $(LIBSRCFILES) $(COMMON_INCLUDES) $(COMMON_LIBS) -DCOMPILE_KERNEL

clean:
	@rm -f test_intranode test_internode_twosided test_internode_onesided test_device_api

print_var:
	@echo "INCLUDEDIR: $(INCLUDEDIR)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
	@echo "MPI_INCLUDE: $(MPI_INCLUDE)"
	@echo "MPI_LIB: $(MPI_LIB)"
	@echo "COMPILER: $(COMPILER)"
	@echo "EXTRA_COMPILER_FLAG: $(EXTRA_COMPILER_FLAG)"
