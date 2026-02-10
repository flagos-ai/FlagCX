BUILDDIR ?= $(abspath ./build)

BINDIR := $(BUILDDIR)/bin
OBJDIR := $(BUILDDIR)/obj




INCLUDEDIR := \
        $(abspath include) \
        $(abspath coll/include) \
        $(abspath topo/include) \
        $(abspath ../../third-party/googletest/googletest/include)

LIBSRCFILES := \
        $(wildcard *.cpp) \
        $(wildcard coll/*.cpp) \
        $(wildcard topo/*.cpp)

BINOBJ := $(LIBSRCFILES:%.cpp=$(OBJDIR)/%.o)




GTEST_DIR := ../../third-party/googletest
GTEST_LIB := $(GTEST_DIR)/build/lib


MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib


TARGET = main
all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(BINOBJ)
	@mkdir -p `dirname $@`
	@echo "Linking   $@"
	@g++ $(BINOBJ) -o $@ -L../../build/lib -L$(MPI_LIB) -L/usr/local/lib -L$(GTEST_LIB) -Wl,--no-as-needed -Wl,-rpath,../../build/lib -lflagcx -lmpi -lgtest -lgtest_main -lpthread -g

$(OBJDIR)/%.o: %.cpp
	@mkdir -p `dirname $@`
	@echo "Compiling $@"
	@g++ $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I../../flagcx/include -I$(MPI_INCLUDE) -I/usr/local/include -c -Wvla -Wno-unused-function -Wno-sign-compare -Wall -g

-include $(BINOBJ:.o=.d)

clean:
	@rm -rf $(BINDIR)/$(TARGET) $(OBJDIR)

run-all-tests:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_LOG_LEVEL=TRACE -x FLAGCX_TOPO_FILE="topo_demo.xml" -x FLAGCX_DEBUG=TRACE -x FLAGCX_DEBUG_SUBSYS=ALL ./build/bin/main
