# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# 2025 - Modified by DU. All Rights Reserved.
USE_NVIDIA ?= 0
USE_ASCEND ?= 0
USE_ILUVATAR_COREX ?= 0
USE_CAMBRICON ?= 0
USE_METAX ?= 0
USE_MUSA ?= 0
USE_KUNLUNXIN ?= 0
USE_DU ?= 0
USE_AMD ?= 0
ifeq ($(USE_CAMBRICON), 1)
   DEV="MLU"
else ifeq ($(USE_ASCEND), 1)
   DEV="CANN"
else ifeq ($(USE_METAX), 1)
   DEV="MACA"
else ifeq ($(USE_MUSA), 1)
   DEV="MUSA"
else ifeq ($(USE_KUNLUNXIN), 1)
   DEV="XPU"
else ifeq ($(USE_AMD), 1)
   DEV="HIP"
else
   DEV="CUDA"
endif

MPI_HOME ?= /usr/local/mpi
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib
MPI_LINK = -lmpi
COMPILER = g++
EXTRA_COMPILER_FLAG = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wl,-rpath,../../build/lib -std=c++11 -g

INCLUDEDIR := $(abspath include)
LIBSRCFILES:= $(wildcard *.cc)

all: test-sendrecv test-allreduce test-allgather test-reducescatter test-alltoall test-alltoallv test-broadcast test-gather test-scatter test-reduce test-core-sendrecv test-ipc-sendrecv

test-sendrecv: test_sendrecv.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_sendrecv test_sendrecv.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-allreduce: test_allreduce.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_allreduce test_allreduce.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-allgather: test_allgather.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_allgather test_allgather.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-reducescatter: test_reducescatter.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_reducescatter test_reducescatter.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-alltoall: test_alltoall.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_alltoall test_alltoall.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-alltoallv: test_alltoallv.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_alltoallv test_alltoallv.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-broadcast: test_broadcast.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_broadcast test_broadcast.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-gather: test_gather.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_gather test_gather.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-scatter: test_scatter.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_scatter test_scatter.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-reduce: test_reduce.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_reduce test_reduce.cpp $(LIBSRCFILES) -I../../flagcx/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-core-sendrecv: test_core_sendrecv.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_core_sendrecv test_core_sendrecv.cpp $(LIBSRCFILES) -I../../flagcx/include -I../../flagcx/service/include -I../../flagcx/core/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib/ -L$(MPI_LIB) -lflagcx $(MPI_LINK)

test-ipc-sendrecv: test_ipc_sendrecv.cpp
	@echo "Compiling $@"
	@$(COMPILER) $(EXTRA_COMPILER_FLAG) -o test_ipc_sendrecv test_ipc_sendrecv.cpp $(LIBSRCFILES) -I../../flagcx/include -I../../flagcx/service/include -I../../flagcx/core/include -I$(INCLUDEDIR) -I$(MPI_INCLUDE) -L../../build/lib/ -L$(MPI_LIB) -lflagcx $(MPI_LINK)

clean:
	@rm -f test_sendrecv
	@rm -f test_allreduce
	@rm -f test_allgather
	@rm -f test_reducescatter
	@rm -f test_alltoall
	@rm -f test_alltoallv
	@rm -f test_broadcast
	@rm -f test_gather
	@rm -f test_scatter
	@rm -f test_reduce
	@rm -f test_core_sendrecv
	@rm -f test_ipc_sendrecv

run-sendrecv:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_sendrecv

run-allreduce:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_allreduce

run-allgather:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_allgather

run-reducescatter:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_reducescatter

run-alltoall:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_alltoall

run-alltoallv:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_alltoallv

run-broadcast:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_broadcast

run-gather:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_gather

run-scatter:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_scatter

run-reduce:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_reduce

run-core-sendrecv:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_core_sendrecv

run-ipc-sendrecv:
	@mpirun --allow-run-as-root -np 8 -x UCX_POSIX_USE_PROC_LINK=n -x ${DEV}_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL ./test_ipc_sendrecv

print_var:
	@echo "INCLUDEDIR: $(INCLUDEDIR)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
    @echo "USE_ASCEND: $(USE_ASCEND)"
	@echo "USE_ILUVATAR_COREX: $(USE_ILUVATAR_COREX)"
	@echo "USE_CAMBRICON: $(USE_CAMBRICON)"
	@echo "USE_DU: $(USE_DU)"
	@echo "USE_MUSA: $(USE_MUSA)"
	@echo "MPI_INCLUDE: $(MPI_INCLUDE)"
	@echo "MPI_LIB: $(MPI_LIB)"
	@echo "COMPILER: $(COMPILER)"
	@echo "EXTRA_COMPILER_FLAG: $(EXTRA_COMPILER_FLAG)"
