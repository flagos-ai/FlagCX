#!/bin/bash
BUILD_DIR="build"

mkdir -p $BUILD_DIR

MPI_HOME=/usr/local/mpi 
make USE_NVIDIA=1

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

cd test/perf
make USE_NVIDIA=1

if [ $? -ne 0 ]; then
    echo "Test compilation failed!"
    exit 1
fi

source ../script/_gpu_check.sh
wait_for_gpu

mpirun -np 8 ./test_alltoall -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoall in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_alltoall -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoall in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_alltoallv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoallv in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_alltoallv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoallv in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_sendrecv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_sendrecv in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_sendrecv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_sendrecv in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_allreduce -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allreduce in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_allgather -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allgather in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_allgather -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allgather in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_reducescatter -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_reducescatter in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_broadcast -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_broadcast in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_broadcast -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_broadcast in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_gather -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_gather in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_gather -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_gather in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_scatter -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_scatter in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./test_scatter -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_scatter in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./test_reduce -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_reduce execution failed!"
    exit 1
fi

echo "All tests completed successfully!"
