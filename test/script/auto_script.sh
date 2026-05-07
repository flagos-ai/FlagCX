#!/bin/bash
BUILD_DIR="build"

mkdir -p $BUILD_DIR

export MPI_HOME=/usr/local/mpi
export PATH=$MPI_HOME/bin:$PATH
make -j$(nproc) USE_NVIDIA=1

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

cd test/perf
make -j$(nproc) USE_NVIDIA=1

if [ $? -ne 0 ]; then
    echo "Test compilation failed!"
    exit 1
fi

# Perf binaries are in host_api/build/bin/ with perf_ prefix
PERF_BIN=host_api/build/bin

source ../script/_gpu_check.sh
wait_for_gpu

mpirun -np 8 ./$PERF_BIN/perf_alltoall -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoall in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_alltoall -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoall in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_alltoallv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoallv in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_alltoallv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoallv in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_sendrecv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_sendrecv in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_sendrecv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_sendrecv in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_allreduce -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allreduce in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_allgather -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allgather in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_allgather -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allgather in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_reducescatter -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_reducescatter in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_broadcast -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_broadcast in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_broadcast -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_broadcast in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_gather -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_gather in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_gather -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_gather in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_scatter -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_scatter in homoRunner mode failed!"
    exit 1
fi

mpirun -np 8 \
  -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_USE_HETERO_COMM=1 \
  ./$PERF_BIN/perf_scatter -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_scatter in uniRunner mode failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_reduce -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_reduce execution failed!"
    exit 1
fi

echo "All tests completed successfully!"
