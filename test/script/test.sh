#!/bin/bash

TYPE=$1

BUILD_DIR="build"
MPI_HOME=/usr/local/mpi

mkdir -p $BUILD_DIR

if [[ "$TYPE" == "nvidia" ]]; then
    USE_NVIDIA=1 make -j$(nproc)

elif [[ "$TYPE" == "bi150" ]]; then
    USE_ILUVATAR_COREX=1 make -j$(nproc)

else
    echo "无效的编译类型: $TYPE"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

cd test/perf

if [[ "$TYPE" == "nvidia" ]]; then
    make -j$(nproc) USE_NVIDIA=1

elif [[ "$TYPE" == "bi150"  ]]; then
    make -j$(nproc) USE_ILUVATAR_COREX=1
else
    echo "无效的编译类型: $TYPE"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Test compilation failed!"
    exit 1
fi

# Perf binaries are in host_api/build/bin/ with perf_ prefix
PERF_BIN=host_api/build/bin

mpirun -np 8 ./$PERF_BIN/perf_alltoall -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoall execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_alltoallv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_alltoallv execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_sendrecv -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_sendrecv execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_allreduce -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allreduce execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_allgather -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_allgather execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_reducescatter -b 128M -e 1G -f 2 -p 1
if [ $? -ne 0 ]; then
    echo "test_reducescatter execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_broadcast -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_broadcast execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_gather -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_gather execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_scatter -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_scatter execution failed!"
    exit 1
fi

mpirun -np 8 ./$PERF_BIN/perf_reduce -b 128M -e 1G -f 2 -r 0 -p 1
if [ $? -ne 0 ]; then
    echo "test_reduce execution failed!"
    exit 1
fi

echo "All tests completed successfully!"


