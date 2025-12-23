#!/bin/bash

TYPE=$1
HOSTS=$2

SHARED_DIR="/mnt/share"

BUILD_DIR="build"
MPI_HOME=/usr/local/mpi

mkdir -p $BUILD_DIR

if [[ "$TYPE" == "nvidia" ]]; then
    USE_NVIDIA=1 make -j32

elif [[ "$TYPE" == "bi150" ]]; then
    USE_ILUVATAR_COREX=1 make -j32

else
    echo "Invalid compilation type: $TYPE"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

cd test/perf

if [[ "$TYPE" == "nvidia" ]]; then
    echo "Compiling test programs for NVIDIA configuration"
    make USE_NVIDIA=1

elif [[ "$TYPE" == "bi150" ]]; then
    echo "Compiling test programs for Bi150 configuration"
    make USE_ILUVATAR_COREX=1

else
    echo "Invalid test type: $TYPE"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Test compilation failed!"
    exit 1
fi

TESTS=("test_alltoall" "test_alltoallv" "test_sendrecv" "test_allreduce" "test_allgather"
       "test_reducescatter" "test_broadcast" "test_gather" "test_scatter" "test_reduce")

for TEST in "${TESTS[@]}"
do
    echo "Running $TEST on multiple machines ..."
    mpirun -np 16 -hosts $HOSTS \
        -genv PATH=usr/local/mpi/bin/mpirun:/usr/local/corex/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        -genv LD_LIBRARY_PATH=usr/local/mpi/lib:/usr/local/corex/lib64:/usr/local/cuda/lib64 \
        -genv FLAGCX_DEBUG=INFO \
        -genv FLAGCX_DEBUG_SUBSYS=INIT,NET \
        -genv FLAGCX_IB_HCA=mlx5_0 \
        -genv FLAGCX_USENET=mlx5_0 \
        ./$TEST -b 128M -e 1G -f 2 -p 1
    if [ $? -ne 0 ]; then
        echo "$TEST execution failed!"
        exit 1
    fi
done

echo "All tests completed successfully!"

