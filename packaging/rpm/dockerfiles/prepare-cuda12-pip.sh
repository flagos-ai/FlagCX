#!/bin/bash
set -euo pipefail

PYTHON_SITE="$(python3 -c '
import pathlib
import site

for candidate in site.getsitepackages():
    path = pathlib.Path(candidate)
    if (path / "nvidia").is_dir():
        print(path)
        break
else:
    raise SystemExit("NVIDIA wheel packages were not found in site-packages")
')"
NVIDIA_ROOT="${PYTHON_SITE}/nvidia"

for directory in cuda_runtime cuda_nvcc cuda_cccl nccl; do
    if [ ! -d "${NVIDIA_ROOT}/${directory}" ]; then
        echo "ERROR: missing ${NVIDIA_ROOT}/${directory}" >&2
        exit 1
    fi
done

rm -rf /opt/cuda12 /opt/nccl12
install -d /opt/cuda12/bin /opt/cuda12/include/cccl /opt/cuda12/lib64
install -d /opt/nccl12/include /opt/nccl12/lib

# NVIDIA splits the CUDA 12 headers across three wheels. In particular,
# crt/host_config.h is in cuda_nvcc rather than cuda_runtime.
cp -a "${NVIDIA_ROOT}/cuda_runtime/include/." /opt/cuda12/include/
cp -a "${NVIDIA_ROOT}/cuda_nvcc/include/." /opt/cuda12/include/
cp -a "${NVIDIA_ROOT}/cuda_cccl/include/." /opt/cuda12/include/cccl/
cp -a "${NVIDIA_ROOT}/cuda_nvcc/bin/." /opt/cuda12/bin/
cp -a "${NVIDIA_ROOT}/cuda_runtime/lib/." /opt/cuda12/lib64/

cp -a "${NVIDIA_ROOT}/nccl/include/." /opt/nccl12/include/
cp -a "${NVIDIA_ROOT}/nccl/lib/." /opt/nccl12/lib/

ln -sfn libcudart.so.12 /opt/cuda12/lib64/libcudart.so
ln -sfn libnccl.so.2 /opt/nccl12/lib/libnccl.so

# libcuda is supplied by the NVIDIA driver at runtime. The openEuler CUDA
# image carries a link-time stub; accept the common toolkit layouts.
CUDA_STUB=""
for candidate in \
    /usr/local/cuda/lib64/stubs/libcuda.so \
    /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
    /usr/lib64/libcuda.so.1 \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
    if [ -e "${candidate}" ]; then
        CUDA_STUB="${candidate}"
        break
    fi
done

if [ -z "${CUDA_STUB}" ]; then
    echo "ERROR: no CUDA driver library or linker stub was found" >&2
    exit 1
fi
ln -sfn "${CUDA_STUB}" /opt/cuda12/lib64/libcuda.so

test -f /opt/cuda12/include/cuda_runtime.h
test -f /opt/cuda12/include/crt/host_config.h
test -f /opt/nccl12/include/nccl.h
test -e /opt/cuda12/lib64/libcudart.so
test -e /opt/cuda12/lib64/libcuda.so
test -e /opt/nccl12/lib/libnccl.so

echo "CUDA 12 prefix: /opt/cuda12"
echo "NCCL prefix:    /opt/nccl12"
echo "CUDA stub:      ${CUDA_STUB}"
