FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-metax-maca3.8.1.3@sha256:5e2c950d09e0957e6a5c2f13d7f2862840a48c47459c43f93e6cb7ebe436047e

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build pkg-config openssh-client \
      libgoogle-perftools-dev libibumad-dev libibverbs-dev librdmacm-dev \
      libnuma-dev libunwind-dev ibverbs-utils iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Keep the base image's MetaX Torch in /flagos. Installing upstream Torch here
# replaces it with a CUDA build that cannot initialize MetaX devices.
RUN /flagos/bin/python3 -m pip uninstall -y flagcx \
    && /flagos/bin/python3 -m pip install --no-cache-dir \
      setuptools-scm==9.2.2 coverage==7.10.7 gcovr==8.2 \
    && /flagos/bin/python3 -c \
      'import torch; assert torch.__version__ == "2.10.0+metax3.8.1.0", torch.__version__'

# MACA already supplies Open MPI 4.1.6, including headers and shared libraries.
# Expose it at the path used by FlagCX's existing build and unit-test scripts.
RUN test -x /opt/maca/ompi/bin/mpirun \
    && test -f /opt/maca/ompi/include/mpi.h \
    && ln -s /opt/maca/ompi /usr/local/mpi \
    && /usr/local/mpi/bin/mpirun --version

ENV MPI_HOME=/usr/local/mpi \
    PYTHON_BIN=/flagos/bin/python3 \
    PATH=/flagos/bin:/usr/local/mpi/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:$PATH \
    LD_LIBRARY_PATH=/opt/mxdriver/lib:/opt/maca/lib:/usr/local/mpi/lib:/opt/maca/ucx/lib:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD ["bash"]
