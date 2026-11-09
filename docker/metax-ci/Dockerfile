FROM harbor.baai.ac.cn/flagos-dev/flagcx@sha256:61b0f3c8fff49aeb540fc6a25eb13917059d3c74aafdf0e9da439724b4f6155e AS rdma-provider

FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-metax-maca3.8.1.3@sha256:5e2c950d09e0957e6a5c2f13d7f2862840a48c47459c43f93e6cb7ebe436047e

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build pkg-config openssh-client \
      libgoogle-perftools-dev libibumad-dev libibverbs-dev librdmacm-dev \
      libnuma-dev libunwind-dev ibverbs-utils iproute2 \
    && rm -rf /var/lib/apt/lists/*

# The nodes use Broadcom bnxt_re 235.2.96.0 (kernel ABI 8). Ubuntu's provider
# only accepts ABI 1. Preserve the matching Ubuntu 24.04 / rdma-core 50 vendor
# provider from the previous CI image, after apt has installed the RDMA stack.
COPY --from=rdma-provider \
    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so \
    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
RUN echo '2032c6861b1c4ab38fc2c00597e8c4d927e9b42a8c5beff85516eb14dc3c7be4  /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so' | sha256sum -c - \
    && /flagos/bin/python3 -c \
      'import ctypes; ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so")'

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
