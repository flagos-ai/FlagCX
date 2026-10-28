FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa4.3.6@sha256:5a1207992e6d7fd671e6f5843735a7d0aeb94ea6a3fd834bc06cf0b3287af7bc

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/musa/bin:/usr/local/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64:/usr/local/mpi/lib:/usr/local/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    iproute2 \
    libgtest-dev \
    libibverbs-dev \
    libnuma-dev \
    libopenmpi-dev \
    librdmacm-dev \
    libssl-dev \
    libtool \
    make \
    openmpi-bin \
    openmpi-common \
    python3 \
    python3-dev \
    python3-pip \
    python3-pytest \
    python3-yaml \
    pkg-config \
    ruby \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
