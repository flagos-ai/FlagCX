FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-iluvatar-corex4.5.0@sha256:4aa770618fa0e55058db69593242ffc284c8a47898b35fbc205b98865ea1b106

ENV DEBIAN_FRONTEND=noninteractive
ENV MPI_HOME=/usr/local/openmpi

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-perftools-dev \
      libibumad-dev \
      libibverbs-dev \
      libnuma-dev \
      librdmacm-dev \
      libunwind-dev \
      nlohmann-json3-dev \
      openmpi-bin \
      libopenmpi-dev \
      pkg-config \
      ruby \
    && rm -rf /var/lib/apt/lists/*

RUN multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"; \
    mkdir -p "$MPI_HOME/bin"; \
    for tool in mpicc mpicxx mpirun; do \
      ln -s "$(command -v "$tool.openmpi")" "$MPI_HOME/bin/$tool"; \
    done; \
    ln -s "/usr/lib/$multiarch/openmpi/include" "$MPI_HOME/include"; \
    ln -s "/usr/lib/$multiarch/openmpi/lib" "$MPI_HOME/lib"

ENV PATH=/usr/local/openmpi/bin:/usr/local/corex/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD ["bash", "-lc", "sleep infinity"]
