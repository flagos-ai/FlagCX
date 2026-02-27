%global debug_package %{nil}
%global _build_id_links none

Name:           flagcx
Version:        0.8.0
Release:        1%{?dist}
Summary:        FlagCX scalable cross-chip communication library

License:        ASL 2.0
URL:            https://github.com/flagos-ai/FlagCX
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  cmake
BuildRequires:  patchelf
# nlohmann-json package name varies by distro
%if 0%{?rhel} == 8
BuildRequires:  json-devel
%else
BuildRequires:  nlohmann-json-devel
%endif

# Backend-specific packages will be built with different profiles
# This is the base spec, actual builds use --define 'backend nvidia|metax|ascend'

%description
FlagCX is a scalable and adaptive cross-chip communication library.
It serves as a platform where developers, researchers, and AI engineers
can collaborate on various projects.

%package -n libflagcx-nvidia
Summary:        FlagCX library for NVIDIA GPUs
Requires:       libnccl >= 2.0

%description -n libflagcx-nvidia
FlagCX communication library built for NVIDIA hardware with NCCL backend support.

%package -n libflagcx-nvidia-devel
Summary:        Development files for libflagcx-nvidia
Requires:       libflagcx-nvidia = %{version}-%{release}

%description -n libflagcx-nvidia-devel
Development files (headers and libraries) for libflagcx-nvidia.

%package -n libflagcx-metax
Summary:        FlagCX library for MetaX accelerators

%description -n libflagcx-metax
FlagCX communication library built for MetaX hardware with MCCL backend support.

%package -n libflagcx-metax-devel
Summary:        Development files for libflagcx-metax
Requires:       libflagcx-metax = %{version}-%{release}

%description -n libflagcx-metax-devel
Development files (headers and libraries) for libflagcx-metax.

%package -n libflagcx-ascend
Summary:        FlagCX library for Ascend NPUs

%description -n libflagcx-ascend
FlagCX communication library built for Huawei Ascend NPUs with HCCL backend support.

%package -n libflagcx-ascend-devel
Summary:        Development files for libflagcx-ascend
Requires:       libflagcx-ascend = %{version}-%{release}

%description -n libflagcx-ascend-devel
Development files (headers and libraries) for libflagcx-ascend.

%prep
%setup -q

%build
# Determine which backend to build based on RPM macro
%if "%{?backend}" == "nvidia"
    make USE_NVIDIA=1 PREFIX=%{_prefix}
%endif

%if "%{?backend}" == "metax"
    make USE_METAX=1 PREFIX=%{_prefix}
%endif

%if "%{?backend}" == "ascend"
    make USE_ASCEND=1 PREFIX=%{_prefix}
%endif

%install
rm -rf %{buildroot}

%if "%{?backend}" == "nvidia"
    # Install NVIDIA variant
    install -d %{buildroot}%{_libdir}

    # Install library
    install -m 755 build/lib/libflagcx.so %{buildroot}%{_libdir}/libflagcx.so.0

    # Create symlinks
    ln -s libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so

    install -d %{buildroot}%{_includedir}/flagcx
    cp -r flagcx/include/* %{buildroot}%{_includedir}/flagcx/

    # Fix RPATH and set SONAME
    patchelf --remove-rpath %{buildroot}%{_libdir}/libflagcx.so.0 || true
    patchelf --set-soname libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so.0 || true
%endif

%if "%{?backend}" == "metax"
    # Install MetaX variant
    install -d %{buildroot}%{_libdir}

    # Install library
    install -m 755 build/lib/libflagcx.so %{buildroot}%{_libdir}/libflagcx.so.0

    # Create symlinks
    ln -s libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so

    install -d %{buildroot}%{_includedir}/flagcx
    cp -r flagcx/include/* %{buildroot}%{_includedir}/flagcx/

    patchelf --remove-rpath %{buildroot}%{_libdir}/libflagcx.so.0 || true
    patchelf --set-soname libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so.0 || true
%endif

%if "%{?backend}" == "ascend"
    # Install Ascend variant
    install -d %{buildroot}%{_libdir}

    # Install library
    install -m 755 build/lib/libflagcx.so %{buildroot}%{_libdir}/libflagcx.so.0

    # Create symlinks
    ln -s libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so

    install -d %{buildroot}%{_includedir}/flagcx
    cp -r flagcx/include/* %{buildroot}%{_includedir}/flagcx/

    patchelf --remove-rpath %{buildroot}%{_libdir}/libflagcx.so.0 || true
    patchelf --set-soname libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so.0 || true
%endif

%files -n libflagcx-nvidia
%if "%{?backend}" == "nvidia"
%license LICENSE
%{_libdir}/libflagcx.so.0
%endif

%files -n libflagcx-nvidia-devel
%if "%{?backend}" == "nvidia"
%{_includedir}/flagcx/
%{_libdir}/libflagcx.so
%endif

%files -n libflagcx-metax
%if "%{?backend}" == "metax"
%license LICENSE
%{_libdir}/libflagcx.so.0
%endif

%files -n libflagcx-metax-devel
%if "%{?backend}" == "metax"
%{_includedir}/flagcx/
%{_libdir}/libflagcx.so
%endif

%files -n libflagcx-ascend
%if "%{?backend}" == "ascend"
%license LICENSE
%{_libdir}/libflagcx.so.0
%{_libdir}/libflagcx.so.%{version}
%endif

%files -n libflagcx-ascend-devel
%if "%{?backend}" == "ascend"
%{_includedir}/flagcx/
%{_libdir}/libflagcx.so
%endif

%changelog
* Sat Nov 01 2025 FlagOS Contributors <contact@flagos.io> - 0.7-1
- Added support to TsingMicro, including device adaptor tsmicroAdaptor and CCL adaptor tcclAdaptor.
- Implemented an experimental kernel-free non-reduce collective communication (SendRecv, AlltoAll, AlltoAllv, Broadcast, Gather, Scatter, AllGather) using device-buffer IPC/RDMA.
- Enabled auto-tuning on NVIDIA, MetaX, and Hygon platforms, achieving 1.02×–1.26× speedups for AllReduce, AllGather, ReduceScatter, and AlltoAll.
- Enhanced flagcxNetAdaptor with one-sided primitives (put, putSignal, waitValue) and added retransmission support for reliability improvement.

* Wed Oct 01 2025 FlagOS Contributors <contact@flagos.io> - 0.6-1
- Implemented device-buffer IPC communication to support intra-node SendRecv operations.
- Introduced device-initiated, host-launched device-side primitives, enabling kernel-based communication directly from devices.
- Enhanced auto-tuning with 50% performance improvement on MetaX platforms for the AllReduce operations.

* Mon Sep 01 2025 FlagOS Contributors <contact@flagos.io> - 0.5-1
- Added support for AMD GPUs, including a device adaptor hipAdaptor and a CCL adaptor rcclAdaptor.
- Introduced flagcxNetAdaptor to unify network backends, currently supporting socket, IBRC, UCX and IBUC (experimental).
- Enabled zero-copy device-buffer RDMA (user-buffer RDMA) to boost performance for small messages.
- Supported auto-tuning in homogeneous scenarios via flagcxTuner.
- Added test automation in CI/CD for PyTorch APIs.

* Fri Aug 01 2025 FlagOS Contributors <contact@flagos.io> - 0.4-1
- Supported heterogeneous training of ERNIE4.5 (Baidu) on NVIDIA and Iluvatar GPUs with Paddle + FlagCX.
- Improved heterogeneous communication across arbitrary NIC configurations, with more robust and flexible deployments.
- Introduced an experimental network plugin interface with extended supports for IBRC and SOCKET. Device buffer registration now can be done via DMA-BUF.
- Added an InterOp-level DSL to enable customized C2C algorithm design.
- Provided user documentation under docs/.

* Tue Jul 01 2025 FlagOS Contributors <contact@flagos.io> - 0.3-1
- Integrated three additional native communication libraries: HCCL (Huawei), MUSACCL (Moore Threads) and MPI.
- Enhanced heterogeneous collective communication operations with pipeline optimizations.
- Introduced device-side functions to enable device-buffer RDMA, complementing the existing host-side functions.
- Delivered a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous prefilling-decoding disaggregation.

* Thu May 01 2025 FlagOS Contributors <contact@flagos.io> - 0.2-1
- Integrated 3 additional native communications libraries, including MCCL (Moore Threads), XCCL (Mellanox) and DUCCL (BAAI).
- Improved 11 heterogeneous collective communication operations with automatic topology detection and full support to single-NIC and multi-NIC environments.

* Tue Apr 01 2025 FlagOS Contributors <contact@flagos.io> - 0.1-1
- Added 5 native communications libraries including CCL adaptors for NCCL (NVIDIA), IXCCL (Iluvatar), and CNCL (Cambricon), and Host CCL adaptors GLOO and Bootstrap.
- Supported 11 heterogeneous collective communication operations using the C2C (Cluster-to-Cluster) algorithm.
- Provided a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous training.
- Natively integrated into PaddlePaddle [v3.0.0](https://github.com/PaddlePaddle/Paddle/tree/v3.0.0), with support for both dynamic and static graphs.
