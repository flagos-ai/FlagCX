%global debug_package %{nil}
%global _build_id_links none
# Main "flagcx" package intentionally has no %files of its own; all artifacts
# live in the libflagcx-%%{backend}{,-devel} subpackages. Without this guard,
# rpmbuild treats an empty main package manifest as an error.
%global _empty_manifest_terminate_build 0

# Some distributions do not package the NVIDIA userspace stack. Their build
# container can still provide CUDA/NCCL from another source, while deployment
# supplies ABI-compatible libraries through the site runtime. Keep normal RPM
# dependency generation by default; the openEuler CUDA 12 target opts out only
# for the three vendor libraries that have no RPM provider there.
%if 0%{?external_vendor_runtime}
%global __requires_exclude ^lib(cuda|cudart|nccl)\\.so\\..*
%endif

# Backend must be specified via: rpmbuild --define 'backend nvidia|metax|ascend'
%{!?backend: %{error: backend must be defined (nvidia, metax, or ascend)}}

# Derive uppercase backend name for make flag (USE_NVIDIA=1, etc.)
%global backend_upper %(echo %{backend} | tr a-z A-Z)

# Pin build/install arch by backend. Ascend CANN images are available for
# both x86_64 development hosts and aarch64 deployment hosts; NVIDIA and
# MetaX RPM builds currently target x86_64. ExclusiveArch makes rpmbuild
# refuse to start on unsupported hosts, avoiding CPU-arch-mislabeled RPMs.
%if "%{backend}" == "ascend"
ExclusiveArch:  x86_64 aarch64
%else
ExclusiveArch:  x86_64
%endif

Name:           flagcx
Version:        0.8.0
Release:        1%{?dist}
Summary:        FlagCX scalable cross-chip communication library

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagCX
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  cmake
BuildRequires:  patchelf
# nlohmann-json package name varies by distro:
#   - Fedora and RHEL/Rocky 8/9 (via EPEL): json-devel
#   - openEuler: nlohmann-json-devel
%if 0%{?fedora} || 0%{?rhel}
BuildRequires:  json-devel
%else
BuildRequires:  nlohmann-json-devel
%endif

%description
FlagCX is a scalable and adaptive cross-chip communication library.
It serves as a platform where developers, researchers, and AI engineers
can collaborate on various projects.

# Only the target backend's subpackages are defined
%package -n libflagcx-%{backend}
Summary:        FlagCX library for %{backend}
%if "%{backend}" == "nvidia"
# Group-call API arrived in NCCL 2.10; ncclConfig appeared in 2.14.
# 2.10 is the practical minimum for FlagCX's adaptor today; bump to 2.14
# once we confirm ncclConfig is actually exercised.
%if 0%{?external_vendor_runtime} == 0
Requires:       libnccl >= 2.10
%endif
%endif

%description -n libflagcx-%{backend}
FlagCX communication library built for %{backend} hardware.

%package -n libflagcx-%{backend}-devel
Summary:        Development files for libflagcx-%{backend}
Requires:       libflagcx-%{backend} = %{version}-%{release}

%description -n libflagcx-%{backend}-devel
Development files (headers and libraries) for libflagcx-%{backend}.

%prep
%setup -q

%build
make USE_%{backend_upper}=1 PREFIX=%{_prefix} JSON_INCLUDE_DIR=%{_includedir} \
    %{?device_home:DEVICE_HOME=%{device_home}} \
    %{?ccl_home:CCL_HOME=%{ccl_home}}

%install
rm -rf %{buildroot}

# Install shared library
install -d %{buildroot}%{_libdir}
install -m 755 build/lib/libflagcx.so %{buildroot}%{_libdir}/libflagcx.so.0
ln -s libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so

# Install headers
install -d %{buildroot}%{_includedir}/flagcx
cp -r flagcx/include/* %{buildroot}%{_includedir}/flagcx/

# Fix RPATH and set SONAME — fail loud if patchelf can't normalize the .so,
# otherwise a misconfigured SONAME ships and crashes consumers at runtime.
patchelf --remove-rpath %{buildroot}%{_libdir}/libflagcx.so.0
patchelf --set-soname libflagcx.so.0 %{buildroot}%{_libdir}/libflagcx.so.0

%files -n libflagcx-%{backend}
%license LICENSE
%{_libdir}/libflagcx.so.0

%files -n libflagcx-%{backend}-devel
%{_includedir}/flagcx/
%{_libdir}/libflagcx.so

%changelog
* Wed Jun 24 2026 FlagOS Contributors <contact@flagos.io> - 0.13.0-1
- Add P2P engine perf benchmark (one-sided read/write)
- Replace C++17 features with C++11 equivalents for RPM packaging
- Bootstrap extension
- Replace #ifdef USE_SUNRISE_ADAPTOR with runtime vendor detection
- Fix fep
- Fix triton LSA test library loading and allocator robustness
- [UIL&PAL] FlagCX P2P Engine optimization
- Store winFlags in flagcxWindow to fix non-NVIDIA build failure

* Fri May 22 2026 FlagOS Contributors <contact@flagos.io> - 0.13.0-rc0.1-1
- New upstream release v0.13.0-rc0.1
- Support pool-only registration and optimize regpool containers
- Deprecate flagcxHandlerGroup, separate device handle/uniqueId/comm lifecycle
- Add torch plugin support for Sunrise
- Support Device API IR Bindings
- Add patch file for flagcx integration into nixl v1.1.0
- Add CI workflow for symmetric memory tests
- KV transfer benchmark

* Mon Jun 01 2026 FlagOS Contributors <contact@flagos.io> - 0.13.0-rc2.post1-1
- Support Device API IR Bindings

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.12.0-1
- Add Device API symmem and multicast support
- Support allgather with different message size in torch flagcx backend
- Add & implement P2P interface for integration with nixl
- Make flagcxP2pAccept blocking
- Reuse closed connection slots to fix proxy destroy hang
- Add changelog sync and installation test scripts
- Optimize rmaProxyProgress and Adds batched one-sided PUT operations to improve RDMA throughput
- Rename container directory to docker
- [Docs] Add instructions for executing torch API test
- [Docs] Update README for PCCL
- Using Device API for customAllreduce implementation
- [Docs] add CODEOWNERS
- add CONTRIBUTING.md
- Add Sunrise device and pccl support
- Fix musa stream
- Fix segfault when FLAGCX_P2P_DISABLE=1 by passing proxyConn to PollProxyResponse
- Refactor One-Sided Memory Registration with Global Handle Indexing and HeteroComm Isolation
- P2P topo manager
- Refactor P2P zerocopy
- IBRC p2p adaptor for flagcx p2p engine
- Add Device API multi-FIFO support
- Support RMA Proxy
- Support Device API Transport
- Refactor unittest
- Add Device API DU support
- Introduce traits abstraction and DeviceAPI for unified vendor/fallback support
- Fix enflame torch api test
- Add workflow for syncing code to other sites
- Update social media links

* Sun Mar 01 2026 FlagOS Contributors <contact@flagos.io> - 0.11.0-1
- Enables kernel-based communication on heterogeneous platforms, including NVIDIA and Hygon.
- Adds support for both host-side and device-side one-sided communication semantics.
- Introduces adaptor plugin support, enabling dynamic loading of user-defined Device, CCL, and Net adaptor implementations.

* Sun Feb 01 2026 FlagOS Contributors <contact@flagos.io> - 0.10.0-1
- Implements 11 chip-decoupled collective communication algorithms in uniRunner mode.
- Refactors Device Intra-/Inter-node API and integrates NCCL Device API support on NVIDIA platforms.
- Enhances usability with pip install support for FlagCX and an NCCL wrapper plugin for seamless adoption on NVIDIA platforms.

* Thu Jan 01 2026 FlagOS Contributors <contact@flagos.io> - 0.9.0-1
- Adds support for Enflame, including topsAdaptor and ecclAdaptor.
- Extends flagcxCCLAdaptor to support symmetric operations.
- Introduces the NCCL Device API in ncclAdaptor to enable customized AllReduce operations.
- Refactors glooAdaptor to support both TCP and IB transports, with automatic NIC detection.

* Mon Dec 01 2025 FlagOS Contributors <contact@flagos.io> - 0.8.0-1
- Enables intra-node zero-copy to improve data transfer efficiency for small messages.
- Supports a naive AllReduce implementation in uniRunner mode using a CPU-centric, device-assisted algorithm.
- Adds one-sided communication primitives via the new APIs flagcxHeteroPut and flagcxHeteroPutSignal.
- *[Unreleased]* Test infrastructure restructure and bug fixes (PR #413):
- Fixed NCCL group imbalance in ncclAdaptorGather/ncclAdaptorScatter: errors inside ncclGroupStart()/ncclGroupEnd() no longer skip ncclGroupEnd(), preventing deadlocks.
- Reduced unit-test buffer allocation from 1GB to 4MB per buffer, cutting memory from 32GB to 128MB for 8-rank runs.
- Improved collective test correctness by using rank-dependent data patterns, catching rank-ordering and single-rank-copy bugs.
- Added infinite-loop guard in perfBenchmarkLoop for stepFactor <= 1.
- Wired PERFCOMMONSRC into test/perf/host_api/Makefile build.
- Removed TRACE-level debug logging from CI workflow.

* Sat Nov 01 2025 FlagOS Contributors <contact@flagos.io> - 0.7.0-1
- Added support to TsingMicro, including device adaptor tsmicroAdaptor and CCL adaptor tcclAdaptor.
- Implemented an experimental kernel-free non-reduce collective communication (SendRecv, AlltoAll, AlltoAllv, Broadcast, Gather, Scatter, AllGather) using device-buffer IPC/RDMA.
- Enabled auto-tuning on NVIDIA, MetaX, and Hygon platforms, achieving 1.02×–1.26× speedups for AllReduce, AllGather, ReduceScatter, and AlltoAll.
- Enhanced flagcxNetAdaptor with one-sided primitives (put, putSignal, waitValue) and added retransmission support for reliability improvement.

* Wed Oct 01 2025 FlagOS Contributors <contact@flagos.io> - 0.6.0-1
- Implemented device-buffer IPC communication to support intra-node SendRecv operations.
- Introduced device-initiated, host-launched device-side primitives, enabling kernel-based communication directly from devices.
- Enhanced auto-tuning with 50% performance improvement on MetaX platforms for the AllReduce operations.

* Mon Sep 01 2025 FlagOS Contributors <contact@flagos.io> - 0.5.0-1
- Added support for AMD GPUs, including a device adaptor hipAdaptor and a CCL adaptor rcclAdaptor.
- Introduced flagcxNetAdaptor to unify network backends, currently supporting socket, IBRC, UCX and IBUC (experimental).
- Enabled zero-copy device-buffer RDMA (user-buffer RDMA) to boost performance for small messages.
- Supported auto-tuning in homogeneous scenarios via flagcxTuner.
- Added test automation in CI/CD for PyTorch APIs.

* Fri Aug 01 2025 FlagOS Contributors <contact@flagos.io> - 0.4.0-1
- Supported heterogeneous training of ERNIE4.5 (Baidu) on NVIDIA and Iluvatar GPUs with Paddle + FlagCX.
- Improved heterogeneous communication across arbitrary NIC configurations, with more robust and flexible deployments.
- Introduced an experimental network plugin interface with extended supports for IBRC and SOCKET. Device buffer registration now can be done via DMA-BUF.
- Added an InterOp-level DSL to enable customized C2C algorithm design.
- Provided user documentation under docs/.

* Tue Jul 01 2025 FlagOS Contributors <contact@flagos.io> - 0.3.0-1
- Integrated three additional native communication libraries: HCCL (Huawei), MUSACCL (Moore Threads) and MPI.
- Enhanced heterogeneous collective communication operations with pipeline optimizations.
- Introduced device-side functions to enable device-buffer RDMA, complementing the existing host-side functions.
- Delivered a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous prefilling-decoding disaggregation.

* Thu May 01 2025 FlagOS Contributors <contact@flagos.io> - 0.2.0-1
- Integrated 3 additional native communications libraries, including MCCL (Moore Threads), XCCL (Mellanox) and DUCCL (BAAI).
- Improved 11 heterogeneous collective communication operations with automatic topology detection and full support to single-NIC and multi-NIC environments.

* Tue Apr 01 2025 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Added 5 native communications libraries including CCL adaptors for NCCL (NVIDIA), IXCCL (Iluvatar), and CNCL (Cambricon), and Host CCL adaptors GLOO and Bootstrap.
- Supported 11 heterogeneous collective communication operations using the C2C (Cluster-to-Cluster) algorithm.
- Provided a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous training.
- Natively integrated into PaddlePaddle [v3.0.0](https://github.com/PaddlePaddle/Paddle/tree/v3.0.0), with support for both dynamic and static graphs.
