
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/31746439-e7b0-4391-8418-2f3597a88141" />](https://www.flagopen.ac.cn/)


## Latest News
- **[2025/10]** Released [v0.6](https://github.com/FlagOpen/FlagCX/tree/release/v0.6):
  - Supported device-buffer P2P communication to achieve intra-node SendRecv operations.
  - Introduced Device-initiated, Host-launched device-side primitives, enabling kernel-based communication directly from the device.
  - Enhanced automatic tuning functionality, achieving up to 50% performance improvement on Metax platforms for the AllReduce operation.
- **[2025/09]** Released [v0.5](https://github.com/FlagOpen/FlagCX/tree/release/v0.5):
  - Added AMD support (hipAdaptor and rcclAdaptor).
  - Introduced flagcxNetAdaptor to unify network backends, currently supporting SOCKET, IBRC, UCX and IBUC (experimently).
  - Enabled zero-copy device-buffer RDMA (user-buffer RDMA) to boost small-message performance.
  - Supported automatic tuning in homogeneous scenarios via flagcxTuner.
  - Integrated automated PyTorch API tests into CI/CD.
- **[2025/08]** Released [v0.4](https://github.com/FlagOpen/FlagCX/tree/release/v0.4):
  - Supported heterogeneous training of ERNIE4.5 on Nvidia and Iluvatar GPUs with Paddle + FlagCX.
  - Enabled more robust and flexible deployments with full support of heterogeneous communication across arbitrary NIC configurations (bug fixes). 
  - Introduced an early experimental net plugin interface extending its support for both IBRC and SOCKET, along with the ability to register device buffers via DMA-BUF.
  - Added an InterOp-level DSL to allow users designing customized C2C algorithms.
  - Provided usage documentation under docs/.
- **[2025/07]** Released [v0.3](https://github.com/FlagOpen/FlagCX/tree/release/v0.3):
  - Integrated three additional native communication libraries: HCCL, MUSACCL and MPI.
  - Enhanced heterogeneous collective communication operations with pipeline optimizations. 
  - Introduced a device-side function mechanism to enable device-buffer RDMA, complementing the original host-side function mechanism.
  - Delivered a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous prefilling-decoding disaggregation.
- **[2025/05]** Released [v0.2](https://github.com/FlagOpen/FlagCX/tree/release/v0.2):
  - Integrated three additional native communications libraries, including MCCL, XCCL and DUCCL.
  - Improved 11 heterogeneous collective communication operations with automatic topology detection, fully supporting both single-NIC and multi-NIC environments.
- **[2025/04]** Released [v0.1](https://github.com/FlagOpen/FlagCX/tree/release/v0.1):
  - Integrated five native communications libraries including NCCL, IXCCL, CNCL, BOOTSTRAP and GLOO.
  - Supported 11 heterogeneous collective communication operations using the originally proposed C2C (Cluster-to-Cluster) algorithm.
  - Provided a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous training.
  - Natively integrated into PaddlePaddle [v3.0.0](https://github.com/PaddlePaddle/Paddle/tree/v3.0.0), with support for both dynamic and static graphs.

## About
[FlagCX](https://github.com/FlagOpen/FlagCX.git) is a scalable and adaptive cross-chip communication library developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI).

FlagCX is also a part of [FlagAI-Open](https://flagopen.baai.ac.cn/), an open-source initiative by BAAI that aims to foster an open-source ecosystem for AI technologies. It serves as a platform where developers, researchers, and AI enthusiasts can collaborate on various AI projects, contribute to the development of cutting-edge AI solutions, and share their work with the global community.

FlagCX leverages native collective communications libraries to provide the full support of single-chip communications on different platforms. In addition to its native x-CCL support, FlagCX provides an original device-buffer RDMA design to offer advanced support for cross-chip high-performance sendrecev operations, which can also be integrated with native x-CCL backends to enable optimized cross-chip collective communications. A comprehensive list of currently supported communication backends and their different capabilities are listed as follows:
| Backend       | NCCL        | IXCCL       | CNCL        | MCCL        | XCCL        | DUCCL       | HCCL        | MUSACCL     | RCCL        | TCCL        |
|:--------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|
| Mode          | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero |
| send          | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| recv          | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| broadcast     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| gather        | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ☓/☓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| scatter       | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| reduce        | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| allreduce     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| allgather     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| reducescatter | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| alltoall      | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| alltoallv     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |
| group ops     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/☓         |

Note that `Homo` and `Hetero` modes refer to communications among homogeneous and heterogeneous clusters. All native collective communications libraries can be referenced through the links below:

- [NCCL](https://github.com/NVIDIA/nccl), NVIDIA Collective Communications Library.
- [IXCCL](https://www.iluvatar.com/software?fullCode=cpjs-rj-rjz), Iluvatar Corex Collective Communications Library.
- [CNCL](https://www.cambricon.com/docs/sdk_1.7.0/cncl_1.2.1/user_guide/index.html#), Cambricon Communications Library.
- [MCCL](https://developer.metax-tech.com/softnova/metax), Metax Collective Communications Library.
- [XCCL](WIP), XPU Collective Communications Library.
- [DUCCL](https://developer.sourcefind.cn), DU Collective Communications Library.
- [HCCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/hccl/hcclug/hcclug_000001.html), Ascend Communications Library.
- [MUSACCL](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/Chapter08/), Musa Collective Communications Library.
- [RCCL](https://github.com/ROCm/rccl), ROCm Communication Collectives Library.
- [TCCL](http://www.tsingmicro.com), TsingMicro Communication Collectives Library.

Additionally, FlagCX supports three collective communication libraries for host-side communication: `BOOTSTRAP`, `GLOO`, and `MPI`. Besides `BOOTSTRAP`, which is built using the FlagCX `bootstrap` component, the other two libraries are described as follows:

- [GLOO](https://github.com/facebookincubator/gloo), Gloo Collective Communications Library.
- [MPI](https://www.mpich.org), Message Passing Interface (MPI) standard.

FlagCX also integrates with upper-layer applications such as PyTorch and PaddlePaddle based on its unified APIs. The table below presents all supported frameworks by FlagCX and their related communication operations, where the `batch_XXX` and `XXX_coalesced` ops refer to the usage of group primitives.

| Framework                        | PyTorch                      | PaddlePaddle |
| :------------------------------- | :--------------------------- | :----------- |
| send                             | ✓                            | ✓            |
| recv                             | ✓                            | ✓            |
| batch_isend_irecv                | ✓                            | ✓            |
| broadcast                        | ✓                            | ✓            |
| all_reduce                       | ✓                            | ✓            |
| all_reduce_coalesced             | ✓ (in order, no aggregation) | ☓            |
| reduce                           | ✓                            | ✓            |
| all_gather                       | ✓                            | ✓            |
| all_gather_into_tensor_coalesced | ✓ (in order, no aggregation) | ☓            |
| gather                           | ✓                            | ✓            |
| scatter                          | ✓                            | ✓            |
| reduce_scatter                   | ✓                            | ✓            |
| reduce_scatter_tensor_coalesced  | ✓ (in order, no aggregation) | ☓            |
| all_to_all                       | ✓                            | ✓            |
| all_to_all_single                | ✓                            | ✓            |
| barrier                          | ✓                            | ✓            |

In particular, PyTorch support is enabled via the FlagCX Torch Plugin, which provides native integration with the PyTorch distributed backend. This plugin has undergone comprehensive validation across diverse communication backends and hardware platforms, ensuring robust functionality, consistent performance, and compatibility in heterogeneous multi-chip environments, as summarized below:

| FlagCX Backend  | NCCL | IXCCL | CNCL | MCCL | XCCL | DUCCL | HCCL | MUSACCL | RCCL |
| :-------------- | :--- | :---- | :--- | :--- | :--- | :---- | :--- | :------ | :--- |
| PyTorch Support | ✓    | ✓     | ✓    | ✓    | ✓    | ✓     | ✓    | ✓       | ✓    |

To enable heterogeneous cross-chip communication using the PyTorch DDP FlagCX backend, it is recommended to use identical PyTorch versions across all nodes. Mismatched versions may lead to initialization failures during process group setup. Further compatibility and performance tests will be conducted in future releases, and we warmly welcome community contributions to help expand and strengthen the validation matrix.

## Join our Discussion Channel


<img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/af9f98be-8176-4039-be4a-7f5b15513ff1" />

## Quick Start

### Build 
1. Clone the repository:
    ```sh
    git clone https://github.com/FlagOpen/FlagCX.git
    ```

2. Build the library with different flags targeting to different platforms:
    ```sh
    cd FlagCX
    make [USE_NVIDIA/USE_ILUVATAR_COREX/USE_CAMBRICON/USE_GLOO/USE_MPI/USE_METAX/USE_MUSA/USE_KUNLUNXIN/USE_DU/USE_ASCEND/USE_AMD]=1
    ```
    The default install path is set to `build/`, you can manually set `BUILDDIR` to specify the build path. You may also define `DEVICE_HOME` and `CCL_HOME` to indicate the install paths of device runtime and communication libraries.

### Tests
Tests for FlagCX are maintained in `test/perf`.
```sh
cd test/perf
make [USE_NVIDIA/USE_ILUVATAR_COREX/USE_CAMBRICON/USE_METAX/USE_MUSA/USE_KUNLUNXIN/USE_DU/USE_ASCEND/USE_TSM]=1
mpirun --allow-run-as-root -np 8 ./test_allreduce -b 128K -e 4G -f 2
```
Note that the default MPI install path is set to `/usr/local/mpi`, you may specify the MPI path with:
```sh
make MPI_HOME=<path to mpi install>
```

All tests support the same set of arguments:

* Sizes to scan
  * `-b <min size in bytes>` minimum size to start with. Default: 1M.
  * `-e <max size in bytes>` maximum size to end at. Default: 1G.
  * `-f <increment factor>` multiplication factor between sizes. Default: 2.
* Performance
  * `-w, <warmup iteration count>` number of warmup iterations (not timed). Default: 5.
  * `-n, <iteration count>` number of iterations. Default: 20.
* Test Operation
  * `-R, <0/1>` enable local buffer registration on send/recv buffers. Default: 0.
  * `-s, <OCT/DEC/HEX>` specify MPI communication split mode. Default: 0
* Utils
  * `-p, <0/1>` print buffer info. Default: 0.
  * `-h` print help message. Default: disabled.

### Training Models
After building and testing FlagCX, you can start training models using upper-layer deep learning frameworks such as PyTorch or PaddlePaddle with FlagCX as communication backend. We provide detailed user guides for both **homogeneous** and **heterogeneous** training across different hardware platforms. Please refer to the docs below:  
- [Training Models with PyTorch and FlagCX](docs/user_guide.md).
- [Training Models with Paddle and FlagCX](docs/paddle/README.md).

## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagCX/blob/main/LICENSE).
