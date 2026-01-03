
[<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/31746439-e7b0-4391-8418-2f3597a88141" />](https://www.flagopen.ac.cn/)

## About

FlagCX is part of [FlagOS](https://flagos.io/), a unified, open-source AI system software stack that
aims to foster an open technology ecosystem by seamlessly integrating various models, systems and chips.
By "develop once, migrate across various chips", FlagOS aims to unlock the full computational potential
of hardware, break down the barriers between different chip software stacks, and effectively reduce
migration costs.

[FlagCX](https://github.com/flagos-ai/FlagCX.git) is a scalable and adaptive cross-chip communication library.
It serves as a platform where developers, researchers, and AI engineers can collaborate on various projects,
contribute to the development of cutting-edge AI solutions, and share their work with the global community.

FlagCX leverages native collective communication libraries to provide full single-chip communication support across platforms.
Beyond its native x-CCL integrations, FlagCX introduces original _device-buffer IPC_ and _device-buffer RDMA_ technologies,
enabling high-performance P2P operations for both cross-chip and single-chip scenarios.
These mechanisms can be seamlessly combined with native x-CCL backends to deliver optimized performance
for cross-chip collective communications.

## Backend Support

The following table summarizes the currently supported communication backends and their corresponding capabilities.

| Backend       | NCCL        | IXCCL       | CNCL        | MCCL        | XCCL        | DUCCL       | HCCL        | MUSACCL     | RCCL        | TCCL        |
|:--------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|
| Mode          | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero | Homo/Hetero |
| send          | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| recv          | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| broadcast     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| gather        | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ☓/☓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| scatter       | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| reduce        | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| allreduce     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| allgather     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| reducescatter | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| alltoall      | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| alltoallv     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |
| group ops     | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/✓         | ✓/☓         | ✓/✓         | ✓/✓         | ✓/✓         |

Note that *Homo* and *Hetero* modes refer to communications among homogeneous and heterogeneous clusters.
All native collective communications libraries can be referenced through the links below (in alphabetic order):

- [CNCL](https://www.cambricon.com/docs/sdk_1.7.0/cncl_1.2.1/user_guide/index.html#), Cambricon Communications Library.
- [DUCCL](https://developer.sourcefind.cn), DU Collective Communications Library.
- [HCCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/hccl/hcclug/hcclug_000001.html), Ascend Communications Library.
- [IXCCL](https://www.iluvatar.com/software?fullCode=cpjs-rj-rjz), Iluvatar Corex Collective Communications Library.
- [MCCL](https://developer.metax-tech.com/softnova/metax), Metax Collective Communications Library.
- [MUSACCL](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/Chapter08/), Musa Collective Communications Library.
- [NCCL](https://github.com/NVIDIA/nccl), NVIDIA Collective Communications Library.
- [RCCL](https://github.com/ROCm/rccl), ROCm Communication Collectives Library.
- [TCCL](http://www.tsingmicro.com), TsingMicro Communication Collectives Library.
- [XCCL](WIP), Kunlunxin XPU Collective Communications Library.

Additionally, FlagCX supports three collective communication libraries for host-side communication:

- BOOTSTRAP: Host-side communication library built using the FlagCX `bootstrap` component.
- [GLOO](https://github.com/facebookincubator/gloo): Gloo Collective Communications Library.
- [MPI](https://www.mpich.org): Message Passing Interface (MPI) standard.

### Application Integration

FlagCX integrates with upper-layer applications such as [PyTorch](https://pytorch.org/) and
[PaddlePaddle](https://github.com/PaddlePaddle/).
The table below lists the frameworks supported by FlagCX and their related communication operations,
where the `batch_XXX` and `XXX_coalesced` ops refer to the usage of group primitives.

| Framework                        | PyTorch                      | PaddlePaddle |
| :------------------------------- | :--------------------------- | :----------- |
| send                             | ✓                            | ✓            |
| recv                             | ✓                            | ✓            |
| all_gather                       | ✓                            | ✓            |
| all_gather_into_tensor_coalesced | ✓ (in order, no aggregation) | ☓            |
| all_reduce                       | ✓                            | ✓            |
| all_reduce_coalesced             | ✓ (in order, no aggregation) | ☓            |
| all_to_all                       | ✓                            | ✓            |
| all_to_all_single                | ✓                            | ✓            |
| barrier                          | ✓                            | ✓            |
| batch_isend_irecv                | ✓                            | ✓            |
| broadcast                        | ✓                            | ✓            |
| gather                           | ✓                            | ✓            |
| reduce                           | ✓                            | ✓            |
| reduce_scatter                   | ✓                            | ✓            |
| reduce_scatter_tensor_coalesced  | ✓ (in order, no aggregation) | ☓            |
| scatter                          | ✓                            | ✓            |

Note that PyTorch support is enabled via the FlagCX Torch plugin, which provides native integration with the PyTorch distributed backend.
This plugin has undergone comprehensive validation across diverse communication backends and hardware platforms,
ensuring robust functionality, consistent performance, and compatibility in multi-chip heterogeneous environments.

| FlagCX Backend  | NCCL | IXCCL | CNCL | MCCL | XCCL | DUCCL | HCCL | MUSACCL | RCCL |
| :-------------- | :--- | :---- | :--- | :--- | :--- | :---- | :--- | :------ | :--- |
| PyTorch Support | ✓    | ✓     | ✓    | ✓    | ✓    | ✓     | ✓    | ✓       | ✓    |

> [!TIP]
> To enable heterogeneous cross-chip communication using the PyTorch DDP FlagCX backend,
> it is recommended to use identical PyTorch versions across all nodes.
> Mismatched versions may lead to initialization failures during process group setup.
> Helpful advice for doing things better or more easily.

## Quick Start

Please check the guides on building, testing the software:

<!--TODO(Qiming): Rework the page structure and then the list below.-->
- [Getting started](./docs/getting_started.md)
- [Build](./docs/build.md)
- [Testing](./docs/testing.md)


### Training Models

After building and testing FlagCX, you can start training models using upper-layer deep learning frameworks
such as [PyTorch](https://pytorch.org) or [PaddlePaddle](https://github.com/PaddlePaddle)
using FlagCX as the communication backend.
We provide detailed user guides for both **homogeneous** and **heterogeneous** training across different hardware platforms.
Please refer to the docs below:  

<!--TODO(Qiming): Revise the directory layout-->
- [Training Models with PyTorch and FlagCX](./docs/user_guide.md).
- [Training Models with Paddle and FlagCX](./docs/paddle/README.md).

## Contribution

<!--TODO(Qiming): Add link to CONTRIBUTING page-->
- We warmly welcome community contributions to help expand and strengthen the validation matrix.

- Join our Discussion Channel

  <img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/af9f98be-8176-4039-be4a-7f5b15513ff1" />

## License

This project is licensed under the [Apache License (Version 2.0)](./LICENSE).


