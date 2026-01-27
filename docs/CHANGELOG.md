## Release History

- **[2025/11]** Released [v0.7](https://github.com/flagos-ai/FlagCX/releases/tag/v0.7.0):

  - Added support to TsingMicro, including device adaptor `tsmicroAdaptor` and CCL adaptor `tcclAdaptor`.
  - Implemented an experimental kernel-free non-reduce collective communication
    (*SendRecv*, *AlltoAll*, *AlltoAllv*, *Broadcast*, *Gather*, *Scatter*, *AllGather*)
    using device-buffer IPC/RDMA.
  - Enabled auto-tuning on NVIDIA, MetaX, and Hygon platforms, achieving 1.02×–1.26× speedups for
    *AllReduce*, *AllGather*, *ReduceScatter*, and *AlltoAll*.
  - Enhanced `flagcxNetAdaptor` with one-sided primitives (`put`, `putSignal`, `waitValue`) and added retransmission support for reliability improvement.

- **[2025/10]** Released [v0.6](https://github.com/flagos-ai/FlagCX/releases/tag/v0.6.0):

  - Implemented device-buffer IPC communication to support intra-node *SendRecv* operations.
  - Introduced _device-initiated, host-launched device-side primitives_, enabling kernel-based communication directly from devices.
  - Enhanced auto-tuning with 50% performance improvement on MetaX platforms for the *AllReduce* operations.

- **[2025/09]** Released [v0.5](https://github.com/flagos-ai/FlagCX/releases/tag/v0.5.0):

  - Added support for AMD GPUs, including a device adaptor `hipAdaptor` and a CCL adaptor `rcclAdaptor`.
  - Introduced `flagcxNetAdaptor` to unify network backends, currently supporting socket, IBRC, UCX and IBUC (experimental).
  - Enabled zero-copy device-buffer RDMA (user-buffer RDMA) to boost performance for small messages.
  - Supported auto-tuning in homogeneous scenarios via `flagcxTuner`.
  - Added test automation in CI/CD for PyTorch APIs.

- **[2025/08]** Released [v0.4](https://github.com/flagos-ai/FlagCX/releases/tag/v0.4.0):

  - Supported heterogeneous training of ERNIE4.5 (Baidu) on NVIDIA and Iluvatar GPUs with Paddle + FlagCX.
  - Improved heterogeneous communication across arbitrary NIC configurations,
    with more robust and flexible deployments.
  - Introduced an experimental network plugin interface with extended supports for IBRC and SOCKET.
    Device buffer registration now can be done via DMA-BUF.
  - Added an InterOp-level DSL to enable customized C2C algorithm design.
  - Provided user documentation under `docs/`.

- **[2025/07]** Released [v0.3](https://github.com/flagos-ai/FlagCX/releases/tag/v0.3.0):

  - Integrated three additional native communication libraries: HCCL (Huawei), MUSACCL (Moore Threads) and MPI.
  - Enhanced heterogeneous collective communication operations with pipeline optimizations. 
  - Introduced _device-side_ functions to enable device-buffer RDMA, complementing the existing _host-side_ functions.
  - Delivered a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous prefilling-decoding disaggregation.

- **[2025/05]** Released [v0.2](https://github.com/flagos-ai/FlagCX/releases/tag/v0.2.0):

  - Integrated 3 additional native communications libraries, including MCCL (Moore Threads), XCCL (Mellanox) and DUCCL (BAAI).
  - Improved 11 heterogeneous collective communication operations with automatic topology detection and
    full support to single-NIC and multi-NIC environments.

- **[2025/04]** Released [v0.1](https://github.com/flagos-ai/FlagCX/releases/tag/v0.1.0):

  - Added 5 native communications libraries including CCL adaptors for 
    NCCL (NVIDIA), IXCCL (Iluvatar), and CNCL (Cambricon), 
    and Host CCL adaptors GLOO and Bootstrap.
  - Supported 11 heterogeneous collective communication operations using the C2C (Cluster-to-Cluster) algorithm.
  - Provided a full-stack open-source solution, FlagScale + FlagCX, for efficient heterogeneous training.
  - Natively integrated into PaddlePaddle [v3.0.0](https://github.com/PaddlePaddle/Paddle/tree/v3.0.0),
    with support for both dynamic and static graphs.


