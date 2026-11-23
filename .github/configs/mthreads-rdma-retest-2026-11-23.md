## 2026.11.23 FlagCX 摩尔线程 MUSA RDMA 复测与问题记录

### 结论与复测环境

**结论：** 在同一台摩尔线程节点上，容器内执行的 CI 入口通过了 P2P WRITE MPI、RMA IPC 数据、RMA NET 读、Runner Hybrid P2P、Runner 强制 NET、Adaptor GPU MR 和普通 SymMem IPC/NET。原先主要失败在 MUSA 适配层缺少 GPU 指针识别、IPC 映射和 GPU MR 声明；CI 另有 GDR 可见性执行 0 个用例却返回成功的问题。

**环境：** 单节点 8 张 MTT S5000，测试使用 4 张；MUSA 4.3.6，驱动 3.3.5，`mt_peermem` 1.0。使用 `harbor.baai.ac.cn/flagos-dev/flagcx:mthreads-dev-fixed-20261022`，按 CI 配置用 `--runtime=mthreads --network=host --privileged --ipc=host --shm-size=32g` 启动 Docker 容器。

**当前网卡配置：** `FLAGCX_SOCKET_IFNAME` 默认排除 `lo,docker,br-,veth`，保留显式覆盖；`FLAGCX_IB_HCA` 未设置时从 `/sys/class/infiniband` 发现活动端口。此节点发现 11 个 `ACTIVE/LinkUp` RoCE 端口，`bond0` 是本次自动选择结果。

**原始日志：** 工作区 `.ci-logs-rdma-rerun/` 保存修复前日志及 GDB 栈；`.ci-logs-rdma-fix/` 保存修复后的 `*-final.log`。

### P2P：GPU 指针分类与 WRITE MPI

**修复前日志：**

```text
flagcx/core/flagcx_p2p.cc:3277 FLAGCX WARN P2P Reg: failed to classify addr 0x1000ca00000: 9
[rank 1] P2P WRITE: registering GPU buffer failed
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD
```

**GDB 调用栈（`p2p-gdb.log`，由内到外）：**

```text
#0 flagcxDeviceAdaptorGetPointerTypeNotSupported   flagcx_device_adaptor.h:464
#1 flagcxP2pDetectPointerType                      p2p_pointer.cc:45
#2 operator()                                     flagcx_p2p.cc:3237
#3 flagcxP2pEngineRegEx                            flagcx_p2p.cc:3274
#4 flagcxP2pEngineReg                              flagcx_p2p.cc:3399
#5 main                                          coll_p2p_engine_write.cpp:119
```

**修复与结果：** 接入 MUSA 指针分类和分配区间查询。`p2p-final.log`：111 个单测通过，同节点 RPC 建连 112 ms；2-rank WRITE MPI 的单次和向量传输数据均一致。

```text
FLAGCX INFO Bootstrap : Using bond0:10.124.33.20<0>
[       OK ] P2pEngineRpcTransportTest.ConnectAcceptIsLocalSameHost (112 ms)
[  PASSED  ] 111 tests.
P2P Engine WRITE correctness: single and vector transfers passed
```

### RMA：IPC 映射、NET 与 GDR 可见性

**修复前日志：**

```text
flagcx/adaptor/flagcx_device.cc:798 FLAGCX WARN buildIpcPeerPointers: ipcMemHandleCreate failed
RMA data window is unavailable
[==========] Running 0 tests from 0 test suites.
```

**GDB 调用栈（`rma-ipc-gdb.log`，由内到外）：**

```text
#0 musaAdaptorIpcMemHandleCreate       musa_adaptor.cc:247
#1 buildIpcPeerPointers                flagcx_device.cc:796
#2 flagcxCommWindowRegisterIpcInternal flagcx_device.cc:990
#3 flagcxCommWindowRegisterFresh       flagcx.cc:2742
#4 flagcxCommWindowRegister            flagcx.cc:2911
#5 RmaTest::SetUpTestSuite             rma_test.cpp:146
```

**修复与结果：** 实现 MUSA IPC handle 的创建、导出、导入、关闭和释放，接入普通 64 位 stream 写入和等待。`rma-final.log`：54 个 host 单测通过；两 rank 各通过 4 个 IPC 数据用例、5 个 NET 用例，并在独立的可见性阶段各执行并通过 2 个用例。补上 MUSA consumer kernel 后，CI 会将普通可见性阶段无通过用例视为失败。

```text
IPC Get same-stream: rank=1 stage=flagcxGetAsync result=0
IPC Get same-stream: rank=1 stage=flagcxWaitCounter result=0
[       OK ] RmaTest.IpcGetSameStreamCompletion (0 ms)
[       OK ] RmaTest.DirectConsumerReadVisibility (1 ms)
[       OK ] RmaTest.DirectConsumerWriteVisibility (0 ms)
```

**仍跳过：** `PutSignalSmall`、`PutSignalLarge`、`SignalOnlyNoData`、`MultipleSignals` 提示 `Remote-write visibility flush is unavailable on at least one rank`。SDK 实测普通 `muStreamWriteValue64` 和 `muStreamWaitValue64` 返回 0；`CAN_FLUSH_REMOTE_WRITES=0`，带 `FLUSH` 的等待返回 1。`DirectConsumerReadVisibility` 由 GPU kernel 校验数据；`DirectConsumerWriteVisibility` 验证 public wait 返回 `flagcxNotSupported`，不代表 GPU 已实现远端 WRITE 后的立即读取。普通 NET 阶段另有 3 个仅 IPC 用例和 2 个独立可见性用例按模式跳过，均在各自阶段执行。

### Runner：Hybrid P2P 与强制 NET

**修复前日志（跨线程错误传播，非单一 GDB 栈）：**

```text
flagcx/core/p2p.cc:867 -> 9
flagcx/core/proxy.cc:1349 -> 9
flagcx/runner/c2c_algo.cc:2543 -> 9
flagcx/runner/hybrid_runner.cc:121/253 -> 9
```

**线程诊断：** 原 Hybrid P2P 的 `AllGather`、`AllReduce`、`AlltoAll`、`AlltoAllvAsymmetric` 失败。`runner_heterogeneous.NLXzEY/attempt-1-stacks.log` 捕获了 `futex_wait_queue_me`、`do_poll.constprop.0` 和 `ib_uverbs_event_read.constprop.0`。两次尝试在约 199 s 和 181 s 后被本地观察 watchdog 中断；线程快照不足以判断具体的阻塞点。

**修复与结果：** IPC 映射恢复后，`runner-final.log` 显示 102 个 host 单测通过；4 rank 每 rank 默认 10 个通过，Hybrid P2P 和强制 NET 各 11 个通过，强制 NET 多 QP 条带化 1 个通过。默认阶段按模式跳过 `HybridPlannerRejectsOuterGroup`，异构阶段执行该用例。

### Adaptor：GPU MR 与本地 stream 信号

**修复前日志：** `Selected net adaptor does not advertise GPU MR`。IBRC 没有识别 MUSA 的 peer memory 路径，GPU MR 校验无法通过。

**中间复测问题：** `DeviceAdaptorTest.StreamWaitWriteValue64` 在 `test_device_adaptor.cpp:588` 读到 `actual=0`，期望 `0x123456789abcdef0`。清零在默认 stream 异步执行，测试的写在非阻塞 stream 上，二者可能交错。测试在初始化后增加 device 同步。

**修复与结果：** IBRC 发现 `mt_peermem` 时声明 GPU MR，CI 要求实际 GPU 注册和数据校验。`adaptor-final.log` 中 90 个单测通过，包括 `RegisterGpuMr`、`GpuIput`、`GpuIget`、`GpuBatchPutAndGet`、`GpuBounds`、`GdrGpuIputAndIget`；2 rank 各通过 15 个跨进程或跨卡 IPC 用例。`StreamWaitWriteValue64` 单独重复 25 轮均通过。

**仍跳过：** 平台配置保留 `DeviceAdaptorTest.StreamWaitEvent` 排除项。远端 flush、host memory 映射和 DMA-BUF 等由各测试的能力条件决定；IBUC、CUDA 专用的连接几何不匹配测试不属于本次 IBRC 覆盖。

### SymMem：普通 IPC/NET 与 VMM 路由

**修复前日志：** 即使 RoCE 建链正常，MUSA 没有 VMM 数据映射和 MR 路径。

```text
neither DMA-BUF nor VA can register a VMM allocation
```

**复测结果：** `symmem-final.log`：68 个 host 单测通过；本地 IPC 每 rank 34 个通过、IPC/NET fallback 每 rank 2 个通过、四 rank IPC+NET 每 rank 9 个通过；四 rank 的 `SymMemTest.VmmNetRouteCapabilityUnion` 均通过，验证了无 VMM 注册路由时正确拒绝。

**仍跳过：** VMM 本地/remote-without-NET、VMM+NET 数据和 VMM flat-map 数据。保留普通 IPC/NET 数据阶段及 VMM 拒绝契约，取消原有的 SymMem MPI 整体跳过。

### 其他未恢复的覆盖

**平台级：** `device_api`、`device_api_unified_ir` 缺少 MUSA 的 Device API kernel 和 unified IR 后端；`device_api_host` 仍启用。Shared P2P engine、IBUC 和 PXN 仍按原平台能力配置关闭。本次复测为单节点容器，未验证跨物理节点或其他芯片。
