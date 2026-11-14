# MCCL 原生 Stream Value64 路径移植到 FlagCX

日期：2026-11-12。修改位置为 `D:\LLM\FlagCX\FlagCX`，保持
`DefaultBackend<MusaPlatform>`。本地完成源码移植和静态检查；用户上传到 Linux
服务器后，已重建主库/adaptor 测试并回传真实 GPU 测试结果：2 PASS、1 SKIP。
本机未执行 MUSA SDK 编译或 GPU 测试。

## 1. 复用来源

参考仓库：`D:\LLM\FlagCX\mccl`，commit
`23edec5afda741efe0119410000dd80fd03092ef`。

| MCCL 源码 | 复用的做法 |
| --- | --- |
| `src/ce_coll.cc` | 通过 `MU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS` 查询支持；用完整 `value64` 和 `MU_STREAM_WRITE_VALUE_DEFAULT` 发布信号 |
| `src/rma/rma_ce.cc` | 用 `MU_STREAM_WAIT_VALUE_GEQ` 等待序号；查询 `MU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES` |
| `src/include/musawrap.h`、`src/misc/musawrap.cc` | MCCL 使用的原生 `muStreamWaitValue64` / `muStreamWriteValue64` 入口 |

MCCL 的 RMA 路径通过 batch MemOp 表达部分等待与写入，原生单操作入口也已
在其 wrapper 中声明。FlagCX 的单次接口直接调用同一组 Driver 原语，不需要
引入 MCCL communicator 或改选原生 MCCL backend。

本轮没有自行实现自旋 Kernel、Atomic/Fence 代用品或 Host callback；也没有
修改 B 的 IPC Handle、注册、映射和释放实现。

## 2. 代码改动

### `flagcx/adaptor/device/musa_adaptor.cc`

原来两个接口均直接返回 NotSupported，现在提供原生实现：

- `musaAdaptorStreamWriteValue64`：检查参数、8 字节对齐及写 flags，检查当前
  device 的 64-bit Stream MemOp 能力，然后用 `MU_STREAM_WRITE_VALUE_DEFAULT`
  调用 `muStreamWriteValue64`，保留原生 stream-scoped 写前内存屏障。
- `musaAdaptorStreamWaitValue64`：检查参数、对齐及允许的 flags，检查 64-bit
  Stream MemOp 能力，使用 `MU_STREAM_WAIT_VALUE_GEQ` 调用
  `muStreamWaitValue64`。
- 请求 `FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES` 时，先查询 Flush 能力，
  确认支持后才加入 `MU_STREAM_WAIT_VALUE_FLUSH`。不支持时不入队弱化版等待，
  直接返回 `flagcxNotSupported`。
- 未知能力属性或原生 NotSupported 映射为 `flagcxNotSupported`；其他设备/
  context 错误映射为设备错误。非法外部参数返回 `flagcxInvalidArgument`。

能力检查针对调用时的当前 device，不用全局缓存将某张 GPU 的结果误用到
另一张 GPU。调用方应像其他 Driver adaptor 一样，保证当前 context 与
stream、地址所属/可访问的设备关系一致。

### `test/unittest/adaptor/test_device_adaptor.cpp`

保留已有 equality/full-width 和 Flush capability 测试，新增
`DeviceAdaptorTest.StreamWaitValue64GreaterThan`：先写入大于 `2^32` 的完整
64-bit 值，再等待一个严格较小的阈值，检查拷回值。它能区别 GEQ 与 EQ，
也验证高 32 位未被写入路径截断；不替代负向等待、跨 GPU 或 NIC 可见性测试。

## 3. 原生 GEQ 的边界

SDK 的 `MU_STREAM_WAIT_VALUE_GEQ` 使用循环序号比较：

```text
(int64_t)(*addr - expected) >= 0
```

它适用于相对序号距离小于 `2^63` 的计数协议，与 MCCL 使用的 Driver GEQ
路径一致。它不是对任意跨度两个 uint64_t 值的普通无符号全序比较。特别是
不能把初值 0 与超过半个序号空间的 expected 直接作为合法等待协议。

如果 B 要求的是任意 uint64_t 范围的数学 `>=`，需要先明确协议，不能把这个
原生接口宣称为完全等价。本轮测试使用合法的序号距离，仍需补负向等待和
跨 GPU 实验后再确认前向进度和可见性。

## 4. 上传文件

为保留 B 对同一 Runtime 文件的其他改动，优先上传本轮增量补丁：

```text
本地：D:\LLM\FlagCX\FlagCX\docs\patches\musa_stream_value64_native.patch
服务器：/tmp/musa_stream_value64_native.patch
```

在服务器源码目录先检查再应用：

```bash
cd /workspace/FlagCX
git apply --check /tmp/musa_stream_value64_native.patch &&
git apply /tmp/musa_stream_value64_native.patch
```

补丁只涉及下表两个文件。检查不通过时保留现场并反馈，不强行覆盖或部分应用。
若已单独同步源码完成同一修改，则不再重复应用补丁。

完整文件的对应路径为：

| 本地完整路径 | 服务器完整路径 |
| --- | --- |
| `D:\LLM\FlagCX\FlagCX\flagcx\adaptor\device\musa_adaptor.cc` | `/workspace/FlagCX/flagcx/adaptor/device/musa_adaptor.cc` |
| `D:\LLM\FlagCX\FlagCX\test\unittest\adaptor\test_device_adaptor.cpp` | `/workspace/FlagCX/test/unittest/adaptor/test_device_adaptor.cpp` |

如果服务器上的 `musa_adaptor.cc` 已有 B 未上传的改动，只合入 Stream Value64
相关 helper 和两个接口的增量，不要整文件覆盖 B 的 Runtime/IPC 成果。

## 5. 服务器构建与定向验证

```bash
cd /workspace/FlagCX
export FLAGCX_SRC=/workspace/FlagCX
export FLAGCX_PATH="$FLAGCX_SRC"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH"

timeout --kill-after=5s 600s \
  make -j1 USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/unittest/adaptor all \
  USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

构建成功后，单独执行以下测试并保留 GTest 的 PASS/FAIL/SKIP：

```bash
timeout --kill-after=5s 120s \
  test/unittest/adaptor/build/bin/adaptor_unit_tests \
  --gtest_filter='DeviceAdaptorTest.StreamWaitWriteValue64:DeviceAdaptorTest.StreamWaitValue64GreaterThan:DeviceAdaptorTest.StreamWaitValue64RemoteFlushCapability'
```

若硬件/驱动不支持，测试可能 SKIP。这表示该语义没有通过，不能用进程正常
退出代替功能支持。定向用例通过也只说明基础 Stream Value64 与能力分支，
不能据此宣布跨 GPU System-scope 或 NIC Flush 已验收。

## 6. 服务器验证结果

用户在 S5000 / mp_31 环境执行本说明中的命令，主库、adaptor unit/MPI 测试
程序均构建成功。定向执行的三个测试结果如下：

| 测试 | 结果 | 可交付结论 |
| --- | --- | --- |
| StreamWaitWriteValue64 | PASS，231 ms | 当前测试覆盖的完整 64-bit 写入、相等阈值等待及读回通过 |
| StreamWaitValue64GreaterThan | PASS，1 ms | 发布值严格大于阈值时 GEQ 等待完成，且大于 2^32 的写入值完整读回 |
| StreamWaitValue64RemoteFlushCapability | SKIP，1 ms；`remote-write visibility flush is not supported` | 当前适配路径返回 NotSupported，没有把 Flush 静默降级为普通等待；未证明远端写可见性 |

GTest 汇总为 3 tests、2 PASSED、1 SKIPPED。SKIP 不算 Flush 能力通过。
现有输出不能区分“能力位为 0 / 属性查询不支持 / 原生调用拒绝 Flush”中的
具体哪一种，也不能据此宣称 S5000 在所有驱动/transport 场景都不支持 Flush。

给 B 的当前口径：两个基础 Stream Value64 接口已移植并取得定向 PASS；
跨 GPU System-scope 及 NIC/GDR Remote Flush 尚未验证。若调用方要求
FLUSH_REMOTE_WRITES，必须处理 NotSupported，不能据基础测试 PASS 将请求
改为 flags=0 后继续宣称原语语义等价。

## 7. 后续仍需验证

1. 双 stream 的负向 GEQ：只更新低 32 位、值仍低于目标时不能提前放行；另一个
   stream 发布真正的完整 64 位目标后应恢复执行。
2. GPU0→GPU1、GPU1→GPU0 的 Payload→Signal→等待→Payload，核对 stream
   前后顺序和多轮运行。
3. Flush 支持与不支持两条能力分支；支持路径检查后续工作是否看到真实远端
   payload，不支持路径保持 NotSupported。
4. GDR/NIC 写入需在对应 transport 和资源上做实验，普通 fence 或本地值读回
   不作为其证明。

本轮未提交或推送，未修改已经发布的 A 分支和 B 的未公开工作区。

## 8. 新增阻塞与跨 GPU 验证（待服务器运行）

新增 `test/unittest/musa_platform/musa_stream_value64.cc` 和同目录 Makefile
的 `stream-value64` 目标。程序使用主库的 `deviceAdaptor` 接口，不复制生产
Stream Write/Wait 实现，也不增加软件等待 Kernel。Peer Access 查询及启用
沿用 MCCL `src/transport/p2p.cc` 的原生调用方式。

普通模式覆盖：

- 双 stream 负向 GEQ：初值高 32 位小于目标，但低 32 位为 33、大于目标的
  17；观察 consumer 100 ms 不完成，再由 producer 发布严格大于目标的值。
  consumer 等待后的 D2D 快照必须读到完整发布值。
- GPU0→GPU1 和 GPU1→GPU0，每方向 32 轮、每轮 256 个 64 位 payload。
  producer stream 将本地 payload 拷贝到 peer 显存，再向 peer signal 写值；
  consumer stream 等待本地 signal 后拷贝 payload 到快照并核对。交替使用
  相等及严格大于阈值；payload 发布到快照执行之间不插入 host 同步。

`--flush` 模式对两个方向单独要求 `FLUSH_REMOTE_WRITES`，能力预检返回
NotSupported 时记 SKIP，不降级。普通模式通过只代表测试覆盖的 P2P 拷贝引擎
路径，不能替代 GPU Kernel 的 system-scope Atomic/Fence 证明，也不能证明
NIC/GDR flush。即使 Flush 模式通过，也仅覆盖本次 P2P payload 来源。

完整 PASS 返回 0；出现 SKIP 返回 77；API 错误或数据不匹配返回 1。
异步等待故障可能使同步调用卡住，所以必须使用外部 timeout。程序在 API
失败时立即结束进程，避免清理未满足的等待时再次卡住；正常路径释放分配与
stream。测试仅使用 GPU0/GPU1，不覆盖其余拓扑或跨进程 IPC。

本次新增需同步的两个文件（文档无需上传）：

| 本机源文件 | 服务器目标文件 |
| --- | --- |
| `D:\LLM\FlagCX\FlagCX\test\unittest\musa_platform\musa_stream_value64.cc` | `/workspace/FlagCX/test/unittest/musa_platform/musa_stream_value64.cc` |
| `D:\LLM\FlagCX\FlagCX\test\unittest\musa_platform\Makefile` | `/workspace/FlagCX/test/unittest/musa_platform/Makefile` |

前提是服务器保留上轮已通过基础测试的 `musa_adaptor.cc` 和已重建的主库。
构建新诊断不会重新编译原有 trap/Coop 探针：

```bash
cd /workspace/FlagCX
export LD_LIBRARY_PATH=/workspace/FlagCX/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH

timeout --kill-after=5s 300s \
  make -C test/unittest/musa_platform stream-value64 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

构建成功后分别运行，不使用 `&&` 串接两种模式，因为 SKIP 返回 77：

```bash
timeout --kill-after=5s 120s \
  test/unittest/musa_platform/build/musa_stream_value64
echo "default mode exit=$?"

timeout --kill-after=5s 120s \
  test/unittest/musa_platform/build/musa_stream_value64 --flush
echo "flush mode exit=$?"
```

新增程序随后由用户上传到服务器完成构建和真实 GPU 执行，结果见下一节。

## 9. 阻塞与跨 GPU 验证结果

用户在 worker34079 的 `/workspace/FlagCX` 构建成功并回传以下结果：

| 验证项 | 结果 |
| --- | --- |
| 完整 64-bit GEQ、低于阈值阻塞、独立 stream 发布后推进 | PASS |
| GPU0→GPU1，默认等待的 P2P payload 发布 | PASS，32 轮，每轮 256 个 64-bit word，覆盖相等及严格大于阈值 |
| GPU1→GPU0，默认等待的 P2P payload 发布 | PASS，同上 |
| GPU0→GPU1，要求 Remote Flush | SKIP，adaptor returned NotSupported |
| GPU1→GPU0，要求 Remote Flush | SKIP，adaptor returned NotSupported |

默认模式为 `3 PASS / 0 SKIP`，退出码 0。Flush 模式为
`0 PASS / 2 SKIP`，退出码 77。这表示默认 P2P 拷贝引擎路径的实际发布顺序及
payload 读回已通过；Flush 要求仍不可用，适配层没有静默降级。

当前日志不能区分 Flush 能力位为 0、属性查询不支持和原生操作不支持，不能
据此断言所有硬件/驱动组合均不支持。B 可接入基础 Stream Write64/GEQ Wait，
但要求 FLUSH_REMOTE_WRITES 的路径必须继续处理 NotSupported，不能当作已经
满足远端 acquire/flush 保证。GPU Kernel 的 system-scope Atomic/Fence、
NIC/GDR 写入及跨进程 IPC 不在本次验证覆盖范围内。
