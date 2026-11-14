# MUSA counted barrier 编译异常记录

记录日期为 2026-11-11。异常发生于 Linux 服务器的 Device / Kernel 适配验证。

## 已知环境与现象

| 项目 | 已知信息 |
| --- | --- |
| 设备 | 交接记录为 MTT S5000，目标架构 `mp_31` |
| mcc | `5.1.0` |
| clang | `14.0.0` |
| 工具链 commit | `39eee21af373855ae51b0e906a74b02950384a05` |
| Target | `x86_64-unknown-linux-gnu` |
| InstalledDir | `/usr/local/musa/bin` |
| 触发操作 | `__barrier_sync_count(1, 64)`，对应 MCCL `mcclCoopWarpSpan(0, 2, 0)` |
| 观察结果 | 重复输出 `llvm.musa.barrier.sync.cnt` DAG 节点，编译未取得成功结果 |

用户提供的重复输出包含以下节点。

```text
ch = llvm.musa.barrier.sync.cnt ...,
     TargetConstant:i64<5252>, Constant:i32<1>, Constant:i32<64>
```

用户后续提供了 `named-barrier-repro` 的实际编译命令及日志开头，确认只包含 SDK runtime 的独立 kernel 仍触发相同输出。异常已独立于 FlagCX PlatformTraits、Atomic/Fence 适配和通信后端复现，当前定位到工具链对计数 barrier 相关代码的处理。

当前尚缺普通 barrier 对照的结果和计数 barrier 的最终退出码。它们应在 SDK 问题记录中补充，现有独立复现已可用于工具链侧定位。具体缺陷及修复方式仍待 SDK 侧确认。

实际复现命令如下，工作目录为 `/workspace/FlagCX/test/unittest/musa_platform`。

```bash
/usr/local/musa/bin/mcc -std=c++17 -O2 -c -mtgpu \
  --musa-path=/usr/local/musa --offload-arch=mp_31 \
  -fgpu-rdc -fPIC -MMD -MP -g -DFLAGCX_MUSA_DEVICE_COMPILE=1 \
  -I/usr/local/musa/include -DFLAGCX_MUSA_NAMED_BARRIER_REPRO=1 \
  -o /workspace/FlagCX/test/unittest/musa_platform/build/musa_named_barrier_repro.o \
  musa_named_barrier_repro.cu
```

## 独立复现

源码为 `test/unittest/musa_platform/musa_named_barrier_repro.cu`，包含 `musa_runtime.h`，编译同一个 64-thread kernel。宏值 0 使用 `__syncthreads()`，宏值 1 使用 `__barrier_sync_count(1, 64)`。共享内存写入后跨 warp 读取，使同步具有实际用途。[复现源码](../test/unittest/musa_platform/musa_named_barrier_repro.cu)

在 Linux 服务器执行以下命令。日志大小和时长限制用于约束异常编译的输出与运行时间，退出码需要连同限制一起记录。

```bash
cd /workspace/FlagCX
repro_dir=/workspace/FlagCX/test/unittest/musa_platform

control_status=0
(
  ulimit -f 8192
  timeout --signal=TERM --kill-after=5s 15s \
    make -C "$repro_dir" -B barrier-control \
    > /tmp/musa_barrier_control.log 2>&1
) || control_status=$?
printf 'control exit=%s\n' "$control_status"
head -c 4096 /tmp/musa_barrier_control.log
printf '\n'

named_status=0
(
  ulimit -f 8192
  timeout --signal=TERM --kill-after=5s 15s \
    make -C "$repro_dir" -B named-barrier-repro \
    > /tmp/musa_named_barrier_build.log 2>&1
) || named_status=$?
printf 'counted barrier exit=%s\n' "$named_status"
head -c 8192 /tmp/musa_named_barrier_build.log
printf '\n'
/usr/local/musa/bin/mcc --version
```

重复 DAG 输出可能没有换行。使用 `head -c` 限制显示字节数，避免整条超长日志再次刷屏。

## 结果判断与 SDK 侧问题

- 当前计数 barrier 已在独立源码中复现异常，可将源码、精确命令、初始日志和工具链版本交给 MUSA SDK 侧定位，并补充对照结果与退出码。
- 在修复工具链或原语实现后，先重编独立复现，再验证完整平台探针及计数 barrier 的同步语义。
- 超时或文件大小限制终止属于诊断结果，不能记为编译通过。

请 SDK 侧确认该 mcc/SDK/`mp_31` 组合对计数 named barrier 的支持范围，并提供对应修复、可用工具链版本或已验证的等价原语映射。

可向 SDK 维护方提交以下问题描述。

> 在 S5000 目标架构 `mp_31` 上，mcc 5.1.0 / clang 14.0.0、工具链 commit `39eee21af373855ae51b0e906a74b02950384a05` 编译附件独立 kernel 时，`__barrier_sync_count(1, 64)` 导致 LLVM 后端持续输出相同的 `llvm.musa.barrier.sync.cnt` DAG 节点，无法取得成功产物。复现源码只包含 `musa_runtime.h`，具体编译命令见上文。该操作来自 MCCL `mcclCoopWarpSpan`。请确认当前工具链的支持范围和缺陷情况，并提供修复版本或满足参与线程数、组隔离及内存可见性语义的官方实现。

## 平台映射修正

当前生产适配按实际参与范围选择同步原语。

| 参与范围 | 实现与状态 |
| --- | --- |
| 合法的完整单 warp | `CoopTileSpan::sync()` 使用 `__syncwarp(fullMask)` |
| 从 warp 0 开始覆盖整个一维 CTA | 使用 `__syncthreads()` |
| 部分 CTA 的多个 warp | `supportsSync()` 返回 false，实际调用 `sync()` 时 trap，状态为未支持 |

`Intrin::namedBarrierSync` 仅接受全 CTA 参与。原始 `__barrier_sync_count` 调用保留在独立 SDK 复现源码中，正常平台探针采用上表映射。探针包含完整 CTA span 的反复跨 warp 共享数据交换，以及独立单 warp span 的不同轮数执行，并覆盖 `CoopAny` 调用。修正后的构建和运行结果仍待 Linux 服务器验证。

生产适配仍需满足连续 warp 子组的参与线程数、group id 隔离、重复调用和内存可见性语义。MCCL `src/device/common.h` 的软件 barrier 依赖其 kernel 初始化共享状态，复用前需要设计并验证相应初始化契约。原生计数 barrier 与部分 CTA 的跨 warp 同步能力继续作为待解决项跟踪。
