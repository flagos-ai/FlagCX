# A → B：MUSA 设备平台基础与 Runtime/IPC 交接说明

整理日期：2026-11-12。A 项验证发生于 2026-11-11，结果依据用户回传的 Linux 服务器日志；本日仅复核本地源码状态和整理交接要求。

## 1. 交付清单

B 提出的 Git Commit、真实 GPU Kernel 测试结果、正式编译命令是核心材料。建议连同环境与接口边界一并提供：

| 交付物 | 应包含的内容 | 当前状态 |
| --- | --- | --- |
| A 成果 commit | 仓库地址、分支、包含 A 修改的完整 commit SHA；说明父提交和依赖，覆盖新增文件 | 已在本地生成 `0baef69bafa06e1b4a3171d7482a739a6f2296ab`，尚未推送 |
| GPU 原语测试结果 | 原始输出、命令、退出码、环境及对应源码版本；三个分项的 PASS 摘录 | 实际 GPU PASS 已取得；原始日志与确切成果版本需归档绑定 |
| 正式编译命令 | 主库、正式 Kernel object、公共测试程序、平台 probe 的构建入口 | 已有服务器验证命令，见下文 |
| 环境说明 | GPU、目标架构、SDK/编译器、runtime/driver、库路径、backend | 已记录 |
| A/B 接口约束 | 原子控制字位宽/对齐、资源有效期、指针/stream 传递、数据发布与完成约定 | 按当前实现交接，并与 B 确认 |
| 已知问题和未覆盖范围 | trap、动态 Span、跨 GPU/CPU 可见性以及未完成资源接口 | 已独立记录，不以 A PASS 覆盖 B 的验收 |

## 2. Git Commit：本地提交已生成

本地目录 `D:\LLM\FlagCX\FlagCX`，2026-11-12 复核结果：

```text
branch: mccl
HEAD:   0baef69bafa06e1b4a3171d7482a739a6f2296ab
title:  feat(musa): add device platform and kernel build support
state:  A 源码/构建文件已提交；探针、诊断脚本和记录文档仍未跟踪
```

该成果提交包含预定的 15 个适配源码、正式 Kernel 和构建文件。父基线为
`5e36133e01024b12b1bb6d853e7b9ff0ec405009`。用户已在本机执行提交脚本，
本次核对确认提交成功，尚未推送到远端。

按用户要求，成果提交不包含独立 probe、诊断脚本和记录文档。B 检出该提交
可以取得适配代码及正式 Kernel，但不会取得 `test/unittest/musa_platform`。
本说明中的原语复现命令需要单独提供该目录源码；原始测试日志也应另行交付。

已测服务器工作区与本地需要先核对一致性。服务器曾有独立修改，不能只取服务器 HEAD 或本地 HEAD 就声称测试与代码完全对应。如果日志先于成果提交，应注明“测试时的工作区快照与该提交已核对一致”；未核对前不填写这个结论。

发给 B 的版本信息建议填写为：

```text
仓库地址：<团队可访问的仓库>
分支：mccl 或最终交付分支
A 成果完整 commit：0baef69bafa06e1b4a3171d7482a739a6f2296ab
父基线：5e36133e01024b12b1bb6d853e7b9ff0ec405009
已测服务器源码对应关系：<快照/提交核对结果>
```

本说明本身未纳入成果提交。远端仓库地址和推送状态需在实际交付时补齐。

## 3. 实际 GPU 结果

| 测试入口 | 服务器结果 | 已覆盖内容 |
| --- | --- | --- |
| host layout | PASS：SIMT=32、mask=8、CoopAny=24、align=8 | 本轮检查项下的 host 布局与常量 |
| `run-groups` | PASS，0 failures | Block/Warp/Tile/Lanes/Thread、CoopAny 转换/复制及共享内存同步 |
| `run-atomics` | PASS，0 failures | 32/64 位原子操作、争用、同 CTA 两 warp release/acquire 发布，执行 device/system fence 调用 |
| `run-cross-block` | PASS，8 轮、0 failures | 两个 64 线程 CTA 的 device-scope 原子争用和数据发布 |

原始输出中的关键摘录：

```text
MUSA host layout PASS: SIMT=32 mask=8 CoopAny=24 align=8

anyGroupProbe: threads=64 maxThreadsPerBlock=1024 numRegs=23 sharedBytes=268 localBytes=0
Launching anyGroupProbe (1 block, 64 threads)
MUSA groups-only probe PASS (0 failures)

atomicOnlyProbe: threads=64 maxThreadsPerBlock=1024 numRegs=26 sharedBytes=12 localBytes=0
Launching atomicOnlyProbe (1 block, 64 threads)
MUSA atomics-only probe PASS (0 failures)

crossBlockProbe: threads=64 maxThreadsPerBlock=1024 numRegs=64 sharedBytes=12 localBytes=0
Launching crossBlockProbe (8 rounds, 2 blocks, 64 threads/block)
MUSA cross-block probe PASS (8 rounds, 0 failures)
```

上述资源行省略了每次进程不同的 `kernel=0x...` 地址，其余数字来自用户实际输出。此处是证据摘录，不能替代完整原始日志。

这些测试实际启动 GPU kernel、检查启动/同步错误并拷回校验结果。它们没有执行跨进程 IPC、跨 GPU 或 GPU/CPU 发布握手，也没有验证 GDR。B 不能据此将这些资源能力标为通过。

## 4. 环境与源码加载方式

| 项目 | 已确认值 |
| --- | --- |
| 服务器源码根 | `/workspace/FlagCX` |
| GPU | 8 × MTT S5000，计算能力均为 3.1 |
| 目标架构 | `mp_31`；当前工具链的 mp_43 编译探测失败 |
| warp / 设备线程块上限 | 32 / 1024 |
| 编译器 | mcc 5.1.0，clang 14.0.0 |
| 编译器 commit | `39eee21af373855ae51b0e906a74b02950384a05` |
| 编译器入口 | `/usr/local/musa/bin/mcc` |
| 实际编译器 | `/usr/local/musa-4.3.5/bin/clang-14` |
| runtime/driver 版本 API 返回值 | 均为 `40305` |
| runtime 动态库 | `/usr/local/musa/lib/libmusart.so.4` |
| 当前 backend | `CommTraits<DefaultBackend<MusaPlatform>>`，不是原生 MCCL CommTraits |

在源码目录 make 构建，不通过 pip 安装本次源码；选择本轮动态库：

```bash
cd /workspace/FlagCX
export FLAGCX_SRC=/workspace/FlagCX
export FLAGCX_PATH="$FLAGCX_SRC"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH"
```

将 B 实际运行程序的 `ldd` 一并附上，确认加载本轮 `build/lib/libflagcx.so`，避免使用预装 wheel 或其他 checkout 的产物。

## 5. 正式编译命令

以下为已使用过的源码构建入口。主库、两个正式 Kernel object 和六个公共测试程序均已在服务器构建/链接成功；六个程序的通信运行尚未通过验证。

```bash
timeout --kill-after=5s 600s \
  make -j1 USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/kernel/musa all \
  USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/unittest/device_api mpi \
  USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

设备源码使用 `--cuda-device-only` 编译、SDK lld 显式链接，再通过 `--cuda-host-only` 嵌入镜像。不要将最终 host object 再送入 mcc 自动重复设备链接。具体参数由当前 Makefile 和 `makefiles/musa_kernel_rules.mk` 提供，不需 B 手工拼接。

## 6. 三个原语分项的复现命令

在第 4 节环境下分别执行，保存每条命令的 stdout/stderr 和退出码：

```bash
timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-groups \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31

timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-atomics \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31

timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-cross-block \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

已有完整日志可直接交付。若为了形成“确定 commit + 原始日志”的绑定而补录复现，应明确这一目的，不用反复运行已通过的分项代替推进 B 的接口工作。

不要用无参数 `run` 替代以上三个入口来表述相同验证范围：完整 probe 仍包含尚未闭环的动态 Span/trap 路径。

## 7. B 应接收的接口基线与已知缺口

| 事项 | 交接给 B 的内容 |
| --- | --- |
| Runtime 函数表 | 核对 `flagcx/adaptor/include/adaptor.h` 和 `flagcx/adaptor/device/musa_adaptor.cc` 的真实签名与调用方；通用 launchKernel 表项当前仍为 NULL |
| MUSA 对象表示 | `flagcx/adaptor/include/musa_adaptor.h` 的 stream/event/IPC handle 表示；IPC 改动需和全部创建、交换、打开、关闭、释放调用方配套 |
| 当前 IPC 实现 | Create/Get/Open/Close/Free 均仍返回 `flagcxNotSupported`，没有拿空桩充当 IPC 通过 |
| Host registration | hostRegister/hostUnregister 仍返回 `flagcxNotSupported`；GDR 与 host-mapped 可见性不能使用 A 的单 GPU结果代替 |
| 控制字宽度 | 当前 MUSA Default backend 的 CompletionWord/FIFO 控制为 64 位；共享控制字按 8 字节自然对齐，B 的 host 更新与设备原子不能混用 CoreX 的低 32 位方案 |
| 完成与可见性 | A 已测 device-scope 单 GPU路径；跨进程/跨 GPU/system-scope 发布需要 B 资源映射与 A 设备测试配合验证 |
| DevNet 构造 | 主库已有真实 SizeOf 和设备构造 helper；设备对象使用期间资源有效，相关 kernel 完成前不可销毁或解除映射 |
| Backend 选择 | 当前是 DefaultBackend<MusaPlatform>；若转原生 MCCL，Host 资源、comm/window 表示及 CommTraits 必须配套 |
| runner 范围 | 本地 persistent reduction 当前仅 float/Sum；其他组合在 host 入队前返回 NotSupported，不等于 MCCL 原生能力限制 |

资源模块接口冻结前应确认 handle 传输大小与生命周期、指针有效范围、设备绑定、stream 完成边界、资源所有权、部分失败回收和错误传播。A 提供设备测试配合，不替 B 宣称 Runtime/IPC/GDR 完成。

## 8. 独立问题随交接一并提供

- [动态 trap 资源异常与不可达标记回归](musa_trap_resource_issue.md)：SDK-only 对照已重复复现；不得未经验证再加入 `__builtin_unreachable()` 来消告警。
- [counted barrier 编译异常](musa_counted_barrier_compiler_issue.md)：当前工具链的计数屏障存在独立复现问题，部分 CTA 多 warp 仍未支持。
- 当前 DefaultNetGdaProvider 非 void 失败路径仍有返回值告警；本轮没有伪造返回值或关闭检查。
- A 已通过的正常分项不包括动态 Span 整套、跨 GPU/CPU 可见性、正式通信 Kernel 运行或 T1–T4。

## 9. 给 B 的交接结论

A 已提供 MUSA 构建链、平台基础实现，以及 host layout、协作组、原子和跨 CTA 的真实 GPU 正常路径结果。B 可以在这套基础上开展独立 Runtime/IPC 能力实验与资源实现。

成果 commit 已生成，正式交付还需使 B 能取得该提交，并将原始服务器日志
与该源码快照对应。独立 probe 和本文档按用户要求另行交付。A/B 联调时重点
补跨进程映射、资源释放次序和系统域可见性，不能将 A 的基础分项 PASS 等同于
B 的 IPC/GDR 回归通过。

详细历程见[完整工作总结](FlagCX_MUSA_适配阶段完整工作总结_2026-11-11.md)，构建与联调进度见[主库 Kernel 接入记录](musa_main_library_kernel_integration.md)。
