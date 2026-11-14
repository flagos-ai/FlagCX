# MUSA Default 后端适配日报

日期：2026-11-13

承接：[2026-11-12 日报](MUSA_Default后端_T1验收与T3排查日报_2026-11-12.md)

## 今日进展

完成设备镜像符号绑定、MUSA Host 映射接口和 IPC 集成修复，代码已推送至 `feat/musa-device-platform`。T3 已完成完整验证；此前回传日志输出 `ALL PASS`，验收脚本返回 0，后续完成状态由本人确认。

T2 已由 B 同事完整跑通。此前 256 KB K11 WaitSignal+Flush 停顿属于排查过程，不再作为当前阻塞项；具体处理和最终代码差异待同步归档。当前剩余工作为本人负责的 T4 Unified IR Inter 全部单测，测试用例保持原样。

## 已完成工作

1. 确认主库设备镜像符号被测试程序抢占。8 个 rank 的 `LD_DEBUG=bindings` 输出均显示，`libflagcx.so` 的 `__mt_fatbin` 绑定到了测试可执行文件。增加链接版本脚本，将主库该符号限制在库内，防止注册到错误镜像。
2. 补齐 `hostGetDevicePointer`。MUSA adaptor 原槽位为 NULL，Proxy 在 FIFO 初始化后直接调用，存在空函数指针崩溃路径。现接入 `musaHostGetDevicePointer`，Host 分配改用 `musaHostAllocMapped`，继续通过 `musaFreeHost` 释放；Proxy 和 Reduce FIFO 初始化增加缺失能力检查。
3. 集成 B 提供的补丁。IPC Handle 改为原生 `musaIpcMemHandle_t`，实现导出、打开、关闭和包装对象释放。同时将主库设备构建改为各 `.cu` 独立编译、嵌入镜像，移除这些对象编译时的 RDC。该方案取代此前 C63 的统一镜像方案，最终仍链接为一个 `libflagcx.so`。
4. 修正 Host 映射测试中 `deviceMemcpy` 多传一个参数的编译错误。该错误来自前一提交，对公开 DeviceHandle 和内部 adaptor 的接口签名混淆；差异检查未发现，已随集成补丁修正。
5. 整理 T3 运行和验收脚本，记录 MPI 退出码、用例 PASS 数、最终结果和 FAIL/SKIP 数，并在验收不完整时返回非零。

| 提交 | 内容 |
| --- | --- |
| `4907551` | 主库 `__mt_fatbin` 绑定在库内 |
| `52ea7dd` | Host 映射接口、FIFO 判空及映射读写测试 |
| `918e8f5` | IPC 集成、独立非 RDC 镜像构建、测试参数修正 |

发布目录：`D:\LLM\FlagCX\FlagCX_gzq\FlagCX`。分支已由 `feat/musa-device-platform-a` 改为 `feat/musa-device-platform` 并推送，远端旧分支保留。

## T3 验证结果

服务器工作目录：`/workspace/yx/FlagCX`。此前因仍处于 main，使用空的 MUSA 编译器配置，错误进入 `kernel_dlink.o` 构建；切换至 `918e8f5` 后，确认设备编译器为 mcc、`DEVICE_NEEDS_DLINK=0`，再执行构建与测试。

本轮使用现有 `test_device_ir_unified_intra` 测试程序，8 ranks、单节点、1K～16M、factor 2，执行 INTRA/WORLD 的 S16～S25。运行配置为 Default 后端、VMM 关闭、P2P 开启、IB 关闭、Host SHM signal 模式关闭。

服务器回传结果：

```text
=== Final Result: ALL PASS ===
T3_RESULT=PASS
SCRIPT_EXIT=0
```

日志目录：`/workspace/yx/FlagCX/acceptance-t3-20261113_115313`。

上述已回传轮次按脚本判定，满足 15 个尺寸 × 10 项 × 8 ranks，共 1200 条用例 PASS，最终 ALL PASS 一次，无 FAIL/SKIP。后续本人确认 T3 已完成完整验证，当前状态更新为完成。已知日志路径保留如上，其余轮次日志和对应代码版本待统一归档；本地尚未逐行复核全部日志。

T3 通过证明本轮配置下统一 IR Intra 功能可完成，不扩展为跨节点网卡路径或 T4 已通过。

## T2 完成情况与 T4 待验证项

B 反馈已补齐 `hostRegister/hostUnregister`，并已完整跑通 T2；最新实现和最终日志尚未同步至本地。此前 K11 停顿不再列为当前阻塞项。T2 的 K8 Get 原用例明确跳过，T2 完成不等于已实际验证 Get，后续由 T4 的 S19 覆盖。

此前源码审查发现 K11 多个 CTA 共用 context 0、signal 0 并执行 reset，存在晚到 reset 清掉 Signal 的竞态疑点。该判断保留为排查记录，不据此认定本次停顿的确定根因，也不推断 B 最终采取了何种修复。

遵守测试用例不能修改的边界，不拆分 reset Kernel、不减少 CTA、不跳过用例，也不改变 reset 或等待语义。若运行证据确认公共用例存在竞态，应交由测试维护方处理。

T4 的公共 IB `iput/iget/iputSignal/test` 已有实现，重点仍是统一 IR Net context、S19 Get 的实际执行，以及跨逻辑节点 Barrier、Signal/Counter 和 Flush 的完成与可见性。本地还存在 MUSA GDR 平台识别缺口，DMA-BUF 接口为空；这些状态不能直接代替 B 最新代码的实际能力，需要先同步核对。VMM、multicast、事件计时的未实现接口不属于当前关闭 VMM 配置下的必补项。

## 当前验收状态与下一步

| 测试 | 状态 |
| --- | --- |
| T1 | 此前 B 反馈通过；本次镜像构建变更后的完整回归待确认 |
| T2 | B 同事已完整跑通；最终实现和日志待归档，K8 Get 原用例跳过 |
| T3 | 本人确认已完成完整验证 |
| T4 | 本人负责，尚待完成完整验证 |

下一步由本人推进 T4：先同步 B 已通过 T2 的最新实现和运行配置，以一致的集成版本按 2×4 逻辑拓扑运行完整 S16～S25，重点检查 IR Net context、S19 Get、跨节点 Barrier 及完成可见性，并完成规定的重复运行。T2/T3 的最终日志与代码版本同时归档，T1 在镜像构建变更后的回归记录待核对。

本机完成源码修改、补丁集成与差异检查；MUSA 编译和 GPU 运行均来自服务器反馈。本日报仅保存到本地，未随代码提交或推送。
