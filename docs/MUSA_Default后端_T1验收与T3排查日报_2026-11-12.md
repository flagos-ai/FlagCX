# MUSA Default 后端适配日报

日期：2026-11-12  
范围：AllReduce 设备镜像修复、A+B 集成反馈、T3 Unified IR Intra 故障排查。

## 今日进展

完成 AllReduce 模板 Kernel 导致的设备镜像加载故障定位，提交了普通 float/double Kernel 入口修复。正式测试镜像在 GPU0、GPU1 上的六项 Runtime 函数属性查询均恢复正常。B 同事合并后反馈 T1 已通过。

T3 尚未通过。通过独立 MPI 分段诊断，确认一轮集成运行中的设备 Comm 有 4 个 context，但 `_netContexts` 为空；静态和 IR INTRA Barrier 均通过，IR WORLD Barrier 执行后返回 MUSA error 700。随后一轮复测又发现 S16 启动前已存在 error 98，排查范围转向主库中的 Net context 构造 Kernel 及设备镜像注册。

B 进一步从反汇编中发现：主库两份独立 Fatbin 的初始化函数使用了相同的 `__musa_gpubin_handle` 地址。这是当前最应优先确认的链接线索，尚未独立核实目标文件的符号绑定和 COMDAT 分组。

所有 SDK 编译和 GPU 运行结果均来自服务器回传。本机完成源码修改、日志分析和静态检查，没有本机 GPU 验证结果。

## 1. AllReduce 镜像加载故障

此前，正式 `device_api.mubin` 在 GPU0、GPU1 上直接加载均返回 error 200，而独立探针镜像可以正常加载、查找函数和执行。由此将问题收窄到正式镜像内容，而非普遍不可用的编译链。

先后完成最小源码分阶段验证、Kernel 单项及组合加载测试，最终得到以下对照：

| 对照项 | 服务器结果 |
| --- | --- |
| 最小 CommQueries：加入头文件、设备对象参数和实际查询方法 | 四阶段均加载、查找成功 |
| 其他 26 个 Kernel 单独成镜像 | 全部加载、查找成功 |
| AllReduce float 模板 Kernel | 加载返回 200 |
| AllReduce double 模板 Kernel | 加载返回 200 |
| 展开为普通 float Kernel | 加载、查找成功 |
| 展开为普通 double Kernel | 加载、查找成功 |
| 完整集合包含 float 模板 Kernel | 加载失败 |
| 完整集合使用普通 float Kernel | 镜像及全部目标函数加载、查找成功 |

据此，将正式测试源码中的 AllReduce 设备入口展开为 float、double 两个普通 Kernel，保留公共 launcher 接口、两种数据类型和算法。静态比较确认，两份函数体与原模板做类型替换后的内容一致，其他 26 个 Kernel 函数体未改。

修复后，正式 `device_api.o` 重新完成设备编译、设备链接、镜像嵌入和 Host 链接。GPU0、GPU1 的以下六项属性查询均返回 0：

- AllReduceFloat、AllReduceDouble。
- CommQueries、SymPtr、CoopGroups、NetGetFromComm。

这是正式镜像加载和 Runtime 函数解析恢复的证据。其后，B 同事反馈 A+B 集成的 T1 已通过；本日报未逐项核验 T1 三轮完整验收日志，因此将其记录为集成方反馈，不扩写为 T1–T4 全部验收完成。

## 2. 构建与诊断工具排雷

修复了 `test/make.inc` 的项目根目录计算方式。原先使用递归变量结合 `MAKEFILE_LIST`，读取生成的 `.d` 文件后，可能把项目根目录解析到 `test/kernel/musa/build`，导致找不到 `makefiles/musa.mk`。现改为读取公共配置时立即计算默认根目录，并保留外部显式设置。

批测工具也发现并修正了两类问题：

- 提取 Kernel 时遗漏公共设备函数 `ipcAlltoAll`，造成部分诊断源码编译失败。这些失败不属于正式 Kernel 冲突。
- 符号扫描最初只识别 `T/t`，误将 AllReduce 的 `W` 弱符号判为缺失。修正后复用已有镜像重新加载，才得到有效的模板对照结论。

一次 T3 复测还缺少 `FLAGCX_USE_HETERO_COMM` 等运行变量，在内存分配处返回 NotSupported，未到达 S16。后续批次改为显式向每个 MPI job 传入环境值，避免依赖终端之前的 export 状态。

## 3. T3 首轮定位：S16 执行返回 700

T3 使用 `test_device_ir_unified_intra`，覆盖 INTRA/WORLD teams 的 S16–S25。服务器按 8 ranks、1K–16M、步长 2 运行，第一轮在 1K 处失败，没有完成 S16 结果输出。

INFO 日志显示，8 个 rank 均在 `test_device_ir_unified_intra.cpp:182` 返回 1。该位置是 S16 第一次 BarrierSync Kernel 启动后的 stream 同步。

增加可选原生诊断后，得到：

| 阶段 | 结果 |
| --- | --- |
| `musaFuncGetAttributes` | 0，函数解析成功 |
| Kernel 启动后的即时状态 | 0 |
| `musaStreamSynchronize` | 700，非法内存访问 |

这次故障发生在设备执行阶段，不能沿用此前的 K1 镜像加载结论。日志中的 staging 注册 warning 后，程序仍继续进入测试；目前没有证据证明该 warning 是此次退出的直接原因。

## 4. 六项独立 MPI 诊断

新增独立诊断程序和批处理脚本。诊断程序复用 T3 的 DevComm requirements、内存、Window 和设备对象创建流程，每个诊断项在新的 8-rank MPI job 中执行。某项失败后继续运行其他项，最后运行正式 T3；正式测试失败后不再重复凑轮数。

首轮结果：

| 诊断项 | 结果与解释 |
| --- | --- |
| 读取设备 Comm | 查询、启动、同步成功；rank/size 正常，Host/Device 大小均为 448，但 `contexts=4、netPresent=0`，检查判 FAIL |
| 读取 Net contexts | 检测到空指针后返回，检查判 FAIL；打印的零值不是实际读取到的 Net 对象内容 |
| 按值传入、静态 INTRA Barrier | PASS |
| 按指针读取、静态 INTRA Barrier | PASS |
| IR INTRA Barrier | PASS |
| IR WORLD Barrier | 同步返回 700 |
| 正式 T3 | S16 同步返回 700，0/3 轮通过 |

这一轮确认：Comm 基础字段能够读取，INTRA Barrier 路径可执行；WORLD IR 路径所依赖的 `_netContexts` 缺失，是必须解决的问题。不能据此认定整个 IPC 或全部 PlatformTraits 均有故障。

本地 Default backend 中，Net context 数组由 `defaultDevApiCommGetDevicePtr` 分配，并调用 `flagcxDevNetLaunchConstruct` 构造，再把数组指针写入设备 Comm。缺少 `COMPILE_KERNEL_HOST` 时，代码存在返回 0 的 SizeOf 占位实现，可能跳过构造。该分支被列为编译配置核查项，但未根据现有日志认定 B 的实际产物必然走了此分支。

## 5. 后续复测：初始化阶段已遗留 98

随后更新的诊断包中，六项诊断均未通过；正式 T3 在 S16 启动前打印 `prior=98`，但函数属性查询仍为 0。

这说明本轮初始化阶段已经产生原生错误。各诊断项打印的 `launch=98` 可能包含此前遗留的状态，不能解释为六个 Kernel 分别发生了新的启动错误。本轮没有提供有效的设备对象读回结果，也不能据此确认 `_netContexts` 问题已解决。

排查重点因此前移到：

```text
defaultDevApiCommGetDevicePtr
  → flagcxDevNetLaunchConstruct
  → flagcxDevNetConstructKernel
```

需要在上述步骤记录第一次原生错误，并确保失败不会被包装为成功的设备对象创建结果。仅清除 error 98 后继续运行不构成修复。

## 6. 当前线索：独立 Fatbin 的注册 Handle 合并

B 提供的反汇编分析指出，`libflagcx.so` 中两份 Fatbin Wrapper 地址不同，但对应模块构造函数访问同一个 `__musa_gpubin_handle` 地址。若该分析成立，第二个模块可能因 Handle 已非空而跳过自己的 Fatbin 注册，随后将函数关联到错误的镜像。

本地构建源码核查发现：

| 项目 | 设备镜像组织 |
| --- | --- |
| 当前 FlagCX MUSA 主库 | 每个 `.cu` 独立设备链接、独立嵌入，再将多个 Host 对象链接到同一 `.so` |
| MCCL 参考实现 | 多个设备对象统一设备链接，由一个 Host 对象集中嵌入设备镜像 |

当前 FlagCX Host-only 编译同样使用 `-fgpu-rdc`。这与独立镜像的组织方式一起，构成了需要优先验证的注册机制问题。

目前尚未取得并独立分析 B 的原始目标文件与反汇编。仍需确认：

1. 两个 Host 对象中 Handle 的符号绑定、可见性和 COMDAT 分组。
2. 最终 `.so` 是否确实将原本不同镜像的 Handle 合并。
3. 各构造函数、Fatbin Wrapper 和注册函数调用的实际对应关系。

共享 Handle 本身不一定错误，统一设备链接方案中可能存在这种设计。当前可疑点是独立镜像共享了注册状态。修复方向应在确认后选择统一设备链接/集中嵌入，或验证独立模块注册方式；不盲目改名、修改 `-shared` 或仅调整符号可见性。

该问题与此前 AllReduce 模板镜像直接加载失败是两条独立证据链，不能用注册问题覆盖或否定前面的模板对照结果。

## 7. 已提交代码

仓库：`gao0624/FlagCX`  
分支：`feat/musa-device-platform-a`

| 提交 | 内容 |
| --- | --- |
| `39adbf51612ed9cad18a7aa60dc56fe179a86a59` | AllReduce 普通 float/double 设备入口、属性诊断及默认项目根目录计算修复 |
| `37efd8edf31da32c081db5d967d16b729458458f` | 六项 IR 分段诊断、独立构建入口、自动 MPI 批测与日志打包 |

两次提交均已同步到 `D:\LLM\FlagCX\FlagCX_gzq\FlagCX` 并推送。前序独立镜像批测脚本和记录文档未随上述修复提交；六项 IR 诊断的四个文件是在用户明确要求后提交。

B 的集成工作区还包含其 Runtime/IPC 修改，日志中的 A 分支 HEAD 不足以单独标识全部集成源码。后续验收需要同时记录 B 的提交或工作区差异。

## 8. 下一步及验收边界

1. A+B 核实主库注册 Handle 和 COMDAT 合并，确定设备镜像的最终构建组织方式。
2. 验证 Net context 构造 Kernel 的属性查询、启动、同步均成功，设备 Comm 的 `_netContexts` 非空，各 context 内容正确。
3. 重新运行六项分段诊断，确认 WORLD Barrier 不再发生非法访问。
4. 完整运行 T3：8 ranks、1K–16M、factor 2、S16–S25 全部通过、全局退出码 0，连续三轮通过。
5. 主库构建或注册方式修改后，回归 T1 和 B 的 IPC 测试，再推进 T2/T4。

当前记录为：T1 已收到 B 的通过反馈；T3 仍失败；T2/T4 未获得新的完整验收结果。Remote Flush 之前返回 NotSupported，本阶段没有新增支持证据。

A 继续负责设备 Kernel、编译链接及注册加载问题；B 保留并维护 Runtime/IPC 实现，提供集成环境。当前没有理由用 A 的 `musa_adaptor.cc` 覆盖 B 的 IPC 工作。
