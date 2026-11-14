# FlagCX / MCCL / MUSA 适配阶段完整工作总结

记录日期：2026-11-11。覆盖前期交接与评估、源码适配、构建接入、服务器排障及最近一轮 GPU 原语验证。

本文是截至当前的汇总记录。历史内容依据交接文档；新增验证依据用户在会话中提供的 Linux 服务器输出及当前本地源码。未取得的运行结果不补写为通过，已撤回的方案不计入最终实现。

## 1. 当前结论

目前已从“仅构建配置可用”推进到：MUSA 主库与正式测试 Kernel 可构建，六个公共测试程序可链接，单 GPU 上协作组、原子操作、同 CTA 和跨 CTA 发布检查实际运行通过。

当前设备接口绑定为 `CommTraits<DefaultBackend<MusaPlatform>>`。这轮成果是 MCCL 接入所需的 MUSA 设备平台基础和 Kernel 构建接入，尚不能表述为“原生 MCCL 通信后端移植完成”。

用户最终明确的本人范围为 **A：设备编译与 PlatformTraits**：`COMPILE_KERNEL=1` 路径、必要 Intrin/Atomic/Fence/Coop、最小 Kernel 正确执行及编译/原语测试结果。该范围已有真实 GPU 正常路径证据，可以整理阶段验收材料。必要原语的边界、未支持项和证据归档仍需随材料明确，不代表已经完成正式验收签署。

| 项目 | 当前状态 | 结论边界 |
| --- | --- | --- |
| MUSA 平台识别与设备编译链 | 服务器已通过 | 主库和测试均进入实际设备编译、链接与镜像嵌入 |
| MUSA PlatformTraits 基础实现 | 已实现并完成部分 GPU 验证 | 不覆盖所有扩展接口和所有同步范围 |
| 主库设备 helper / launcher | 编译和链接通过 | 尚未取得生产 helper/runner 的运行正确性结果 |
| 两个正式测试 Kernel object | 构建通过 | 不等于其全部通信 case 运行通过 |
| 六个 Device API / IR 程序 | 编译链接通过 | T1–T4 尚未执行 |
| 协作组基础分项 | GPU PASS，0 failures | Block/Warp/Tile/Lanes/Thread 和 CoopAny 正常路径 |
| 原子与同 CTA 发布分项 | GPU PASS，0 failures | 32/64 位操作、争用及同 CTA 两 warp 发布 |
| 跨 CTA 发布分项 | GPU PASS，8 轮、0 failures | 单 GPU、两个 CTA 的 device-scope 原子和发布可见性 |
| counted barrier / 动态 trap | 独立复现并归档 | 尚未获得厂商修复或已验证替代原语 |
| 原生 MCCL Host/CommTraits 与完整通信 | 未完成 | 需要后续配套接入与联调 |

## 2. 工作约束、源码与环境

### 2.1 开发与运行方式

| 位置 | 用途 |
| --- | --- |
| `D:\LLM\FlagCX\FlagCX` | 本地 FlagCX 源码，当前改动所在目录 |
| `D:\LLM\FlagCX\mccl` | 本地 MCCL 参考源码 |
| `D:\LLM\FlagCX` | 交接、规划、图和历史补丁所在目录 |
| `/workspace/FlagCX` | 用户 Linux 服务器上的构建与测试根目录 |

适配基于本地官方仓库 checkout 和已同步的服务器有效修改继续进行。服务器历史 checkout 带有前期修改，不能只比较旧 commit 就认定其状态落后，也不能用本地整个目录覆盖服务器已有 submodule 和工作区成果。

所有 MUSA SDK 编译、设备链接及 GPU 测试都由用户在 Linux 服务器执行。本地进行了源码修改、接口静态核对和文档整理，没有在 Windows 上执行 MUSA 编译或 GPU 测试。

不通过 pip 安装本次源码，也未卸载预装 wheel；运行时选择本次源码和动态库产物：

```bash
export FLAGCX_SRC=/workspace/FlagCX
export FLAGCX_PATH="$FLAGCX_SRC"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH"
```

正式结果还需要用目标程序的 `ldd` 核验实际加载的 `libflagcx.so`。仅 `import flagcx` 成功不能作为 Device API 验证证据。

### 2.2 源码基线

| 仓库 | 本地分支 | 当前基线 commit |
| --- | --- | --- |
| FlagCX | `mccl` | `5e36133e01024b12b1bb6d853e7b9ff0ec405009` |
| MCCL | `release_musa_5.3.0` | `23edec5afda741efe0119410000dd80fd03092ef` |

FlagCX 的适配目前仍包含未提交修改及新增未跟踪文件。基线 commit 不包含本轮补丁，交付时需要同时保存 diff 和新增文件；只保存 `git diff` 会漏掉尚未跟踪的文件。

交接记录中的服务器历史基线为 `36bf2c56a1de599d4fd09ad134f0593babd0ae16`，本轮未重新取得服务器完整 git 状态，不能用本地 HEAD 替代服务器实际源码清单。

### 2.3 服务器重新探测结果

| 项目 | 实测信息 |
| --- | --- |
| 系统 | Linux，工作目录 `/workspace/FlagCX` |
| GPU | 8 张 MTT S5000 |
| 每张 GPU 计算能力 | 3.1 |
| warpSize | 32 |
| 设备级线程块上限 | 1024 |
| 设备目标 | `mp_31` |
| mcc / clang | mcc 5.1.0 / clang 14.0.0 |
| 编译器 commit | `39eee21af373855ae51b0e906a74b02950384a05` |
| 编译器入口 / 实际程序 | `/usr/local/musa/bin/mcc` / `/usr/local/musa-4.3.5/bin/clang-14` |
| runtime / driver 版本 API 返回值 | 均为 `40305` |
| runtime 库解析 | `/usr/local/musa/lib/libmusart.so.4` |
| 编译 `mp_31` 最小源码 | 成功，生成非空设备 object |
| 编译 `mp_43` 最小源码 | 失败：`No available targets are compatible with triple "mthg-mt-musa"` |

架构目标不是 SDK 版本号。此次重探确认当前机器和工具链应继续使用 `mp_31`；不能把 `mp_43` 当成升级参数替换。结果也不构成对其他 MUSA 版本全部架构支持情况的判断。

交接中安装的 MCCL 历史版本与本地 `release_musa_5.3.0` 源码属于不同的版本证据。本轮没有证明这份新 MCCL 源码已在服务器完整构建并通过原生 Device API 通信测试。

## 3. 从头推进的阶段记录

| 阶段 | 做过的工作 | 得到的结果 |
| --- | --- | --- |
| 交接与基线核对 | 阅读交接、v3 验收和已有构建记录，区分本地官方基线与服务器适配状态 | 明确不在本机运行、不通过 pip 部署、不覆盖服务器有效改动 |
| MCCL 评估与规划 | 分析可复用通信能力，梳理阶段、串行依赖、并行工作和验收范围 | 形成推进计划与接口缺口清单，未把静态判断当作实测支持 |
| 图与说明 | 优化通信链路 draw.io，导出 PDF；另画本人职责与进度流程图 | 形成可编辑图和阶段性展示材料 |
| 构建配置承接 | 用户同步已在服务器验证的 `musa.mk` / `test/make.inc` | 在已验证配置上继续设备适配，没有重做或抹掉 W-03/W-03a |
| 平台基础实现 | MUSA 编译识别、PlatformTraits、Atomic/Fence、Coop 与 Default 绑定 | 从默认占位平台切换到明确的 MusaPlatform 实现 |
| 平台 probe 排障 | 处理 host/device 头文件边界、barrier 编译异常、trap、函数指针和设备链接问题 | 独立 probe 可构建；SDK 兼容问题有最小复现 |
| 正式 Kernel 接入 | 新增 `test/kernel/musa` 的真实源码和构建规则 | 两个正式 object 在服务器编译、设备链接和镜像嵌入通过 |
| 主库符号补齐 | 增加生产 helper、collective launcher 和 MUSA 主库分步规则 | 主库和六个公共测试程序全部链接通过 |
| 分项 GPU 验证 | 独立运行协作组、原子和跨 CTA 分项 | 均为真实 GPU PASS；完整通信验收仍待开展 |

## 4. 前期评估、规划与图示工作

### 4.1 MCCL 复用判断

按用户限定，接入方案围绕摩尔通信库，核心参考为 MCCL；其他平台只用于核对 FlagCX 已有接口和测试模板，不作为额外厂商适配目标。

前期对照了以下链路及接口：

- 同节点：Window、LSA pointer、Team 与 LSA barrier。
- 跨节点：Proxy GIN Put/Get、Signal、Counter、Wait、Flush。
- 公共对象：DevComm、window、requirements、context、内存注册与生命周期。
- 语义缺口：Get 的 backend 选择、SignalSet、混合 signal 操作、多 context 隔离、重复调用与完成可见性。

源码中 Proxy GIN 存在 Get 实现，GDAKI Get 有自身限制，不能将不同 backend 的能力混为一谈。这些属于静态评估，仍需目标服务器的原生能力验证。

推进计划经过调整，把分阶段流程放在前面，并区分能并行开发的 Host 资源、CommTraits、Kernel 准备，以及必须在接口和资源一致后汇合的运行验证。

`test/kernel/musa` 的定位也已明确：开发和模块联调阶段就需要接入，不能等到最终 T1–T4 才补。完整验收仍使用公共 Host driver、case 编号和 PASS/FAIL 判定。

### 4.2 形成的规划材料

下列资料位于源码目录的上一级，链接用于追溯工作，不表示其中的历史进度已全部更新：

- [交接入口](../../HANDOFF_FlagCX_MUSA_适配交接.md)。
- [MCCL 接入评估与推进计划](../../MCCL接入FlagCX_评估与推进计划.md)。
- [W-03/W-03a 验证步骤](../../W03-W03a_验证步骤.md)与[补丁](../../W03-W03a.patch)。
- [W-26/W-27 Kernel 移植替换清单](../../W26-W27_MUSA移植替换清单.md)。
- [通信链路 draw.io](../../FlagCX_MUSA_通信链路.drawio)：目标链路、接口覆盖、通信路径与缺口、Linux 推进流程、v3 验收矩阵，共五页。
- [通信链路 PDF](../../output/pdf/FlagCX_MUSA_通信链路.pdf)：此前导出的阶段快照。
- [Device/Kernel 职责与进度 draw.io](../../Device_Kernel_MUSA_职责与进度.drawio)：两页职责和进度说明。

图和 PDF 形成于不同阶段。最新代码与服务器状态以本总结和专项记录为准，不能把旧图中的“待验证”直接当作当前结论。

## 5. 实际完成的源码适配

### 5.1 MUSA 编译识别与平台绑定

在 `device_utils.h` 中加入 MUSA 编译器识别，兼顾 `__MUSACC__`、`__MTGPU__`、`__MUSA_ARCH__` 及显式编译标记；设备编译参数增加 `FLAGCX_MUSA_DEVICE_COMPILE=1`。

补齐 MUSA 的 host/device 装饰、runtime、线程/块索引、SIMT 宽度、stream 类型、同步和 fence 映射。普通 g++ 与 mcc 的头文件及能力边界分别处理。

`platform_traits.h` 选择 `PlatformTraits<MusaPlatform>`；`comm_traits.h` 选择 `musa_comm_traits.h`。后者当前明确绑定 Default backend，没有把占位声明写成原生 MCCL 后端已接入。

### 5.2 PlatformTraits、Atomic 与 Fence

平台基础包含 lane、活动 mask、低位 lane mask、popcount、warp 同步、CTA 同步、spin backoff 和字节拷贝等适配。

Atomic 基于 MCCL 使用的 `musa::atomic_ref`：映射五种 memory order、四种 scope，实现 load/store、fetchAdd/Sub/Or/And、exchange、compareExchange，并用 CAS 补充整数 fetchMax。独立 probe 覆盖 32/64 位整数操作。

Fence 使用原生 scoped fence。device/system fence 调用已随单 GPU probe 执行；system scope 的枚举映射和调用成功不能替代 GPU/CPU 或 GPU 间实际可见性验证。

### 5.3 Coop 与 CoopAny

实现 Block、Tile、Thread、Warp、Lanes、TileSpan 及 CoopAny。

当前 Span 的支持边界为：完整单 warp 使用 `__syncwarp`；覆盖整个一维 CTA 的 span 使用 `__syncthreads()`；部分 CTA 的多个 warp 尚未支持，不能扩大为全 CTA 屏障。

CoopAny 从公共函数指针表改为 MUSA 内部直接分派，保存 rank、size、mask 和同步类型，保持 24 字节、8 字节对齐。直接分派避免当前设备链接器对静态函数指针表的重定位问题。

动态 Span 包含运行时失败路径，完整动态 Span probe 尚未通过。`CoopOps`、`Clock` 当前为空类型，未对这些扩展能力宣称完成。

### 5.4 正式测试 Kernel

新增 `test/kernel/musa/Makefile`、`device_api.cu`、`device_ir.cu`。测试主体复用当前仓库的公共用例，实现 MUSA adaptor/runtime、stream、错误类型和启动适配。

静态核对了 27 个 Device API 和 48 个 IR 公共 launcher：有效声明均有唯一对应定义，返回类型和参数类型一致。这是源码级核对，最终链接则由后续服务器构建取得证据。

没有复制 CoreX 私有诊断 kernel 和 AS1 launcher 重载桥。MUSA 的 IR 指针限定为空，保留这些桥会产生重复定义。发现模板中 Tile<8> 用固定成功值代替测试后，恢复了真实 rank、size 和同步检查，没有沿用假通过。

### 5.5 主库设备侧 helper 与 runner

新增 MUSA 生产编译单元，补齐三个实际定义：

```text
flagcxDevNetSizeOf
flagcxDevNetLaunchConstruct
flagcxLaunchCollectiveKernel
```

DevNet helper 实际返回对象大小，并启动 kernel 为各 context placement-new 构造设备对象。collective launcher 实际启动 persistent FIFO kernel，执行任务消费、归约和完成发布，不是空函数。

MUSA 的 FIFO 控制与完成标志使用与 Host 一致的 64 位原子，没有复制 CoreX 的低 32 位方案。

当前 runner 设备侧本地归约只支持 **float/Sum**，计算主体与参考实现一致；Host 入队前拒绝其他 datatype/redop，返回 `flagcxNotSupported`。这项限制不代表 MCCL 原生 collective 的能力范围，完整 datatype/redop 支持还需扩展。

### 5.6 设备镜像的最终构建方式

独立 probe、正式测试 Kernel 及主库 MUSA Kernel 均采用以下分工：

```text
MUSA .cu
  → mcc --cuda-device-only
  → SDK lld 显式设备链接
  → mcc --cuda-host-only，嵌入已链接镜像
  → host object
  → host 链接器生成 libflagcx.so / 测试程序
```

CUDA 拼写的这些选项来自 MCCL 的 MUSA 构建做法。RDC/PIC 保留；不再让最终 host object 经过 mcc 自动重复设备链接。主库规则同时跟踪头文件和构建配置变化。

## 6. 排障记录及最终处置

### 6.1 交接中已有的环境与配置问题

| 问题 | 定位与处理 | 当前结论 |
| --- | --- | --- |
| 多份 FlagCX 库和预装 wheel | 设置 `FLAGCX_PATH` / `LD_LIBRARY_PATH`，要求目标程序 `ldd` 核验，不以 Python import 作为设备测试 | 运行路径选择规则已明确；每次正式记录仍需保存实际库解析 |
| `nlohmann/json.hpp` 缺失 | 服务器官方 checkout 的 submodule 未补齐；前期通过初始化 submodule 解决 | 交接阶段已解决，不能整体上传本地不完整 third-party 覆盖服务器 |
| `dlink: No such file` / 缺少 `kernel_dlink.o` | MUSA 工具链字段和 dlink 配置不完整；用户已验证并同步 W-03 | 已承接；当前实际设备镜像采用显式链接和嵌入 |
| 测试 `No platform selected` | `test/make.inc` 缺少 USE_MUSA include 分支 | 用户同步的 W-03a 已解决 |
| `missing test/kernel/musa` | 当时不存在 MUSA 测试实现 | 已以真实源码和规则补齐，两个 object 已构建成功 |

上述前期事实来自交接和用户同步，不把它们写成本轮重新完成的服务器操作。

### 6.2 本轮设备适配中遇到的问题

| 现象 | 排查与尝试 | 最终状态 |
| --- | --- | --- |
| g++ 编译 SDK atomic 时缺少完整 half/BF16 类型 | 先补浮点头文件，仍不能单独解决；随后将原生 atomic 头文件、枚举和实现限制在 MUSA 编译边界，host 单独检查布局 | 已解决，host layout 编译和运行通过 |
| `llvm.musa.barrier.sync.cnt` 持续刷屏 | SDK-only counted barrier 也复现；按实际参与范围映射单 warp/全 CTA，保留原语复现 | 正常路径不再调用该原语；部分 CTA 多 warp 仍未支持 |
| 后端 `Cannot select ... ExternalSymbol'abort'` | 可疑来源为通用 builtin trap；MUSA 设备映射改为 MCCL 使用的原生 `__trap()`，增加独立编译检查 | 后续设备编译通过；不等于原生 trap 资源或运行行为已通过 |
| CoopAny 的 `R_MTGPU_ABS64` 重定位错误 | 编译和链接已带 PIC；检查发现公共类型擦除使用静态函数指针表，改为直接分派 | 已解决，后续构建及 groups-only GPU 检查通过 |
| 设备链接 `invalid buffer: size (0)` | `-v -save-temps=obj` 显示 mcc 连续两次设备链接，第一次已生成非空产物；改用显式分步构建 | 当前构建和正常分项 GPU 执行通过；具体哪一个空 buffer 未被单独确定 |
| `LLVMgold.so` 路径不存在 | 先修正到编译器真实目录，但文件仍不存在；确认它是驱动传给 lld 的参数，撤去 Makefile 强制文件依赖 | 链接通过；保留 SDK 驱动参数，不宣称已经安装插件或独立证明其内部处理方式 |
| `too many resources requested for launch` | 增加与实际启动一致的 64 线程 launch bounds、资源查询和逐 kernel 启动记录 | 更清楚地暴露动态 trap 路径异常；launch bounds 本身没有解决它 |
| `warpSpanProbe maxThreadsPerBlock=0` | 三个 kernel 资源不同；统一查询 API、零初始化、返回值检查和不同函数地址均已核对 | 不是测试项分别用了不同查询方式；继续 SDK-only 对照 |
| 尝试改变最终设备镜像链接模式 | 去掉来自中间链接的 `-shared --as-needed`，按 MCCL 最终镜像方式添加 O2 | 构建方式得到修正，但资源异常未消失，不能把该尝试记为 trap 修复 |
| 想换 `mp_43` 规避 | 脚本重探全部 GPU、runtime/driver 和两种目标编译 | 8 卡均 mp_31；当前 mp_43 编译失败，继续 mp_31 |
| 非 void GDA 失败分支缺少返回值告警 | 曾在原生 trap 后加 `__builtin_unreachable()` 以说明不返回，随后触发 `failed to process CFG` | 改动已撤回；强制重编恢复成功。告警仍独立跟踪，不返回伪造值、不静默抑制 |
| libflagcx.so 缺少三个 helper/launcher 符号 | MUSA 生产 Kernel 目录为空，共享库阶段未强制全部符号闭合，问题在测试最终链接暴露 | 已补实际实现和主库构建规则；主库及六个测试程序链接成功 |

### 6.3 动态 trap 的独立对照结果

五个 SDK-only kernel 使用同一构建链、同一个属性查询函数和 64 线程 launch bound。结果如下，强制重建后重复一致：

| 对照 | maxThreadsPerBlock | numRegs | sharedBytes | localBytes |
| --- | ---: | ---: | ---: | ---: |
| plain | 1024 | 2 | 12 | 0 |
| warp | 1024 | 1 | 268 | 0 |
| block | 1024 | 1 | 268 | 0 |
| trap | 0 | 256 | 12 | 16384 |
| guarded-warp | 0 | 256 | 268 | 16384 |

这组程序只查询 `musaFuncGetAttributes`，不启动 kernel、不执行 trap。可以确认当前构建/运行链路中动态 trap 路径与异常资源值相关，不能解释成“某线程运行失败后，系统把上限调为 0”。

它也不能仅凭这些数值把根因完全归于某个 SDK 层：编译器降级、trap handler 链接、运行时元数据解析和版本配套还需要厂商核对。

“把 trap 摘出去”指将问题从整个移植工作中单独跟踪，并增加分项验证入口；没有全局删除生产检查，也没有把不支持同步改成空操作。

### 6.4 仍然存在的告警

- 设备编译中的 `-g` unused：本轮实际编译仍成功。
- 显式设备链接中的 `_start` 缺失警告：当前设备镜像仍成功生成，正常分项 GPU 测试已经通过；不把它描述为所有镜像场景都已证明无影响。
- `DefaultNetGdaProvider::call` 非 void 返回路径告警：尚未解决，与错误终止契约一起跟踪；不能再次未经验证加入不可达标记。

## 7. 已取得的服务器验证证据

### 7.1 编译与链接

服务器实际完成以下构建：

- `libflagcx.so`，包含新增 MUSA 设备 helper / collective launcher。
- `test/kernel/musa/build/obj/device_api.o`。
- `test/kernel/musa/build/obj/device_ir.o`。
- `test_device_api_intra`、`test_device_api_inter`。
- `test_device_ir_intra`、`test_device_ir_inter`。
- `test_device_ir_unified_intra`、`test_device_ir_unified_inter`。

测试程序位于 `test/unittest/device_api/build/bin/`。六个可执行程序生成是构建证据，尚无其全部通信 case 运行结果。

### 7.2 最小 Kernel 与原语运行

| 测试 | 实际输出/资源 | 已证明的内容 |
| --- | --- | --- |
| host layout | `PASS: SIMT=32 mask=8 CoopAny=24 align=8` | 本轮检查项下的 host 布局与常量 |
| groups-only | `PASS (0 failures)`；64 线程；23 registers、268 shared bytes、0 local bytes | 基础协作组、CoopAny 转换/复制和共享内存交换 |
| atomics-only | `PASS (0 failures)`；64 线程；26 registers、12 shared bytes、0 local bytes | 32/64 位原子检查、争用计数和同 CTA 发布/获取 |
| cross-block-only | `PASS (8 rounds, 0 failures)`；两个 64 线程 CTA；64 registers、12 shared bytes、0 local bytes | 跨 CTA 的 32/64 位原子争用及 device-scope 数据发布 |

三个 GPU 分项的 kernel 属性线程上限均为 1024。调用流程包含资源预检、kernel launch、错误检查、设备同步、结果拷回和内容核验，不是只编译或只查询属性。

跨 CTA 每轮分别清零，用不同 payload 检查生产 CTA 的 64 个数据经 release/acquire 向消费 CTA 发布；消费者等待有上限，不设置跨 grid 屏障。

当前没有证明：动态 Span 整套通过、非法路径处理正常、跨 GPU/GPU-CPU 可见性、完整通信 Kernel 运行、T1–T4。设备内存上的 system-scope 操作成功，不扩展解释为跨系统域发布成功。

## 8. 源码与工具交付清单

以下路径相对于本地 `D:\LLM\FlagCX\FlagCX`，对应服务器根目录为 `/workspace/FlagCX`。

| 路径 | 作用 |
| --- | --- |
| `Makefile` | MUSA 主库设备源文件采用分步构建规则 |
| `makefiles/musa.mk` | MUSA 工具链、架构、runtime 链接、Default backend、主库设备源文件选择 |
| `makefiles/musa_kernel_rules.mk` | 主库设备编译、链接及镜像嵌入 |
| `test/make.inc` | 用户原先同步的 USE_MUSA 平台选择 |
| `flagcx/adaptor/include/device_utils.h` | MUSA 编译识别、装饰、runtime、同步/fence、原生 trap 映射 |
| `flagcx/adaptor/include/device_api/platform_traits.h` | MUSA 平台选择 |
| `flagcx/adaptor/include/device_api/comm_traits.h` | MUSA DeviceAPI 选择 |
| `flagcx/adaptor/include/device_api/musa_platform_traits.h` | Intrin、Atomic/Fence、Coop 和直接分派的 CoopAny |
| `flagcx/adaptor/include/device_api/musa_comm_traits.h` | 当前 DefaultBackend<MusaPlatform> 绑定 |
| `flagcx/adaptor/kernel/musa/device_api_host_helpers.cu` | 设备侧 DevNet 构造与大小 helper |
| `flagcx/adaptor/kernel/musa/flagcx_reduce_kernel_device.cu` | persistent FIFO 和 float/Sum collective launcher |
| `flagcx/core/flagcx_reduce_kernel_host.cc` | MUSA runner 入队前支持范围检查 |
| `test/kernel/musa/Makefile` | 正式测试 object 构建 |
| `test/kernel/musa/device_api.cu` | Device API 测试 kernel 与 launcher |
| `test/kernel/musa/device_ir.cu` | IR 测试 kernel 与 launcher |
| `test/kernel/musa/README.md` | Kernel 移植说明与验证边界 |
| `test/unittest/musa_platform/Makefile` | 独立平台 probe、诊断与分项入口 |
| `test/unittest/musa_platform/musa_host_layout.cc` | g++ host 布局检查 |
| `test/unittest/musa_platform/musa_platform_probe.cu` | 完整 probe 及三个已通过的 GPU 分项 |
| `test/unittest/musa_platform/musa_named_barrier_repro.cu` | 普通/计数 barrier SDK 对照源码 |
| `test/unittest/musa_platform/musa_native_trap_repro.cu` | 原生 trap 编译检查；不是错误路径运行通过证据 |
| `test/unittest/musa_platform/musa_launch_resources.cu` | 五组 SDK-only 资源属性对照 |
| `test/unittest/musa_platform/detect_arch.sh` | GPU/runtime/driver 属性和候选架构编译探测 |
| `test/unittest/musa_platform/README.md` | 独立测试使用方式与历史结果 |

公共 Host driver、launcher 声明、case 编号和 expected mask 没有因这轮适配被改写。已有的 `musa_adaptor.cc` IPC 等空槽位也没有在本轮伪造成功实现。

## 9. 按本人 A 项范围判断完成度

| A 项要求 | 当前证据 | 判断 |
| --- | --- | --- |
| COMPILE_KERNEL=1 的 MUSA 构建路径可用 | 主库、正式 Kernel object 和六个公共测试程序的服务器构建结果 | 已满足本轮构建范围 |
| 必要 Intrin、Atomic、Fence、Coop | 平台实现及三个独立 GPU 分项 | 基础正常路径已覆盖；必要项范围须包含能力边界说明 |
| 最小 Kernel 正确执行 | groups-only、atomics-only、cross-block-only 实际 GPU PASS | 已满足，非仅编译成功 |
| 输出编译日志与原语测试结果 | 会话中已有完整/片段终端输出、专项记录及复现命令 | 技术结果已有，原始日志和版本快照仍需正式归档 |

如果 A 项要求的是本轮已覆盖的基础原语和最小正常路径，当前可申请阶段验收。若验收双方把部分 CTA 多 warp、跨 GPU/CPU 可见性或其他扩展 API 列为必要项，需补对应实现/结果后再判断，不以目前 PASS 自动覆盖。

T1–T4 和完整 MCCL 通信联调属于后续项目集成验收，不自动变成本人 A 项独立负责的全部交付条件。设备侧问题仍需要本人配合定位和修复。

## 10. 未完成项与下一阶段安排

| 项目 | 尚缺内容 | 推进关系 |
| --- | --- | --- |
| 动态 trap / noreturn 路径 | 支持的终止原语、正确链接方式或修复版本 | 带最小复现与摩尔核对；独立跟踪，不阻塞所有源码准备 |
| counted barrier / 部分 CTA 多 warp | 真实参与数、组隔离及可见性语义 | 需要原生可用原语或带初始化契约的软件实现 |
| CoopOps / Clock | 按实际接口需求确认并实现 | 当前为空，未声明扩展能力完成 |
| GPU 间、GPU/CPU 可见性 | peer / host-mapped 内存上的实际数据发布 | 依赖可用资源映射与配套测试 |
| 当前 Default 资源路径 | IPC、host registration、通用 launch 等接口仍有空槽位 | 由相应资源模块与 Device/Kernel 配合，不单凭 Kernel 编译推进为运行完成 |
| 原生 MCCL 后端 | Host 资源、comm/window/context 表示、CommTraits 及完成语义 | 与对应模块负责人对齐；切换后重新配套构建和验证 |
| runner 扩展 | float/Sum 之外的类型与操作及运行验证 | 独立扩展，不绕过 Host 支持范围检查 |
| 正式测试运行 | 最小通信、逐项验证、最终 T1–T4 | 资源与 backend 配套就绪后推进 |
| 材料归档 | 原始编译日志、测试退出码、源码快照、实际库解析 | 作为 A 项阶段交付收尾 |

建议先完成 A 项的结果与能力清单归档，再与资源/通信模块对齐对象、backend、context、所有权和错误处理。资源就绪后从小规模单 size 通信用例开始，随后扩展同节点、跨节点及重复 size round，最后按 v3 执行正式验收。

## 11. 可复现命令与证据归档说明

下面是当前流程的复现入口，供交接使用，不要求为本总结立即重复已通过测试。

### 11.1 主库与公共测试构建

```bash
cd /workspace/FlagCX
export FLAGCX_SRC=/workspace/FlagCX
export FLAGCX_PATH="$FLAGCX_SRC"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH"

timeout --kill-after=5s 600s \
  make -j1 USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/kernel/musa all \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/unittest/device_api mpi \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

### 11.2 三个已通过的 GPU 分项

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

无参数的 `run` 仍是原完整 probe，包含尚未闭环的动态 Span；它与三个独立分项不是同一个通过范围。

### 11.3 已记录的服务器诊断位置

| 路径/来源 | 内容与注意 |
| --- | --- |
| `/tmp/musa_named_barrier_build.log` | 计数 barrier 刷屏日志；显示时应按字节截断 |
| `/tmp/musa_platform_link.log` | 设备链接 verbose 输出 |
| `test/unittest/musa_platform/build/link_diag/` | 当时保存的设备链接中间产物 |
| `test/unittest/musa_platform/build/arch_detect.zyR6N8/report.log` | 用户回传的架构重探报告位置 |
| `/tmp/device_api-c0c614/device_api-mp_31.cu` 与 `.sh` | 后端 CFG 崩溃时编译器给出的复现材料 |
| 本轮会话中用户回传的构建和 PASS 输出 | 本总结采用的服务器验证证据，尚未全部收集为仓库内原始日志文件 |

上述路径是用户当时输出中的位置，临时文件目前是否仍存在需在服务器确认。本地没有伪造这些服务器日志，也没有以本地检查替代服务器执行。

正式归档建议包含：服务器和本地 commit/branch、工作区 diff 与新增文件、编译器版本、GPU/runtime/driver 信息、实际构建命令、完整 stdout/stderr、退出码、目标程序 `ldd`、能力范围与遗留项。

## 12. 专项文档索引

- [MUSA 主库 Kernel 接入与链接验证](musa_main_library_kernel_integration.md)。
- [动态 trap 资源异常与 CFG 回归](musa_trap_resource_issue.md)。
- [counted barrier 编译异常](musa_counted_barrier_compiler_issue.md)。
- [正式测试 Kernel 移植说明](../test/kernel/musa/README.md)。
- [平台 probe 与 GPU 分项说明](../test/unittest/musa_platform/README.md)。
- [项目评估与推进计划](../../MCCL接入FlagCX_评估与推进计划.md)。
- [v3 验收文档](../../FlagCX厂商适配（v3）.md)。

验收口径以 v3 和双方明确的 A 项边界为准；本总结用于说明工作、证据与范围，不替代正式验收记录。
