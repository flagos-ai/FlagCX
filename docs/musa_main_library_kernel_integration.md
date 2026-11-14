# MUSA 主库设备 Kernel 接入与链接验证

2026-11-11。服务器已在撤回 `__builtin_unreachable()` 后强制重建两个正式
测试 object 成功。公共测试随后在链接 `test_device_ir_intra` 时失败：

```text
libflagcx.so: undefined reference to flagcxDevNetLaunchConstruct
libflagcx.so: undefined reference to flagcxLaunchCollectiveKernel(...)
libflagcx.so: undefined reference to flagcxDevNetSizeOf
```

**最新结果：本轮主库 Kernel/helper 已在服务器完成设备编译、设备链接和
host 镜像嵌入，`libflagcx.so` 重建成功。六个公共测试程序全部链接成功：**
`test_device_api_intra/inter`、`test_device_ir_intra/inter`、
`test_device_ir_unified_intra/inter`。三个缺失符号导致的链接失败已解决。
未收到 GPU 通信运行通过的结果，不能把构建完成记为 T1–T4 通过。

## 原因与本轮实现

此前 MUSA 的 `PLATFORM_KERNEL_SRCS` 通配目录为空，主库在
`COMPILE_KERNEL=1` 下包含了调用这些接口的 Host 实现，却没有对应设备
编译单元。共享库构建未强制所有符号闭合，因此问题直到测试程序最终链接
时才暴露。缺失的是主库的生产 Kernel/helper，不是已经构建成功的两个
`test/kernel/musa` object。

| 文件 | 实现 |
| --- | --- |
| `flagcx/adaptor/kernel/musa/device_api_host_helpers.cu` | 实际返回 DevNet 大小，启动 kernel 在设备内存逐 context placement-new 构造 DevNet |
| `flagcx/adaptor/kernel/musa/flagcx_reduce_kernel_device.cu` | 实际 persistent FIFO 消费、float/Sum 计算、release 完成标志及 collective launcher |
| `makefiles/musa_kernel_rules.mk` | 将上述主库源文件分为设备编译、显式链接、host 编译/镜像嵌入三步，跟踪设备头文件和构建配置依赖 |
| `Makefile` | 仅 MUSA 路径选择新规则，其他平台仍使用原来的设备编译规则 |
| `flagcx/core/flagcx_reduce_kernel_host.cc` | MUSA 的 collective FIFO 入队前检查 datatype/operation，未支持组合返回 `flagcxNotSupported` |

helper 和 persistent runner 参考当前仓库的生产实现。MUSA FIFO 的
consumed/produced/terminate 和完成标志使用与 Host 一致的 64 位原子，
没有照搬 CoreX 的低 32 位控制字方案。消费使用 acquire、CAS 使用 acq_rel、
完成发布使用 release；跨 Host/GPU 可见性仍需要服务器运行验证。

## 明确的支持范围

当前 persistent reduction kernel 只实现 **float/Sum**，与本仓库参考 kernel
的计算主体相同。其他 datatype 或 redop 在 Host 发布 FIFO 描述符之前返回
`flagcxNotSupported`，没有用 float 加法冒充支持其他操作。

这个范围仅针对 unified runner 的设备侧本地归约，不是对 MCCL 原生
collective 能力的限制。完整 datatype/redop 分派需要后续扩展。当前实现
也不代表原生 MCCL CommTraits、通信资源生命周期和 T1–T4 已完成。

现有原生 trap 保护没有关闭；相关 SDK 资源异常仍在
[独立文档](musa_trap_resource_issue.md)中跟踪。本轮解决符号缺失和设备源码
构建接入，不宣称 GPU 运行或异常路径已验证。

## 服务器验证顺序

同步上表五个代码文件到 `/workspace/FlagCX` 下对应路径，不需要 pip 安装。

```bash
cd /workspace/FlagCX
export FLAGCX_SRC=/workspace/FlagCX
export FLAGCX_PATH="$FLAGCX_SRC"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH"

timeout --kill-after=5s 600s \
  make -j1 USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

主库成功后确认三个符号均为定义：

```bash
nm -D -C --defined-only build/lib/libflagcx.so | \
  grep -E 'flagcxDevNet(SizeOf|LaunchConstruct)|flagcxLaunchCollectiveKernel'
```

应看到三个对应定义，不能仍为 `U`（上述 `--defined-only` 已过滤未定义项）。
再构建公共测试程序：

```bash
timeout --kill-after=5s 300s \
  make -C test/unittest/device_api mpi \
  USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

主库与六个程序的编译链接已在服务器通过，GPU 运行尚待验证。编译和链接后
核验可执行程序实际加载的库，再开展最小通信运行，最后按验收规范执行完整
T1–T4。本地仅核对符号定义、FIFO 位宽、构建引用和补丁格式。

## 平台基础能力的 GPU 验证结果

当前本地 MUSA adaptor 的 IPC Create/Get/Open/Close/Free 仍返回
`flagcxNotSupported`，hostRegister/hostUnregister 也未实现，通用 launchKernel
函数表项仍为 NULL。公共多 rank 用例可能在资源初始化阶段失败；运行联调
需要与相应资源模块配合。这些缺口没有在本轮通过空桩掩盖。

独立平台 probe 新增 `run-groups` / `run-atomics` 两个分项入口，先在单 GPU
验证协作组、原子和 fence，避免把 Host IPC 当作这些基础能力的运行前置。
默认完整入口仍保留，含动态 Span 的检查没有删除；分项结果不代表完整 probe
或通信验收通过。同步 probe 的 Makefile 和 `.cu` 后分别执行：

```bash
timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-groups \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31

timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-atomics \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

上述两个分项已在服务器通过：

| 分项 | 服务器结果 | 结论范围 |
| --- | --- | --- |
| host layout | PASS，SIMT=32 / mask=8 / CoopAny=24 / align=8 | Host 对象布局检查 |
| groups-only | PASS，0 failures；23 registers / localBytes=0 | 64 线程下 Block/Warp/Tile/Lanes/Thread/CoopAny 的转换及同步 |
| atomics-only | PASS，0 failures；26 registers / localBytes=0 | 32/64 位原子操作、争用与同 CTA 两 warp release/acquire 发布，执行 device/system fence 调用 |
| cross-block-only | PASS，8 轮、0 failures；64 registers / sharedBytes=12 / localBytes=0 | 两个 64 线程 CTA 的设备级原子争用与 release/acquire payload 发布 |

scope 映射和单 GPU 上的 system-scope 操作不能替代跨 GPU / GPU-CPU 的
可见性验证。动态 Span 和负向 trap 路径仍在独立问题中跟踪。

跨 CTA 分项的复现命令：

```bash
timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-cross-block \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

服务器已通过：两个 64 线程 CTA，8 轮分别清零，检查 32/64 位
争用计数，以及生产 CTA 的 64 个 payload 经 device-scope release/acquire
向消费 CTA 发布。它不创建 Host DevComm/window/IPC，也不替代通信验收。

## 当前阶段与联调前置清单

Device / Kernel 已取得以下证据：MUSA 构建链、主库和六个测试程序编译链接、
Host ABI 检查、协作组基础分项、同 CTA 原子分项及跨 CTA 发布分项通过。
这是单 GPU 基础正常路径的阶段成果，不能写成全部 PlatformTraits、完整
Fence 语义或 MCCL 通信接入已经验收。

| 后续事项 | 当前状态与配合关系 |
| --- | --- |
| 动态 Span / 非法操作处理 | Device/Kernel 继续落实固定约束前移和合法路径；动态 trap 仍独立跟踪，部分 CTA 多 warp span 仍缺真实同步方案 |
| 跨 GPU、GPU/CPU 可见性 | 尚未验证；需要可用 peer 映射或 host-mapped 内存，单 GPU PASS 不覆盖这些可见性域 |
| 当前 Default 路径的 IPC / host registration | 对应 adaptor 仍有未实现接口；由资源层与 Device/Kernel 对接后开展运行联调 |
| MCCL Host 资源 / 原生 CommTraits | 需要配套提供 comm/window/context 等对象并选择一致 backend；本轮仍为 DefaultBackend<MusaPlatform> |
| 正式 Kernel 联调 | 资源闭环后先跑小规模单 size 用例，记录最早失败位置、rank 和退出码，再扩展同节点/跨节点组合 |
| CoopOps / Clock | 当前为空类型；按实际 Device API/IR 调用范围列出需求并补齐，尚未对扩展接口声明完成 |
| T1–T4 | 尚未执行；最后统一配置构建并按验收范围运行 |

对接时需明确：使用当前 Default 路径还是原生 MCCL 路径、资源所有权和销毁
顺序、comm/window 的设备表示、context 编号及错误传播。若切换到原生 MCCL，
应在新的配套配置下重新构建和验证，不能沿用 Default 路径的结果当作通信证明。
