# MUSA Device API / Unified IR 测试 Kernel

2026-11-11：Linux 服务器已完成两个正式测试 object 的设备编译、设备链接和
host 编译/镜像嵌入，产出 `device_api.o`、`device_ir.o`。主库补齐设备
helper/launcher 后已重建成功，六个公共测试程序也已全部链接成功。
通信运行尚未验证，修复与验证命令见
[主库 Kernel 接入说明](../../../docs/musa_main_library_kernel_integration.md)。

## 本轮实现

| 文件 | 内容 |
| --- | --- |
| `Makefile` | 设备编译、显式设备链接、host 编译并嵌入镜像，产出公共测试请求的 `device_api.o` / `device_ir.o` |
| `device_api.cu` | Device API 测试 kernel 与 27 个公共 launcher，MUSA runtime 类型、stream、错误检查和启动 |
| `device_ir.cu` | Scalar / Unified IR 测试 kernel 与 48 个公共 launcher，使用 MUSA adaptor 和通用指针 ABI |

测试主体取自本仓库 `test/kernel/iluvatar` 的公共测试实现，平台原语由前序从
MCCL 移植的 MUSA PlatformTraits 提供。没有把 Iluvatar 私有的负向测试、原子
契约检查 kernel 或 CoreX AS1 launcher 桥接复制到 MUSA。MUSA 的
`FLAGCX_IR_GLOBAL_PTR` 为空，不需要额外的地址空间重载。

K10 的 Tile<8> 恢复了与 NVIDIA 参考相同的 rank、size 和真实同步检查，没有
沿用模板中的固定成功结果。S2 的 span 按完整 CTA 构造，128 个线程在 S5000
对应四个 warp。公共 Host driver、case 编号、expected mask 和 launcher 头文件
均未修改；没有缩减 S16–S25 的期望覆盖。

`makefiles/musa.mk` 同时补入 `DEVICE_LINK := -lmusart -lmusa`，供公共测试的
host 链接器解析 MUSA kernel object 中直接引用的 runtime 符号。主库原有
平台选择和 Default backend 选择保持不变，尚未接入原生 MCCL CommTraits。

## 本地检查与验证边界

- 已静态核对 27 个 Device API、48 个 IR 公共 launcher 的存在、唯一性、返回
  类型及参数类型；对应公共头文件中的有效声明均有定义。
- 已检查无残留 CUDA runtime 名称、CoreX 专用 ABI 桥和固定 Tile<8> 成功值。
- 显式分步编译方式已在独立平台 probe 和两个正式 kernel object 的服务器
  构建中通过。完整通信正确性尚未验证。
- 不在本机安装 MUSA SDK，不在本机运行编译或 GPU 测试。

服务器构建仍报告 DefaultNetGdaProvider 的非 void 返回路径告警。该分支
调用失败原语后没有返回；mcc 未将 SDK `__trap()` 识别为不返回。为消除告警
加入 `__builtin_unreachable()` 后，服务器在 `MTGPU annotate control flow`
处理 `flagcxInterTwoSidedAlltoAllKernel` 时崩溃，报告 `failed to process CFG`。
本地已撤回这一改动，服务器强制重编已恢复两个 object 构建成功。未抑制告警，
也未伪造 Signal/Counter 返回值；告警与错误终止契约
继续在独立 SDK 问题中跟踪。

## object 构建复现

同步本目录三个代码文件和 `makefiles/musa.mk` 到 `/workspace/FlagCX`。
本次改动是在已有 MUSA PlatformTraits / device_utils 等前序补丁基础上继续。

```bash
cd /workspace/FlagCX
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

timeout --kill-after=5s 180s \
  make -C test/kernel/musa device_api.o \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31

timeout --kill-after=5s 180s \
  make -C test/kernel/musa device_ir.o \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

两个命令分别执行，保留各自输出及退出码，不把第一个失败隐藏为第二个成功。
默认产物位于 `test/kernel/musa/build/obj/`。每个最终 `.o` 嵌入该源文件已经
显式链接的设备镜像，不能再送给 mcc 自动重复设备链接。

同步最新 `flagcx/adaptor/include/device_utils.h` 后，先重建 object，再按当前
主库配置构建公共测试程序（含四个正式验收入口及两个 Scalar IR 入口）：

```bash
timeout --kill-after=5s 300s \
  make -B -C test/kernel/musa all \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 &&
timeout --kill-after=5s 300s \
  make -C test/unittest/device_api mpi \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

该命令还依赖同一配置的 libflagcx、MPI、测试依赖及配套 Host 资源实现。
先检查编译、链接和实际加载的库；不因可执行文件生成就宣布通信接入完成。

## 后续运行工作

1. 对齐平台基础能力、Host 资源和通信 CommTraits，先运行最小用例。
2. 逐步验证同节点通信、跨节点通信及 S16–S25；不改公共 PASS/FAIL 判定。
3. 统一构建后，按验收文档执行 T1–T4，保存全 rank 结果。

动态 trap 的 SDK 兼容问题在
[独立问题说明](../../../docs/musa_trap_resource_issue.md)中跟踪。已有动态范围检查
仍然存在；含此路径的 kernel 可能继续出现资源异常。固定约束前移与设备错误
传播尚待实现，当前 object 移植不能作为这些路径已通过的证据。部分 CTA 多
warp span、CoopOps / Clock 的支持范围也需要继续落实。
