# MUSA 动态 __trap 路径导致 kernel 资源属性异常

## 跟踪状态与移植边界

2026-11-11：作为独立 SDK 兼容问题归档。强制重新构建 SDK-only 对照后，
五组资源属性与前次完全一致，异常可以重复。尚未获得厂商修复或安全替代原语。

该问题不作为 Kernel 源码移植、launcher 对接和构建接入的全局前置条件。
这些工作继续推进；调用动态 trap 的设备路径仍受影响，不能标为运行验证通过。
目前保留原有检查，没有全局关闭 trap，没有改变通信或同步的参与范围。

后续可以把可证明的固定约束放到编译期，把启动维度等约束放到 host 启动前
检查。涉及运行时设备数据的检查需要单独设计错误传播，不能通过移除检查
宣布支持。Kernel 接入进度与下一步命令见
[MUSA 测试 Kernel 说明](../test/kernel/musa/README.md)。

## 追加：trap 后显式不可达标记触发编译器崩溃

两个正式测试 object 已在只调用 `__trap()` 的包装下完成构建，但 mcc 对
DefaultNetGdaProvider 的非 void 失败分支报告缺少返回值。随后将 MUSA 宏
改为 `do { __trap(); __builtin_unreachable(); } while (0)`，设备编译出现回归：

- mcc 退出码 70：`fatal error: error in backend: failed to process CFG`。
- Pass：`MTGPU annotate control flow`。
- 函数：`flagcxInterTwoSidedAlltoAllKernel`。
- 原编译参数、架构和 kernel 源码未因这一补丁改变。
- 编译器保存的复现文件：
  `/tmp/device_api-c0c614/device_api-mp_31.cu`、
  `/tmp/device_api-c0c614/device_api-mp_31.sh`（位于用户服务器）。

本地已撤回不可达标记，服务器强制重编后两个正式测试 object 已恢复构建成功。
改动与回归有
直接时间关联，仍需最小化 IR 复现才能确定后端内部根因。这也说明在当前
工具链中不能未经验证就将原生 trap 包装为 noreturn 路径。

## 已确认的现象

在目标 Linux 服务器上，SDK-only 对照不包含任何 FlagCX 头文件。五个
kernel 使用相同的 `__launch_bounds__(64)`，在同一进程内通过同一个函数调用
`musaFuncGetAttributes`，结构零初始化，检查 API 返回值。

| kernel | 差异 | maxThreadsPerBlock | numRegs | sharedBytes | localBytes |
| --- | --- | ---: | ---: | ---: | ---: |
| plain | 写出 threadIdx | 1024 | 2 | 12 | 0 |
| warp | 共享内存 + __syncwarp | 1024 | 1 | 268 | 0 |
| block | 共享内存 + __syncthreads | 1024 | 1 | 268 | 0 |
| trap | 写出 threadIdx 前增加 if(fail) __trap() | 0 | 256 | 12 | 16384 |
| guarded-warp | warp 前增加相同的条件 trap | 0 | 256 | 268 | 16384 |

`fail` 为 kernel 运行时参数，防止条件 trap 被编译器删除。对照程序只查询
属性，没有启动 kernel，也没有实际触发 trap。不能将查询命令成功退出解释
为 GPU 正确性通过。

原 FlagCX probe 的资源预检结果：

| kernel | maxThreadsPerBlock | numRegs | sharedBytes | localBytes |
| --- | ---: | ---: | ---: | ---: |
| platformProbe | 128 | 256 | 268 | 16384 |
| warpSpanProbe | 0 | 256 | 268 | 16384 |
| anyGroupProbe | 1024 | 23 | 268 | 0 |

host layout 已 PASS。早先完整 probe 首次启动曾返回
`too many resources requested for launch`；增加资源预检后，在发现线程上限
为 0 时退出，GPU 正确性检查尚未执行。

## 环境与复现材料

- GPU：MTT S5000，目标架构 mp_31，已记录 warpSize=32。
- mcc 5.1.0 / clang 14.0.0。
- 编译器 commit：`39eee21af373855ae51b0e906a74b02950384a05`。
- 驱动日志识别 MUSA 4.3，工具实际目录 `/usr/local/musa-4.3.5`。
- 编译器入口 `/usr/local/musa/bin/mcc`。
- 独立架构探测已确认：8 张可见 GPU 全部为 MTT S5000，计算能力 3.1、
  warpSize=32、设备最大线程块大小 1024。
- `musaRuntimeGetVersion` 与 `musaDriverGetVersion` 均返回 `40305`。
- `ldd` 确认探测程序加载 `/usr/local/musa/lib/libmusart.so.4`。
- 同一最小 device-only 编译：`mp_31` 成功；`mp_43` 失败，报告
  `No available targets are compatible with triple "mthg-mt-musa"`。
  该结果说明当前编译器/目标组合无法生成此测试的 mp_43 对象，不代表列举了
  所有 MUSA 工具链的架构支持范围。当前服务器继续使用 mp_31。
- 最小对照源码：`test/unittest/musa_platform/musa_launch_resources.cu`。
- 构建规则：`test/unittest/musa_platform/Makefile` 的 `launch-resources` 目标。

```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
timeout --kill-after=5s 60s \
  make -C test/unittest/musa_platform launch-resources \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

## 当前设备镜像构建方式

使用与 MCCL 源码相同的 device-only / 显式设备链接 / host-only 镜像嵌入
分工，最终由 host 编译器链接运行时。具体命令以复现日志为准：

1. mcc `-std=c++17 -O2 -c -mtgpu --musa-path=/usr/local/musa
   --offload-arch=mp_31 -fgpu-rdc -fPIC --cuda-device-only`。
2. SDK lld `-flavor gnu --no-undefined -plugin <toolchain>/lib/LLVMgold.so
   -plugin-opt=mcpu=mp_31 -plugin-opt=O2 <device.o> <resource>/lib/linux/trap_handler.o
   -o <image>`。
3. 同一 mcc 执行 host-only 编译，使用
   `-Xclang -fcuda-include-gpubinary -Xclang <image>` 嵌入镜像。
4. g++ 链接 host object，`-L/usr/local/musa/lib -lmusart -lmusa`。

`LLVMgold.so` 在该服务器上不存在，但 SDK lld 接受驱动同样传入的这个参数，
并完成链接。最终镜像链接有 `_start` 缺失警告。此前使用
`-shared --as-needed` 时也出现了原 probe 的资源异常；去掉后没有消除异常。
显式镜像构建是否完全符合该 SDK 的要求，仍需要厂商核对。

## 结论边界与待厂商确认

对照证明：在当前构建和运行链路下，加入动态 `__trap()` 足以使简单
SDK-only kernel 出现相同资源异常。它排除了必须依赖 FlagCX 复杂逻辑才能
复现的假设，但不能独立区分编译器降级、trap handler 链接和运行时元数据
解析，也不能仅凭这些属性彻底排除驱动版本或 ABI 兼容问题。

请摩尔确认：

1. mcc 5.1.0 / mp_31 是否支持在 kernel 动态错误路径调用 `__trap()`？
2. 256 registers / 16384 local bytes / maxThreadsPerBlock=0 是否是已知问题？
3. 对该版本，`trap_handler.o` 应如何链接？上述最终镜像和嵌入方式是否正确？
4. 是否需要配套编译器、runtime 或驱动升级？请给出可验证版本组合。
5. 若原生 trap 路径不可用，支持哪一种能终止错误 kernel 并向 host 报告失败
   的原语？需要保留非法协作组/同步操作的失败语义。

FlagCX 当前不删除范围检查，不把 trap 改为空操作，不以提前 return
替代集体同步，不改变同步参与线程来绕过问题。尚无本地源码证据可以确认
一个安全、可用的替代终止原语。

## 可继续提取的定位材料

以下操作仅导出 IR 和实际动态库解析结果，不执行错误路径：

```bash
timeout --kill-after=5s 60s /usr/local/musa/bin/mcc \
  -std=c++17 -O2 -mtgpu --musa-path=/usr/local/musa \
  --offload-arch=mp_31 -fgpu-rdc -fPIC --cuda-device-only \
  -S -emit-llvm test/unittest/musa_platform/musa_launch_resources.cu \
  -o /tmp/musa_launch_resources.ll

grep -nE 'define.*resource|call.*(trap|abort|assert)|declare.*(trap|abort|assert)' \
  /tmp/musa_launch_resources.ll

ldd test/unittest/musa_platform/build/musa_launch_resources
```

如需区分构建模式，应由厂商提供其认可的最小程序构建命令，并对同一源码
比较资源属性；不能用移除错误检查后的结果宣布平台适配完成。
