# pto.vexp

`pto.vexp` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`pto.vexp` 在每个活跃 lane 上计算指数函数。活跃 lane 计算 `e^src[i]`；非活跃 lane 保持目标寄存器原值。

## 机制

`pto.vexp` 是标准的 `pto.v*` 计算指令，它对每个活跃 lane 独立应用指数函数：

对每个谓词位为 1 的 lane `i`：

$$ \mathrm{dst}_i = \exp(\mathrm{src}_i) $$

活跃 lane 会把指数值写入 `dst[i]`；非活跃 lane 则保持目标原值，不会被自动清零。

## 语法

### PTO 汇编形式

```text
%result = vexp %input, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vexp %input, %mask : (!pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vexp ins(%input, %mask : !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

声明位于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename VecDst, typename VecSrc, typename MaskT, typename... WaitEvents>
PTO_INST RecordEvent VEXP(VecDst& dst, const VecSrc& src,
                          const MaskT& mask, WaitEvents&... events);
```

## C 语义

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = expf(src[i]);
    // else: dst[i] unchanged
```

其中 `N` 由元素类型决定：

| 元素类型 | lane 数（N） | 说明 |
|----------|:------------:|------|
| `f32` | 64 | |
| `f16`、`bf16` | 128 | |
| `i8`、`u8` | 256 | 这里只是向量宽度示意，不表示 `vexp` 支持整数类型 |

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `exp(input[i])`；非活跃 lane 保持原值 |

## 副作用

无额外架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 源和目标寄存器的元素类型必须一致。
- 源和目标寄存器的向量宽度必须一致。
- 掩码宽度必须与 `N` 相同。
- 非活跃 lane 的目标值保持不变，不能假定它们被清零。
- 溢出、下溢和 NaN 处理遵循目标 profile 的浮点异常规则。

## 异常与非法情形

- verifier 会拒绝非法的类型不匹配、宽度不匹配或掩码宽度不匹配。
- 数值溢出、NaN 等异常由目标平台定义；除非文档明确收窄，否则代码不应依赖某个特定异常值。
- [一元向量操作](../../unary-vector-ops_zh.md)页声明的额外非法情形，同样适用于 `pto.vexp`。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|----------|:-------------:|:-----:|:--:|
| `f32` | 模拟 | 模拟 | 支持 |
| `f16` | 模拟 | 模拟 | 支持 |

A5 是当前向量特殊函数最具体的实现 profile。CPU 模拟器和 A2/A3 类目标会以标量循环或等效路径模拟这些超越函数，但必须保留可见的 PTO 契约。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 16 | `RV_VEXP` |
| `f16` | 21 | `RV_VEXP` |

### A2/A3 吞吐

| 指标 | f32 | f16 |
|------|-----|-----|
| 启动时延 | 13（`A2A3_STARTUP_REDUCE`） | 13 |
| 完成时延 | 26（`A2A3_COMPL_FP32_EXP`） | 28（`A2A3_COMPL_FP16_EXP`） |
| 每次 repeat 吞吐 | 2 | 4 |
| 流水间隔 | 18 | 18 |

示例：1024 个 f32 元素，16 次迭代：

```text
A5（流水重叠）：16 + 15×2 = 46 周期
A2/A3：13 + 26 + 32 + 270 = 341 周期
```

**性能说明：** 对数值稳定的 softmax，更推荐使用 `vexpdiff` 融合形式，而不是先 `vsub` 再 `vexp`。

## 示例

### Softmax 分子（数值稳定形式）

```mlir
%max_bc = pto.vlds %ub_max[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%sub = pto.vsub %x, %max_bc, %mask
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

### C++ 用法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void exp_vector(VReg<64, float>& dst, const VReg<64, float>& src, Mask<64>& mask) {
    VEXP(dst, src, mask);
}
```

## 详细说明

指数函数 `exp(x)` 计算的是 `e^x`，其中 `e ≈ 2.71828`。对浮点目标而言：

- `exp(+INF) = +INF`
- `exp(-INF) = +0`
- `exp(NaN) = NaN`
- 很大的正输入可能溢出为 `+INF`
- 很大的负输入可能下溢为 `+0`

对 softmax 这类数值敏感路径，优先计算 `exp(x - max(x))` 而不是直接 `exp(x)`。

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vneg](./vneg_zh.md)
- 下一条指令：[pto.vln](./vln_zh.md)
- 向量指令总览：[向量指令面](../../../instruction-surfaces/vector-instructions_zh.md)
- 类型系统：[类型系统](../../../state-and-types/type-system_zh.md)
