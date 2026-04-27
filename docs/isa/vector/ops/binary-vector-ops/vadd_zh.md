# pto.vadd

`pto.vadd` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vadd` 对两个向量寄存器做逐 lane 加法，结果写入目标向量寄存器。只有谓词掩码选中的 lane 参与计算；非活跃 lane 不参与加法，目标寄存器对应位置保持不变。

## 机制

`pto.vadd` 是标准的 `pto.v*` 计算指令。它按 lane 同时读取两个源向量寄存器，把对应元素相加，再写回目标向量寄存器。逻辑迭代域覆盖全部 `N` 个 lane，真正参与计算的是谓词位为真的那一部分。

对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i + \mathrm{rhs}_i $$

谓词为假的 lane 处于非活跃状态，目标寄存器在这些位置上不被修改。

## 语法

### PTO 汇编形式

```text
vadd %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vadd %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vadd ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

声明位于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename VecDst, typename VecLhs, typename VecRhs, typename MaskT, typename... WaitEvents>
PTO_INST RecordEvent VADD(VecDst& dst, const VecLhs& lhs, const VecRhs& rhs,
                          const MaskT& mask, WaitEvents&... events);
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 左操作数向量寄存器 |
| `%rhs` | `!pto.vreg<NxT>` | 右操作数向量寄存器 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与运算 |

两个源寄存器必须有相同的元素类型和相同的向量宽度 `N`。掩码宽度也必须与 `N` 一致。

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%dst` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 和；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `%lhs`、`%rhs` 和 `%dst` 的元素类型必须完全一致。
- 三个寄存器必须具有相同的向量宽度 `N`。
- `%mask` 的宽度必须等于 `N`。
- 只有谓词位为 1 的 lane 参与加法。
- 非活跃 lane 对应的目标元素保持不变。

## 异常与非法情形

- verifier 会拒绝非法的元素类型不匹配、向量宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面声明的额外非法情形，同样属于 `pto.vadd` 的契约。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|-----------|:-------------:|:-----:|:--:|
| `f32` | 模拟 | 模拟 | 支持 |
| `f16` / `bf16` | 模拟 | 模拟 | 支持 |
| `i8`–`i64`、`u8`–`u64` | 模拟 | 模拟 | 支持 |

A5 是当前文档中最具体的向量实现 profile。CPU 模拟器和 A2/A3 类目标通过标量循环或等效路径模拟 `pto.v*`，但仍需保留可见的 PTO 语义。任何依赖具体时延或吞吐的代码，都应把这类依赖视为 profile 相关行为。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 7 | `RV_VADD` |
| `f16` | 7 | `RV_VADD` |
| `i32` | 7 | `RV_VADD` |
| `i16` | 7 | `RV_VADD` |
| `i8` | 7 | `RV_VADD` |

### A2/A3 吞吐

| 指标 | 数值 | 适用范围 |
|------|------|----------|
| 启动时延 | 14（`A2A3_STARTUP_BINARY`） | 全部 FP / INT 二元运算 |
| 完成时延：FP32 | 19（`A2A3_COMPL_FP_BINOP`） | f32、i32 |
| 完成时延：INT16 | 17（`A2A3_COMPL_INT_BINOP`） | int16 |
| 每次 repeat 吞吐 | 2（`A2A3_RPT_2`） | 全部二元运算 |
| 流水间隔 | 18（`A2A3_INTERVAL`） | 全部向量运算 |
| 周期模型 | `14 + C + 2R + (R-1)×18` | `C` 为完成时延，`R` 为 repeats |

示例：1024 个 f32 元素，16 次迭代（`R=16`）：

```text
A5 总周期（流水重叠后）：7 + 15×2 = 37
A2/A3 总周期：14 + 19 + 32 + 270 = 335
```

`vadd` 的每次 repeat 吞吐高于不少一元操作，因此在纯逐元素 kernel 里通常是最容易饱和流水线的一类运算。

## 示例

### 全向量加法（全部 lane 活跃）

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

Mask<64> mask;
mask.set_all(true);

VADD(vdst, va, vb, mask);
```

### 部分谓词化

```mlir
%result = pto.vadd %va, %vb, %cond
    : (!pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask) -> !pto.vreg<128xf16>
```

### 完整的 load / compute / store 链

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vector_add(Ptr<ub_space_t, ub_t> ub_a, Ptr<ub_space_t, ub_t> ub_b,
                Ptr<ub_space_t, ub_t> ub_out, size_t count) {
    VReg<64, float> va, vb, vdst;
    Mask<64> mask;
    mask.set_all(true);

    VLDS(va, ub_a, "NORM");
    VLDS(vb, ub_b, "NORM");
    VADD(vdst, va, vb, mask);
    VSTS(vdst, ub_out);
}
```

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 下一条指令：[pto.vsub](./vsub_zh.md)
- 向量指令总览：[向量指令面](../../README_zh.md)
- 类型系统：[类型系统](../../../state-and-types/type-system_zh.md)
