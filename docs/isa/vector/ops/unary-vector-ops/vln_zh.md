# pto.vln

`pto.vln` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`%result` 在每个活跃 lane 上保存自然对数结果。

## 机制

`pto.vln` 逐 lane 计算自然对数：`dst[i] = ln(src[i])`。在实数语义下，活跃输入最好严格大于 0；非正输入的异常或 NaN 行为由目标平台决定。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vln %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`f16`、`f32`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器；在每个活跃 lane 上读取 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `ln(src[i])`；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 只支持浮点元素类型。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 除零、域外输入等数值异常，仍由所选 backend profile 决定。
- 约束部分列出的额外非法情形，同样属于 `pto.vln` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 18 | `RV_VLN` |
| `f16` | 23 | `RV_VLN` |

### A2/A3 吞吐

| 指标 | f32 | f16 |
|------|-----|-----|
| 启动时延 | 13（`A2A3_STARTUP_REDUCE`） | 13 |
| 完成时延 | 26（f32）/ 28（f16） | `A2A3_COMPL_FP32_EXP` / `A2A3_COMPL_FP16_EXP` |
| 每次 repeat 吞吐 | 2 | 4 |
| 流水间隔 | 18 | 18 |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

**输入定义域：** `logf(x)` 只在 `x > 0` 时有实数意义。非正活跃输入会产生 NaN 或目标定义的异常结果。

### 数值稳定性说明

对于 softmax 分母这类 `log(sum(exp(x - max)))` 模式，通常优先通过 `vexpdiff` 等融合路径避免不必要的数值风险。

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vexp](./vexp_zh.md)
- 下一条指令：[pto.vsqrt](./vsqrt_zh.md)
