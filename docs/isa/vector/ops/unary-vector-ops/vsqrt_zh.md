# pto.vsqrt

`pto.vsqrt` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`%result` 在每个活跃 lane 上保存平方根结果。

## 机制

`pto.vsqrt` 逐 lane 计算平方根。对每个谓词位为真的 lane `i`，执行 `dst[i] = sqrt(src[i])`。负输入的定义域行为由目标平台决定。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vsqrt %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
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
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `sqrt(src[i])`；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 只支持浮点元素类型。
- 负的活跃输入会触发目标定义的异常或 NaN 规则。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 域外输入、NaN 等数值异常由目标 profile 决定。
- 约束部分列出的额外非法情形，同样属于 `pto.vsqrt` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 17 | `RV_VSQRT` |
| `f16` | 22 | `RV_VSQRT` |

### A2/A3 吞吐

| 指标 | f32 | f16 |
|------|-----|-----|
| 启动时延 | 13（`A2A3_STARTUP_REDUCE`） | 13 |
| 完成时延 | 27（`A2A3_COMPL_FP32_SQRT`） | 29（`A2A3_COMPL_FP16_SQRT`） |
| 每次 repeat 吞吐 | 2 | 4 |
| 流水间隔 | 18 | 18 |

### 执行说明

`vsqrt` 走 SFU（Special Function Unit）路径。对于 1024 元素、16 次迭代的向量循环，可以近似理解为：

```text
A5 f32：首发 17 周期，之后约每次迭代 2 周期推进
A5 f16：首发 22 周期，之后约每次迭代 4 周期推进
```

`vrsqrt` 与它共享同一类 SFU 路径，因此成本也接近。

## 示例

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

`sqrtf` 遵循 IEEE 754 的平方根语义。对浮点类型，负的活跃输入会产生 NaN 或目标定义的异常结果。

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vln](./vln_zh.md)
- 下一条指令：[pto.vrsqrt](./vrsqrt_zh.md)
