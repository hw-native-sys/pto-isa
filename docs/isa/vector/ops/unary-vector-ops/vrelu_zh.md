# pto.vrelu

`pto.vrelu` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`%result` 在每个活跃 lane 上保存 `max(input[i], 0)` 的结果。

## 机制

`pto.vrelu` 逐 lane 应用 ReLU：`dst[i] = max(src[i], 0)`。对浮点类型，负值被压成 0，零和正值原样通过。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vrelu %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
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
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `max(src[i], 0)`；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 当前 A5 形式只支持浮点元素类型。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vrelu` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 5 | `RV_VRELU` |
| `f16` | 5 | `RV_VRELU` |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 19 | `A2A3_COMPL_FP_BINOP` |
| 每次 repeat 吞吐 | 1 | `A2A3_RPT_1` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

**性能说明：** `vrelu` 是 A5 上时延最低的一类一元向量操作。若需要带斜率的版本，应使用 `vlrelu`。

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vrec](./vrec_zh.md)
- 下一条指令：[pto.vnot](./vnot_zh.md)
