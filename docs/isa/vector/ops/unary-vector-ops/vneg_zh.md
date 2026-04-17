# pto.vneg

`pto.vneg` 属于[一元向量操作](../../unary-vector-ops_zh.md)指令集。

## 概述

`%result` 在每个活跃 lane 上接收算术取负结果。

## 机制

`pto.vneg` 逐 lane 计算算术负值。对每个谓词位为真的 lane `i`，执行 `dst[i] = -src[i]`。当前实现通过标量乘法硬件路径，用 `-1` 乘数完成这一操作。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vneg %result, %input, %mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 当前文档化支持的类型：`i8-i32`、`f16`、`f32`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器；在每个活跃 lane 上读取 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 为活跃 lane |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到 `-src[i]`；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。

## 约束

- 源与结果的元素类型必须一致。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vneg` 的契约。

## 目标 Profile 限制

- A5 当前文档化支持：`i8-i32`、`f16`、`f32`。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 8 | `RV_VMULS`（标量乘法路径） |
| `f16` | 8 | `RV_VMULS`（标量乘法路径） |
| `i32` | 8 | `RV_VMULS`（标量乘法路径） |
| `i16` | 8 | `RV_VMULS`（标量乘法路径） |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 20（FP）/ 18（INT） | `A2A3_COMPL_FP_MUL` / `A2A3_COMPL_INT_MUL` |
| 每次 repeat 吞吐 | 1 | `A2A3_RPT_1` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

`vneg` 借用标量乘法路径，因此其时延与标量乘法型路径一致。

## 相关页面

- 指令集总览：[一元向量操作](../../unary-vector-ops_zh.md)
- 上一条指令：[pto.vabs](./vabs_zh.md)
- 下一条指令：[pto.vexp](./vexp_zh.md)
