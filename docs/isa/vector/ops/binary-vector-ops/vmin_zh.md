# pto.vmin

`pto.vmin` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vmin` 在每个活跃 lane 上取两个输入里的较小值：

```text
dst[i] = (lhs[i] < rhs[i]) ? lhs[i] : rhs[i]
```

## 机制

这条指令对两个源向量寄存器做逐 lane 比较，再把每个 lane 上的较小值写入目标寄存器。

对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \min(\mathrm{lhs}_i, \mathrm{rhs}_i) $$

比较规则依赖元素类型：有符号整数按有符号比较，无符号整数按无符号比较，浮点类型则按浮点排序规则处理。非活跃 lane 保持目标原值。

## 语法

### PTO 汇编形式

```text
vmin %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vmin %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vmin ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

A5 当前记录的支持元素类型包括 `i8-i32`、`f16`、`bf16`、`f32`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 第一个比较操作数 |
| `%rhs` | `!pto.vreg<NxT>` | 第二个比较操作数 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与比较 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上写入较小值；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 三个寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 一致。
- 只有掩码位为 1 的 lane 参与比较。
- 浮点情况下，如果任一操作数是 NaN，结果也是 NaN。

## 异常与非法情形

- verifier 会拒绝非法的类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面列出的额外非法情形，同样适用于 `pto.vmin`。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|-----------|:-------------:|:-----:|:--:|
| `f32` | 模拟 | 模拟 | 支持 |
| `f16` / `bf16` | 模拟 | 模拟 | 支持 |
| `i8`–`i32` | 模拟 | 模拟 | 支持 |

A5 是当前文档里最具体的向量 profile。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 7 | `RV_VMAX` |
| `f16` | 7 | `RV_VMAX` |
| `i32` | 7 | `RV_VMAX` |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延：FP32 | 19 | `A2A3_COMPL_FP_BINOP` |
| 完成时延：INT | 17 | `A2A3_COMPL_INT_BINOP` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

### MLIR 用法

```mlir
%result = pto.vmin %a, %b, %active
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>

%clamped = pto.vmin %input, %upper, %active
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vmax](./vmax_zh.md)
- 下一条指令：[pto.vand](./vand_zh.md)
