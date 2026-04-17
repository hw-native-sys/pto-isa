# pto.vxor

`pto.vxor` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vxor` 在每个活跃 lane 上做按位异或：

```text
dst[i] = lhs[i] ^ rhs[i]
```

## 机制

这条指令对两个源向量寄存器做逐 lane 按位异或。对每个满足谓词的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i \oplus \mathrm{rhs}_i $$

非活跃 lane 保持目标原值。它是纯整数指令。

## 语法

### PTO 汇编形式

```text
vxor %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vxor %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vxor ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

支持的元素类型是全部整数类型：`i8`–`i64`、`u8`–`u64`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 左操作数向量寄存器 |
| `%rhs` | `!pto.vreg<NxT>` | 右操作数向量寄存器 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与运算 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 按位异或；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 只支持整数元素类型。
- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 三个寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 一致。
- 非活跃 lane 保持目标原值。

## 异常与非法情形

- verifier 会拒绝非整数元素类型、类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面列出的额外非法情形，同样适用于 `pto.vxor`。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|-----------|:-------------:|:-----:|:--:|
| 整数类型 | 模拟 | 模拟 | 支持 |

A5 是当前文档里最具体的向量 profile。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `i32` | 7 | `RV_VXOR` |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延 | 17 | `A2A3_COMPL_INT_BINOP` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

### MLIR 用法

```mlir
%result = pto.vxor %a, %b, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>

%toggled = pto.vxor %data, %toggle, %active
    : (!pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask) -> !pto.vreg<64xi32>
```

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vor](./vor_zh.md)
- 下一条指令：[pto.vshl](./vshl_zh.md)
