# pto.vsub

`pto.vsub` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`pto.vsub` 在每个活跃 lane 上执行 `dst[i] = lhs[i] - rhs[i]`。

## 机制

`pto.vsub` 对两个源向量寄存器做逐 lane 相减。对于谓词为真的 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i - \mathrm{rhs}_i $$

非活跃 lane 不参与计算，目标寄存器这些位置保持不变。减法的具体解释依赖元素类型：有符号整数用有符号减法，无符号整数用无符号减法，浮点类型则按浮点减法语义处理。

## 语法

### PTO 汇编形式

```text
vsub %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vsub %lhs, %rhs, %mask : (!pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vsub ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

A5 当前记录的支持类型包括 `i8-i64`、`f16`、`bf16`、`f32`。

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 被减数 |
| `%rhs` | `!pto.vreg<NxT>` | 减数 |
| `%mask` | `!pto.mask` | 谓词掩码；掩码位为 1 的 lane 参与减法 |

两个源寄存器必须有相同的元素类型和相同的向量宽度 `N`。掩码宽度必须与 `N` 相同。

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 差值；非活跃 lane 保持原值 |

## 副作用

这条指令除了产生目标向量寄存器外，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- `%lhs`、`%rhs` 和 `%result` 的元素类型必须一致。
- 三个寄存器的向量宽度必须一致。
- `%mask` 的宽度必须与 `N` 相同。
- 只有掩码位为 1 的 lane 参与减法。
- 非活跃 lane 对应的目标元素保持不变。

## 异常与非法情形

- verifier 会拒绝非法的类型不匹配、宽度不匹配或掩码宽度不匹配。
- [二元向量操作](../../binary-vector-ops_zh.md)页面中声明的额外非法情形，同样适用于 `pto.vsub`。

## 目标 Profile 限制

| 元素类型 | CPU Simulator | A2/A3 | A5 |
|-----------|:-------------:|:-----:|:--:|
| `f32` | 模拟 | 模拟 | 支持 |
| `f16` / `bf16` | 模拟 | 模拟 | 支持 |
| `i8`–`i64`、`u8`–`u64` | 模拟 | 模拟 | 支持 |

A5 是当前文档里最具体的向量 profile。CPU 模拟器和 A2/A3 类目标在保留可见契约的前提下模拟这类运算。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 7 | `RV_VSUB` |
| `f16` | 7 | `RV_VSUB` |
| `i32` | 7 | `RV_VSUB` |
| `i16` | 7 | `RV_VSUB` |

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
    dst[i] = src0[i] - src1[i];
```

### MLIR 用法

```mlir
%result = pto.vsub %lhs, %rhs, %active
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>

%diff = pto.vsub %a, %b, %cond
    : (!pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask) -> !pto.vreg<128xf16>
```

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vadd](./vadd_zh.md)
- 下一条指令：[pto.vmul](./vmul_zh.md)
