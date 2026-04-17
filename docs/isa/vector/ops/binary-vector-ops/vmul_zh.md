# pto.vmul

`pto.vmul` 属于[二元向量操作](../../binary-vector-ops_zh.md)指令集。

## 概述

`%result` 是 `%lhs` 和 `%rhs` 的逐 lane 乘积。

## 机制

`pto.vmul` 是标准的 `pto.v*` 计算指令。它逐 lane 读取两个源向量寄存器，在活跃 lane 上计算乘积，再把结果写到目标向量寄存器。

对每个 lane `i`：

$$ \mathrm{dst}_i = \mathrm{lhs}_i \times \mathrm{rhs}_i $$

当前页记录的谓词行为采用 zero-merge 模型：`mask[i] == 0` 的 lane 在结果中写成零，而不是保持旧值。

## 语法

### PTO 汇编形式

```asm
vmul %dst, %lhs, %rhs, %mask : !pto.vreg<NxT>
```

### AS Level 1（SSA）

```mlir
%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vmul ins(%lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask)
    outs(%result : !pto.vreg<NxT>)
```

## C++ Intrinsic

```cpp
PTO_VMUL_IMPL(result, lhs, rhs, mask);
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%lhs` | `!pto.vreg<NxT>` | 第一个源向量 |
| `%rhs` | `!pto.vreg<NxT>` | 第二个源向量，在每个 lane 上与 `%lhs` 相乘 |
| `%mask` | `!pto.mask<G>` | 谓词掩码；非活跃 lane 在结果中写零 |

A5 当前记录的类型包括 `i16-i32`、`f16`、`bf16`、`f32`，明确**不包含** `i8` / `u8`。

## 预期输出

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%result` | `!pto.vreg<NxT>` | `%lhs` 与 `%rhs` 的逐 lane 乘积 |

## 副作用

这条指令除了产生 SSA 结果，没有其他架构副作用。它不会隐式占用 buffer、发送事件，也不会建立内存栅栏。

## 约束

- 源和结果的元素类型必须一致。
- A5 profile 不支持 `i8` / `u8` 形式。
- `mask[i] == 0` 的 lane 在结果中写零。

## 异常与非法情形

- verifier 会拒绝非法的向量形状、不支持的元素类型，以及不合法的属性组合。
- 约束部分列出的非法情形同样属于 `pto.vmul` 的契约。

## 目标 Profile 限制

- A5 当前记录的支持范围：`i16-i32`、`f16`、`bf16`、`f32`，**不支持** `i8` / `u8`。
- A5 是手册中最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以用更窄的子集或等效模拟来保留可见契约。
- 如果代码依赖具体类型清单、分布模式或融合行为，这类依赖都应视为 profile 相关，而不是跨目标保证。

## 性能

### A5 时延

| 元素类型 | 时延（周期） | A5 RV |
|----------|--------------|-------|
| `f32` | 8 | `RV_VMUL` |
| `f16` | 8 | `RV_VMUL` |
| `i32` | 8 | `RV_VMUL` |
| `i16` | 8 | `RV_VMUL` |

### A2/A3 吞吐

| 指标 | 数值 | 常量 |
|------|------|------|
| 启动时延 | 14 | `A2A3_STARTUP_BINARY` |
| 完成时延：FP | 20 | `A2A3_COMPL_FP_MUL` |
| 完成时延：INT | 18 | `A2A3_COMPL_INT_MUL` |
| 每次 repeat 吞吐 | 2 | `A2A3_RPT_2` |
| 流水间隔 | 18 | `A2A3_INTERVAL` |

## 示例

### C 语义

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

整数溢出遵循目标平台定义的行为。被谓词关闭的 lane 在结果中写零。

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vmul`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vmul` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- 指令集总览：[二元向量操作](../../binary-vector-ops_zh.md)
- 上一条指令：[pto.vsub](./vsub_zh.md)
- 下一条指令：[pto.vdiv](./vdiv_zh.md)
