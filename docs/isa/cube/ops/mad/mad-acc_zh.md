# pto.mad_acc

`pto.mad_acc` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

累加 cube 矩阵乘加：`dst[m, n] = dst[m, n] + sum_k(lhs[m, k] * rhs[k, n])`。

## 机制

与 [`pto.mad`](./mad_zh.md) 相同，但把新计算的乘积**累加**到 L0C 已有的累加器状态上，而不是覆盖。典型用法是 K 方向分块：连续的 MAD 沿 K 累加部分和，直到整个归约完成。

## 语法

```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## 输入

参数形状与语义同 [`pto.mad`](./mad_zh.md#输入)。可选 clauses 见 [MAD 通用 Clauses](./mad_zh.md#mad-通用-clauses)。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 就地更新 L0C 中已有的 `M x N` tile，没有 SSA 结果。 |

## 副作用

占用 CUBE 流水线，并读写 L0C。调用方需要保证 L0C tile 已被初始化（通常由该 `%dst` 上的首次 [`pto.mad`](./mad_zh.md) 或 [`pto.mad_bias`](./mad-bias_zh.md) 完成，之后才允许多次 `pto.mad_acc`）。

## 约束

同 [`pto.mad`](./mad_zh.md#约束)。

## 示例

```mlir
// K 方向分块：先 pto.mad 初始化，再多次 pto.mad_acc。
pto.mad %l0a_k0, %l0b_k0, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64

pto.mad_acc %l0a_k1, %l0b_k1, %l0c, %c16_i64, %c16_i64, %c32_i64 unit_flag(check_only)
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## 相关指令

- 零初始化形式：[pto.mad](./mad_zh.md)
- 偏置初始化形式：[pto.mad_bias](./mad-bias_zh.md)
- MX 累加形式：[pto.mad_mx_acc](./mad-mx-acc_zh.md)
