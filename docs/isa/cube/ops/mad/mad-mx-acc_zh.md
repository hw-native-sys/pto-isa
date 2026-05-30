# pto.mad_mx_acc

`pto.mad_mx_acc` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

累加 **MX（微缩放）** cube 矩阵乘加：`dst[m, n] = dst[m, n] + mx_product[m, n]`。

MX 分组乘加的语义见 [MX Matmul 模型](./mad-mx_zh.md#mx-matmul-模型)。

## 机制

与 [`pto.mad_mx`](./mad-mx_zh.md) 相同，但把 MX 缩放后的乘积**累加**到 L0C 已有状态上。典型用法是 MX GEMM 的 K 方向分块。

## 语法

```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## 输入

参数形状同 [`pto.mad_mx`](./mad-mx_zh.md#输入)。可选 clauses 见 [MAD 通用 Clauses](./mad_zh.md#mad-通用-clauses)，但不接受 `tf32_mode(...)`。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 用 MX 缩放结果就地更新 L0C 中已有的 `M x N` tile。 |

## 副作用

同 [`pto.mad_mx`](./mad-mx_zh.md#副作用)。调用方需保证 L0C tile 已被初始化（通常由该 `%dst` 上的首次 [`pto.mad_mx`](./mad-mx_zh.md) 或 [`pto.mad_mx_bias`](./mad-mx-bias_zh.md) 完成）。

## 约束

同 [`pto.mad_mx`](./mad-mx_zh.md#约束)。

## 示例

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## 相关指令

- 零初始化 MX 形式：[pto.mad_mx](./mad-mx_zh.md)
- MX 偏置初始化形式：[pto.mad_mx_bias](./mad-mx-bias_zh.md)
- 非 MX 累加形式：[pto.mad_acc](./mad-acc_zh.md)
