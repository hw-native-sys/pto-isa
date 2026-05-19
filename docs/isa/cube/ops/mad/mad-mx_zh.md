# pto.mad_mx

`pto.mad_mx` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

零初始化 **MX（微缩放）** cube 矩阵乘加：`dst[m, n] = mx_product[m, n]`。

## MX Matmul 模型

`pto.mad_mx*` 在 MAD 之外额外应用微缩放。Scale payload 通过 [`pto.mte_l1_l0a_mx`](../data-movement/mte-l1-l0a-mx_zh.md) / [`pto.mte_l1_l0b_mx`](../data-movement/mte-l1-l0b-mx_zh.md) 加载，**不**作为本指令的直接操作数。

K 维按 32 个元素分组：

```text
k_group = floor(k / 32)

mx_product[m, n] =
  sum k in 0 .. K-1:
    (lhs[m, k] * lhs_scale[m, k_group]) *
    (rhs[k, n] * rhs_scale[k_group, n])
```

当前目标 profile 的 MX 数据 tile 使用 `f8E4M3FN`。`%k` 必须满足 MX 分组规则；目前的目标 profile 要求 MX matmul 消费的 K 是 64 的倍数（包含两个 32 元素 scale 组）。

## 机制

功能上等同于 [`pto.mad`](./mad_zh.md)，但在乘加过程中应用 MX scale。与 `pto.mad` 一样，结果覆盖 L0C。

## 语法

```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## 输入

参数形状同 [`pto.mad`](./mad_zh.md#输入)。本指令发起前，`%lhs` / `%rhs` 必须已经加载好匹配的 MX scale payload。

可选 clauses 见 [MAD 通用 Clauses](./mad_zh.md#mad-通用-clauses)（注意：`tf32_mode(...)` **不**是 MX MAD 的 clause）。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 MX 缩放后的 `M x N` 结果 tile 写到 L0C。 |

## 副作用

占用 CUBE 流水线；读取与 `%lhs` / `%rhs` 关联的 scale payload；写 L0C。

## 约束

!!! warning "约束"
    - 操作数必须使用目标支持的 MX 数据类型组合（当前 profile 是 `f8E4M3FN`）。
    - 在本指令前必须通过 [`pto.mte_l1_l0a_mx`](../data-movement/mte-l1-l0a-mx_zh.md) / [`pto.mte_l1_l0b_mx`](../data-movement/mte-l1-l0b-mx_zh.md) 加载好匹配的左右 MX scale payload。
    - `%k` 必须满足 [MX Matmul 模型](#mx-matmul-模型) 描述的分组规则。
    - `tf32_mode(...)` 不是 MX MAD 的 clause。
    - 其它约束同 [`pto.mad`](./mad_zh.md#约束)。

## 示例

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## 相关指令

- 非 MX 形式：[pto.mad](./mad_zh.md)
- MX 累加形式：[pto.mad_mx_acc](./mad-mx-acc_zh.md)
- MX 偏置初始化形式：[pto.mad_mx_bias](./mad-mx-bias_zh.md)
- MX scale 装载：[pto.mte_l1_l0a_mx](../data-movement/mte-l1-l0a-mx_zh.md)、[pto.mte_l1_l0b_mx](../data-movement/mte-l1-l0b-mx_zh.md)
