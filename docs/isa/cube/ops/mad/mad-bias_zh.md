# pto.mad_bias

`pto.mad_bias` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

偏置初始化 cube 矩阵乘加：`dst[m, n] = sum_k(lhs[m, k] * rhs[k, n]) + bias[n]`。

## 机制

与 [`pto.mad`](./mad_zh.md) 相同，但用 N 维 bias 向量作为累加器种子，而不是 0。常用作 K 分块序列里的第一条 MAD；后续部分和可以用 [`pto.mad_acc`](./mad-acc_zh.md) 继续累加。

## 语法

```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```

## 输入

| 参数 | 类型 | 描述 |
|-----------|------|------|
| `%lhs`、`%rhs`、`%dst`、`%m`、`%n`、`%k` | — | 同 [`pto.mad`](./mad_zh.md#输入) |
| `%bias` | `!pto.ptr<C, bt>` | BT 中的 bias 向量，解读为 `N` 个值并沿 M 维广播 |

可选 clauses 见 [MAD 通用 Clauses](./mad_zh.md#mad-通用-clauses)。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把以 bias 为初值的 `M x N` 结果 tile 写到 L0C。 |

## 副作用

占用 CUBE 流水线，从 BT 读 `%bias`，写 L0C。调用方需在本指令之前通过 [`pto.mte_l1_bt`](../data-movement/mte-l1-bt_zh.md) 把 `%bias` 加载到 BT。

## 约束

!!! warning "约束"
    - `%bias` 必须位于 `bt` 地址空间。
    - `%bias` 的元素类型必须与 `%dst` 相同。
    - 只消费 `N` 个 bias 值；`%bias` 不是 `M x N` 矩阵。
    - 其它约束同 [`pto.mad`](./mad_zh.md#约束)。

## 示例

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

## 相关指令

- 零初始化形式：[pto.mad](./mad_zh.md)
- 累加形式：[pto.mad_acc](./mad-acc_zh.md)
- MX 偏置初始化形式：[pto.mad_mx_bias](./mad-mx-bias_zh.md)
- Bias 装载：[pto.mte_l1_bt](../data-movement/mte-l1-bt_zh.md)
