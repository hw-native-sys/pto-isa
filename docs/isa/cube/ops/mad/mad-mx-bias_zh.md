# pto.mad_mx_bias

`pto.mad_mx_bias` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

偏置初始化 **MX（微缩放）** cube 矩阵乘加：`dst[m, n] = mx_product[m, n] + bias[n]`。

MX 分组乘加的语义见 [MX Matmul 模型](./mad-mx_zh.md#mx-matmul-模型)。

## 机制

把 [`pto.mad_mx`](./mad-mx_zh.md) 的 MX 缩放与 [`pto.mad_bias`](./mad-bias_zh.md) 的偏置初始化结合在一起。累加器从 `bias[n]` 开始，而不是 0。

## 语法

```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```

## 输入

参数形状同 [`pto.mad_bias`](./mad-bias_zh.md#输入)，且 `%lhs` / `%rhs` 需带有 [`pto.mad_mx`](./mad-mx_zh.md) 所要求的 MX scale payload。

可选 clauses 见 [MAD 通用 Clauses](./mad_zh.md#mad-通用-clauses)，但不接受 `tf32_mode(...)`。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 以 bias 为种子，把 MX 缩放后的 `M x N` 结果写到 L0C。 |

## 副作用

占用 CUBE 流水线；从 BT 读 `%bias`，从与 `%lhs` / `%rhs` 关联的 MX scale 状态读 scale；写 L0C。

## 约束

!!! warning "约束"
    - [`pto.mad_mx`](./mad-mx_zh.md#约束) 的全部约束（MX 类型组合、scale payload 前置条件、K 分组规则）。
    - [`pto.mad_bias`](./mad-bias_zh.md#约束) 中关于 `%bias` 的所有约束：`%bias` 必须在 `bt` 空间，元素类型与 `%dst` 相同，只消费 `N` 个值。

## 示例

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

## 相关指令

- 零初始化 MX 形式：[pto.mad_mx](./mad-mx_zh.md)
- MX 累加形式：[pto.mad_mx_acc](./mad-mx-acc_zh.md)
- 非 MX 偏置初始化形式：[pto.mad_bias](./mad-bias_zh.md)
- Bias 装载：[pto.mte_l1_bt](../data-movement/mte-l1-bt_zh.md)
- MX scale 装载：[pto.mte_l1_l0a_mx](../data-movement/mte-l1-l0a-mx_zh.md)、[pto.mte_l1_l0b_mx](../data-movement/mte-l1-l0b-mx_zh.md)
