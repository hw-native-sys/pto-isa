# pto.mte_l1_l0b_mx

`pto.mte_l1_l0b_mx` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

为逻辑 `%k x %n` 右数据 tile 加载右侧 MX scale 片段。生成的 payload 供 [`pto.mad_mx`](../mad/mad-mx_zh.md) / [`pto.mad_mx_acc`](../mad/mad-mx-acc_zh.md) / [`pto.mad_mx_bias`](../mad/mad-mx-bias_zh.md) 消费。

## MX Scale 加载模型

每个 scale 项对应一个 32 元素的 K 组。

- 右侧 scale 逻辑形状：`[ceil(K / 32), N]`
- L1 源数据按 32B scale 片段组织，逻辑顺序与对应数据 tile 一致。

## 语法

```mlir
pto.mte_l1_l0b_mx %src, %dst, %k, %n
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | L1 中的 MX scale 源 |
| `%dst` | ptr | 与 `l0b` 关联的右侧 MX payload 目标 |
| `%k` | i64 | K 范围；scale 按 32 个 K 元素分组 |
| `%n` | i64 | 对应右数据 tile 的 N 范围 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把与 L0B 操作数 tile 关联的 MX scale payload 写好。 |

## 副作用

读 L1，写与 L0B 关联的 MX scale 状态。结果由后续读取该 `%rhs` 的 `pto.mad_mx*` 消费。

## 约束

!!! warning "约束"
    - `%src` 必须在 `l1`，`%dst` 必须在 `l0b`。
    - `%src` 与 `%dst` 必须满足 32B MX scale 片段对齐要求。

## 示例

```mlir
pto.mte_l1_l0b_mx %l1_b_scale, %l0b_scale, %c64_i64, %c16_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64
```

## 相关指令

- 数据 tile 加载：[pto.mte_l1_l0b](./mte-l1-l0b_zh.md)
- 左侧 scale 加载：[pto.mte_l1_l0a_mx](./mte-l1-l0a-mx_zh.md)
- MX MAD 消费者：[pto.mad_mx](../mad/mad-mx_zh.md)、[pto.mad_mx_acc](../mad/mad-mx-acc_zh.md)、[pto.mad_mx_bias](../mad/mad-mx-bias_zh.md)
