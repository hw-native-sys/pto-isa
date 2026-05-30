# pto.mte_l1_l0a

`pto.mte_l1_l0a` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

把 L1 `l1` 中的逻辑 `%m x %k` 左 tile 加载到 `l0a`，供 `pto.mad*` 消费。源必须已经是 cube fractal NZ 布局；本指令不会把任意行优先矩阵自动转换。若源是 ND/DN 原始数据，请先用 [`pto.mte_gm_l1_frac`](./mte-gm-l1-frac_zh.md) 完成重排。

## 机制

将 L1 cube fractal tile 移入 L0A 操作数域。目标布局遵循 [NZ Fractal 布局 — 各缓冲的 NZ 布局](../../nz-fractal-layout_zh.md#各缓冲的-nz-布局)：L0A 为 `K1 M1 M0 K0`（A5 上 FRACTAL_NZ / A3 上 FRACTAL_ZZ）。

若 `transpose = true`，被选中的逻辑源 tile 会在置入目标操作数域之前转置；省略该属性等同于 `transpose = false`。

## 语法

```mlir
pto.mte_l1_l0a %src, %dst, %m, %k
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | L1 中的 cube fractal 源 tile |
| `%dst` | ptr | L0A 左操作数目标 |
| `%m` | i64 | 逻辑 M 范围 |
| `%k` | i64 | 逻辑 K 范围 |
| `transpose` | attr | 可选布尔属性，置入目标域前是否对源 tile 做转置 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 L0A tile 写好，供后续 `pto.mad*` 读取。 |

## 副作用

读 L1，写 L0A。占用 AIC MTE1 流水线。

## 约束

!!! warning "约束"
    - `%src` 必须在 `l1`，`%dst` 必须在 `l0a`。
    - `%src` 与 `%dst` 必须满足 Cube tile load 的目标对齐要求。
    - `transpose = true` 要求 tile 形状被元素类型的转置粒度支持。

## 示例

```mlir
pto.mte_l1_l0a %l1_a, %l0a, %c16_i64, %c32_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64
```

## 相关指令

- 右操作数加载：[pto.mte_l1_l0b](./mte-l1-l0b_zh.md)
- MX scale 加载：[pto.mte_l1_l0a_mx](./mte-l1-l0a-mx_zh.md)
- 上游重排：[pto.mte_gm_l1_frac](./mte-gm-l1-frac_zh.md)
- MAD 消费者：[pto.mad](../mad/mad_zh.md)、[pto.mad_acc](../mad/mad-acc_zh.md)、[pto.mad_bias](../mad/mad-bias_zh.md)
