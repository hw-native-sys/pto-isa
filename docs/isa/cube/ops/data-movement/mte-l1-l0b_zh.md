# pto.mte_l1_l0b

`pto.mte_l1_l0b` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

把 L1 `l1` 中的逻辑 `%k x %n` 右 tile 加载到 `l0b`，供 `pto.mad*` 消费。源必须已经是 cube fractal NZ 布局；本指令不会把任意行优先矩阵自动转换。

L0B 使用 `K1 N1 N0 K0` FRACTAL_ZN，**K 在最内层**，这样 cube 硬件每个 cycle 都能读到完整的 `K0` 个元素而不跨 stride。从 L1 的 `K1 N1 K0 N0` 到 L0B 的 `K1 N1 N0 K0` 的内层 box 转置在本指令搬运过程中完成，用户层看不到额外的转置 pass。

## 机制

若 `transpose = true`，选中的逻辑源 tile 在置入目标操作数域前转置；省略等同于 `transpose = false`。

布局原理见 [NZ Fractal 布局 — 为什么 L0B 必须 K-innermost](../../nz-fractal-layout_zh.md#为什么-l0b-必须-k-innermost)。

## 语法

```mlir
pto.mte_l1_l0b %src, %dst, %k, %n
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | L1 中的 cube fractal 源 tile |
| `%dst` | ptr | L0B 右操作数目标 |
| `%k` | i64 | 逻辑 K 范围 |
| `%n` | i64 | 逻辑 N 范围 |
| `transpose` | attr | 可选布尔属性，置入目标域前是否对源 tile 做转置 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 L0B tile 写好，供后续 `pto.mad*` 读取。 |

## 副作用

读 L1，写 L0B。占用 AIC MTE1 流水线。

## 约束

!!! warning "约束"
    - `%src` 必须在 `l1`，`%dst` 必须在 `l0b`。
    - `%src` 与 `%dst` 必须满足 Cube tile load 的目标对齐要求。
    - `transpose = true` 要求 tile 形状被元素类型的转置粒度支持。

## 示例

```mlir
pto.mte_l1_l0b %l1_b, %l0b, %c32_i64, %c16_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64
```

## 相关指令

- 左操作数加载：[pto.mte_l1_l0a](./mte-l1-l0a_zh.md)
- MX scale 加载：[pto.mte_l1_l0b_mx](./mte-l1-l0b-mx_zh.md)
- 上游重排：[pto.mte_gm_l1_frac](./mte-gm-l1-frac_zh.md)
- MAD 消费者：[pto.mad](../mad/mad_zh.md)、[pto.mad_acc](../mad/mad-acc_zh.md)、[pto.mad_bias](../mad/mad-bias_zh.md)
