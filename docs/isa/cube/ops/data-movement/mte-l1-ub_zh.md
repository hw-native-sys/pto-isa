# pto.mte_l1_ub

`pto.mte_l1_ub` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

结构化 L1→UB 拷贝。从 `%src`（L1）读取分组字节区间并写入 `%dst`（UB）。这是 [`pto.mte_ub_l1`](../../../scalar/ops/dma-copy/mte-ub-l1_zh.md) 的反向通路，即 L1→Vector 数据通路。

## 机制

使用与 [`pto.mte_gm_l1`](./mte-gm-l1_zh.md) 相同的分组 `nburst(...) [loop(...)]*` 模型。每个 `nburst` 行后，源/目标按 `src_stride` / `dst_stride` 前进；外层 `loop(...)` 把内层传输打包。

## 语法

```mlir
pto.mte_l1_ub %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, l1>, !pto.ptr<T, ub>, i64, i64, i64, i64
```

## 输入

与 [`pto.mte_gm_l1`](./mte-gm-l1_zh.md#输入) 同样的分组字节模型，源/目标地址空间反向为 `l1 -> ub`。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把数据写入 UB 目标区域。 |

## 副作用

读 L1 可见存储，写 UB 可见存储。传输由 AIC 侧发起并经过 cube-to-vector 数据通路；AIV 侧的 UB 消费者需相应同步。

## 约束

!!! warning "约束"
    - `%src` 必须在 `l1`，`%dst` 必须在 `ub`。
    - `nburst(...)` 必须存在。
    - 每个 `loop(...)` 子句出现时必须给出完整三元组。

## 示例

```mlir
pto.mte_l1_ub %l1_src, %ub_dst, %c64_i64
  nburst(%c2_i64, %c128_i64, %c64_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

## 相关指令

- 反向通路（UB → L1）：[pto.mte_ub_l1](../../../scalar/ops/dma-copy/mte-ub-l1_zh.md)
- GM → L1：[pto.mte_gm_l1](./mte-gm-l1_zh.md)、[pto.mte_gm_l1_frac](./mte-gm-l1-frac_zh.md)
