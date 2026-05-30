# pto.mte_l1_bt

`pto.mte_l1_bt` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

把 L1 中的 bias payload 加载到 `bt` 地址空间，供后续 [`pto.mad_bias`](../mad/mad-bias_zh.md) / [`pto.mad_mx_bias`](../mad/mad-mx-bias_zh.md) 消费。消费者把结果解读为 `N` 元素的 bias 向量 `bias[n]`。

## 机制

一次 burst 从 `%src` 读 `%len_burst` 个 bias-load 单位，写到 `%dst`。除最后一次外，每次 burst 之后源/目标按 burst 长度加上对应 gap 前进。每个单位的宽度由配置的类型对决定。

## 语法

```mlir
pto.mte_l1_bt %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, bt>, i64, i64, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | L1 源指针，位于 `l1` |
| `%dst` | ptr | bias 目标指针，位于 `bt` |
| `%len_burst` | i64 | 每个 burst 加载的 bias-load 单位数 |
| `%count` | i64 | Burst 数量 |
| `%src_gap` | i64 | 跨 burst 的源间隔（bias-load 单位） |
| `%dst_gap` | i64 | 跨 burst 的目标间隔（bias-load 单位） |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 bias 值写入 `bt` 目标区域。 |

## 副作用

读 L1，写 BT。结果由后续的 `pto.mad_bias` / `pto.mad_mx_bias` 消费。

## 约束

!!! warning "约束"
    - 支持的类型对：`f32 -> f32`、`i32 -> i32`、`f16 -> f32`、`bf16 -> f32`。
    - 对 `bf16 -> f32`，bf16 源会被恒定地扩展到 f32 bias 值；对 `f16 -> f32`，作为 f32 bias 使用时会扩展，否则 f16 payload 写入 32-bit bias slot，高位未用。
    - 只加载消费 tile 实际需要的通道 bias；bias payload 不是 result-shaped。

## 示例

```mlir
pto.mte_l1_bt %l1_bias, %bt, %c1_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f32, bt>, i64, i64, i64, i64
```

## 相关指令

- 偏置初始化 MAD：[pto.mad_bias](../mad/mad-bias_zh.md)、[pto.mad_mx_bias](../mad/mad-mx-bias_zh.md)
- FIXPIPE 辅助 payload：[pto.mte_l1_fb](./mte-l1-fb_zh.md)
