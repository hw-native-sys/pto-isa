# pto.mte_gm_l1

`pto.mte_gm_l1` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

结构化 GM→L1（cube CBUF）拷贝。把 `%src`（GM）中的分组字节区间原样写入 `%dst`（L1），不做任何布局变换——源字节按字面顺序写入 L1。

如果源是行优先 ND 数据，需要在送给 cube 之前做 ND→NZ fractal 重排，请用 [`pto.mte_gm_l1_frac`](./mte-gm-l1-frac_zh.md)。

## 机制

与标量 [`pto.mte_gm_ub`](../../../scalar/ops/dma-copy/copy-gm-to-ubuf_zh.md) 一样，使用分组 `nburst(...) [loop(...)]*` 模型。每个 `nburst` 行结束后，源/目标按 `src_stride` / `dst_stride` 前进；可选的外层 `loop(...)` 把内层传输打包。

## 语法

```mlir
pto.mte_gm_l1 %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | GM 源基址 |
| `%dst` | ptr | L1 目标基址（`!pto.ptr<T, l1>`） |
| `%len_burst` | i64 | 每个 burst 行复制的字节数 |
| `nburst(%count, %src_stride, %dst_stride)` | i64 三元组 | 最内层 burst 个数与字节步长 |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 三元组 | 可选外层重复；字节步长 |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把数据写入 L1 目标区域。 |

## 副作用

读 GM 可见存储，写 L1 可见存储。占用 AIC MTE2 流水线。

## 约束

!!! warning "约束"
    - `nburst(...)` 必须存在。
    - 每个 `loop(...)` 子句出现时必须给出完整三元组。
    - 所有步长以字节为单位。对一个连续 16 元素的 f16 向量，使用 `%len_burst = 32`。

## 示例

```mlir
pto.mte_gm_l1 %bias_gm, %l1_bias, %c32_i64
  nburst(%c4_i64, %c64_i64, %c32_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64
```

## 相关指令

- ND→NZ 重排：[pto.mte_gm_l1_frac](./mte-gm-l1-frac_zh.md)
- L1 → UB：[pto.mte_l1_ub](./mte-l1-ub_zh.md)
- L1 → cube 操作数 tile：[pto.mte_l1_l0a](./mte-l1-l0a_zh.md)、[pto.mte_l1_l0b](./mte-l1-l0b_zh.md)
