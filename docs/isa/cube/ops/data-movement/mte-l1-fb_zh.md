# pto.mte_l1_fb

`pto.mte_l1_fb` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。

## 摘要

把 FIXPIPE 参数 payload 从 L1 加载到 `fb`。`pto.mte_l0c_*` 回写指令族里的 `pre_quant(...)` 与 `pre_relu(...)` 子句稍后会通过 `fb` 指针消费这些 payload。FIXPIPE 流水线全貌见 [FIXPIPE 模型](../../fixpipe-model_zh.md)。

## 机制

一次 burst 从 `%src` 读 `%len_burst` 个 parameter-load 单位，写到 `%dst`。本指令的拷贝单位是它的 parameter-load 单位——与 `mte_l0c_*` 向量 payload 行的大小不同。`%len_burst` 与 `nburst(...)` 的 gap 都以这个单位计数，不是字节，也不是目标元素。

`pto.mte_l1_fb` 在 `fb` 中物化 payload 之后，pre-ReLU 向量消费者按 64B 参数行读取，pre-quant 向量消费者按 128B 参数行读取。传给 `mte_l0c_*` 的 payload 指针必须指向该 store 对应逻辑输出 tile 的第一行；后续行按照逻辑累加器元素相同的通道 / NZ 顺序前进。

## 语法

```mlir
pto.mte_l1_fb %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, fb>, i64, i64, i64, i64
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src` | ptr | L1 源指针，位于 `l1` |
| `%dst` | ptr | 缩放/参数目标指针，位于 `fb` |
| `%len_burst` | i64 | 每个 burst 加载的 parameter-load 单位数 |
| `%count` | i64 | Burst 数量 |
| `%src_gap` | i64 | 跨 burst 的源间隔（parameter-load 单位） |
| `%dst_gap` | i64 | 跨 burst 的目标间隔（parameter-load 单位） |

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 FIXPIPE 参数 payload 写入 `fb` 目标区域。 |

## 副作用

读 L1，写 FB。结果由后续 `pto.mte_l0c_*` 回写指令里引用 `fb` 指针的 `pre_quant` / `pre_relu` 子句消费。

## 约束

!!! warning "约束"
    - `%src` 必须在 `l1`，`%dst` 必须在 `fb`。
    - 向量 `pre_quant` 与 `pre_relu` 消费者要求按 [FIXPIPE 模型](../../fixpipe-model_zh.md) 文档化的行序准备参数数据。

## 示例

```mlir
pto.mte_l1_fb %l1_fp, %fb_fp, %c2_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, l1>, !pto.ptr<f32, fb>, i64, i64, i64, i64
```

## 相关指令

- FIXPIPE 回写消费者：[pto.mte_l0c_l1](./mte-l0c-l1_zh.md)、[pto.mte_l0c_gm](./mte-l0c-gm_zh.md)、[pto.mte_l0c_ub](./mte-l0c-ub_zh.md)
- FIXPIPE 流水线概览：[FIXPIPE 模型](../../fixpipe-model_zh.md)
