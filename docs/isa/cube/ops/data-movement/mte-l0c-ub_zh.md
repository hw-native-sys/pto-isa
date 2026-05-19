# pto.mte_l0c_ub

`pto.mte_l0c_ub` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。它是三条 FIXPIPE 回写指令之一；共享的回写流水线见 [FIXPIPE 模型](../../fixpipe-model_zh.md)。这是 [1→2 Cube-to-Vector tile 分发](../../fixpipe-model_zh.md#双目标广播1--2-cube-to-vector) 的架构基础。

## 摘要

把 L0C 中的结果 FIXPIPE 回写到 UB。数据变换 clauses 与 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md) 完全一致；UB 特有的操作数用于选择单目标或双目标（split-M / split-N）行为。

## 语法

```mlir
pto.mte_l0c_ub %src, %dst, %m, %n, %src_stride, %dst_stride,
    dst_mode(%sub_blockid | split_m | split_n)
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src`、`%m`、`%n`、`%src_stride` | — | 同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md#输入) |
| `%dst` | buffer-like | UB 目标 |
| `%dst_stride` | i64 | UB 目标步长，单位为目标元素 |
| `dst_mode(%sub_blockid)` | i64 operand | 单目标模式。`%sub_blockid` 选择 UB 子块 `0` 或 `1`；该值可以动态。 |
| `dst_mode(split_m)` | keyword | 双目标模式，沿 M 切分逻辑 tile。 |
| `dst_mode(split_n)` | keyword | 双目标模式，沿 N 切分逻辑 tile。 |
| 其它可选 clauses | — | 同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md)；**不**支持 `atomic(...)`。 |

`dst_mode(%sub_blockid)` 时，整个逻辑结果 tile 都写到选中的 UB 子块；`%dst` 是该子块的基址。

`dst_mode(split_m)` 时，逻辑 tile 沿 M 维切成两段 `[0, m/2)` 和 `[m/2, m)`，分别写到 UB 子块 0 和子块 1。每个子块都看到自己的目标原点 `%dst`，每个子块内写入的逻辑 tile 形状为 `(m / 2) x n`。

`dst_mode(split_n)` 时，逻辑 tile 沿 N 维切成两段 `[0, n/2)` 和 `[n/2, n)`，分别写到 UB 子块 0 和子块 1。每个子块都看到自己的目标原点 `%dst`，每个子块内写入的逻辑 tile 形状为 `m x (n / 2)`。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把转换后的 `M x N` 结果写到 UB；双目标模式下可同时写到 AIV0 / AIV1 子块。 |

## 副作用

读 L0C，写 UB。占用 AIC FIXP 流水线；双目标模式下还会使用专用 1→2 cube-to-vector 数据通路。AIV 块上的 UB 消费者需要通过跨块信号量原语同步。

## 约束

!!! warning "约束"
    - `pto.mte_l0c_ub` 不支持 `atomic(...)`。
    - `dst_mode(%sub_blockid)` 把整个逻辑 tile 写到一个 UB 子块。运行时 `%sub_blockid` 必须为 `0` 或 `1`；常量值在编译期检查。
    - `dst_mode(split_m)` 沿 M 切分成两个等高子块区域；`%m` 必须是偶数，每个子块收到 `(m / 2) x n` tile。
    - `dst_mode(split_n)` 沿 N 切分成两个等宽子块区域；`%n` 必须是 32 的倍数，每个子块收到 `m x (n / 2)` tile。
    - 双目标 split 模式仅在目标支持的 normal 或 `nz2nd` 回写场景下有效，并且必须省略 pre-quant、pre-ReLU/clip 等数据变换 clauses。
    - 其它约束同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md#约束)。

## 示例

```mlir
pto.mte_l0c_ub %l0c, %ub_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  dst_mode(%c1_i64),
  nz2nd
  : !pto.ptr<f32, l0c>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
```

## 相关指令

- FIXPIPE 回写兄弟指令：[pto.mte_l0c_l1](./mte-l0c-l1_zh.md)、[pto.mte_l0c_gm](./mte-l0c-gm_zh.md)
- 参数 payload 装载：[pto.mte_l1_fb](./mte-l1-fb_zh.md)
- MAD 生产者：[pto.mad](../mad/mad_zh.md) 及其变体
- Cluster 广播模型：[FIXPIPE 模型 — 双目标广播](../../fixpipe-model_zh.md#双目标广播1--2-cube-to-vector)
