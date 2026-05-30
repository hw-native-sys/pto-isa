# pto.mte_l0c_gm

`pto.mte_l0c_gm` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。它是三条 FIXPIPE 回写指令之一；共享的回写流水线见 [FIXPIPE 模型](../../fixpipe-model_zh.md)。

## 摘要

把 L0C 中的结果 FIXPIPE 回写到 GM。数据变换 clauses 与 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md) 完全一致；GM 特有的操作数用于选择 GM 写路径及可选的原子更新行为。

## 语法

```mlir
pto.mte_l0c_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```

## 输入

| 参数 | 位宽 | 描述 |
|-----------|-------|-------------|
| `%src`、`%m`、`%n`、`%src_stride` | — | 同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md#输入) |
| `%dst` | buffer-like | GM 目标 |
| `%dst_stride` | i64 | GM 目标步长，单位为目标元素 |
| `%sid` | i64 | GM stream/session 提示，不影响写入数值 |
| `%l2_cache_ctrl` | i64 | GM store cache 提示，不影响写入数值 |
| `atomic(type = ..., op = ...)` | clause | 可选 GM 读-改-写 |
| 其它可选 clauses | — | 同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md#语法) |

`%sid` 与 `%l2_cache_ctrl` 仅影响内存路径，不会改变逻辑结果、目标布局、数值转换或原子操作。对当前 profile，常量 `%sid` 必须在 `[0, 3]`（若内存系统没有特别配置，使用 `0` 即可）。常量 `%l2_cache_ctrl` 必须在目标 cache 控制范围 `[0, 15]` 内。

`atomic(type = T, op = add|max|min)` 在每个 GM 目标元素上执行原子读-改-写。`add` 把转换后的值累加到 GM 原值上；`max` 与 `min` 按类型 `T` 比较后写入较大/较小者。支持的原子类型：`f32`、`f16`、`bf16`、`s32`、`s16`、`s8`。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把转换后的 `M x N` 结果写到 GM。 |

## 副作用

读 L0C，写 GM。占用 AIC FIXP 流水线。带 `atomic(...)` 时，GM 更新是原子读-改-写。

## 约束

!!! warning "约束"
    - `atomic(...)` 仅在 `pto.mte_l0c_gm` 上有效。
    - `atomic` 必须同时提供 `type` 和 `op`。
    - 原子 op 取值为 `add`、`max`、`min`。
    - `%sid` 或 `%l2_cache_ctrl` 若为常量，必须落在上文给出的目标取值范围内。
    - 其它约束同 [`pto.mte_l0c_l1`](./mte-l0c-l1_zh.md#约束)。

## 示例

```mlir
pto.mte_l0c_gm %l0c, %out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  %c0_i64, %c0_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  nz2nd,
  atomic(type = f16, op = add)
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64, f32
```

## 相关指令

- FIXPIPE 回写兄弟指令：[pto.mte_l0c_l1](./mte-l0c-l1_zh.md)、[pto.mte_l0c_ub](./mte-l0c-ub_zh.md)
- 参数 payload 装载：[pto.mte_l1_fb](./mte-l1-fb_zh.md)
- MAD 生产者：[pto.mad](../mad/mad_zh.md) 及其变体
