# pto.mte_l0c_l1

`pto.mte_l0c_l1` 属于 [Cube 数据搬运指令](../../README_zh.md#cube-数据搬运指令)。它是三条 FIXPIPE 回写指令之一；共享的回写流水线、布局模式与 clause 语义见 [FIXPIPE 模型](../../fixpipe-model_zh.md)。

## 摘要

把 L0C 中的结果 FIXPIPE 回写到 L1 `l1`。在写到 L1 前按规范顺序依次应用可选的 pre-quant、pre-ReLU/clip、布局变换、外层 loop3 重复以及饱和行为。

## 语法

```mlir
pto.mte_l0c_l1 %src, %dst, %m, %n, %src_stride, %dst_stride
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
| `%src` | buffer-like | L0C 中的累加器源 |
| `%dst` | buffer-like | L1 目标 |
| `%m` | i64 | 逻辑 M 元素数 |
| `%n` | i64 | 逻辑 N 元素数 |
| `%src_stride` | i64 | 源步长，单位 C0（1 个 C0 = 32 字节） |
| `%dst_stride` | i64 | 目标步长，单位为目标元素 |

可选 clauses 见 [FIXPIPE 通用 Clauses](../../fixpipe-model_zh.md) 与 [FIXPIPE 布局模型](../../fixpipe-model_zh.md)。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把转换后的 `M x N` 结果写到 L1。 |

## 副作用

读 L0C，写 L1。占用 AIC FIXP 流水线。L1 中的下游消费者需通过 pipe 事件同步。

## 约束

!!! warning "约束"
    - Clauses 必须按规范顺序：`unit_flag` → `pre_quant` → `pre_relu` → layout → `loop3` → `sat`/`nosat`。
    - `pre_quant` 要求 payload 与 mode 一起出现。
    - 向量 `pre_quant` 模式要求 `fb` 指针的元素类型为 `f16`、`bf16` 或 `f32`。
    - 标量 `pre_quant` 模式要求 `f16`、`bf16` 或 `f32` 的标量 payload。
    - `pre_quant` 的源元素类型必须是 `f32` 或 `i32`，所选 mode 必须兼容源与目标元素类型。
    - `no_relu` 与 `normal_relu` 不接受 payload。
    - `scalar_relu` 要求 `f16` / `bf16` / `f32` 标量 payload。
    - `vector_relu` 要求 `fb` 指针，元素类型 `f16` / `bf16` / `f32`。
    - `clip` 只能出现在 `pre_relu(...)` 里。
    - `clip` 支持的目标包括 `f16`、`ui8` 以及有符号/无符号 4/8/16 位整型；payload 类型必须匹配目标家族。
    - `nz2dn` 需要 `%loop0_src_stride`；`nz2nd` 与 `nz2nz` 不接受该参数。
    - 当 `nz2dn(%loop0_src_stride)` 取值不为 1 时，必须省略 `unit_flag`。
    - `nz2nz` 要求 `f32` 目标元素类型，且不接受 `loop3`。
    - `sat`、`sat(preserve_nan)`、`nosat` 三者互斥。

## 示例

```mlir
pto.mte_l0c_l1 %l0c, %l1_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  pre_relu(%c025_f32, mode = scalar_relu),
  nz2nd,
  sat
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64, i64, i64, i64, f32, f32
```

## 相关指令

- FIXPIPE 回写兄弟指令：[pto.mte_l0c_gm](./mte-l0c-gm_zh.md)、[pto.mte_l0c_ub](./mte-l0c-ub_zh.md)
- 参数 payload 装载：[pto.mte_l1_fb](./mte-l1-fb_zh.md)
- MAD 生产者：[pto.mad](../mad/mad_zh.md) 及其变体
