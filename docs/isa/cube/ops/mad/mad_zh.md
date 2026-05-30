# pto.mad

`pto.mad` 属于 [Cube MAD 指令](../../README_zh.md#矩阵乘加mad指令)。

## 摘要

零初始化 cube 矩阵乘加：`dst[m, n] = sum_k(lhs[m, k] * rhs[k, n])`。

## 机制

从 L0A、L0B 读取已分块的操作数，在 cube MMAD 流水线中相乘后，把累加 tile 写到 L0C。结果**覆盖** L0C（不会与原有 L0C 状态累加；如需累加请使用 [`pto.mad_acc`](./mad-acc_zh.md)，需要偏置初始化请用 [`pto.mad_bias`](./mad-bias_zh.md)）。

矩阵元素类型从 `%lhs`、`%rhs`、`%dst` 三个指针的元素类型推断，没有单独的类型选择器。不被支持的类型组合属于非法程序。

## 语法

```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## 输入

| 参数 | 类型 | 描述 |
|-----------|------|------|
| `%lhs` | `!pto.ptr<A, l0a>` | L0A 中的左操作数 tile，逻辑形状 `M x K` |
| `%rhs` | `!pto.ptr<B, l0b>` | L0B 中的右操作数 tile，逻辑形状 `K x N` |
| `%dst` | `!pto.ptr<C, l0c>` | L0C 中的累加器目标 tile，逻辑形状 `M x N` |
| `%m` | `i64` | 逻辑 M 元素数 |
| `%n` | `i64` | 逻辑 N 元素数 |
| `%k` | `i64` | 逻辑 K 元素数 |

可选 clauses 见 [MAD 通用 Clauses](#mad-通用-clauses)。

## 预期输出

| 结果 | 类型 | 描述 |
| --- | --- | --- |
| 无 | `—` | 把 `M x N` 结果 tile 写到 L0C，没有 SSA 结果。 |

## 副作用

占用 CUBE 流水线并写 L0C。下游 FIXPIPE 消费者必须通过 `pto.set_flag` / `pto.wait_flag`（`PIPE_CUBE` → `PIPE_FIXP`）做同步。

## 约束

!!! warning "约束"
    - `%lhs`、`%rhs`、`%dst` 分别必须在 `l0a`、`l0b`、`l0c`。
    - `%m`、`%n`、`%k` 必须为正，并满足所选元素类型组合的目标形状上限。
    - `tf32_mode(...)` 仅在 `f32` lhs、rhs、dst 上有效。
    - `sat` / `nosat` 仅对浮点元素类型组合有效。
    - 打包 4-bit 整型数据要求 `%k` 选择偶数个 K 元素。

## MAD 通用 Clauses

| Clause | 取值 | 作用 |
|--------|------|------|
| `unit_flag(...)` | `check_only`、`check_and_set` | 参与生产者侧的 tile 同步。`check_only` 检查生产者 slot 可用；`check_and_set` 还会把生产出的 `%dst` tile 发布给后续消费者。如果调度不使用 unit flag，省略即可。 |
| `disable_gemv` | flag | 仅在 `%m = 1` 时生效。省略意味着 GEMV A-vector 消费：`%lhs` 必须按目标的 GEMV 左 tile 组织包含逻辑的 `1 x K` 行；带上则使用普通 matmul 左 tile 组织。`%m != 1` 时使用普通 matmul 组织。 |
| `sat` / `nosat` | flag | 浮点与 MX MAD 的浮点异常值行为。`sat` 时异常乘数会先做规范化（`±inf` 映射到有限类型极值，`nan` 映射到 0），有限溢出饱和到有限范围；`nosat` 保留异常输入，溢出可能产生异常输出。两者都不带时使用外部选择的执行模式。整型 MAD 不接受这两个 flag。 |
| `tf32_mode(...)` | `round_even`、`round_away` | 仅适用于非 MX 的 `f32 x f32 -> f32`。FP32 输入在相乘前先舍入到 TF32 精度；累加与输出仍为 FP32。 |
| `n_dir` | flag | 请求按 N 方向产出结果，方便与 unit flag 调度和后续布局移动配合。不会改变 `dst[m, n]`。 |

## 示例

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## 相关指令

- 累加形式：[pto.mad_acc](./mad-acc_zh.md)
- 偏置初始化形式：[pto.mad_bias](./mad-bias_zh.md)
- MX 变体：[pto.mad_mx](./mad-mx_zh.md)、[pto.mad_mx_acc](./mad-mx-acc_zh.md)、[pto.mad_mx_bias](./mad-mx-bias_zh.md)
- 操作数搬运：[pto.mte_l1_l0a](../data-movement/mte-l1-l0a_zh.md)、[pto.mte_l1_l0b](../data-movement/mte-l1-l0b_zh.md)
- 结果回写：[FIXPIPE 模型](../../fixpipe-model_zh.md)
