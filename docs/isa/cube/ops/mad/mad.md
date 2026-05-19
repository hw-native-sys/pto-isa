# pto.mad

`pto.mad` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Zero-init cube matrix multiply: `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n])`.

## Mechanism

Reads tiled operands from L0A and L0B, multiplies them in the cube MMAD pipe, and writes the accumulator tile in L0C. The result overwrites L0C (no accumulation with prior L0C state — use [`pto.mad_acc`](./mad-acc.md) for accumulation, or [`pto.mad_bias`](./mad-bias.md) for bias-init).

The matrix element types are inferred from `%lhs`, `%rhs`, and `%dst` pointer element types — there is no separate type selector. Unsupported type combinations are invalid programs.

## Syntax

```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `%lhs` | `!pto.ptr<A, l0a>` | Left operand tile in L0A, interpreted as logical `M x K` |
| `%rhs` | `!pto.ptr<B, l0b>` | Right operand tile in L0B, interpreted as logical `K x N` |
| `%dst` | `!pto.ptr<C, l0c>` | Accumulator destination tile in L0C, interpreted as logical `M x N` |
| `%m` | `i64` | Logical M element count |
| `%n` | `i64` | Logical N element count |
| `%k` | `i64` | Logical K element count |

See [MAD Common Clauses](#mad-common-clauses) for the optional clauses.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the produced `M x N` tile to L0C. No SSA result. |

## Side Effects

Engages the CUBE pipe and writes to L0C. Downstream FIXPIPE consumers must synchronize through `pto.set_flag` / `pto.wait_flag` (`PIPE_CUBE` → `PIPE_FIXP`).

## Constraints

!!! warning "Constraints"
    - `%lhs`, `%rhs`, and `%dst` must be in `l0a`, `l0b`, and `l0c`.
    - `%m`, `%n`, and `%k` must be positive and satisfy the target shape limits for the selected element-type combination.
    - `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
    - `sat` / `nosat` requires a floating element-type combination.
    - Packed 4-bit integer data requires `%k` to select an even number of K elements.

## MAD Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag(...)` | `check_only`, `check_and_set` | Participates in producer-side tile synchronization. `check_only` checks that the producer slot can be used. `check_and_set` also publishes the produced `%dst` tile for later consumers. Omit when the schedule does not use unit flags for this tile. |
| `disable_gemv` | flag | Applies only when `%m = 1`. Omitted means GEMV A-vector consumption: `%lhs` must contain the logical `1 x K` row in the target GEMV left-tile organization. Present means normal matmul left-tile organization. The mathematical result is still `lhs @ rhs`; only the required `%lhs` organization changes. For `%m != 1`, normal matmul organization is used. |
| `sat` / `nosat` | flags | Floating exceptional-value mode for floating and MX MAD forms. With `sat`, exceptional multiply inputs are normalized before arithmetic (`+/-inf` to finite type extrema, `nan` to 0) and finite overflow saturates to the finite type range. With `nosat`, exceptional inputs are preserved and overflow may produce exceptional outputs. Omit both to use the execution mode selected outside this op. Integer MAD forms do not accept these flags. |
| `tf32_mode(...)` | `round_even`, `round_away` | Valid only for non-MX `f32 x f32 -> f32`. FP32 inputs are rounded to TF32 precision before multiplication; accumulation and output remain FP32. |
| `n_dir` | flag | Requests N-direction result production order for schedules that combine compute with unit flags and later layout movement. It does not change `dst[m, n]`. |

## Examples

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## Related Ops

- Accumulating form: [pto.mad_acc](./mad-acc.md)
- Bias-init form: [pto.mad_bias](./mad-bias.md)
- MX variants: [pto.mad_mx](./mad-mx.md), [pto.mad_mx_acc](./mad-mx-acc.md), [pto.mad_mx_bias](./mad-mx-bias.md)
- Operand staging: [pto.mte_l1_l0a](../data-movement/mte-l1-l0a.md), [pto.mte_l1_l0b](../data-movement/mte-l1-l0b.md)
- Result writeback: [FIXPIPE Model](../../fixpipe-model.md)
