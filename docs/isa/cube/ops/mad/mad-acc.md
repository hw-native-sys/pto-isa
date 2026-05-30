# pto.mad_acc

`pto.mad_acc` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Accumulating cube matrix multiply: `dst[m, n] = dst[m, n] + sum_k(lhs[m, k] * rhs[k, n])`.

## Mechanism

Like [`pto.mad`](./mad.md), but adds the freshly-computed product to the existing L0C accumulator state instead of overwriting it. Typical use is K-axis tiling, where successive MAD calls accumulate partial sums along K until the full reduction is complete.

## Syntax

```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## Inputs

Same parameter shape and semantics as [`pto.mad`](./mad.md#inputs). See [MAD Common Clauses](./mad.md#mad-common-clauses) for the optional clauses.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Updates the existing `M x N` tile in L0C in place. No SSA result. |

## Side Effects

Engages the CUBE pipe; reads from and writes to L0C. The caller is responsible for ensuring the L0C tile has been initialized (typically by an initial [`pto.mad`](./mad.md) or [`pto.mad_bias`](./mad-bias.md) on the same `%dst` before the first `pto.mad_acc`).

## Constraints

Same as [`pto.mad`](./mad.md#constraints).

## Examples

```mlir
// K-axis tiling: initial pto.mad then repeated pto.mad_acc.
pto.mad %l0a_k0, %l0b_k0, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64

pto.mad_acc %l0a_k1, %l0b_k1, %l0c, %c16_i64, %c16_i64, %c32_i64 unit_flag(check_only)
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## Related Ops

- Zero-init form: [pto.mad](./mad.md)
- Bias-init form: [pto.mad_bias](./mad-bias.md)
- MX accumulating form: [pto.mad_mx_acc](./mad-mx-acc.md)
