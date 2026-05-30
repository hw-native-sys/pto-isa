# pto.mad_bias

`pto.mad_bias` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Bias-init cube matrix multiply: `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n]) + bias[n]`.

## Mechanism

Like [`pto.mad`](./mad.md), but seeds the accumulator with a per-N bias vector instead of zero. Useful as the first MAD in a K-tiled sequence where the bias is known up front; subsequent partial sums can accumulate via [`pto.mad_acc`](./mad-acc.md).

## Syntax

```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `%lhs`, `%rhs`, `%dst`, `%m`, `%n`, `%k` | — | Same as [`pto.mad`](./mad.md#inputs) |
| `%bias` | `!pto.ptr<C, bt>` | Bias vector in BT, interpreted as `N` values broadcast across M |

See [MAD Common Clauses](./mad.md#mad-common-clauses) for the optional clauses.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the produced `M x N` tile to L0C with bias-init seed. |

## Side Effects

Engages the CUBE pipe, reads `%bias` from BT, writes to L0C. The caller is responsible for staging `%bias` into BT via [`pto.mte_l1_bt`](../data-movement/mte-l1-bt.md) prior to this op.

## Constraints

!!! warning "Constraints"
    - `%bias` must be in `bt` address space.
    - `%bias` element type must match `%dst` element type.
    - Only `N` bias values are consumed; `%bias` is not an `M x N` matrix.
    - Other constraints match [`pto.mad`](./mad.md#constraints).

## Examples

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

## Related Ops

- Zero-init form: [pto.mad](./mad.md)
- Accumulating form: [pto.mad_acc](./mad-acc.md)
- MX bias-init form: [pto.mad_mx_bias](./mad-mx-bias.md)
- Bias staging: [pto.mte_l1_bt](../data-movement/mte-l1-bt.md)
