# pto.mad_mx_bias

`pto.mad_mx_bias` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Bias-init **MX (microscaled)** cube matrix multiply: `dst[m, n] = mx_product[m, n] + bias[n]`.

See [MX Matmul Model](./mad-mx.md#mx-matmul-model) for the per-K-group scaled multiply-accumulate.

## Mechanism

Combines the MX scaling of [`pto.mad_mx`](./mad-mx.md) with the bias-init seed of [`pto.mad_bias`](./mad-bias.md). The accumulator starts from `bias[n]` instead of zero.

## Syntax

```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```

## Inputs

Same parameter shape as [`pto.mad_bias`](./mad-bias.md#inputs), with MX `%lhs` / `%rhs` scale payload requirements from [`pto.mad_mx`](./mad-mx.md).

See [MAD Common Clauses](./mad.md#mad-common-clauses) for the optional clauses. `tf32_mode(...)` is not accepted.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the produced `M x N` MX-scaled tile to L0C with bias-init seed. |

## Side Effects

Engages the CUBE pipe; reads `%bias` from BT and MX scale payloads associated with `%lhs` / `%rhs`; writes to L0C.

## Constraints

!!! warning "Constraints"
    - All constraints from [`pto.mad_mx`](./mad-mx.md#constraints) (MX dtype combination, scale payload prerequisites, K grouping rule).
    - All `%bias` constraints from [`pto.mad_bias`](./mad-bias.md#constraints): `%bias` must be in `bt` space with element type matching `%dst`; only `N` values are consumed.

## Examples

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

## Related Ops

- Zero-init MX form: [pto.mad_mx](./mad-mx.md)
- Accumulating MX form: [pto.mad_mx_acc](./mad-mx-acc.md)
- Non-MX bias-init form: [pto.mad_bias](./mad-bias.md)
- Bias staging: [pto.mte_l1_bt](../data-movement/mte-l1-bt.md)
- MX scale loaders: [pto.mte_l1_l0a_mx](../data-movement/mte-l1-l0a-mx.md), [pto.mte_l1_l0b_mx](../data-movement/mte-l1-l0b-mx.md)
