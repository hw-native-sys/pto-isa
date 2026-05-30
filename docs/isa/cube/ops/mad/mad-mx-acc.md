# pto.mad_mx_acc

`pto.mad_mx_acc` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Accumulating **MX (microscaled)** cube matrix multiply: `dst[m, n] = dst[m, n] + mx_product[m, n]`.

See [MX Matmul Model](./mad-mx.md#mx-matmul-model) for the per-K-group scaled multiply-accumulate.

## Mechanism

Like [`pto.mad_mx`](./mad-mx.md) but adds the MX-scaled product to existing L0C state. Typical use is K-axis tiling for MX GEMM.

## Syntax

```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## Inputs

Same parameter shape as [`pto.mad_mx`](./mad-mx.md#inputs).

See [MAD Common Clauses](./mad.md#mad-common-clauses) for the optional clauses. `tf32_mode(...)` is not accepted.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Updates the existing `M x N` tile in L0C with MX-scaled accumulation. |

## Side Effects

Same as [`pto.mad_mx`](./mad-mx.md#side-effects). The caller is responsible for ensuring the L0C tile has been initialized (typically by an initial [`pto.mad_mx`](./mad-mx.md) or [`pto.mad_mx_bias`](./mad-mx-bias.md) on the same `%dst`).

## Constraints

Same as [`pto.mad_mx`](./mad-mx.md#constraints).

## Examples

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## Related Ops

- Zero-init MX form: [pto.mad_mx](./mad-mx.md)
- Bias-init MX form: [pto.mad_mx_bias](./mad-mx-bias.md)
- Non-MX accumulating form: [pto.mad_acc](./mad-acc.md)
