# pto.mad_mx

`pto.mad_mx` is part of the [Cube MAD Ops](../../README.md#matrix-multiply-mad-ops).

## Summary

Zero-init **MX (microscaled)** cube matrix multiply: `dst[m, n] = mx_product[m, n]`.

## MX Matmul Model

`pto.mad_mx*` additionally applies microscaling. The scale payloads are loaded with [`pto.mte_l1_l0a_mx`](../data-movement/mte-l1-l0a-mx.md) / [`pto.mte_l1_l0b_mx`](../data-movement/mte-l1-l0b-mx.md) and are associated with the selected `%lhs` / `%rhs` tiles; they are **not** direct operands of `pto.mad_mx*`.

The K dimension is partitioned into 32-element groups:

```text
k_group = floor(k / 32)

mx_product[m, n] =
  sum k in 0 .. K-1:
    (lhs[m, k] * lhs_scale[m, k_group]) *
    (rhs[k, n] * rhs_scale[k_group, n])
```

Current target-profile MX data tiles use `f8E4M3FN`. `%k` must be compatible with MX grouping. On the current target profile, MX matmul consumes K in 64-element multiples, which contain two 32-element scale groups.

## Mechanism

Functionally equivalent to [`pto.mad`](./mad.md) but with the MX scaling applied during the multiply-accumulate. Like `pto.mad`, the result overwrites L0C.

## Syntax

```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```

## Inputs

Same parameter shape as [`pto.mad`](./mad.md#inputs). `%lhs` and `%rhs` must additionally have matching MX scale payloads loaded into L0A / L0B before this op is issued.

See [MAD Common Clauses](./mad.md#mad-common-clauses) for the optional clauses (note: `tf32_mode(...)` is **not** a clause of MX MAD).

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the produced `M x N` MX-scaled tile to L0C. |

## Side Effects

Engages the CUBE pipe; reads scale payloads associated with `%lhs` / `%rhs`; writes to L0C.

## Constraints

!!! warning "Constraints"
    - Operands must use a target-supported MX dtype combination (currently `f8E4M3FN` on the supported profile).
    - Matching left and right MX scale payloads must be loaded before this op via [`pto.mte_l1_l0a_mx`](../data-movement/mte-l1-l0a-mx.md) / [`pto.mte_l1_l0b_mx`](../data-movement/mte-l1-l0b-mx.md).
    - `%k` must satisfy the MX grouping rule described in [MX Matmul Model](#mx-matmul-model).
    - `tf32_mode(...)` is not a clause of MX MAD.
    - Other constraints match [`pto.mad`](./mad.md#constraints).

## Examples

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

## Related Ops

- Non-MX form: [pto.mad](./mad.md)
- Accumulating MX form: [pto.mad_mx_acc](./mad-mx-acc.md)
- Bias-init MX form: [pto.mad_mx_bias](./mad-mx-bias.md)
- MX scale loaders: [pto.mte_l1_l0a_mx](../data-movement/mte-l1-l0a-mx.md), [pto.mte_l1_l0b_mx](../data-movement/mte-l1-l0b-mx.md)
