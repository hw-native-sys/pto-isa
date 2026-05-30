# pto.mte_l1_l0a_mx

`pto.mte_l1_l0a_mx` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load left-side MX scale fragments for a logical `%m x %k` left data tile. The fragments prepare the scale payload consumed by [`pto.mad_mx`](../mad/mad-mx.md) / [`pto.mad_mx_acc`](../mad/mad-mx-acc.md) / [`pto.mad_mx_bias`](../mad/mad-mx-bias.md).

## MX Scale Load Model

Each scale entry applies to one 32-element K group.

- Left scale logical shape: `[M, ceil(K / 32)]`
- L1 source data is organized as 32B scale fragments in the same logical order as the associated data tile.

## Syntax

```mlir
pto.mte_l1_l0a_mx %src, %dst, %m, %k
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Left-side MX payload destination associated with `l0a` |
| `%m` | i64 | M extent of the associated left data tile |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the MX scale payload associated with the L0A operand tile. |

## Side Effects

Reads L1; writes MX scale state associated with L0A. The result is consumed by the next `pto.mad_mx*` op that reads from this `%lhs`.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `l0a`.
    - `%src` and `%dst` must satisfy 32B MX scale-fragment alignment.

## Examples

```mlir
pto.mte_l1_l0a_mx %l1_a_scale, %l0a_scale, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0a>, i64, i64
```

## Related Ops

- Data tile load: [pto.mte_l1_l0a](./mte-l1-l0a.md)
- Right-side scale loader: [pto.mte_l1_l0b_mx](./mte-l1-l0b-mx.md)
- MX MAD consumers: [pto.mad_mx](../mad/mad-mx.md), [pto.mad_mx_acc](../mad/mad-mx-acc.md), [pto.mad_mx_bias](../mad/mad-mx-bias.md)
