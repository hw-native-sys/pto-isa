# pto.mte_l1_l0b_mx

`pto.mte_l1_l0b_mx` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load right-side MX scale fragments for a logical `%k x %n` right data tile. The fragments prepare the scale payload consumed by [`pto.mad_mx`](../mad/mad-mx.md) / [`pto.mad_mx_acc`](../mad/mad-mx-acc.md) / [`pto.mad_mx_bias`](../mad/mad-mx-bias.md).

## MX Scale Load Model

Each scale entry applies to one 32-element K group.

- Right scale logical shape: `[ceil(K / 32), N]`
- L1 source data is organized as 32B scale fragments in the same logical order as the associated data tile.

## Syntax

```mlir
pto.mte_l1_l0b_mx %src, %dst, %k, %n
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Right-side MX payload destination associated with `l0b` |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |
| `%n` | i64 | N extent of the associated right data tile |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the MX scale payload associated with the L0B operand tile. |

## Side Effects

Reads L1; writes MX scale state associated with L0B. The result is consumed by the next `pto.mad_mx*` op that reads from this `%rhs`.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `l0b`.
    - `%src` and `%dst` must satisfy 32B MX scale-fragment alignment.

## Examples

```mlir
pto.mte_l1_l0b_mx %l1_b_scale, %l0b_scale, %c64_i64, %c16_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64
```

## Related Ops

- Data tile load: [pto.mte_l1_l0b](./mte-l1-l0b.md)
- Left-side scale loader: [pto.mte_l1_l0a_mx](./mte-l1-l0a-mx.md)
- MX MAD consumers: [pto.mad_mx](../mad/mad-mx.md), [pto.mad_mx_acc](../mad/mad-mx-acc.md), [pto.mad_mx_bias](../mad/mad-mx-bias.md)
