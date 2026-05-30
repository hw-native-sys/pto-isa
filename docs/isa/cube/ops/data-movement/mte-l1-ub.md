# pto.mte_l1_ub

`pto.mte_l1_ub` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Structured L1→UB copy. Reads grouped byte ranges from `%src` in L1 and writes them to `%dst` in UB. This is the L1→Vector data path complement to [`pto.mte_ub_l1`](../../../scalar/ops/dma-copy/mte-ub-l1.md).

## Mechanism

Uses the same grouped `nburst(...) [loop(...)]*` model as [`pto.mte_gm_l1`](./mte-gm-l1.md). For each `nburst` row the source and destination advance by `src_stride` / `dst_stride`. Outer `loop(...)` groups wrap the inner transfer pattern.

## Syntax

```mlir
pto.mte_l1_ub %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, l1>, !pto.ptr<T, ub>, i64, i64, i64, i64
```

## Inputs

Same grouped byte model as [`pto.mte_gm_l1`](./mte-gm-l1.md#inputs), with source and destination address spaces reversed to `l1 -> ub`.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes data into the UB destination region. |

## Side Effects

Reads L1-visible storage; writes UB-visible storage. The transfer is issued on the AIC side and the cube-to-vector data path; UB consumers on the AIV side must synchronize accordingly.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `ub`.
    - `nburst(...)` is required.
    - Each `loop(...)` group must provide all three operands.

## Examples

```mlir
pto.mte_l1_ub %l1_src, %ub_dst, %c64_i64
  nburst(%c2_i64, %c128_i64, %c64_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

## Related Ops

- Reverse direction (UB → L1): [pto.mte_ub_l1](../../../scalar/ops/dma-copy/mte-ub-l1.md)
- GM → L1: [pto.mte_gm_l1](./mte-gm-l1.md), [pto.mte_gm_l1_frac](./mte-gm-l1-frac.md)
