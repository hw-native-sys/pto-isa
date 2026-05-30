# pto.mte_l1_bt

`pto.mte_l1_bt` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load an L1 bias payload into the `bt` address space for later [`pto.mad_bias`](../mad/mad-bias.md) or [`pto.mad_mx_bias`](../mad/mad-mx-bias.md) consumption. The consumer interprets the result as an `N`-element bias vector `bias[n]`.

## Mechanism

One burst loads `%len_burst` bias-load units from `%src` and writes the corresponding bias values to `%dst`. After each burst except the last, source and destination advance by the burst length plus the corresponding gap. Each unit is the bias-element width for the configured type pair.

## Syntax

```mlir
pto.mte_l1_bt %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, bt>, i64, i64, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Bias destination pointer in `bt` |
| `%len_burst` | i64 | Number of bias-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in bias-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in bias-load units |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes bias values into the `bt` destination region. |

## Side Effects

Reads L1-visible storage; writes BT-visible storage. The result is consumed by `pto.mad_bias` / `pto.mad_mx_bias` later in the cube pipeline.

## Constraints

!!! warning "Constraints"
    - Supported type pairs: `f32 -> f32`, `i32 -> i32`, `f16 -> f32`, `bf16 -> f32`.
    - For `bf16 -> f32`, compact bf16 source values are always widened to f32 bias values. For `f16 -> f32`, compact f16 source values are widened when the load is used as an f32 bias payload; otherwise the f16 payload is stored in the 32-bit bias slot with unused high bits.
    - Load exactly the channel bias values needed by the consumer tile; the bias payload is not result-shaped.

## Examples

```mlir
pto.mte_l1_bt %l1_bias, %bt, %c1_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f32, bt>, i64, i64, i64, i64
```

## Related Ops

- Bias-init MAD: [pto.mad_bias](../mad/mad-bias.md), [pto.mad_mx_bias](../mad/mad-mx-bias.md)
- FIXPIPE auxiliary payload: [pto.mte_l1_fb](./mte-l1-fb.md)
