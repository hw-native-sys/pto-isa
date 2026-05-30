# pto.mte_l0c_ub

`pto.mte_l0c_ub` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops). It is one of the three FIXPIPE writeback ops; see [FIXPIPE Model](../../fixpipe-model.md) for the shared writeback pipeline. This is also the architectural basis for [1→2 Cube-to-Vector tile distribution](../../fixpipe-model.md#dual-destination-broadcast-1--2-cube-to-vector).

## Summary

FIXPIPE writeback from `l0c` to UB. The data transform clauses match [`pto.mte_l0c_l1`](./mte-l0c-l1.md); UB-specific operands select single-destination or dual-destination (split-M / split-N) behavior.

## Syntax

```mlir
pto.mte_l0c_ub %src, %dst, %m, %n, %src_stride, %dst_stride,
    dst_mode(%sub_blockid | split_m | split_n)
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | — | Same as [`pto.mte_l0c_l1`](./mte-l0c-l1.md#inputs) |
| `%dst` | buffer-like | UB destination |
| `%dst_stride` | i64 | UB destination stride in destination elements |
| `dst_mode(%sub_blockid)` | i64 operand | Single-destination mode. `%sub_blockid` selects UB sub-block `0` or `1`; the value may be dynamic. |
| `dst_mode(split_m)` | keyword | Dual-destination mode that splits the logical tile along M. |
| `dst_mode(split_n)` | keyword | Dual-destination mode that splits the logical tile along N. |
| other optional clauses | — | Same as [`pto.mte_l0c_l1`](./mte-l0c-l1.md); `atomic(...)` is **not** supported |

In `dst_mode(%sub_blockid)`, the whole logical result tile is written to the selected UB sub-block using the selected layout mode and `%dst` as that sub-block's base destination pointer.

In `dst_mode(split_m)`, the logical tile is split into two M ranges: `[0, m/2)` and `[m/2, m)`. The first range is written to UB sub-block 0 and the second range is written to UB sub-block 1. Each sub-block sees its own destination origin at `%dst`; within each sub-block, the written logical tile has shape `(m / 2) x n`.

In `dst_mode(split_n)`, the logical tile is split into two N ranges: `[0, n/2)` and `[n/2, n)`. The first range is written to UB sub-block 0 and the second range is written to UB sub-block 1. Each sub-block sees its own destination origin at `%dst`; within each sub-block, the written logical tile has shape `m x (n / 2)`.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes converted `M x N` result to UB, possibly split between AIV0 / AIV1 sub-blocks. |

## Side Effects

Reads L0C; writes UB. Engages the AIC FIXP pipe and (for dual-destination modes) the dedicated 1→2 cube-to-vector data path. UB-side consumers on the AIV blocks must synchronize via cross-block sema primitives.

## Constraints

!!! warning "Constraints"
    - `atomic(...)` is not supported on `pto.mte_l0c_ub`.
    - `dst_mode(%sub_blockid)` writes the whole logical tile to one UB sub-block. Runtime `%sub_blockid` values must be `0` or `1`; constant values are checked statically when available.
    - `dst_mode(split_m)` splits the logical tile along M into two equal-height sub-block regions. `%m` must be even; each sub-block receives an `(m / 2) x n` tile.
    - `dst_mode(split_n)` splits the logical tile along N into two equal-width sub-block regions. `%n` must be a multiple of 32; each sub-block receives an `m x (n / 2)` tile.
    - Dual-destination split modes are valid only for target-supported normal or `nz2nd` writeback cases with pre-quant, pre-ReLU/clip, and other transform clauses omitted.
    - Other constraints match [`pto.mte_l0c_l1`](./mte-l0c-l1.md#constraints).

## Examples

```mlir
pto.mte_l0c_ub %l0c, %ub_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  dst_mode(%c1_i64),
  nz2nd
  : !pto.ptr<f32, l0c>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
```

## Related Ops

- FIXPIPE writeback siblings: [pto.mte_l0c_l1](./mte-l0c-l1.md), [pto.mte_l0c_gm](./mte-l0c-gm.md)
- Parameter payload loader: [pto.mte_l1_fb](./mte-l1-fb.md)
- MAD producers: [pto.mad](../mad/mad.md) and variants
- Cluster broadcast model: [FIXPIPE Model — Dual-Destination Broadcast](../../fixpipe-model.md#dual-destination-broadcast-1--2-cube-to-vector)
