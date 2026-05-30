# pto.mte_l1_fb

`pto.mte_l1_fb` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load FIXPIPE parameter payloads from L1 into `fb`. Vector `pre_quant(...)` and `pre_relu(...)` clauses in the `pto.mte_l0c_*` writeback family later consume these payloads through `fb` pointers. See [FIXPIPE Model](../../fixpipe-model.md) for the writeback pipeline context.

## Mechanism

One burst loads `%len_burst` parameter-load units from `%src` to `%dst`. The copy unit is the parameter-load unit of this op — separate from the row size consumed by `mte_l0c_*` vector payloads. `%len_burst` and the `nburst(...)` gaps are counted in these load units, not in bytes and not in destination elements.

After `pto.mte_l1_fb` materializes the payload in `fb`, vector pre-ReLU consumers read it as 64B parameter rows and vector pre-quant consumers read it as 128B parameter rows. The payload pointer passed to `mte_l0c_*` must point at the first row for the logical output tile, and rows must follow the same channel/NZ order consumed by that store.

## Syntax

```mlir
pto.mte_l1_fb %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, fb>, i64, i64, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Scaling/parameter destination pointer in `fb` |
| `%len_burst` | i64 | Number of parameter-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in parameter-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in parameter-load units |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes FIXPIPE parameter payload into the `fb` destination region. |

## Side Effects

Reads L1-visible storage; writes FB-visible storage. The result is consumed by subsequent `pto.mte_l0c_*` writeback ops that reference an `fb` pointer in their `pre_quant` or `pre_relu` clauses.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `fb`.
    - Vector `pre_quant` and `pre_relu` consumers require parameter data prepared in the row order documented under [FIXPIPE Model](../../fixpipe-model.md).

## Examples

```mlir
pto.mte_l1_fb %l1_fp, %fb_fp, %c2_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, l1>, !pto.ptr<f32, fb>, i64, i64, i64, i64
```

## Related Ops

- FIXPIPE writeback consumers: [pto.mte_l0c_l1](./mte-l0c-l1.md), [pto.mte_l0c_gm](./mte-l0c-gm.md), [pto.mte_l0c_ub](./mte-l0c-ub.md)
- FIXPIPE pipeline overview: [FIXPIPE Model](../../fixpipe-model.md)
