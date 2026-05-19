# pto.mte_l0c_gm

`pto.mte_l0c_gm` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops). It is one of the three FIXPIPE writeback ops; see [FIXPIPE Model](../../fixpipe-model.md) for the shared writeback pipeline.

## Summary

FIXPIPE writeback from `l0c` to GM. The data transform clauses match [`pto.mte_l0c_l1`](./mte-l0c-l1.md); GM-specific operands select the GM write path and optional atomic update behavior.

## Syntax

```mlir
pto.mte_l0c_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | — | Same as [`pto.mte_l0c_l1`](./mte-l0c-l1.md#inputs) |
| `%dst` | buffer-like | GM destination |
| `%dst_stride` | i64 | GM destination stride in destination elements |
| `%sid` | i64 | GM stream/session hint; does not change written values |
| `%l2_cache_ctrl` | i64 | GM store cache hint; does not change written values |
| `atomic(type = ..., op = ...)` | clause | Optional GM read-modify-write |
| other optional clauses | — | Same as [`pto.mte_l0c_l1`](./mte-l0c-l1.md#syntax) |

`%sid` and `%l2_cache_ctrl` affect the memory path only — they do not change the logical result, destination layout, numeric conversion, or atomic operation. For target-profile GM writeback, constant `%sid` values must be in `[0, 3]` (use `0` unless the surrounding memory system deliberately assigns a different stream/session hint). Constant `%l2_cache_ctrl` values must fit in the target cache-control hint range `[0, 15]`.

`atomic(type = T, op = add|max|min)` performs an atomic read-modify-write at each GM destination element. `add` accumulates the converted value into the existing GM value. `max` and `min` compare using `T` and write the selected value. Supported atomic types: `f32`, `f16`, `bf16`, `s32`, `s16`, `s8`.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes converted `M x N` result to GM. |

## Side Effects

Reads L0C; writes GM. Engages the AIC FIXP pipe. If `atomic(...)` is present, the GM update is read-modify-write.

## Constraints

!!! warning "Constraints"
    - `atomic(...)` is valid only on `pto.mte_l0c_gm`.
    - `atomic` requires both `type` and `op`.
    - Atomic op values are `add`, `max`, and `min`.
    - If `%sid` or `%l2_cache_ctrl` is a constant, it must be in the target range described above.
    - Other constraints match [`pto.mte_l0c_l1`](./mte-l0c-l1.md#constraints).

## Examples

```mlir
pto.mte_l0c_gm %l0c, %out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  %c0_i64, %c0_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  nz2nd,
  atomic(type = f16, op = add)
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64, f32
```

## Related Ops

- FIXPIPE writeback siblings: [pto.mte_l0c_l1](./mte-l0c-l1.md), [pto.mte_l0c_ub](./mte-l0c-ub.md)
- Parameter payload loader: [pto.mte_l1_fb](./mte-l1-fb.md)
- MAD producers: [pto.mad](../mad/mad.md) and variants
