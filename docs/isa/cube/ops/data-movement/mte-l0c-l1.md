# pto.mte_l0c_l1

`pto.mte_l0c_l1` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops). It is one of the three FIXPIPE writeback ops; see [FIXPIPE Model](../../fixpipe-model.md) for the shared writeback pipeline, layout modes, and clause semantics.

## Summary

FIXPIPE writeback from `l0c` to L1 `l1`. Applies optional pre-quant, pre-ReLU/clip, layout transform, outer-loop repeat, and saturation behavior in canonical order before storing the converted result to L1.

## Syntax

```mlir
pto.mte_l0c_l1 %src, %dst, %m, %n, %src_stride, %dst_stride
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
| `%src` | buffer-like | Accumulator source in `l0c` |
| `%dst` | buffer-like | L1 destination in `l1` |
| `%m` | i64 | Logical M element count |
| `%n` | i64 | Logical N element count |
| `%src_stride` | i64 | Source stride in C0-size units (1 unit = 32 bytes) |
| `%dst_stride` | i64 | Destination stride in destination elements |

See [FIXPIPE Common Clauses](../../fixpipe-model.md#fixpipe-common-clauses) and [FIXPIPE Layout Model](../../fixpipe-model.md#fixpipe-layout-model) for the optional clauses.

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes converted `M x N` result to L1. |

## Side Effects

Reads L0C; writes L1. Engages the AIC FIXP pipe. Consumers in L1 must synchronize through pipe events.

## Constraints

!!! warning "Constraints"
    - Clauses must appear in canonical order: `unit_flag` → `pre_quant` → `pre_relu` → layout → `loop3` → `sat`/`nosat`.
    - `pre_quant` requires payload and mode together.
    - Vector `pre_quant` modes require a `fb` pointer with `f16`, `bf16`, or `f32` element type.
    - Scalar `pre_quant` modes require an `f16`, `bf16`, or `f32` scalar payload.
    - `pre_quant` source element type must be `f32` or `i32`, and the selected mode must be compatible with the source and destination element types.
    - `no_relu` and `normal_relu` do not accept a payload.
    - `scalar_relu` requires an `f16`/`bf16`/`f32` scalar payload.
    - `vector_relu` requires a `fb` pointer with `f16`/`bf16`/`f32` element type.
    - `clip` can appear only inside `pre_relu(...)`.
    - `clip` is supported for destination `f16`, `ui8`, and signed/signless 4/8/16-bit integer destinations; payload must match the destination family.
    - `nz2dn` requires `%loop0_src_stride`; `nz2nd` and `nz2nz` do not accept it.
    - `unit_flag` must be omitted when `nz2dn(%loop0_src_stride)` uses a value other than 1.
    - `nz2nz` requires `f32` destination element type and does not accept `loop3`.
    - `sat`, `sat(preserve_nan)`, and `nosat` are mutually exclusive.

## Examples

```mlir
pto.mte_l0c_l1 %l0c, %l1_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  pre_relu(%c025_f32, mode = scalar_relu),
  nz2nd,
  sat
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64, i64, i64, i64, f32, f32
```

## Related Ops

- FIXPIPE writeback siblings: [pto.mte_l0c_gm](./mte-l0c-gm.md), [pto.mte_l0c_ub](./mte-l0c-ub.md)
- Parameter payload loader: [pto.mte_l1_fb](./mte-l1-fb.md)
- MAD producers: [pto.mad](../mad/mad.md) and variants
