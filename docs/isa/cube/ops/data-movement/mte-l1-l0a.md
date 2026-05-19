# pto.mte_l1_l0a

`pto.mte_l1_l0a` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load a logical `%m x %k` left tile from L1 `l1` into `l0a` for `pto.mad*` consumption. The source must already be in cube-fractal NZ layout; this op does not convert arbitrary row-major matrices. Use [`pto.mte_gm_l1_frac`](./mte-gm-l1-frac.md) to repack ND/DN source data first.

## Mechanism

The op moves an L1 cube-fractal tile into the L0A operand domain. The destination layout follows [NZ Fractal Layout](../../nz-fractal-layout.md#per-buffer-nz-layouts) for L0A (`K1 M1 M0 K0`, FRACTAL_NZ on A5 / FRACTAL_ZZ on A3).

If `transpose = true`, the selected logical source tile is transposed before placement in the destination operand domain. Omitting the attribute means `transpose = false`.

## Syntax

```mlir
pto.mte_l1_l0a %src, %dst, %m, %k
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Left operand destination in `l0a` |
| `%m` | i64 | Logical M extent |
| `%k` | i64 | Logical K extent |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the L0A tile that subsequent `pto.mad*` will read. |

## Side Effects

Reads L1; writes L0A. Engages the AIC MTE1 pipe.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `l0a`.
    - `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
    - `transpose = true` requires a tile shape supported by the element-type transpose granularity.

## Examples

```mlir
pto.mte_l1_l0a %l1_a, %l0a, %c16_i64, %c32_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64
```

## Related Ops

- Right operand load: [pto.mte_l1_l0b](./mte-l1-l0b.md)
- MX scale loader: [pto.mte_l1_l0a_mx](./mte-l1-l0a-mx.md)
- Upstream repack: [pto.mte_gm_l1_frac](./mte-gm-l1-frac.md)
- MAD consumers: [pto.mad](../mad/mad.md), [pto.mad_acc](../mad/mad-acc.md), [pto.mad_bias](../mad/mad-bias.md)
