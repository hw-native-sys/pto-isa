# pto.mte_l1_l0b

`pto.mte_l1_l0b` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Load a logical `%k x %n` right tile from L1 `l1` into `l0b` for `pto.mad*` consumption. The source must already be in cube-fractal NZ layout; this op does not convert arbitrary row-major matrices.

L0B uses `K1 N1 N0 K0` FRACTAL_ZN with **K innermost** so the cube hardware reads all `K0` elements per cycle without striding. The inner-box transpose from L1's `K1 N1 K0 N0` to L0B's `K1 N1 N0 K0` is performed as part of this movement; no separate user-visible pass is required.

## Mechanism

If `transpose = true`, the selected logical source tile is transposed before placement in the destination operand domain. Omitting the attribute means `transpose = false`.

See [NZ Fractal Layout — Why K-Innermost on L0B](../../nz-fractal-layout.md#why-k-innermost-on-l0b) for the layout rationale.

## Syntax

```mlir
pto.mte_l1_l0b %src, %dst, %k, %n
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Right operand destination in `l0b` |
| `%k` | i64 | Logical K extent |
| `%n` | i64 | Logical N extent |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes the L0B tile that subsequent `pto.mad*` will read. |

## Side Effects

Reads L1; writes L0B. Engages the AIC MTE1 pipe.

## Constraints

!!! warning "Constraints"
    - `%src` must be in `l1`, `%dst` must be in `l0b`.
    - `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
    - `transpose = true` requires a tile shape supported by the element-type transpose granularity.

## Examples

```mlir
pto.mte_l1_l0b %l1_b, %l0b, %c32_i64, %c16_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64
```

## Related Ops

- Left operand load: [pto.mte_l1_l0a](./mte-l1-l0a.md)
- MX scale loader: [pto.mte_l1_l0b_mx](./mte-l1-l0b-mx.md)
- Upstream repack: [pto.mte_gm_l1_frac](./mte-gm-l1-frac.md)
- MAD consumers: [pto.mad](../mad/mad.md), [pto.mad_acc](../mad/mad-acc.md), [pto.mad_bias](../mad/mad-bias.md)
