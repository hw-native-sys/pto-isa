<!-- Generated from `docs/isa/instruction-families/tile-families.md` -->

# Tile Families

Tile-family documentation explains how `pto.t*` groups behave. Each family describes the shared mechanism, operand model, constraints, and target-profile narrowing before the reader drops into the standalone per-op pages under `tile/ops/`.

## Overview

| Family | Prefix | Description |
|--------|--------|-------------|
| [Sync and Config](./sync-and-config.md) | `pto.tassign`, `pto.tsync`, `pto.tset*` | Resource binding, event setup, mode control |
| [Elementwise Tile-Tile](./elementwise-tile-tile.md) | `pto.tadd`, `pto.tmul`, `pto.tcmp`, `pto.tcvt` | Lane-wise binary and unary operations |
| [Tile-Scalar and Immediate](./tile-scalar-and-immediate.md) | `pto.tadds`, `pto.tmuls`, `pto.tmins` | Tile combined with scalar or immediate operand |
| [Reduce and Expand](./reduce-and-expand.md) | `pto.trowsum`, `pto.tcolmax`, `pto.trowexpand` | Row/column reductions and expansions |
| [Memory and Data Movement](./memory-and-data-movement.md) | `pto.tload`, `pto.tstore`, `pto.mgather` | GM↔tile transfer, gather/scatter |
| [Matrix and Matrix-Vector](./matrix-and-matrix-vector.md) | `pto.tgemv`, `pto.tmatmul`, `pto.tmatmul_bias` | GEMV, matmul, and variants |
| [Layout and Rearrangement](./layout-and-rearrangement.md) | `pto.tmov`, `pto.ttrans`, `pto.textract` | Reshape, transpose, extract, insert |
| [Irregular and Complex](./irregular-and-complex.md) | `pto.tmrgsort`, `pto.tquant`, `pto.tprint` | Sort, quantize, histogram, print |

## Shared Constraints

All tile families must state:

1. **Valid-region interaction** — How the family interprets source tile valid regions relative to the destination.
2. **Layout and role restrictions** — Which tile layouts, TileTypes, and roles the family accepts.
3. **Target-profile restrictions** — Where A2/A3 and A5 differ from each other and from the portable ISA contract.
4. **Cases that are not allowed** — Conditions that are illegal across the family.

## Valid Region Compatibility

All elementwise tile-tile operations iterate over the **destination tile's valid region**. For each lane `(r, c)` in the destination's valid region:

- The corresponding lane `(r, c)` from each source tile is read, **regardless of whether that lane is within the source tile's own valid region**
- Source tiles whose valid region does not cover `(r, c)` read **implementation-defined values**
- Programs MUST NOT rely on any particular value being read from an out-of-region source lane unless the operation explicitly documents the behavior

Producers that need defined behavior when valid regions differ SHOULD either:

- Ensure all operands have matching valid regions, or
- Use a fill/pad operation to expand the source before the elementwise operation

## Saturating Variants

Operations with the `_c` suffix perform saturating arithmetic:

| Variant | Base Op | Overflow/Underflow Behavior |
|---------|---------|---------------------------|
| `TADD` | Addition | Wrapping: result wraps around the type's representable range |
| `TADDC` | Addition | Saturating: result is clamped to the type's min/max representable value |
| `TSUB` | Subtraction | Wrapping: result wraps around the type's representable range |
| `TSUBC` | Subtraction | Saturating: result is clamped to the type's min/max representable value |

Programs MUST NOT assume that `TADDC` and `TADD` produce identical results when overflow does not occur; they MAY differ even for in-range values due to implementation precision choices.

## Type Support by Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8/u8 | Yes | Yes | Yes |
| i16/u16 | Yes | Yes | Yes |
| i32/u32 | Yes | Yes | Yes |
| i64/u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |
| hifloat8_t / float4_e* | No | No | Yes |

## Navigation

See the [Tile ISA reference](./tile/README.md) for the full per-op reference under `tile/ops/`.

## See Also

- [Tile instruction surface](./instruction-surfaces/tile-instructions.md) — High-level surface description
- [Instruction families](./README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
