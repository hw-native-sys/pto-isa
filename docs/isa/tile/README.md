# Tile ISA Reference

`pto.t*` is the tile-centric execution surface of the PTO instruction set architecture. It covers tile data loading, elementwise compute, reduction and expansion, layout rearrangement, matrix multiply, explicit synchronization, and a small set of irregular specialized operations.

This group of documents is organized as "read the family page first, then the individual instruction page." Family pages explain shared mechanisms, roles, constraints, and profile boundaries. Leaf pages under `tile/ops/` give the per-instruction contracts.

## Instruction Families

| Family | Description | Typical Operations |
|--------|-------------|-------------------|
| [Sync and Config](./sync-and-config.md) | Resource binding, event setup, tile-side mode control | `TASSIGN`, `TSYNC` |
| [Elementwise Tile-Tile](./elementwise-tile-tile.md) | Tile-to-tile elementwise arithmetic, comparison, and selection | `TADD`, `TMUL`, `TSEL` |
| [Tile-Scalar and Immediate](./tile-scalar-and-immediate.md) | Tile combined with scalar or immediate operand | `TADDS`, `TMULS` |
| [Reduce and Expand](./reduce-and-expand.md) | Row/column reductions and axis-wise expansion | `TROWSUM`, `TROWEXPAND` |
| [Memory and Data Movement](./memory-and-data-movement.md) | GM↔tile transfer and tile-side gather/scatter | `TLOAD`, `TSTORE` |
| [Matrix and Matrix-Vector](./matrix-and-matrix-vector.md) | Cube-path matrix multiply, GEMV, and variants | `TMATMUL`, `TGEMV` |
| [Layout and Rearrangement](./layout-and-rearrangement.md) | Reshape, transpose, extract, insert, img2col | `TTRANS`, `TIMG2COL` |
| [Irregular and Complex](./irregular-and-complex.md) | Sort, quantization, indexed movement, partial reduction | `TSORT32`, `TQUANT` |

## Common Tile Roles

PTO tile roles are architectural abstractions and should not be conflated with a single physical implementation on a given backend. When reading tile instructions, first distinguish the role, then examine dtype, shape, layout, and valid region.

| Role | Meaning | Typical Use |
|------|---------|-------------|
| `Vec` | Vector tile buffer abstraction | Elementwise, reduction, movement, rearrangement |
| `Left` | Left matrix operand tile, L0A path | matmul / GEMV left input |
| `Right` | Right matrix operand tile, L0B path | matmul / GEMV right input |
| `Acc` | Accumulator / output tile | matmul / GEMV result |
| `Bias` | Bias tile | `*_bias` variants |
| `ScaleLeft` / `ScaleRight` | Left/right scale tile for MX block-scale | `*_mx` variants |

## Memory Capacities (A5)

| Tile Type | Memory | Capacity | Alignment |
|-----------|--------|----------|----------|
| `Vec` | UB | 256 KB | 32 B |
| `Mat` | L1 | 512 KB | 32 B |
| `Left` | L0A | 64 KB | 32 B |
| `Right` | L0B | 64 KB | 32 B |
| `Acc` | L0C | 256 KB | 32 B |
| `Bias` | Bias | 4 KB | 32 B |

## Reading Order

If you are new to PTO tile instructions, read in this order:

1. Read [Tile instruction surface](../instruction-surfaces/tile-instructions.md) to understand the boundary between the tile path and scalar/vector paths.
2. Read [Location intent and legality](../state-and-types/location-intent-and-legality.md) and [Layout](../state-and-types/layout.md) to build the role and layout constraints.
3. Then enter the corresponding family page.
4. Finally read the specific leaf page.

## Shared Constraints

- Tile `dtype`, `shape`, `layout`, `role`, and `valid region` all may enter legality checking.
- Most elementwise and rearrangement operations iterate over the destination tile's valid region.
- Matrix multiply operations are additionally constrained by `Left`/`Right`/`Acc`/`Bias`/scale tile roles.
- Some high-performance or special-format paths are only available on specific profiles (e.g., A5-only MX block-scale).

## See Also

- [Tile instruction surface](../instruction-surfaces/tile-instructions.md)
- [Tile instruction families](../instruction-families/tile-families.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
