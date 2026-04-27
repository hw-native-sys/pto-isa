# Tile ISA Reference

The `pto.t*` tile instruction set of PTO ISA is organized by instruction set, with standalone per-op pages under `tile/ops/`.

## Instruction Sets

| Instruction Set | Description | Operations |
|--------|-------------|------------|
| [Sync and Config](./sync-and-config.md) | Resource binding, event setup, mode control | 9 |
| [Elementwise Tile-Tile](./elementwise-tile-tile.md) | Lane-wise binary and unary operations | 29 |
| [Tile-Scalar and Immediate](./tile-scalar-and-immediate.md) | Tile combined with scalar operand | 21 |
| [Reduce and Expand](./reduce-and-expand.md) | Row/column reductions and expansions | 28 |
| [Memory and Data Movement](./memory-and-data-movement.md) | GM↔tile transfer, gather/scatter | 5 |
| [Matrix and Matrix-Vector](./matrix-and-matrix-vector.md) | GEMV, matmul, and variants | 8 |
| [Layout and Rearrangement](./layout-and-rearrangement.md) | Reshape, transpose, extract, insert, concatenate, pack | 11 |
| [Irregular and Complex](./irregular-and-complex.md) | Sort, quantize, generated state, index movement, partial reductions | 16 |

## Quick Reference

### Common Tile Types

| Type | Location | Typical Use |
|------|----------|-------------|
| `TileType::Vec` | UB | General elementwise operations |
| `TileType::Mat` | L1 | Matrix multiply operations |
| `TileType::Left` | L0A | Matrix multiply A operand |
| `TileType::Right` | L0B | Matrix multiply B operand |
| `TileType::Acc` | L0C | Matrix multiply accumulator |

### Memory Capacities (A5)

| Tile Type | Memory | Capacity | Alignment |
|-----------|--------|----------|----------|
| `Vec` | UB | 256 KB | 32 B |
| `Mat` | L1 | 512 KB | 32 B |
| `Left` | L0A | 64 KB | 32 B |
| `Right` | L0B | 64 KB | 32 B |
| `Acc` | L0C | 256 KB | 32 B |
| `Bias` | Bias | 4 KB | 32 B |

## Navigation

The left sidebar provides standalone per-op pages for all tile instructions. Use the instruction set overviews above to understand shared constraints and mechanisms before reading individual opcode pages.

## See Also

- [Tile instructions](../instruction-families/tile-families.md)
- [Tile Instruction Set](../instruction-families/tile-families.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
