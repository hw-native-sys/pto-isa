<!-- Generated from `docs/isa/vector/README.md` -->

# Vector ISA Reference

This section documents the `pto.v*` vector micro-instruction surface of PTO ISA. Pages are organized by family, with standalone per-op pages under `vector/ops/`.

## Families

| Family | Description | Operations |
|--------|-------------|------------|
| [Vector Load Store](./vector-load-store.md) | UB↔vector register transfer, gather/scatter | ~25 |
| [Predicate and Materialization](./predicate-and-materialization.md) | Vector broadcast and duplication | 2 |
| [Unary Vector Ops](./unary-vector-ops.md) | Single-input elementwise operations | 12 |
| [Binary Vector Ops](./binary-vector-ops.md) | Two-input elementwise operations | 14 |
| [Vec-Scalar Ops](./vec-scalar-ops.md) | Vector combined with scalar operand | 14 |
| [Conversion Ops](./conversion-ops.md) | Type conversion between numeric types | 3 |
| [Reduction Ops](./reduction-ops.md) | Cross-lane reductions | 6 |
| [Compare and Select](./compare-select.md) | Comparison and conditional selection | 5 |
| [Data Rearrangement](./data-rearrangement.md) | Lane permutation and packing | 10 |
| [SFU and DSA Ops](./sfu-and-dsa-ops.md) | Special function units and DSA ops | 11 |

## Quick Reference

### Common Vector Types

| Type | Description |
|------|-------------|
| `!pto.vreg<NxT>` | Vector register with N lanes of type T |
| `!pto.mask` | Predicate mask (width matches vector length) |
| `!pto.scalar<T>` | Scalar register |

### Vector Lengths

Vector length `N` is a power of 2. Common values depend on the target profile.

## Navigation

The left sidebar provides standalone per-op pages for all vector surface instructions. Use the family overviews above to understand shared constraints and mechanisms before reading individual opcode pages.

## See Also

- [Vector instruction surface](../instruction-surfaces/vector-instructions.md)
- [Vector families](../instruction-families/vector-families.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
