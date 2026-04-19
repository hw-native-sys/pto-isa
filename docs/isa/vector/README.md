# Vector ISA Reference

`pto.v*` is the PTO ISA vector micro-instruction set. It directly exposes the vector pipeline, vector registers, predicates, and vector-visible UB movement.

## Organization

The vector reference is organized by instruction family, with individual per-op pages under `vector/ops/`.

## Instruction Families

| Family | Description | Operations |
|--------|-------------|-----------|
| [Vector Load Store](./vector-load-store.md) | GM↔UB and UB↔vreg data movement | `vlds`, `vldas`, `vldus`, `vldx2`, `vsld`, `vsldb`, `vgather2`, `vgatherb`, `vgather2_bc`, `vsts`, `vstx2`, `vsst`, `vsstb`, `vscatter`, `vsta`, `vstas`, `vstar`, `vstu`, `vstus`, `vstur` |
| [Predicate and Materialization](./predicate-and-materialization.md) | Predicate broadcast and duplication | `vbr`, `vdup` |
| [Unary Vector Ops](./unary-vector-ops.md) | Single-operand vector operations | `vabs`, `vneg`, `vexp`, `vln`, `vsqrt`, `vrsqrt`, `vrec`, `vrelu`, `vnot`, `vbcnt`, `vcls`, `vmov` |
| [Binary Vector Ops](./binary-vector-ops.md) | Two-operand vector operations | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`, `vshl`, `vshr`, `vaddc`, `vsubc` |
| [Vector-Scalar Ops](./vec-scalar-ops.md) | Vector combined with scalar operand | `vadds`, `vsubs`, `vmuls`, `vmaxs`, `vmins`, `vands`, `vors`, `vxors`, `vshls`, `vshrs`, `vlrelu`, `vaddcs`, `vsubcs` |
| [Conversion Ops](./conversion-ops.md) | Type conversion | `vci`, `vcvt`, `vtrc` |
| [Reduction Ops](./reduction-ops.md) | Cross-lane reduction | `vcadd`, `vcmax`, `vcmin`, `vcgadd`, `vcgmax`, `vcgmin`, `vcpadd` |
| [Compare and Select](./compare-select.md) | Predicate generation and conditional selection | `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2` |
| [Data Rearrangement](./data-rearrangement.md) | Lane permutation and packing | `vintlv`, `vdintlv`, `vslide`, `vshift`, `vsqz`, `vusqz`, `vperm`, `vpack`, `vsunpack`, `vzunpack`, `vintlvv2`, `vdintlvv2` |
| [SFU and DSA](./sfu-and-dsa-ops.md) | Special function units and DSA operations | `vprelu`, `vexpdiff`, `vaddrelu`, `vsubrelu`, `vaxpy`, `vaddreluconv`, `vmulconv`, `vmull`, `vmula`, `vtranspose`, `vsort32`, `vbitsort`, `vmrgsort` |

## Common Constraints

- Vector width is determined by element type.
- Predicate width must match vector width.
- Alignment, distribution, and advanced forms depend on the target profile.
- There is no tile-level valid-region semantics at the vector layer.

## Quick Reference

### Common Vector Types

| Type | Width/Element | Total Elements/vreg |
|------|---------------|-------------------|
| f32 / i32 | 8 | 64 |
| f16 / bf16 / i16 | 16 | 128 |
| i8 / si8 / ui8 | 32 | 256 |

### Mask Types

| Mask Type | Bytes/Element Slot | Total Lanes |
|-----------|-------------------|-------------|
| `mask<b32>` | 4 | 64 |
| `mask<b16>` | 2 | 128 |
| `mask<b8>` | 1 | 256 |

## See Also

- [Vector instruction surface](../instruction-surfaces/vector-instructions.md) — High-level description
- [Vector instruction families](../instruction-families/vector-families.md) — Normative contracts
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
- [Micro-instruction summary](./micro-instruction-summary.md) — Scalar micro-instructions for vector scope
