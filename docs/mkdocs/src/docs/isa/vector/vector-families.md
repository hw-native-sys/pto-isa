<!-- Generated from `docs/isa/vector/vector-families.md` -->

# Vector Families

Vector-family documentation explains how `pto.v*` groups behave. Each family describes the shared mechanism, operand model, constraints, and target-profile narrowing before the reader drops into the standalone per-op pages under `vector/ops/`.

## Overview

| Family | Prefix | Description |
|--------|--------|-------------|
| [Vector Load/Store](./vector-load-store.md) | `pto.vlds`, `pto.vsts`, `pto.vgather2` | UB↔vector register transfer, gather/scatter |
| [Predicate and Materialization](./predicate-and-materialization.md) | `pto.vbr`, `pto.vdup` | Vector broadcast and duplication |
| [Unary Vector Ops](./unary-vector-ops.md) | `pto.vabs`, `pto.vneg`, `pto.vexp`, `pto.vsqrt` | Single-input elementwise operations |
| [Binary Vector Ops](./binary-vector-ops.md) | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vcmp` | Two-input elementwise operations |
| [Vec-Scalar Ops](./vec-scalar-ops.md) | `pto.vadds`, `pto.vmuls`, `pto.vshls` | Vector combined with scalar operand |
| [Conversion Ops](./conversion-ops.md) | `pto.vci`, `pto.vcvt`, `pto.vtrc` | Type conversion between numeric types |
| [Reduction Ops](./reduction-ops.md) | `pto.vcadd`, `pto.vcmax`, `pto.vcgadd` | Cross-lane reductions |
| [Compare and Select](./compare-select.md) | `pto.vcmp`, `pto.vsel`, `pto.vselr` | Comparison and conditional selection |
| [Data Rearrangement](./data-rearrangement.md) | `pto.vintlv`, `pto.vslide`, `pto.vpack` | Lane permutation and packing |
| [SFU and DSA Ops](./sfu-and-dsa-ops.md) | `pto.vprelu`, `pto.vaxpy`, `pto.vtranspose` | Special function units and DSA ops |

## Shared Constraints

All vector families must state:

1. **Vector length** — The lane count `N` for vector registers in this family
2. **Predication model** — How inactive lanes are treated (zeroed, preserved, or undefined)
3. **Type support** — Which element types are legal (varies by A2/A3 vs A5)
4. **Target-profile narrowing** — Where profiles differ from each other and from the portable ISA contract

## Common Operand Model

All vector operations share a common operand model:

- **`%input` / `%src0` / `%src1`** — Source vector register operands (`!pto.vreg<NxT>`)
- **`%mask`** — Predicate operand for masking inactive lanes (`!pto.mask`)
- **`%result` / `%dst`** — Destination vector register operand
- **Scalar operands** — Immediate values, rounding modes, or scalar register operands

Vector length `N` is a power of 2. The predicate mask width must match `N`.

## Navigation

See the [Vector ISA reference](./README.md) for the full per-op reference under `vector/ops/`.

## See Also

- [Vector instruction surface](../instruction-surfaces/vector-instructions.md) — High-level surface description
- [Instruction families](./README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
