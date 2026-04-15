# Vector Instruction Set

Vector-instruction set documentation explains how `pto.v*` groups behave. Each instruction set describes the shared mechanism, operand model, constraints, and target-profile narrowing before the standalone per-op pages under `vector/ops/`.

## Overview

| Instruction Set | Prefix | Description |
|--------|--------|-------------|
| [Vector Load/Store](./vector-load-store.md) | `pto.vlds`, `pto.vsts`, `pto.vgather2` | UB↔vector register transfer, gather/scatter |
| [Predicate and Materialization](./predicate-and-materialization.md) | `pto.vbr`, `pto.vdup` | Vector broadcast and duplication |
| [Unary Vector Instructions](./unary-vector-ops.md) | `pto.vabs`, `pto.vneg`, `pto.vexp`, `pto.vsqrt` | Single-input elementwise operations |
| [Binary Vector Instructions](./binary-vector-ops.md) | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vcmp` | Two-input elementwise operations |
| [Vector-Scalar Instructions](./vec-scalar-ops.md) | `pto.vadds`, `pto.vmuls`, `pto.vshls` | Vector combined with scalar operand |
| [Conversion Ops](./conversion-ops.md) | `pto.vci`, `pto.vcvt`, `pto.vtrc` | Type conversion between numeric types |
| [Reduction Instructions](./reduction-ops.md) | `pto.vcadd`, `pto.vcmax`, `pto.vcgadd` | Cross-lane reductions |
| [Compare and Select](./compare-select.md) | `pto.vcmp`, `pto.vsel`, `pto.vselr` | Comparison and conditional selection |
| [Data Rearrangement](./data-rearrangement.md) | `pto.vintlv`, `pto.vslide`, `pto.vpack` | Lane permutation and packing |
| [SFU and DSA Instructions](./sfu-and-dsa-ops.md) | `pto.vprelu`, `pto.vaxpy`, `pto.vtranspose` | Special function units and DSA ops |

## Shared Constraints

Every vector instruction set must state:

1. **Vector length** — The lane count `N` for vector registers in this instruction set
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

## Timing Coverage Policy

The standalone micro-instruction pages under `vector/ops/` now all carry an explicit timing section.
That section is sourced in this order:

1. Public timing statements in `~/visa.txt`
2. VPTO semantic context from `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch

Where those sources do not publish a numeric latency or steady-state throughput, the per-op page states that the timing is not publicly published and must be measured on the concrete backend.


## See Also

- [Vector instruction set](../instruction-surfaces/vector-instructions.md) — High-level instruction set description
- [Instruction sets](./README.md) — All instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
