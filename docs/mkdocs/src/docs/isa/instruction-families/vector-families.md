<!-- Generated from `docs/isa/instruction-families/vector-families.md` -->

# Vector Families

Vector-family documentation follows the PTOAS VPTO family shape. Vector families cover the complete set of `pto.v*` micro-instructions that operate on vector registers.

## Overview

| Family | Description | Examples |
|--------|-------------|----------|
| [Vector Load/Store](./vector/vector-load-store.md) | UB↔vector register transfer with distribution modes | `vlds`, `vldas`, `vgather2`, `vsld`, `vsst`, `vscatter` |
| [Predicate and Materialization](./vector/predicate-and-materialization.md) | Vector broadcast and duplication | `vbr`, `vdup` |
| [Unary Vector Ops](./vector/unary-vector-ops.md) | Single-operand lane-wise operations | `vabs`, `vneg`, `vexp`, `vsqrt`, `vrec`, `vrelu`, `vnot` |
| [Binary Vector Ops](./vector/binary-vector-ops.md) | Two-operand lane-wise operations | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin` |
| [Vec-Scalar Ops](./vector/vec-scalar-ops.md) | Vector combined with scalar operand | `vadds`, `vmuls`, `vshls`, `vlrelu` |
| [Conversion Ops](./vector/conversion-ops.md) | Type conversion between numeric types | `vci`, `vcvt`, `vtrc` |
| [Reduction Ops](./vector/reduction-ops.md) | Cross-lane reductions | `vcadd`, `vcmax`, `vcmin`, `vcgadd`, `vcgmax` |
| [Compare and Select](./vector/compare-select.md) | Comparison and conditional lane selection | `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2` |
| [Data Rearrangement](./vector/data-rearrangement.md) | Lane permutation, interleaving, packing | `vintlv`, `vdintlv`, `vslide`, `vshift`, `vpack`, `vzunpack` |
| [SFU and DSA Ops](./vector/sfu-and-dsa-ops.md) | Special function units and DSA-style operations | `vprelu`, `vexpdiff`, `vaxpy`, `vtranspose`, `vsort32` |

## Shared Constraints

All vector families must state:

1. **Vector width** — `N` is determined by the element type (f32: N=64, f16/bf16: N=128, i8/u8: N=256).
2. **Predicate behavior** — How masked-off lanes behave in compute and load/store ops.
3. **Pointer space** — All vector load/store addresses are in UB space (`!pto.ptr<T, ub>`).
4. **Pipeline handoff** — How data moves between DMA (GM↔UB) and vector register ops.
5. **Target-profile narrowing** — A5-only ops (FP8 types, unaligned store, pair select).

## Vector Width and Type Support by Profile

| Element Type | Vector Width N | CPU Sim | A2/A3 | A5 |
|-------------|:-------------:|:-------:|:------:|:--:|
| f32 | 64 | Yes | Yes | Yes |
| f16 | 128 | Yes | Yes | Yes |
| bf16 | 128 | Yes | Yes | Yes |
| i16 / u16 | 128 | Yes | Yes | Yes |
| i8 / u8 | 256 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | 256 | No | No | Yes |
| hifloat8_t / float4_e* | 256 | No | No | Yes |

## Mask Behavior

Vector ops gated by a predicate mask follow these rules:

- Lanes where mask bit = **1**: operation executes normally.
- Lanes where mask bit = **0**: result is defined but operation-specific; programs MUST NOT rely on any specific value.

## Pipeline Handoff

Vector surface data movement requires explicit synchronization between DMA and vector register ops:

```
copy_gm_to_ubuf ──(set_flag)──► wait_flag ──► vlds ──► vadd ──► vsts ──(set_flag)──► wait_flag ──► copy_ubuf_to_gm
```

## A5-Only Operations

The following ops require A5 profile:

- `vstu`, `vstus`, `vstur` — Vector unaligned store with alignment state
- `vselr`, `vselrv2` — Pair select operations
- `thistogram`, `tpack`, `trandom` — Tile ops but listed under SFU category
- All ops using FP8 element types (e4m3, e5m2)

## Navigation

See the [Vector ISA reference](./vector/README.md) for the full per-op reference under `vector/ops/`.

## See Also

- [Vector instruction surface](./instruction-surfaces/vector-instructions.md) — High-level surface description
- [Instruction families](./README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
