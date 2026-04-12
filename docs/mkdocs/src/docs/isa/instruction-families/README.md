<!-- Generated from `docs/isa/instruction-families/README.md` -->

# Instruction Families

Family pages describe shared contracts that apply across related PTO operations. They sit between the model chapters and the per-op reference pages. For how individual opcode pages are structured, see [format of instruction descriptions](../reference/format-of-instruction-descriptions.md).

## Overview

PTO ISA organizes its instruction set into four families, each corresponding to one instruction surface:

| Family | Prefix | Surface | Description |
|--------|--------|---------|-------------|
| [Tile Families](./tile-families.md) | `pto.t*` | Tile | Primary tile-oriented compute, data movement, layout operations |
| [Vector Families](./vector-families.md) | `pto.v*` | Vector | Micro-instructions for vector pipeline execution |
| [Scalar And Control Families](./scalar-and-control-families.md) | `pto.*` | Scalar/Control | Configuration, synchronization, DMA, predicate operations |
| [Other Families](./other-families.md) | `pto.*` | Communication | Collective communication and runtime support |

## What A Family Contract Must State

Each family page provides the following:

1. **Mechanism** — What the family is for, explained in one short section.
2. **Shared operand model** — Common input/output roles and how they interact.
3. **Common side effects** — Synchronization, ordering, or configuration effects shared by all ops in the family.
4. **Shared constraints** — Legality rules that apply across the family.
5. **Cases that are not allowed** — Conditions that are illegal for all ops in the family.
6. **Target-profile narrowing** — Where A2/A3 and A5 differ in what the family accepts.
7. **Operation list** — Pointers to each per-op page under `ops/`.

Family pages do not repeat per-op details; they set the contract for the group.

## Navigation Map

```
Instruction Families
├── Tile Families
│   ├── Sync and Config            → pto.tassign, pto.tsync, pto.tsethf32mode, pto.tsetfmatrix, etc.
│   ├── Elementwise Tile-Tile      → pto.tadd, pto.tmul, pto.tcmp, pto.tcvt, pto.tsel, etc.
│   ├── Tile-Scalar and Immediate  → pto.tadds, pto.tmuls, pto.tmins, pto.texpands, etc.
│   ├── Reduce and Expand          → pto.trowsum, pto.tcolmax, pto.trowexpand, pto.tcolexpand, etc.
│   ├── Memory and Data Movement   → pto.tload, pto.tstore, pto.tstore_fp, pto.mgather, pto.mscatter
│   ├── Matrix and Matrix-Vector    → pto.tgemv, pto.tgemv_mx, pto.tmatmul, pto.tmatmul_acc, pto.tmatmul_bias, etc.
│   ├── Layout and Rearrangement   → pto.tmov, pto.ttrans, pto.textract, pto.tinsert, pto.timg2col, etc.
│   └── Irregular and Complex      → pto.tmrgsort, pto.tsort32, pto.tquant, pto.tprint, pto.tci, pto.ttri, etc.
│
├── Vector Families
│   ├── Vector Load Store          → pto.vlds, pto.vldas, pto.vgather2, pto.vsld, pto.vsst, pto.vscatter, etc.
│   ├── Predicate and Materialization → pto.vbr, pto.vdup
│   ├── Unary Vector Ops          → pto.vabs, pto.vneg, pto.vexp, pto.vsqrt, pto.vrec, pto.vrelu, pto.vnot, etc.
│   ├── Binary Vector Ops          → pto.vadd, pto.vsub, pto.vmul, pto.vmax, pto.vmin, pto.vand, pto.vor, etc.
│   ├── Vec-Scalar Ops            → pto.vadds, pto.vmuls, pto.vshls, pto.vlrelu, etc.
│   ├── Conversion Ops             → pto.vci, pto.vcvt, pto.vtrc
│   ├── Reduction Ops              → pto.vcadd, pto.vcmax, pto.vcmin, pto.vcgadd, pto.vcgmax, pto.vcpadd, etc.
│   ├── Compare and Select         → pto.vcmp, pto.vcmps, pto.vsel, pto.vselr, pto.vselrv2
│   ├── Data Rearrangement         → pto.vintlv, pto.vdintlv, pto.vslide, pto.vshift, pto.vpack, pto.vzunpack, etc.
│   └── SFU and DSA Ops           → pto.vprelu, pto.vexpdiff, pto.vaxpy, pto.vtranspose, pto.vsort32, etc.
│
├── Scalar And Control Families
│   ├── Control and Configuration  → pto.nop, pto.barrier, pto.yield, etc.
│   ├── Pipeline Sync             → pto.set_flag, pto.wait_flag, pto.pipe_barrier, pto.mem_bar, etc.
│   ├── DMA Copy                  → pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm, pto.copy_ubuf_to_ubuf, etc.
│   ├── Predicate Load Store       → pto.pld, pto.plds, pto.pldi, pto.pst, pto.psts, pto.psti, pto.pstu
│   ├── Predicate Generation       → pto.pset_b8, pto.pge_b8, pto.plt_b8, pto.pand, pto.por, pto.pxor, pto.pnot, etc.
│   ├── Shared Arithmetic          → Scalar arithmetic ops shared across surfaces
│   └── Shared SCF               → Scalar structured control flow
│
└── Other Families
    ├── Communication and Runtime  → pto.tbroadcast, pto.tget, pto.tput, pto.treduce, pto.tscatter, pto.tgather, pto.tnotify, pto.ttest, pto.twait, etc.
    └── Non-ISA Supporting Ops    → pto.talias, pto.tconcat, pto.tfree, pto.tquant, pto.tdequant, pto.tpack, pto.thistogram, pto.tpop, pto.tpush, pto.trandom, etc.
```

## Normative Language

Family pages use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction surfaces](./instruction-surfaces/README.md) — High-level surface descriptions
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
- [Diagnostics and illegal cases](../reference/diagnostics-and-illegal-cases.md) — What makes a PTO program illegal
