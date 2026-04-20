# Instruction Set Contracts

Instruction set pages describe shared contracts that apply across related PTO operations. They sit between the model chapters and the per-op reference pages.

## Four Named Instruction Sets

| Instruction Set | Prefix | Families | Description |
|----------------|--------|----------|-------------|
| [Tile Instruction Set](./tile-families.md) | `pto.t*` | 8 | Tile-oriented compute, data movement, layout operations |
| [Vector Instruction Set](./vector-families.md) | `pto.v*` | 9 | Micro-instructions for vector pipeline execution |
| [Scalar and Control Instruction Set](./scalar-and-control-families.md) | `pto.*` | 6 | Configuration, synchronization, DMA, predicate operations |
| [Other Instruction Set](./other-families.md) | `pto.*` | 2 | Collective communication and runtime support |

## Navigation Map

```
Tile Instruction Set
├── Sync and Config             → tassign, tsync, tsetf32mode, tsetfmatrix, tset_img2col_*, tsubview
├── Elementwise Tile-Tile      → tadd, tsub, tmul, tdiv, tmin, tmax, tcmp, tcvt, tsel, etc.
├── Tile-Scalar and Immediate  → tadds, tsubs, tmuls, tdivs, tcmps, tsels, texpands, etc.
├── Reduce and Expand          → trowsum, tcolmax, trowexpand, tcolexpand, etc.
├── Memory and Data Movement   → tload, tprefetch, tstore, tstore_fp, mgather, mscatter
├── Matrix and Matrix-Vector    → tgemv, tgemv_mx, tmatmul, tmatmul_acc, tmatmul_bias, etc.
├── Layout and Rearrangement   → tmov, ttrans, textract, tinsert, timg2col, tfillpad, etc.
└── Irregular and Complex     → tprint, tsort32, tgather, tscatter, tquant, etc.

Vector Instruction Set
├── Vector Load Store           → vlds, vldas, vgather2, vsld, vsst, vscatter, etc.
├── Predicate and Materialization → vbr, vdup
├── Unary Vector Ops            → vabs, vneg, vexp, vln, vsqrt, vrec, vrelu, etc.
├── Binary Vector Ops            → vadd, vsub, vmul, vdiv, vmax, vmin, vand, vor, etc.
├── Vector-Scalar Ops           → vadds, vsubs, vmuls, vshls, vlrelu, etc.
├── Conversion Ops               → vci, vcvt, vtrc
├── Reduction Ops               → vcadd, vcmax, vcmin, vcgadd, vcgmax, etc.
├── Compare and Select          → vcmp, vcmps, vsel, vselr, vselrv2
├── Data Rearrangement          → vintlv, vslide, vshift, vpack, vzunpack, etc.
└── SFU and DSA                → vprelu, vexpdiff, vaxpy, vtranspose, vsort32, etc.

Scalar and Control Instruction Set
├── Control and Configuration   → nop, barrier, yield; tsetf32mode, tsetfmatrix
├── Pipeline Sync              → set_flag, wait_flag, pipe_barrier, mem_bar, get_buf
├── DMA Copy                  → copy_gm_to_ubuf, copy_ubuf_to_gm, copy_ubuf_to_ubuf
├── Predicate Load Store       → pld, plds, psts, pst, pstu
├── Predicate Generation        → pset_b8/b16/b32, pge_b8/b16/b32, plt_b8/b16/b32
│                               → pand, por, pxor, pnot, psel, ppack, punpack
├── Shared Arithmetic          → Scalar arithmetic shared across instruction sets
└── Shared SCF                → Scalar structured control flow

Communication Instruction Set
├── Collective Ops              → tbroadcast, tget, tput, tgather, tscatter, treduce, tnotify, ttest, twait
└── Non-ISA Supporting Ops     → talias, taxpy, tconcat, tdequant, tfree, thistogram, tpack, tpop, tpush, trandom
```

## What an Instruction Set Contract Must State

Each instruction set page provides:

1. **Mechanism** — What the instruction set is for, explained in one short section.
2. **Shared operand model** — Common input/output roles and how they interact.
3. **Common side effects** — Synchronization, ordering, or configuration effects shared by all instructions in the set.
4. **Shared constraints** — Legality rules that apply across the set.
5. **Cases that are not allowed** — Conditions that are illegal for all instructions in the set.
6. **Target-profile narrowing** — Where A2/A3 and A5 differ in what the set accepts.
7. **Operation list** — Pointers to each per-op page under `ops/`.

## Normative Language

Instruction set pages use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction overview](../instruction-surfaces/README.md) — High-level map of all four instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
- [Diagnostics and illegal cases](../reference/diagnostics-and-illegal-cases.md) — What makes a PTO program illegal
