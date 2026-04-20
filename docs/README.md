# PTO ISA Documentation

<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO ISA" width="200" />
</p>

**PTO ISA** (Parallel Tile Operation Instruction Set Architecture) defines a stable, machine-independent instruction set for Huawei Ascend NPUs. It sits between high-level frontends (C/C++, Python, TileLang, PyPTO) and target-specific backends, providing one versioned instruction language across Ascend generations.

> **Documentation version:** PTO ISA 1.0
> **Applicable targets:** CPU Simulator · A2/A3 (Ascend 910B/910C) · A5 (Ascend 950)

---

## Quick Navigation

Use this page as a **reading guide**, not a table of contents. The manual is organized into five layers — start at the layer that matches your goal.

### Five-Layer Structure

| Layer | Contents | Audience |
|-------|----------|----------|
| **1. Foundations** | Introduction, programming model, machine model | Everyone — start here |
| **2. Syntax and Semantics** | Assembly model, operands, types, memory model | Kernel authors, compiler developers |
| **3. Instruction Surface** | Instruction-set overview and contracts | All users |
| **4. Reference Manual** | Tile, vector, scalar, and communication reference | Performance engineers, kernel authors |
| **5. Appendices** | Format guidelines, diagnostics, glossary, portability | Everyone |

### By Instruction Set

| Instruction Set | Prefix | Role | Count | Reference |
|----------------|--------|------|-------|-----------|
| **Tile** | `pto.t*` | Tile-oriented compute, data movement, layout transforms, synchronization | ~120 ops | [Tile reference](isa/tile/README.md) |
| **Vector** | `pto.v*` | Low-level vector micro-instructions, per-lane masking, pipeline control | ~99 ops | [Vector reference](isa/vector/README.md) |
| **Scalar & Control** | `pto.*` | Configuration, control flow, DMA setup, predicate operations | ~60 ops | [Scalar reference](isa/scalar/README.md) |
| **Communication** | `pto.*` | Multi-NPU collective operations and runtime support | ~24 ops | [Communication reference](isa/other/README.md) |

### By Task

| What you're doing | Start here |
|-------------------|------------|
| Understanding PTO's place in the stack | [What is PTO ISA?](isa/introduction/what-is-pto-visa.md) |
| Writing a matrix multiplication kernel | [Tile → Matrix ops](isa/tile/matrix-and-matrix-vector.md) |
| Optimizing elementwise operations | [Tile → Elementwise ops](isa/tile/elementwise-tile-tile.md) |
| Implementing a convolution kernel | [Tile → img2col](isa/tile/ops/layout-and-rearrangement/timg2col.md) |
| Setting up data movement (GM ↔ tile) | [Tile memory ops](isa/tile/memory-and-data-movement.md) |
| Hand-tuning vector kernels | [Vector instructions](isa/vector/README.md) |
| Using per-lane masking and predicates | [Vector → Predicate ops](isa/vector/predicate-and-materialization.md) |
| Implementing collective communication | [Communication instructions](isa/other/README.md) |
| Sorting, quantization, or histogram ops | [Irregular ops](isa/tile/irregular-and-complex.md) |
| Letting the compiler manage synchronization | [Auto vs Manual mode](isa/programming-model/auto-vs-manual.md) |
| Managing pipeline synchronization manually | [Synchronization model](isa/machine-model/ordering-and-synchronization.md) |
| Checking which types/features are on A5 vs A2/A3 | [Target profiles](isa/machine-model/execution-agents.md) |
| Reading a per-instruction page for the first time | [Format of instruction descriptions](isa/reference/format-of-instruction-descriptions.md) |

---

## Get Started

New to PTO? Follow this path:

1. **[What is PTO ISA?](isa/introduction/what-is-pto-visa.md)** — Core concepts, design rationale, and where PTO fits in the software stack
2. **[Programming Model: Tiles and Valid Regions](isa/programming-model/tiles-and-valid-regions.md)** — The tile abstraction that makes PTO tile-first
3. **[Machine Model: Execution Agents and Profiles](isa/machine-model/execution-agents.md)** — Execution hierarchy, pipelines, target profiles, and synchronization
4. **[Instruction Set Overview](isa/instruction-surfaces/README.md)** — High-level map of all four instruction sets and when to use each
5. **[Per-Instruction Reference](isa/README.md)** — Complete catalog organized by category

---

## What is PTO ISA?

PTO ISA is the stable instruction language for Ascend NPU software. It abstracts away hardware differences across A2/A3/A5 generations while preserving enough control for performance tuning.

```
Source Languages
(C/C++, Python, TileLang, PyPTO, code generators)
        │
        ▼
   PTO instructions (.pto text)
        │
        ├──► ptoas ──► C++ ──► bisheng ──► binary   (Flow A: via C++ intermediate)
        │
        └──► ptoas ────────────────────► binary        (Flow B: direct assemble)

Targets: CPU simulation / A2A3 (Ascend 910B / 910C) / A5 (Ascend 950 PR / 950 DT)
```

### Tile vs Vector: When To Use Which?

| Criteria | Tile Instructions (`pto.t*`) | Vector Instructions (`pto.v*`) |
|----------|-------------------------------|--------------------------------|
| **Typical use** | Dense tensor algebra, matmul, elementwise operations | Fine-grained vector-pipe control, per-lane masking |
| **Data movement** | `TLOAD`/`TSTORE` (implicit tile↔UB) | `copy_gm_to_ubuf` + `vlds`/`vsts` + `copy_ubuf_to_gm` |
| **Synchronization** | `TSYNC`, `set_flag`/`wait_flag` | `set_flag`/`wait_flag` on vector pipe, `mem_bar` |
| **Layout control** | Via tile layout parameters (`RowMajor`, `ColMajor`, fractal) | Via distribution mode (`NORM`, `BRC`, `DS`, etc.) |
| **Predication** | No per-lane masking (valid region is coarse-grained) | Full per-lane predicate mask on every operation |
| **Target portability** | All profiles (CPU, A2/A3, A5) | A5 hardware; emulated on CPU/A2/A3 |
| **Abstraction level** | High-level tile semantics, valid regions | Low-level vector registers, explicit UB staging |

> **Rule of thumb:** Start with tile instructions for tensor operations. Drop to vector instructions only when you need per-lane masking, custom data layouts, or micro-optimization that tile instructions cannot express.

---

## Core Concepts

Understanding these concepts is essential before reading per-instruction pages.

### Tile

A **tile** is a bounded multi-dimensional array fragment with architecturally visible shape, layout, and valid-region metadata. Tiles are the primary programming objects in PTO.

```cpp
Tile<Vec, float, 16, 16> a;  // 16×16 f32 tile in vector tile buffer (UB)
Tile<Left, f16, 64, 64> b;   // 64×64 f16 left operand (L0A)
Tile<Acc, i32, 128, 128> c; // 128×128 i32 accumulator (L0C)
```

[Learn more →](isa/programming-model/tiles-and-valid-regions.md)

### Valid Region

The **valid region** `(Rv, Cv)` is the subset of a tile's declared shape that contains meaningful data. Operations iterate over the destination tile's valid region; source tiles with smaller valid regions yield implementation-defined values outside their valid region.

### TileType (Location Intent)

The **TileType** determines which hardware buffer backs a tile:

| TileType | Hardware Buffer | Capacity | Typical Use |
|----------|----------------|----------|-------------|
| `Vec` | Unified Buffer (UB) | 256 KB | General elementwise operations |
| `Left` | L0A | 64 KB | Matmul A operand |
| `Right` | L0B | 64 KB | Matmul B operand |
| `Acc` | L0C | 256 KB | Matmul accumulator/output |
| `Mat` | L1 | 512 KB | 2D matrix operands |

### GlobalTensor

A **GlobalTensor** is a view of off-chip device memory (`__gm__` address space). All data movement between GM and tile buffers happens through explicit `TLOAD`/`TSTORE` or DMA operations.

[Learn more →](isa/programming-model/globaltensor-and-data-movement.md)

### Auto vs Manual Mode

| Mode | Resource Binding | Synchronization | Data Movement | Who manages it? |
|------|-----------------|-----------------|---------------|-----------------|
| **Auto** | Compiler inserts `TASSIGN` | Compiler inserts `TSYNC` | Compiler inserts `TLOAD`/`TSTORE` | Compiler |
| **Manual** | Author writes `TASSIGN` explicitly | Author writes `TSYNC` explicitly | Author writes `TLOAD`/`TSTORE` explicitly | You |

[Auto vs Manual →](isa/programming-model/auto-vs-manual.md)

### Target Profiles

PTO ISA is instantiated by concrete **target profiles** that narrow the accepted subset for a specific backend.

| Feature | CPU Simulator | A2/A3 Profile | A5 Profile |
|---------|:--------------:|:--------------:|:----------:|
| Tile instructions (`pto.t*`) | Full | Full | Full |
| Vector instructions (`pto.v*`) | Emulated | Emulated | Full hardware |
| Matmul / CUBE ops | Software fallback | Hardware | Hardware |
| Vector width (f32 / f16,bf16 / i8) | Configurable | 64 / 128 / 256 | 64 / 128 / 256 |
| FP8 types (`f8e4m3`, `f8e5m2`) | — | — | Supported |
| Fractal layouts (NZ/ZN/FR/RN) | Simulated | Simulated | Full |
| Block-scoped collective comm | — | Supported | Supported |

---

## Instruction-Set Navigation Map

PTO groups its instructions into four named instruction sets. Each set has a **contract page** (shared rules) and **per-op pages** (individual instructions).

### Tile Instruction Set — `pto.t*`

```
Tile Instruction Set
├── Sync and Config             → tassign, tsync, tsettf32mode, tsetfmatrix, tset_img2col_*, tsubview, tget_scale_addr
├── Elementwise Tile-Tile       → tadd, tsub, tmul, tdiv, tmin, tmax, tcmp, tcvt, tsel, tlog, trecip, texp, tsqrt, trsqrt, trem, tfmod, tabs, tand, tor, txor, tnot, tneg, tprelu, taddc, tsubc, tshl, tshr
├── Tile-Scalar and Immediate   → tadds, tsubs, tmuls, tdivs, tminmaxs, tcmps, tsels, texpands, tfmods, trems, tands, tors, txors, tshls, tshrs, tlrelu, taddsc, tsubsc
├── Reduce and Expand           → trowsum, tcolsum, trowprod, tcolprod, tcolmax, tcolmin, trowmax, trowmin, tcolargmax, tcolargmin, trowargmax, trowargmin
│                               → trowexpand, trowexpandadd, trowexpanddiv, trowexpandmul, trowexpandsub, trowexpandmax, trowexpandmin, trowexpandexpdif
│                               → tcolexpand, tcolexpandadd, tcolexpanddiv, tcolexpandmul, tcolexpandsub, tcolexpandmax, tcolexpandmin, tcolexpandexpdif
├── Memory and Data Movement   → tload, tprefetch, tstore, tstore_fp, mgather, mscatter
├── Matrix and Matrix-Vector    → tgemv, tgemv_mx, tgemv_acc, tgemv_bias, tmatmul, tmatmul_mx, tmatmul_acc, tmatmul_bias
├── Layout and Rearrangement   → tmov, tmov_fp, ttrans, textract, textract_fp, tinsert, tinsert_fp, timg2col, tfillpad, tfillpad_inplace, tfillpad_expand, treshape
└── Irregular and Complex      → tprint, tmrgsort, tsort32, tgather, tgatherb, tscatter, tci, ttri, tpartadd, tpartmul, tpartmax, tpartmin, tquant
```

[Tile instruction set contract →](isa/instruction-families/tile-families.md)

### Vector Instruction Set — `pto.v*`

```
Vector Instruction Set
├── Vector Load Store           → vlds, vldas, vldus, vldx2, vsld, vsldb, vgather2, vgatherb, vgather2_bc
│                               → vsts, vstx2, vsst, vsstb, vscatter, vsta, vstas, vstar, vstu, vstus, vstur
├── Predicate and Materialization → vbr, vdup
├── Unary Vector Ops            → vabs, vneg, vexp, vln, vsqrt, vrsqrt, vrec, vrelu, vnot, vbcnt, vcls, vmov
├── Binary Vector Ops            → vadd, vsub, vmul, vdiv, vmax, vmin, vand, vor, vxor, vshl, vshr, vaddc, vsubc
├── Vector-Scalar Ops           → vadds, vsubs, vmuls, vmaxs, vmins, vands, vors, vxors, vshls, vshrs, vlrelu, vaddcs, vsubcs
├── Conversion Ops               → vci, vcvt, vtrc
├── Reduction Ops               → vcadd, vcmax, vcmin, vcgadd, vcgmax, vcgmin, vcpadd
├── Compare and Select          → vcmp, vcmps, vsel, vselr, vselrv2
├── Data Rearrangement          → vintlv, vdintlv, vslide, vshift, vsqz, vusqz, vperm, vpack, vsunpack, vzunpack, vintlvv2, vdintlvv2
└── SFU and DSA                 → vprelu, vexpdiff, vaddrelu, vsubrelu, vaxpy, vaddreluconv, vmulconv, vmull, vmula, vtranspose, vsort32, vbitsort, vmrgsort
```

[Vector instruction set contract →](isa/instruction-families/vector-families.md)

### Scalar and Control Instruction Set — `pto.*`

```
Scalar and Control Instruction Set
├── Control and Configuration   → nop, barrier, yield; tsetf32mode, tsethf32mode, tsetfmatrix
├── Pipeline Sync              → set_flag, wait_flag, wait_flag_dev, pipe_barrier, mem_bar, get_buf, rls_buf, set_cross_core, set_intra_block, wait_intra_core
├── DMA Copy                   → set_loop_size_outtoub, set_loop1/2_stride_outtoub
│                               → set_loop_size_ubtoout, set_loop1/2_stride_ubtoout
│                               → copy_gm_to_ubuf, copy_ubuf_to_gm, copy_ubuf_to_ubuf
├── Predicate Load Store        → pld, plds, pldi, psts, pst, psti, pstu
├── Predicate Generation        → pset_b8/b16/b32, pge_b8/b16/b32, plt_b8/b16/b32
│                               → pand, por, pxor, pnot, psel, ppack, punpack
│                               → pdintlv_b8, pintlv_b16
├── Shared Arithmetic           → Scalar arithmetic ops shared across instruction sets
├── Shared SCF                 → Scalar structured control flow
└── Micro-Instructions          → BlockDim queries, pointer ops, vector scope, alignment state
    [Micro-instruction summary →](isa/vector/micro-instruction-summary.md)
```

[Scalar instruction set contract →](isa/instruction-families/scalar-and-control-families.md)

### Communication Instruction Set — `pto.*`

```
Communication Instruction Set
├── Collective Ops              → tbroadcast, tget, tget_async, tput, tput_async
│                               → tscatter, tgather, treduce, ttest, twait, tnotify
└── Non-ISA Supporting Ops      → talias, taxpy, tconcat, tdequant, tfree, thistogram
                                → tpack, tpop, tpush, trandom
```

[Communication instruction set contract →](isa/instruction-families/other-families.md)

---

## Compilation Flows

### Flow A: High-Level Compile (ptoas → C++ → bisheng → binary)

High-level frontends emit `.pto` text files. `ptoas` parses, validates, and lowers these to C++ code calling the `pto-isa` C++ library. A backend compiler (bisheng) then produces the final binary.

**Who uses this:** Compiler developers, library authors, high-level framework integrators. The `.pto` format is portable and cacheable.

### Flow B: Direct Assemble (ptoas → binary)

`ptoas` assembles directly to target binary, bypassing the C++ intermediate step.

**Who uses this:** Performance engineers who need direct control over the final instruction stream, or toolchains that embed `ptoas` as a pure assembler.

[Learn more about the compilation flows →](isa/introduction/what-is-pto-visa.md#two-compilation-flows)

---

## Key References

| Reference | What it covers |
|-----------|---------------|
| **[PTO-AS Specification](assembly/PTO-AS.md)** | Assembly syntax and grammar for `.pto` text files |
| **[Tile Programming Model](coding/Tile.md)** | Tile shape, tile mask, and data organization |
| **[Events and Synchronization](coding/Event.md)** | set/wait flag and pipeline synchronization |
| **[Performance Optimization](coding/opt.md)** | Bottleneck analysis and tuning guidance |
| **[Auto Mode Overview](auto_mode/Auto_Mode_Overview.md)** | Compiler-driven resource management and synchronization |
| **[Micro-Instruction Summary](isa/vector/micro-instruction-summary.md)** | Scalar micro-instructions: BlockDim, pointer ops, vector scope |
| **[Portability and Target Profiles](isa/reference/portability-and-target-profiles.md)** | Which features exist on which target |
| **[Glossary](isa/reference/glossary.md)** | Terminology reference |
| **[Source of Truth](isa/reference/source-of-truth.md)** | Which files define authoritative semantics |
| **[Build the Docs](mkdocs/README.md)** | Generate this site locally |

---

## Contributing

This documentation is generated from the canonical PTO ISA specification at [github.com/PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa). Report issues and submit changes there.

---

*PTO ISA is part of the Ascend software stack. Copyright © Huawei Technologies Co., Ltd.*
