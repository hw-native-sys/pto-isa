# Parallel Tile Operation ISA

## Overview

**PTO ISA** (Parallel Tile Operation Instruction Set Architecture) defines a machine-independent ISA for Huawei Ascend NPU software. PTO ISA provides a stable low-level programming contract above generation-specific hardware instruction sets, serving as the assembly-language layer of the PTO software stack.

PTO ISA is not the native binary ISA of any single Ascend implementation. It defines the architecture-visible meaning of legal PTO programs and the instruction vocabulary shared by frontends, code generators, verifiers, simulators, and target backends.

## Why Tile-First

Most Ascend kernels are authored in terms of **tiles** — bounded multi-dimensional array fragments with layout and valid-region metadata — not anonymous lanes or opaque buffers. A generic SIMD or SIMT model can describe the hardware eventually, but it pushes the important questions into backend-specific folklore:

- Shape and layout legality
- Which elements are meaningful (valid regions)
- When two tiles may alias
- Where synchronization must appear

PTO lifts these questions into the ISA so programs, verifiers, and backends share one testable, portable contract.

See [Goals Of PTO](./goals-of-pto.md) for product goals and [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) for how tiles work in programs.

## Two Compilation Flows

PTO programs can be compiled to hardware through two supported paths. Both paths share the same PTO instruction semantics; they differ in how the final binary is produced.

### Flow A: High-Level Compile (ptoas → C++ → bisheng → binary)

High-level frontends (TileLang, PyPTO, custom DSLs) emit PTO programs as `.pto` text files. The `ptoas` tool parses, validates, and lowers these to C++ code that calls the `pto-isa` C++ library. A backend compiler (bisheng) then compiles this C++ to the target binary.

```
High-level Frontend
(TileLang, PyPTO, C/C++, ...)
        │
        ▼
   .pto file
   (PTO program text)
        │
        ▼
   ptoas
   (PTO assembler & optimizer)
   ┌─────────────────────────────────────┐
   │ Parse, validate, optimize            │
   │ Lower PTO instructions to C++ calls  │
   │ Insert synchronization (auto-sync)   │
   └─────────────────────────────────────┘
        │
        ▼
   C++ kernel code
   (calls pto-isa C++ intrinsics)
        │
        ▼
   bisheng (or backend C++ compiler)
   ┌─────────────────────────────────────┐
   │ Compile to target binary             │
   │ Target: A2A3 (Ascend 910B / 910C)  │
   │ Target: A5 (Ascend 950 PR / DT)    │
   │ Target: CPU simulator                │
   └─────────────────────────────────────┘
        │
        ▼
   Binary
```

**Who uses this flow:** Compiler developers, library authors, high-level framework integrators. The `.pto` text format is portable and can be cached/distributed as bytecode.

### Flow B: Direct Assemble (ptoas → binary)

PTO programs can also be assembled directly to binary via `ptoas` with an appropriate backend target. This bypasses the C++ intermediate step.

```
High-level Frontend
(TileLang, PyPTO, C/C++, ...)
        │
        ▼
   .pto file
   (PTO program text)
        │
        ▼
   ptoas --target=a3|a5|cpu
   ┌─────────────────────────────────────┐
   │ Parse, validate, lower to binary     │
   │ Directly emit target instructions    │
   └─────────────────────────────────────┘
        │
        ▼
   Binary
```

**Who uses this flow:** Performance engineers who need direct control over the final instruction stream, or toolchains that embed `ptoas` as a pure assembler without a full C++ toolchain.

### Which Flow to Use

| Criterion | Flow A (ptoas → C++ → bisheng) | Flow B (ptoas → binary) |
|-----------|--------------------------------|--------------------------|
| Debugging | Full C++ debugging available | Binary only |
| Portability | C++ code is source portable | Binary is target-specific |
| Integration | Easy with existing C++ codebases | Requires custom binary packaging |
| Performance | Depends on C++ compiler | Direct, predictable instruction stream |
| Typical user | Library authors, compiler devs | Kernel engineers, performance tuners |

## A Minimal Example

The smallest end-to-end PTO program loads two tiles from global memory, adds them element-wise, and stores the result:

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c, const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);           // Load from global memory
    TLOAD(b, gb);           // Load from global memory
    TADD(c, a, b);          // Element-wise addition
    TSTORE(gc, c);          // Store to global memory
}
```

Even this fragment depends on valid regions, dtype and layout rules, and explicit data movement — ideas the manual unpacks in the programming model, machine model, and per-instruction reference.

## Key Terms

| Term | Definition |
|------|------------|
| **PTO** | The programming and instruction model built around tiles, explicit data movement, explicit synchronization, and machine-visible execution structure |
| **PTO ISA** | The instruction set architecture defined by this manual |
| **PTO-AS** | The textual assembly syntax for PTO ISA (e.g., `tadd %dst, %src0, %src1`) |
| **ptoas** | The assembler and optimizer tool that parses `.pto` files and lowers them to C++ or directly to binary |
| **PTOBC** | The bytecode representation used to package PTO programs for transport, caching, and distribution |
| **Tile** | A bounded multi-dimensional array fragment with shape, layout, and valid-region metadata that is architecturally visible |
| **Valid Region** | The subset of a tile's declared shape that contains meaningful data, expressed as `(Rv, Cv)` — valid rows and valid columns |
| **Global Memory (GM)** | Off-chip device memory (`__gm__` address space) shared by all blocks and accessible via `GlobalTensor` views |
| **Vector Tile Buffer** | The local tile buffer used for `TileType::Vec`. On current hardware this is implemented by the Unified Buffer (UB), but PTO treats it as one tile-buffer concept rather than two separate architectural objects. |
| **Tile Buffer** | On-chip storage for one tile, chosen by `TileType`: `Vec` uses the vector tile buffer (hardware UB), `Left` maps to L0A, `Right` maps to L0B, `Acc` maps to L0C, and scale tiles map to the corresponding left/right scale buffers. |
| **Location Intent** | The declared role of a tile operand: `Left` (L0A-backed left matmul operand), `Right` (L0B-backed right matmul operand), `Acc` (accumulator/output), `Vec` (vector tile buffer), `ScaleLeft`, and `ScaleRight` |
| **Block Layout (BLayout)** | The in-memory storage order of a tile: `RowMajor` (row-major, C-contiguous) or `ColMajor` (column-major, Fortran-contiguous) |
| **Stripe Layout (SLayout)** | The layout of sub-elements within a tile: `NoneBox` (uniform rectangular), `RowMajor` (fractal/strided), `ColMajor` (fractal/strided) |
| **Fractal Layout** | A strided layout encoding non-uniform strides for 2D tiles: `NZ` (row-major fractal), `ZN` (col-major fractal), `FR` (row-fractal), `RN` (row-N-fractal) |
| **TileType** | Classification of tile buffer role: `Vec` (vector pipe), `Mat` (matrix/CUBE pipe), `Acc` (accumulator), `Scalar` (scalar tile), `Left`/`Right` (matmul operands) |
| **MTE** | DMA engine sub-unit: `MTE1` (GM→UB), `MTE2` (UB→GM for loads), `MTE3` (tile→GM for stores) |
| **Target Profile** | A concrete instantiation of PTO ISA for a specific backend: `CPU` (reference simulator), `A2A3` (Ascend 910B / Ascend 910C), `A5` (Ascend 950 PR / Ascend 950 DT) |
| **Instruction Set** | One of the four ISA instruction sets: `pto.t*` (tile instructions), `pto.v*` (vector micro-instruction set), `pto.*` (scalar and control instructions), collective ops (communication instructions) |
| **pto.t*** | The tile compute instruction set (`pto.tadd`, `pto.tmul`, etc.) that operates on tile buffers |
| **pto.v*** | The low-level vector micro-instruction set (`pto.v*`) that operates on vector registers after an explicit GM→UB→vector data flow |
| **Element Type** | The dtype of a tile's elements: floating-point (`f16`, `bf16`, `f32`, `f8e4m3`, `f8e5m2`), integer (`i8`–`i64`, `u8`–`u64`), or specialized (`hifloat8_t`, `float4_e*`) |
| **Auto Mode** | Execution mode where the compiler/runtime automatically inserts `TASSIGN`, `TSYNC`, and data-movement operations |
| **Manual Mode** | Execution mode where the author explicitly binds tile resources with `TASSIGN` and manages synchronization explicitly |
| **pto.tget / TGET** | Inter-NPU remote read: reads data from a remote NPU's GM to local GM. Both spellings (`pto.tget` in IR, `TGET` in C++) refer to the same operation. |

## Position In The Software Stack

PTO ISA sits between source-level frontends and target-specific lowering. Frontends and code generators target PTO ISA; target backends lower PTO ISA to CPU simulation or to supported Ascend NPU targets.

```
Source Languages
(C/C++, Python, TileLang, PyPTO, code generators)
        │
        ▼
   PTO instructions (.pto text)
        │
        ├──► ptoas ──► C++ ──► bisheng ──► binary  (Flow A)
        │
        └──► ptoas ──────────────────► binary        (Flow B)

Targets: CPU simulation / A2A3 (Ascend 910B / 910C) / A5 (Ascend 950 PR / 950 DT) / future Ascend NPUs
```

This structure gives the software stack one versioned instruction language even when native hardware instruction sets and low-level programming rules change across generations.

## Hierarchical Abstractions

PTO ISA uses **hierarchical abstractions** rather than one flat opcode space. The ISA is organized into four instruction sets:

```
PTO ISA
├── Tile Instructions (pto.t*)              Primary tile-oriented compute instruction set
│   ├── Sync and Config                Resource binding, event setup, tile-local config
│   ├── Elementwise Tile-Tile           Lane-wise binary and unary operations
│   ├── Tile-Scalar and Immediate       Tile combined with scalar or immediate
│   ├── Reduce and Expand             Row/column reductions and expansions
│   ├── Memory and Data Movement       GM↔tile transfer, gather/scatter
│   ├── Matrix and Matrix-Vector       GEMV, matmul, and variants
│   ├── Layout and Rearrangement       Reshape, transpose, extract, insert
│   └── Irregular and Complex          Sort, quantize, print, and others
│
├── Vector Instructions (pto.v*)             Micro-instruction set for vector pipe
│   ├── Vector Load/Store              Predicate-based vector memory access
│   ├── Unary Vector Instructions              abs, neg, exp, sqrt, rec, relu, not, etc.
│   ├── Binary Vector Instructions             add, sub, mul, div, max, min, shl, shr, etc.
│   ├── Vector-Scalar Instructions                Vector combined with scalar operands
│   ├── Conversion Ops                Type conversion between numeric types
│   ├── Reduction Instructions                 Cross-lane reductions (cadd, cmax, etc.)
│   ├── Compare and Select            Comparison and conditional selection
│   ├── Data Rearrangement            Interleave, slide, shift, permute, pack
│   └── SFU and DSA Instructions              Special function units and DSA ops
│
├── Scalar and Control Instruction Set (pto.*)  State setup and control shell
│   ├── Pipeline Sync                 Event and barrier synchronization
│   ├── DMA Copy                     GM↔vector-tile-buffer transfer configuration
│   ├── Predicate Load/Store         Mask-based scalar memory access
│   ├── Predicate Generation         pset, pge, plt, pand, por, pxor, pnot, etc.
│   ├── Control and Configuration    legacy tile-prefixed mode/config ops such as tsethf32mode and tsetfmatrix
│   └── Shared Arithmetic/SCF         Scalar arithmetic and structured control flow
│
└── Communication Instructions (pto.*)       Collective and runtime operations
    ├── Collective Communication        TBROADCAST, TGET, TPUT, TREDUCE, etc.
    └── Supporting Operations          TALIAS, TAXPY, TCONCAT, TFREE, etc.
```

The **tile instructions** is the primary programming instruction set. The **vector instructions** exists for fine-grained vector-pipe control. The **scalar and control instructions** sets up the execution shell around tile payload regions. The **communication instructions** handles inter-rank communication and runtime support.

## Machine Model

PTO programs run on a hierarchical execution structure:

```
Grid (whole kernel invocation)
└── Block  (AI Core / NPU)
    ├── Host Interface
    ├── Scalar Unit          (control flow, address calculation)
    ├── Local Tile Buffers   (typed by TileType)
    │   ├── Vec buffer   = hardware UB
    │   ├── Left buffer  = L0A
    │   ├── Right buffer = L0B
    │   └── Acc buffer   = L0C
    ├── Tile Registers       ISA abstraction over those local tile buffers
    ├── DMA Engine
    │   ├── MTE1: GM ──► UB  (GM→UB, prefetch)
    │   ├── MTE2: GM ──► UB  (GM→UB, load staging)
    │   └── MTE3: UB ──► GM  (UB→GM, store)
    └── Vector Pipeline (V)  (unary/binary/reduce on vector regs)
```

### Execution Hierarchy

| Level | Description | PTO Visibility |
|-------|-------------|---------------|
| **Grid** | Entire kernel invocation across all participating AI Cores | `GetBlockNum()`, `GetBlockIdx()` |
| **Block** | Single AI Core with local UB, tile regs, and compute units | `GetSubBlockNum()`, `GetSubBlockIdx()` |
| **Tile Buffer** | Per-core on-chip storage for one tile (typed by `TileType`) | `!pto.tile_buf<...>` |
| **Vector Register** | Per-lane on-chip storage for vector compute (N lanes) | `!pto.vreg<NxT>` |
| **Vector Tile Buffer (hardware UB)** | On-chip buffer used by `TileType::Vec` and by the vector micro-instruction path | `!pto.ptr<T, ub>` |
| **Global Memory (GM)** | Off-chip device memory shared by all AI Cores | `__gm__ T*`, `!pto.partition_tensor_view<...>` |

### Target Profiles

PTO ISA is instantiated by concrete **target profiles** that narrow the ISA to the capabilities of a specific backend. Profiles do NOT introduce new ISA semantics; they only restrict which subsets are available.

| Feature | CPU Simulator | A2A3 Profile | A5 Profile |
|---------|:------------:|:-------------:|:----------:|
| Tile instructions (`pto.t*`) | Full | Full | Full |
| Vector instructions (`pto.v*`) | Emulated | Emulated | Full |
| Matmul / CUBE ops | Software fallback | Hardware | Hardware |
| MX format (int8→acc int32) | Not applicable | Not applicable | Supported |
| Fractal layout (NZ/ZN/FR/RN) | Simulated | Simulated | Full |
| Vector tile buffer size | Configurable | 256 KB/core | 256 KB/core |
| Vector width (f32 / f16,bf16 / i8) | N=64 / N=128 / N=256 | N=64 / N=128 / N=256 | N=64 / N=128 / N=256 |
| FP8 types (e4m3 / e5m2) | Not supported | Not supported | Supported |
| Vector unaligned store (`vstu`) | Not supported | Not supported | Supported |
| Block-scoped collective comm | Not supported | Supported | Supported |

## Instruction Syntax Overview

PTO instructions use a consistent textual syntax. Three forms are commonly shown:

### Assembly Form (PTO-AS)

The human-readable assembly spelling — the preferred form for documentation and portable pseudocode:

```asm
# Scalar operand suffix: immediate added to each tile element
tadds %dst, %src, 0x3F800000  : !pto.tile<f32, 16, 16>

# Saturating carry variant
taddc %dst, %src0, %src1       : !pto.tile<f32, 16, 16>

# Tile with explicit memory operand: load from GlobalTensor view
tload %tile, %gtensor[%r, %c]  : (!pto.tile<f32,16,16>, !pto.memref<f32,1x16x16x16>) -> !pto.tile<f32,16,16>
```

### SSA Form (AS Level 1)

MLIR-style SSA form with explicit types and a named result:

```mlir
// Tile compute: element-wise addition
%dst = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>

// Tile load: from GlobalTensor partition view
%tile = pto.tload %mem : !pto.partition_tensor_view<1x1x1x16x16xf32> -> !pto.tile_buf<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>

// Scalar tile comparison
%cmp = pto.tcmps %src, 0 : !pto.tile<f32, 16, 16>, i32 -> !pto.tile<predicate, 16, 16>
```

### DPS Form (AS Level 2)

Functional-style form with explicit `ins(...)` and `outs(...)` blocks — closest to the C++ intrinsic instruction set:

```mlir
// Tile compute (DPS)
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%dst : !pto.tile_buf<f32, 16, 16>)

// Tile load (DPS)
pto.tload ins(%mem : !pto.partition_tensor_view<1x1x1x16x16xf32>)
          outs(%tile : !pto.tile_buf<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>)

// Tile store (DPS)
pto.tstore ins(%tile : !pto.tile_buf<f32, 16, 16>)
          outs(%mem : !pto.partition_tensor_view<1x1x1x16x16xf32>)
```

See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full syntax specification.

## Tile Instructions And Vector Instructions

PTO distinguishes two complementary data-flow paths from GM to computed result. Both are architecturally visible; neither is a backend-only detail.

### Tile Instructions (pto.t*)

The tile instructions operates on tile buffers directly. The complete data path is:

```
GM ──(MTE2)──► UB ──(implicit)──► Tile Buffer ──(Tile Compute)──► Tile Buffer ──(MTE3)──► GM
                      │                                                              ▲
                      └──(vlds/vsts on vector instructions before/after tile instructions)─────────┘
```

- `TLOAD` copies data from GM into a tile buffer (via MTE2 → UB → tile)
- Tile compute (`TADD`, `TMATMUL`, etc.) operates directly on tile buffers
- `TSTORE` copies data from a tile buffer to GM through the corresponding local store path
- Valid regions, layout, and tile type are explicit at every step

### Vector Instructions (pto.v*)

The vector instructions operates on vector registers after an explicit UB staging step. The data path is:

```
GM ──(copy_gm_to_ubuf)──► UB ──(vlds)──► Vector Register ──(Vector Compute)──► Vector Register ──(vsts)──► UB ──(copy_ubuf_to_gm)──► GM
```

- `copy_gm_to_ubuf` / `copy_ubuf_to_gm`: DMA engine moves data between GM and UB
- `vlds` / `vsld` / `vgather2`: Vector load brings data from UB into vector registers
- Vector compute (`vadd`, `vmul`, etc.): operates on vector registers with predicate masking
- `vsts` / `vsst` / `vscatter`: Vector store writes data from vector registers back to UB
- An explicit `sync` or `set_flag` / `wait_flag` sequence establishes producer-consumer ordering between DMA and vector compute

### When To Use Which Instruction Set

| Criteria | Tile Instructions (`pto.t*`) | Vector Instructions (`pto.v*`) |
|----------|-------------------------|---------------------------|
| Typical use | Dense tensor algebra, matmul, elementwise | Fine-grained vector-pipe control, per-lane masking |
| Data movement | TLOAD/TSTORE (implicit tile↔UB) | copy_gm_to_ubuf / copy_ubuf_to_gm + vlds/vsts |
| Synchronization | TSYNC, set_flag/wait_flag | set_flag/wait_flag on vector pipe, mem_bar |
| Layout control | Via tile layout parameters | Via distribution mode (NORM, BRC, DS, etc.) |
| Predicate support | No per-lane masking | Yes — `%mask : !pto.mask` on every vector op |
| Target portability | All profiles | A5 hardware; emulated on CPU/A2/A3 |

## Audience: Who Reads This Manual

This manual serves two primary audiences with different needs:

### Compiler Backend Developers

You are building or maintaining a compiler that targets PTO ISA. You need to understand:

- The complete instruction inventory and its legality rules
- How PTO-AS maps to your backend's native instructions
- Target profile restrictions (which ops are available on A2/A3 vs A5)
- Layout constraints (which tile layouts are legal for which operations)
- Synchronization contracts (when to insert `set_flag`/`wait_flag` pairs)
- The two compilation flows and when to use each

### Kernel Writers

You are writing PTO programs directly, either in C++ (using `pto-isa` intrinsics) or in `.pto` text (using `ptoas`). You need to understand:

- Tile and valid region semantics (what data is meaningful)
- The tile instructions programming model (TLOAD, TSTORE, TADD, TMATMUL, etc.)
- GlobalTensor and memory layout (how data maps from GM to tiles)
- Auto vs. Manual mode (when the compiler helps vs. when you control everything)
- The synchronization model (TSYNC, set_flag/wait_flag, RecordEvent)
- Collective communication (`pto.tbroadcast`, `pto.tget`, `pto.tput`) for multi-NPU kernels

## Scope Of This Manual

This manual defines:

- The architecture-visible meaning of PTO instructions
- The programming model, machine model, and memory model of PTO ISA
- The distinction between tile, vector, scalar/control, and communication instructions
- The boundary between core ISA guarantees and target-profile restrictions

This manual is written for:

- Library and kernel authors
- Compiler and code generator developers
- Backend and runtime implementers
- Performance engineers
- Architecture and conformance test authors

## See Also

- [Document structure](./document-structure.md) — Full chapter map
- [Goals Of PTO](./goals-of-pto.md) — Design objectives
- [Scope And Boundaries](./design-goals-and-boundaries.md) — ISA scope and boundaries
- [PTO ISA Version 1.0](./pto-isa-version-1-0.md) — Version baseline decisions
- [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) — Tile semantics
- [Auto Vs Manual](../programming-model/auto-vs-manual.md) — Execution modes
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — How individual opcode pages are structured
