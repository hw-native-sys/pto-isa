<!-- Generated from `docs/isa/machine-model/execution-agents.md` -->

# Execution Agents And Target Profiles

PTO uses an architecture-visible three-level execution hierarchy: host, device, and core. This structure is not a direct hardware block diagram — it is an abstraction that makes explicit where work is prepared, dispatched, and executed, and where target profiles may differ in capability.

## Execution Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                        HOST                              │
│  CPU: prepares kernel arguments, submits graphs,         │
│  manages runtime orchestration and memory allocation       │
└───────────────────────┬─────────────────────────────────┘
                        │ RPC / AOE / custom transport
                        ▼
┌─────────────────────────────────────────────────────────┐
│                       DEVICE                            │
│  Scheduler: dispatches legal PTO work to cores in       │
│  dependence order, manages device-level memory (GM)     │
└───────────────────────┬─────────────────────────────────┘
                        │ Block dispatch
                        ▼
┌─────────────────────────────────────────────────────────┐
│          BLOCK / AI CORE (one per physical core)       │
│                                                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Scalar Unit                                       │ │
│  │  - Control flow, address calculation               │ │
│  │  - System query: GetBlockIdx, GetSubBlockIdx, ...│ │
│  ├────────────────────────────────────────────────────┤ │
│  │  Unified Buffer (UB) — 256 KB on-chip SRAM         │ │
│  │  - GM↔tile DMA staging area                      │ │
│  │  - Shared by all tile buffers and vector regs       │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  Tile Register File                                │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┐   │ │
│  │  │ Vec slots│ Mat slots│ Acc slots│Scalar slt│   │ │
│  │  │ 16×16×N │ 16×16×N │ 16×16×N │   1×1    │   │ │
│  │  └────┬─────┴────┬─────┴────┬─────┴──────────┘   │ │
│  ├───────┼──────────┼──────────┼───────────────────┤ │
│  │  ┌────▼────┐ ┌───▼───┐ ┌────▼────┐              │ │
│  │  │ Vector  │ │Matrix │ │  DMA    │              │ │
│  │  │Pipeline │ │  M /  │ │ Engine  │              │ │
│  │  │   (V)   │ │ CUBE  │ │MTE1/2/3 │              │ │
│  │  └────┬────┘ └───┬───┘ └─────────┘              │ │
│  └───────┼──────────┼────────────────────────────────┘ │
└──────────┼──────────┼────────────────────────────────────┘
           │          │
           ▼          ▼
        GM (off-chip device memory, shared by all blocks)
```

## Host

The **host** (typically a CPU or the host portion of a heterogeneous SoC):

- Prepares kernel arguments and memory descriptors
- Submits PTO programs to the device scheduler
- Manages graph-level or runtime orchestration (stream queuing, event tracking)
- Owns host-side memory used for argument staging

The host does NOT execute PTO instructions directly. It prepares and submits.

## Device

The **device** is the architecture-visible scheduling layer. A backend may implement it differently, but it is responsible for:

- Dispatching legal PTO work units to AI Core blocks
- Maintaining device-level memory (GM) and coherency with host memory
- Enforcing dependence order across blocks when required
- Managing device-side memory allocation

## Core (AI Core)

The **core** (one physical AI Core / NPU) is where PTO instructions execute. It contains:

| Component | Description | PTO Visibility |
|-----------|-------------|---------------|
| **Scalar Unit** | Control flow, address calculation, system queries | `GetBlockIdx()`, `GetBlockNum()`, `GetSubBlockIdx()` |
| **Unified Buffer (UB)** | 256 KB on-chip SRAM; shared staging area for GM↔tile DMA | `!pto.ptr<T, ub>` |
| **Tile Register File** | On-chip tile buffer storage, typed by `TileType` | `!pto.tile_buf<...>` |
| **Vector Pipeline (V)** | Executes `pto.v*` vector micro-instructions on vector registers | `!pto.vreg<NxT>` |
| **Matrix Multiply Unit (M/CUBE)** | Executes `pto.tmatmul` and `pto.tgemv` | Via `TileType::Mat`, `TileType::Left`, `TileType::Right`, `TileType::Acc` |
| **DMA Engine (MTE1/MTE2/MTE3)** | Moves data between GM and UB; coordinates with pipelines | `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `TLOAD`, `TSTORE` |

## Vector Register Architecture (VLane)

On A5 (Ascend 9xx-class), the vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Elements per VLane by data type:

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| i8 / u8 | 32 | 256 |
| i16 / u16 / f16 / bf16 | 16 | 128 |
| i32 / u32 / f32 | 8 | 64 |
| i64 / u64 | 4 | 32 |

The VLane concept is architecturally visible: group reduction operations (`vcgadd`, `vcgmax`, `vcgmin`) reduce within each VLane independently, producing one result per VLane.

## MTE Pipeline Detail

The DMA engine uses three sub-units that operate concurrently with compute pipelines:

| MTE | Direction | Role in Tile Surface | Role in Vector Surface |
|-----|-----------|---------------------|----------------------|
| `MTE1` | GM → UB | Optional: explicit prefetch | Pre-stage data before vector load |
| `MTE2` | GM → UB | Load staging: GM→UB→tile buffer (via TLOAD) | DMA copy: GM→UB (via `copy_gm_to_ubuf`) |
| `MTE3` | UB → GM | Store: tile→UB→GM (via TSTORE) | DMA copy: UB→GM (via `copy_ubuf_to_gm`) |

MTE1, MTE2, and MTE3 can operate in parallel with the Vector Pipeline and Matrix Multiply Unit when proper `set_flag`/`wait_flag` synchronization is used.

## System Query Operations

The following operations query the position of the current block within the grid:

| Operation | Return | Description |
|-----------|--------|-------------|
| `GetBlockIdx(dim)` | `i32` | 0-based index of current block along dimension `dim` |
| `GetSubBlockIdx(dim)` | `i32` | 0-based index of current sub-block within its parent block |
| `GetBlockNum(dim)` | `i32` | Total number of blocks along dimension `dim` |
| `GetSubBlockNum(dim)` | `i32` | Total number of sub-blocks within the parent block |

These are the only operations that depend on the grid topology. All other tile/vector/scalar operations are block-local.

## Target Profiles

PTO ISA is instantiated by **target profiles** that narrow the ISA to the capabilities of a specific backend. A profile does NOT introduce new ISA semantics — it only documents which subsets are available and may add implementation-defined variation points.

Three target profiles are currently defined:

### CPU Simulator

The **CPU simulator** (also called the reference simulator) executes PTO programs on the host CPU. Its goals are correctness and debuggability, not performance.

- All `pto.t*` tile surface operations are emulated in software
- All `pto.v*` vector surface operations are emulated with scalar loops
- Matmul operations use a reference GEMM implementation
- Fractal layouts are simulated with strided memory access
- UB is allocated from heap memory
- The UB size is configurable via build flags

### A2/A3 Profile

The **A2/A3 profile** targets Ascend 2/3-class NPUs. These targets support:

- Full `pto.t*` tile surface on hardware
- `pto.v*` vector surface emulated through a tile-vector bridge (`SimdTileToMemrefOp`, `SimdVecScopeOp`)
- Hardware matmul via the Matrix Multiply Unit (CUBE)
- Fractal layout support on hardware, but with software fallback paths
- UB: 256 KB per AI Core
- Vector width: N=64 (f32), N=128 (f16/bf16), N=256 (i8)
- Support for `textract` compact modes (ND2NZ, NZ2ND, ND, ND2NZ2)

### A5 Profile

The **A5 profile** targets Ascend 9xx-class NPUs (Ascend 910, 910B, 920, etc.). These targets support:

- Full `pto.t*` tile surface on hardware
- Full native `pto.v*` vector surface on the vector pipeline
- Hardware matmul with MX format support (int8 input → int32 accumulator)
- Full fractal layout support (NZ, ZN, FR, RN) on hardware
- UB: 256 KB per AI Core
- FP8 support: `float8_e4m3_t` (E4M3) and `float8_e5m3fn` (E5M2)
- Native vector unaligned store (`vstu` / `vstus`) and alignment state threading
- Block-scoped collective communication primitives (`TBROADCAST`, `TGET`, `TPUT`, etc.)
- 8 VLanes per vector register (group reduction atomic unit)

### Target Profile Comparison

| Feature | CPU Simulator | A2/A3 Profile | A5 Profile |
|---------|:-------------:|:-------------:|:----------:|
| Tile surface (`pto.t*`) | Full (emulated) | Full (hardware) | Full (hardware) |
| Vector surface (`pto.v*`) | Emulated (scalar loops) | Emulated (tile-vector bridge) | Full native |
| Matmul (`TMATMUL`) | Software fallback | Hardware CUBE | Hardware CUBE |
| MX format (int8→int32 acc) | Not applicable | Not applicable | Supported |
| Fractal layouts (NZ/ZN/FR/RN) | Simulated | Simulated | Full hardware |
| UB size | Configurable | 256 KB/core | 256 KB/core |
| Vector width (f32 / f16,bf16 / i8) | N=64 / N=128 / N=256 | N=64 / N=128 / N=256 | N=64 / N=128 / N=256 |
| FP8 types (e4m3 / e5m2) | Not supported | Not supported | Supported |
| Vector unaligned store (`vstu`) | Not supported | Not supported | Supported |
| Vector alignment state (`vstu`/`vstas`) | Not supported | Not supported | Supported |
| `hifloat8_t`, `float4_e*` types | Not supported | Not supported | Supported |
| Block-scoped collective comm | Not supported | Supported | Supported |
| Atomic store variants | Not supported | Supported | Supported |
| `vselr`, `vselrv2` (pair select) | Not supported | Not supported | Supported |
| TEXTRACT compact modes | Simulated | Supported | Supported |
| VLane group reduction | Not applicable | Not applicable | Supported |

## Constraints

- Architecture-visible dependence order MUST survive target scheduling
- Target profiles may narrow support, but MUST NOT redefine legal PTO semantics
- Machine-model documentation MUST state clearly which facts are portable and which are profile-specific
- Programs that depend on profile-specific features (e.g., MX format, FP8, unaligned vector store) are NOT portable across profiles

## Cases That Are Not Allowed

- Documenting A5-only features as general PTO guarantees
- Assuming the CPU simulator's emulation behavior matches hardware performance or cycle-accurate timing
- Treating a profile restriction as a contradiction of the ISA (profiles only narrow, never contradict)

## See Also

- [Ordering And Synchronization](./ordering-and-synchronization.md)
- [Vector Instruction Surface](../instruction-surfaces/vector-instructions.md)
- [Tile Instruction Surface](../instruction-surfaces/tile-instructions.md)
- [Portability And Target Profiles](../reference/portability-and-target-profiles.md)
- [PTO ISA Version 1.0](../introduction/pto-isa-version-1-0.md)
