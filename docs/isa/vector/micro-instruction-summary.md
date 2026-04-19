# Vector Micro-Instruction Reference Summary

This document summarizes the micro-architectural details for vector instructions (`pto.v*`) from the PTO micro-instruction SPEC. It supplements the per-operation reference pages with hardware-specific timing, pipeline behavior, and implementation notes for the Ascend 950 (A5) architecture.

> **Scope:** This information applies to the A5 profile (Ascend 950). The CPU simulator and A2/A3 profiles emulate vector instructions using scalar loops; performance numbers below are specific to A5 hardware.

## Architecture Overview

### Vector Core and Pipelines

The Ascend 950 vector core operates asynchronously with three primary pipelines:

| Pipeline | Role | Instructions |
|----------|------|-------------|
| **PIPE_MTE2** | DMA inbound: GM → UB | `copy_gm_to_ubuf` |
| **PIPE_V** | Vector compute: UB ↔ vreg + operations | All `pto.v*` instructions |
| **PIPE_MTE3** | DMA outbound: UB → GM | `copy_ubuf_to_gm` |

Synchronization between these pipelines is explicit via `pto.set_flag` / `pto.wait_flag` or `pto.get_buf` / `pto.rls_buf`.

## Element Types (Extended, A5)

The `vreg<NxT>` type has exactly 256 bytes total (2048 bits). `N × bitwidth(T) = 2048`:

|| Type | Bits | Description |
|------|------|-------------|
| `i8` / `si8` / `ui8` | 8 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | Signless/signed/unsigned 16-bit integer |
| `i32` / `si32` / `ui32` | 32 | Signless/signed/unsigned 32-bit integer |
| `i64` / `si64` / `ui64` | 64 | Signless/signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |
| `f8e4m3` | 8 | Float8 E4M3 (A5+) |
| `f8e5m2` | 8 | Float8 E5M2 (A5+) |

### Predicate Masks

The mask type `!pto.mask<G>` models an A5 predicate register (256-bit) under a typed granularity view.

`G` MUST be one of `b32`, `b16`, `b8`:

|| Mask Type | Bytes/Element | Typical Element | Logical Lanes |
|-----------|--------------|-----------------|---------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit family | 256 |

The physical predicate register is always 256 bits. The `G` parameter records how VPTO interprets the register for matching mask-producing and mask-consuming ops, and for verifier legality rules.

**Predication Mode — ZEROING:** Inactive lanes produce zero, not preserved destination values:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

This is intentionally different from a lane-vector model such as `mask<64xi1>`.

## Architecture Overview

Each 256-byte vector register (`vreg`) is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

**Elements per VLane by data type:**

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| `i8`/`u8` | 32 | 256 |
| `i16`/`u16`/`f16`/`bf16` | 16 | 128 |
| `i32`/`u32`/`f32` | 8 | 64 |
| `i64`/`u64` | 4 | 32 |

### Unified Buffer (UB)

- **Capacity:** 256 KB on-chip SRAM per core
- **Address space:** `!pto.ptr<T, ub>` distinguishes UB from GM (`!pto.ptr<T, gm>`)
- **Role:** Staging area between DMA and vector registers; the only valid source for vector load instructions

### Predicate Masks

- **Type:** `!pto.mask` (256-bit width for A5)
- **Width:** Must match vector register width `N`
- **Semantics:** Mask bit `1` = lane active; `0` = lane inactive (destination preserved)

## Instruction Groups and Timing

### Group 3: Vector Load/Store (`pto.vlds`, `pto.vsts`, etc.)

**Category:** UB ↔ Vector Register data movement
**Pipeline:** PIPE_V

Vector loads move data from UB to vector registers; stores move data from vector registers back to UB. All vector compute operates only on `vreg`; UB staging is explicit.

#### Common Operand Model

- `%source`/`%dest`: base address in UB space (`!pto.ptr<T, ub>`)
- `%offset`: displacement; encoding is instruction-specific
- `%mask`: predicate operand; inactive lanes do not issue memory requests
- `!pto.align`: alignment state for unaligned operations

#### A5 Latency and Throughput

Cycle-accurate simulator (Ascend910_9599 CA) issue→retire timings. **These are simulator results, not guaranteed silicon values.**

| Instruction | A5 mnemonic | Mode / note | Issue→Retire (cycles) |
|-------------|-------------|-------------|----------------------|
| `pto.vlds` | `RV_VLD` | `dist:NORMAL` / `NORAML` | **9** |
| `pto.vldsx2` | `RV_VLDI` | `dist:DINTLV` (dual vreg) | **9** |
| `pto.vsts` / `pto.vstx2` | `RV_VST` / `RV_VSTI` | `dist:NORM` | **9** (12 for `INTLV`) |
| `pto.vgather2` | `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `pto.vgatherb` | `RV_VGATHERB` | indexed byte gather | **~21** |
| `pto.vscatter` | `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `pto.vadd` | `RV_VADD` | F32 between UB-backed ops | **7** |

**Dual-issue capability:** `pto.vlds` is dual-issue capable — two independent `vlds` can issue in the same cycle. Alternatively, one `vlds` + one `vsts` can issue together in a **1+1** cycle. These modes are mutually exclusive.

**Throughput summary:**

| `dist:` token (load) | RV op | Cycles |
|----------------------|-------|--------|
| `NORM`, `UNPK`, `DINTLV`, `BRC`, `BRC_BLK`, `BDINTLV`, `US`, `DS`, `SPLT4CHN`, `SPLT2CHN` | `RV_VLD`/`RV_VLDI` | **9** |
| `NORM`/`PK` (store) | `RV_VSTI` | **9** |
| `INTLV` (`vstx2`) | `RV_VSTI` | **12** |

**Gather/scatter latency:**

| PTO op | A5-level | Latency |
|--------|----------|---------|
| `pto.vgather2` | `RV_VGATHER2` | 27–28 cycles (pattern-dependent) |
| `pto.vgather2_bc` | broadcast gather | 27–28 cycles |
| `pto.vgatherb` | `RV_VGATHERB` | ~21 cycles |
| `pto.vscatter` | `RV_VSCATTER` | ~17 cycles (B16) |

### Group 6: Unary Vector Ops (`pto.vabs`, `pto.vexp`, etc.)

**Category:** Single-input element-wise operations
**Pipeline:** PIPE_V

#### Operand Model

- `%input`: source vector register
- `%mask`: predicate mask (active lanes participate)
- `%result`: destination vector register (same width/type as input unless specified)

Zeroing forms (`-z` suffix variants) zero-fill inactive lanes; merging forms preserve destination values.

#### A5 Latency (Cycle-Accurate Simulator)

| PTO op | RV mnemonic | fp32 | fp16 | bf16 | Types |
|--------|-------------|------|------|------|-------|
| `pto.vabs` | `RV_VABS_FP` | **5** | **5** | — | i8–i32, f16, f32 |
| `pto.vneg` | `RV_VMULS` | **8** | **8** | — | i8–i32, f16, f32 |
| `pto.vexp` | `RV_VEXP` | **16** | **21** | — | f16, f32 |
| `pto.vln` | `RV_VLN` | **18** | **23** | — | f16, f32 |
| `pto.vsqrt` | `RV_VSQRT` | **17** | **22** | — | f16, f32 |
| `pto.vrelu` | `RV_VRELU` | **5** | **5** | — | i8–i32, f16, f32 |
| `pto.vnot` | `RV_VNOT` | — | int-only | — | integer types |
| `pto.vmov` | `RV_VLD` proxy | **9** | **9** | — | all |

**Notes:**
- Integer overflow follows ISA default truncation for `pto.vabs`.
- Transcendental ops (`vexp`, `vln`, `vsqrt`) are hardware-accelerated SFU operations.

### Group 7: Binary Vector Ops (`pto.vadd`, `pto.vmul`, etc.)

**Category:** Two-input element-wise operations
**Pipeline:** PIPE_V

#### Operand Model

- `%lhs`, `%rhs`: source vector registers
- `%mask`: predicate mask
- `%result`: destination (same width and type as sources)

#### A5 Latency

| PTO op | RV mnemonic | fp32 | fp16 | i32 | i16 | i8 |
|--------|-------------|------|------|-----|-----|----|
| `pto.vadd` | `RV_VADD` | **7** | **7** | **7** | **7** | **7** |
| `pto.vsub` | `RV_VSUB` | **7** | **7** | **7** | **7** | **7** |
| `pto.vmul` | `RV_VMUL` | **7** | **7** | **7** | **7** | **7** |
| `pto.vdiv` | `RV_VDIV` | ~**14–28** | varies | — | — | — |
| `pto.vmax`/`pto.vmin` | `RV_VMAX`/`RV_VMIN` | **7** | **7** | **7** | **7** | **7** |
| `pto.vand`/`pto.vor`/`pto.vxor` | bitwise | **7** | **7** | **7** | **7** | **7** |
| `pto.vshl`/`pto.vshr` | shift | **7** | **7** | **7** | **7** | **7** |
| `pto.vaddc` | carry add | **7–10** | — | — | — | — |
| `pto.vsubc` | borrow subtract | **7–10** | — | — | — | — |

**Throughput:** Binary ops have 2× the per-repeat throughput of unary ops. Dual-issue paths exist for certain type combinations.

### Group 8: Vec-Scalar Ops (`pto.vadds`, `pto.vmuls`, etc.)

**Category:** Vector combined with scalar operand
**Pipeline:** PIPE_V

Vector-scalar operations broadcast a scalar to all lanes before computing.

#### Operand Model

- `%vector`: vector register
- `%scalar`: scalar value (broadcast to all lanes)
- `%mask`: predicate mask

#### Latency

Similar to binary ops but with scalar broadcast overhead (typically +1–2 cycles depending on type).

### Group 9: Conversion Ops (`pto.vcvt`, `pto.vtrc`)

**Category:** Type conversion with rounding/saturation control
**Pipeline:** PIPE_V

#### Key Instructions

- `pto.vci`: Convert with implementation-defined rounding
- `pto.vcvt`: Explicit rounding mode controlled by attribute
- `pto.vtrc`: Truncate toward zero (round-to-nearest-even vs truncation)

**Rounding modes:** `RN` (round-to-nearest-even), `RZ` (round-toward-zero), `RP` (round-toward-positive), `RM` (round-toward-negative)

### Group 10: Reduction Ops (`pto.vcadd`, `pto.vcmax`, etc.)

**Category:** Cross-lane reduction and prefix operations
**Pipeline:** PIPE_V

Reductions combine elements across lanes. Two categories:

- **Full vector reductions** (`vcadd`, `vcmax`, `vcmin`): Reduce entire vector to a single value distributed to all lanes
- **Per-VLane (Group) reductions** (`vcgadd`, `vcgmax`, `vcgmin`, `vcpadd`): Reduce within each VLane group (8-lane chunk)

**VLane grouping:** The 256-bit vector is logically split into 8 VLanes of 32 bits each; group reductions operate within each VLane independently.

### Group 11: Compare & Select (`pto.vcmp`, `pto.vsel`, etc.)

**Category:** Comparison and conditional selection
**Pipeline:** PIPE_V

- `pto.vcmp`: Compare two vectors, produce predicate mask
- `pto.vcmps`: Compare vector with scalar
- `pto.vsel`: Select between two vectors based on predicate
- `pto.vselr`: Select with reversal (invert condition)
- `pto.vselrv2`: Select variant (not available on A5)

### Group 12: Data Rearrangement (`pto.vintlv`, `pto.vdintlv`, etc.)

**Category:** In-register data movement and permutation
**Pipeline:** PIPE_V

Lane-level permutation, interleave/deinterleave, pack/unpack operations.

**Available on A5:** `vintlv`, `vdintlv`
**Not A5:** `vintlvv2`, `vdintlvv2` (removed from A5 surface)

### Group 13: DSA/SFU Ops (`pto.vprelu`, `pto.vexpdiff`, `pto.vaxpy`, `pto.vsort32`, etc.)

**Category:** Specialized domain-specific and special function unit operations
**Pipeline:** PIPE_V

Fused operations, transcendental helpers, sorting, and index generation:

| Instruction | Semantics |
|-------------|-----------|
| `pto.vprelu` | Parametric ReLU with broadcast scalar slope |
| `pto.vexpdiff` | `exp(src0 - src1)` — fused exponential difference |
| `pto.vaddrelu` | `max(src0 + src1, 0)` — fused add + ReLU |
| `pto.vsubrelu` | `max(src0 - src1, 0)` — fused subtract + ReLU |
| `pto.vaxpy` | `a*x + y` — fused multiply-add |
| `pto.vaddreluconv` | `max(src0 + src1, 0) + src2` — complex fused pattern |
| `pto.vmulconv` | `(src0 * src1) + src2` — multiply + add |
| `pto.vmull` | Extended-precision multiply (wider product) |
| `pto.vmula` | Multiply-accumulate variant |
| `pto.vtranspose` | In-register matrix transpose (4×4 or 8×8 blocks) |
| `pto.vsort32` | Sort 32-element vector block |
| `pto.vbitsort` | Bitonic sort variant |
| `pto.vmrgsort` | Merge sort for pre-sorted sequences |

## Execution Scope (`__VEC_SCOPE__`)

Vector instructions must be enclosed in a `pto.vecscope` or `pto.strict_vecscope` region. This defines the vector execution interval and establishes producer-consumer ordering with surrounding DMA operations.

```mlir
pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

**`pto.strict_vecscope`** requires all vector-scope inputs to be explicit region arguments, rejecting implicit capture.

## Synchronization Patterns

### Producer-Consumer (MTE2 → Vector)

```mlir
// DMA: GM → UB
pto.copy_gm_to_ubuf %gm, %ub, ...
// Signal: MTE2 → Vector
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
// Wait: Vector sees data
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.vecscope {
  %v = pto.vlds %ub[...]
  // compute
}
```

### Consumer-Producer (Vector → MTE3)

```mlir
pto.vecscope {
  // vector compute
  pto.vsts %v, %ub, %mask
}
// Signal: Vector → MTE3
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
// Wait: MTE3 sees data
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
// DMA: UB → GM
pto.copy_ubuf_to_gm %ub, %gm, ...
```

### Barrier Synchronization

`pto.pipe_barrier "PIPE_V"` drains all pending vector operations. Use when ordering within a single pipeline matters (e.g., two stores to the same GM address).

### Resource-Based Sync (`get_buf`/`rls_buf`)

Buffer IDs provide finer-grained producer-consumer coordination than event flags. The producer calls `rls_buf` after writing; the consumer calls `get_buf` before reading. This tracks *which* buffer slot is ready, not just *that* something is ready.

## Memory Barriers Within VecScope

When UB addresses alias between vector load/store operations within the same `vecscope`, explicit memory barriers are required:

```c
pto.mem_bar "VV_ALL"      // All prior vector ops complete before subsequent
pto.mem_bar "VST_VLD"     // All prior vector stores visible before subsequent loads
pto.mem_bar "VLD_VST"     // All prior vector loads complete before subsequent stores
```

Without proper barriers, loads may see stale data or stores may be reordered.

## Common Usage Patterns

### Full-Vector Compute (All Lanes Active)

```cpp
Mask<64> mask;
mask.set_all(true);
VADD(vdst, va, vb, mask);
```

### Partial Predication

```mlir
%result = pto.vadd %va, %vb, %cond_mask : (!pto.vreg<128xf16>, ...) -> !pto.vreg<128xf16>
```

Only lanes where `%cond_mask` is true participate; inactive lanes preserve destination.

### Double-Buffering (Ping-Pong)

```mlir
// Buffer A
pto.copy_gm_to_ubuf %gm_a, %ub_a, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_A"]
// Buffer B (overlap with A copy)
pto.copy_gm_to_ubuf %gm_b, %ub_b, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_B"]

scf.for %iter = 0 to %N step 1 {
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_A"]
  pto.vecscope { pto.vlds %v_a, %ub_a[...] ... compute ... pto.vsts ... }
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_A"]

  // Overlap compute on A with copy for B
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_B"]
  // ... process B ...
}
```

## Type Support Matrix (A5)

| Element Type | Vector Load/Store | Unary | Binary | Vec-Scalar | Conversion | Reduction |
|--------------|------------------|-------|--------|------------|------------|-----------|
| `f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `f16` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bf16` | ✓ (limited) | — | — | — | — | — |
| `i8`/`u8` | ✓ | ✓ | ✓ | ✓ | ✓ (with restrictions) | ✓ |
| `i16`/`u16` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `i32`/`u32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `i64`/`u64` | — | — | — | — | — | — |
| `f8e4m3`/`f8e5m2` | ✓ (A5+) | A5+ | A5+ | A5+ | A5+ | A5+ |

**Note:** BF16 coverage varies by instruction; integer types support saturating variants where documented.

## Performance Tuning Tips

1. **Maximize dual-issue:** Structure loops to issue two `vlds` per cycle or `vlds`+`vsts` pairs.
2. **Hide latency with double-buffering:** Overlap `copy_gm_to_ubuf` with vector compute using two buffer slots.
3. **Predicate early:** Apply masks as early as possible to avoid unnecessary computation on inactive lanes.
4. **Align UB addresses:** Misaligned accesses incur alignment-state overhead; prefer 32B alignment for contiguous patterns.
5. **Group reductions by VLane:** Use group reduction ops (`vcg*`, `vcpadd`) when data locality within 8-lane groups is exploitable.
6. **Prefer fused ops:** `vaddrelu`, `vexpdiff`, `vaxpy` reduce register pressure and instruction count.
7. **Gather/scatter sparingly:** `vgather2` latency is ~27–28 cycles; ensure sufficient computation to amortize.

## Relationship to Tile Instructions

Vector instructions provide fine-grained control where tile instructions expose higher-level tile semantics:

| Aspect | Tile (`pto.t*`) | Vector (`pto.v*`) |
|--------|----------------|------------------|
| **Abstraction** | Tile buffers with valid regions | Raw vector registers, no valid region |
| **Data movement** | Implicit TLOAD/TSTORE (GM → UB → tile) | Explicit `copy_gm_to_ubuf` + `vlds`/`vsts` |
| **Predication** | No per-lane masking | Full per-lane predicate on every op |
| **Typical use** | High-level tensor algebra | Hand-tuned kernels, per-element control |
| **Target profile** | All (A2/A3/A5/CPU) | A5 native; emulated on others |

Many high-level tile operations are internally lowered to vector micro-instructions plus DMA orchestration. Understanding both levels enables hand-tuning critical kernels.

## References

- **[PTO ISA Manual](../README.md)** — Full instruction reference
- **[Vector Instruction Reference](../vector/README.md)** — Per-op reference pages
- **[Format of Instruction Descriptions](../reference/format-of-instruction-descriptions.md)** — Page structure standard
- **[PTO-Gym micro-instruction SPEC](https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-micro-Instruction-SPEC.md)** — Source specification
