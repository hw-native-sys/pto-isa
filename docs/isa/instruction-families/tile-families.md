# Tile Instruction Set

`pto.t*` is the tile-oriented instruction set of PTO ISA. It defines how tile payloads are loaded from global memory, transformed element-wise, reduced, expanded, synchronized, and written back to global memory.

## Instruction Overview

Tile instructions operate over tiles whose shapes, layouts, roles, and valid regions are architecturally visible. The result is usually another tile, a changed valid-region interpretation, or a synchronization edge.

**Tile operands** (`!pto.tile<T, R, C>` or `!pto.tile_buf<...>`) are the primary operands of this instruction set. Unlike vector registers or scalar registers, tiles carry explicit metadata about which elements are valid and what layout the data uses.

### Data Flow

```
GlobalMemory
    │
    │  pto.tload: GM → local tile buffer
    ▼
Tile Buffer ──► Tile Compute ──► Tile Buffer ──► pto.tstore: Tile Buffer → GM
(Vec/Mat/Acc)    (pto.tadd, pto.tmatmul, etc.)       (Vec/Mat/Acc)
```

## Instruction Classes

| | Class | Description | Examples |
|-|-------|-------------|----------|
| | Sync and Config | Resource binding, tile address assignment, alias views, pipeline synchronization | `pto.tassign`, `pto.tsync`, `pto.talias`, `pto.settf32mode`, `pto.setfmatrix`, `pto.set_img2col_rpt`, `pto.set_img2col_padding`, `pto.subview`, `pto.get_scale_addr` |
| | Elementwise Tile-Tile | Lane-wise binary and unary operations | `pto.tadd`, `pto.tmul`, `pto.tpow`, `pto.tcmp`, `pto.tcvt`, `pto.tsel`, `pto.trelu` |
| | Tile-Scalar and Immediate | Tile combined with scalar or immediate operands | `pto.tadds`, `pto.taxpy`, `pto.tmuls`, `pto.tpows`, `pto.tlrelu`, `pto.tcmps` |
| | Reduce and Expand | Row/column reductions and expansions | `pto.trowsum`, `pto.tcolmax`, `pto.trowexpand`, `pto.tcolexpand` |
| | Memory and Data Movement | GM↔tile transfer, gather/scatter | `pto.tload`, `pto.tstore`, `pto.mgather`, `pto.mscatter` |
| | Matrix and Matrix-Vector | GEMV, matmul, and variants | `pto.tgemv`, `pto.tgemv_mx`, `pto.tmatmul`, `pto.tmatmul_acc`, `pto.tmatmul_bias` |
| | Layout and Rearrangement | Reshape, transpose, extract, insert, concatenate, pack | `pto.tmov`, `pto.ttrans`, `pto.tconcat`, `pto.tpack`, `pto.treshape`, `pto.textract`, `pto.tinsert`, `pto.timg2col` |
| | Irregular and Complex | Sort, quantize, dequantize, generated state, index movement, partial reductions | `pto.tmrgsort`, `pto.tsort32`, `pto.tquant`, `pto.tdequant`, `pto.trandom`, `pto.thistogram`, `pto.tgather`, `pto.tpartadd` |

> **Configuration tile-side modes** (`pto.settf32mode`, `pto.setfmatrix`, `pto.set_img2col_rpt`, `pto.set_img2col_padding`, `pto.subview`, `pto.get_scale_addr`) are Tile ISA instructions that program tile-mode registers. They are architecturally tile-state configuration, not scalar control-shell ops. They are documented in the [Sync and Config](../tile/sync-and-config.md) group.

## Inputs

Tile instructions consume combinations of:

- Source tiles (read-only operands): `!pto.tile<...>`
- Destination tiles (write-only or read-write operands): `!pto.tile_buf<...>`
- Scalar modifiers or immediate operands
- GM-facing views: `!pto.partition_tensor_view<...>`
- Optional `RecordEvent` event tokens or `WaitEvents...` for chaining

## Expected Outputs

Tile instructions produce:

- Destination tile payloads carrying the result
- Changed valid-region interpretations
- Explicit state updates (e.g., assigned addresses) or synchronization edges (`RecordEvent`)

## Side Effects

Some tile instructions have architectural side effects beyond the destination tile:

| | Class | Side Effects |
|-|-------|-------------|
| | Memory and Data Movement | Reads from or writes to GM-visible storage (`pto.tload`, `pto.tstore`) |
| | Sync and Config | Establishes synchronization edges or binds tile addresses (`pto.tassign`, `pto.tsync`) |
| | Irregular and Complex | May produce debug output (`pto.tprint`) or modify allocation state |

## Shared Constraints

All tile instruction groups must state:

1. **Valid-region interaction** — How the instruction set interprets source tile valid regions relative to the destination.
2. **Layout and role restrictions** — Which tile layouts, TileTypes, and roles the instruction set accepts.
3. **Target-profile restrictions** — Where A2/A3 and A5 differ from each other and from the portable ISA contract.
4. **Cases that are not allowed** — Conditions that are illegal across the instruction set.

## Valid Region Model

All tile elementwise operations iterate over the **destination tile's valid region**. For each lane `(r, c)` in the destination's valid region:

- The corresponding lane `(r, c)` from each source tile is read, **regardless of whether that lane is within the source tile's own valid region**
- Source tiles whose valid region does not cover `(r, c)` read **target-specific undefined values**:
  - On A2/A3 and A5: Reads produce undefined values; the underlying tile buffer SRAM content is unpredictable
  - On CPU simulator: Reads produce whatever bits are present in the backing memory
- Programs MUST NOT rely on any particular value being read from an out-of-region source lane unless the operation explicitly documents the behavior

See [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) for the full model.

## Constraints

- **Tile legality** depends on more than dtype; shape, layout, role, and valid region all matter.
- Operations with multiple tiles must define valid-region interaction explicitly.
- Some tile instruction groups are profile-gated: MX block-scale tiles (`Left`, `Right`, `ScaleLeft`, `ScaleRight`) are A5-only; FP8/FP4-family element types are A5-only.
- Tile instructions do **not** inherit vector-register semantics; they operate on architecturally visible tile state.
- No implicit broadcasting: all source tiles must have shapes compatible with the destination tile.

## Cases That Are Not Allowed

- Reading undefined out-of-valid-region data as if it were meaningful.
- Assuming tile instructions inherit vector-register semantics.
- Relying on target-specific support gaps as universal architecture rules.
- Assuming implicit broadcasting, reshaping, or valid-region repair unless documented.
- Using MX format tiles (`TileType::Left`/`Right`) on CPU or A2/A3 profiles.
- Using FP8 element types on CPU or A2/A3 profiles.

## Saturating Variants

Operations with the `_c` suffix perform saturating arithmetic:

| | Variant | Base Op | Overflow/Underflow Behavior |
|-|---------|---------|---------------------------|
| | `pto.taddc` | Addition | Saturating: result is clamped to the type's min/max representable value |
| | `pto.tsubc` | Subtraction | Saturating: result is clamped to the type's min/max representable value |

Programs MUST NOT assume that `taddc` and `tadd` produce identical results when overflow does not occur; they MAY differ even for in-range values due to implementation precision choices.

## Type Support by Profile

| | Element Type | CPU Simulator | A2/A3 | A5 |
|-|------------|:-------------:|:------:|:--:|
| | f32 (float) | Yes | Yes | Yes |
| | f16 (half) | Yes | Yes | Yes |
| | bf16 (bfloat16_t) | Yes | Yes | Yes |
| | i8/u8 | Yes | Yes | Yes |
| | i16/u16 | Yes | Yes | Yes |
| | i32/u32 | Yes | Yes | Yes |
| | i64/u64 | Yes | Yes | Yes |
| | f8e4m3 / f8e5m2 | No | No | Yes |
| | hifloat8_t / float4_e* | No | No | Yes |

## Syntax

### Assembly Form (PTO-AS)

```asm
pto.tadd %dst, %src0, %src1 : !pto.tile<f32, 16, 16>
```

### SSA Form (AS Level 1)

```mlir
%dst = pto.tadd %src0, %src1
    : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>)
    -> !pto.tile<f32, 16, 16>
```

### DPS Form (AS Level 2)

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%dst : !pto.tile_buf<f32, 16, 16>)
```

See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full syntax specification.

## C++ Intrinsic

Tile instructions are available as C++ intrinsics declared in `include/pto/common/pto_instr.hpp`:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Elementwise: tile = tile op tile
template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TADD(TileDst& dst, TileSrc0& src0, TileSrc1& src1);

// Memory transfer: tile = GM view
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events);

// Matmul: acc = matmul(lhs, rhs)
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix,
                             WaitEvents&... events);
```

## Navigation

See the [Tile ISA reference](../tile/README.md) for the full per-op reference under `tile/ops/`.

## See Also

- [Instruction sets](./README.md) — All instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
