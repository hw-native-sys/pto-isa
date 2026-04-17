# Tile Instruction Set

`pto.t*` is the tile-oriented instruction set of PTO ISA. It defines how tile payloads are loaded from global memory, transformed element-wise, reduced, expanded, synchronized, and written back to global memory.

## Instruction Overview

Tile instructions operate over tiles whose shapes, layouts, roles, and valid regions are architecturally visible. The result is usually another tile, a changed valid-region interpretation, or a synchronization edge.

**Tile operands** (`!pto.tile<T, R, C>` or `!pto.tile_buf<...>`) are the primary operands of this instruction set. Unlike vector registers or scalar registers, tiles carry explicit metadata about which elements are valid and what layout the data uses.

### Data Flow

```
GlobalMemory
    │
    │  TLOAD: GM → local tile buffer
    ▼
Tile Buffer ──► Tile Compute ──► Tile Buffer ──► TSTORE: Tile Buffer → GM
(Vec/Mat/Acc)    (pto.tadd, TMATMUL, etc.)       (Vec/Mat/Acc)
```

## Instruction Classes

| Class | Description | Examples |
|-------|-------------|----------|
| Sync and Config | Resource binding, event setup, tile-local config | `tassign`, `tsync`, `tsettf32mode`, `tset_img2col_rpt` |
| Elementwise Tile-Tile | Lane-wise binary and unary operations | `tadd`, `tmul`, `tcmp`, `tcvt`, `tsel`, `trelu` |
| Tile-Scalar and Immediate | Tile combined with scalar or immediate operands | `tadds`, `tmuls`, `tlrelu`, `tcmps` |
| Reduce and Expand | Row/column reductions and expansions | `trowsum`, `tcolmax`, `trowexpand`, `tcolexpand` |
| Memory and Data Movement | GM↔tile transfer, gather/scatter, fix-pipe store variants | `tload`, `tstore`, `tstore_fp`, `mgather`, `mscatter` |
| Matrix and Matrix-Vector | GEMV, matmul, and variants | `tgemv`, `tgemv_mx`, `tmatmul`, `tmatmul_acc`, `tmatmul_bias` |
| Layout and Rearrangement | Reshape, transpose, extract, insert | `tmov`, `ttrans`, `treshape`, `textract`, `tinsert`, `timg2col` |
| Irregular and Complex | Sort, quantize, histogram, print | `tmrgsort`, `tsort32`, `tquant`, `thistogram`, `tprint` |

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

| Class | Side Effects |
|-------|-------------|
| Memory and Data Movement | Reads from or writes to GM-visible storage (`TLOAD`, `TSTORE`) |
| Sync and Config | Establishes synchronization edges or binds tile addresses (`TASSIGN`, `TSYNC`) |
| Irregular | May produce debug output (`TPRINT`) or modify allocation state |

## Valid Region Model

All tile elementwise operations iterate over the **destination tile's valid region**. For each lane `(r, c)` in the destination's valid region:

- The corresponding lane `(r, c)` from each source tile is read, **regardless of whether that lane is within the source tile's own valid region**
- Source tiles whose valid region does not cover `(r, c)` read **implementation-defined values**
- Programs MUST NOT rely on any particular value being read from an out-of-region source lane unless the operation explicitly documents the behavior

See [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) for the full model.

## Constraints

- **Tile legality** depends on more than dtype; shape, layout, role, and valid region all matter.
- Operations with multiple tiles must define valid-region interaction explicitly.
- Some tile instruction set are profile-gated: MX block-scale tiles (`Left`, `Right`, `ScaleLeft`, `ScaleRight`) are A5-only; FP8/FP4-family element types are A5-only.
- `tsethf32mode` and `tsetfmatrix` are documented in the scalar/control lane even though their historic API names still carry the `t` prefix; they configure scalar-visible mode state rather than tile payload state.
- Tile instructions do **not** inherit vector-register semantics; they operate on architecturally visible tile state.
- No implicit broadcasting: all source tiles must have shapes compatible with the destination tile.

## Cases That Are Not Allowed

- Reading undefined out-of-valid-region data as if it were meaningful.
- Assuming tile instructions inherit vector-register semantics.
- Relying on target-specific support gaps as universal architecture rules.
- Assuming implicit broadcasting, reshaping, or valid-region repair unless documented.
- Using MX format tiles (`TileType::Left`/`Right`) on CPU or A2/A3 profiles.
- Using FP8 element types on CPU or A2/A3 profiles.

## Syntax

### Assembly Form (PTO-AS)

```asm
tadd %dst, %src0, %src1 : !pto.tile<f32, 16, 16>
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

## See Also

- [Tile ISA reference](../tile/README.md) — Full tile instruction set reference
- [Tile Instruction Set](../instruction-families/tile-families.md) — Instruction-set-level contracts
- [Instruction set contracts](../instruction-families/README.md) — All instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
