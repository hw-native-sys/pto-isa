# TLOAD



## Tile Operation Diagram

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## Introduction

Load data from a GlobalTensor (GM) into a Tile.

## Math Interpretation

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## Assembly Syntax

Synchronous form:

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2 (DPS)

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);

template <TLoadL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

Existing `TLOAD(dst, src)` and `TLOAD<TileT, GTensor>(dst, src)` still work. The `TLoadL2Hint`-first form is an extra overload (no default on `l2Control`).

## L2 cache hint

Optional first-template overload (existing `TLOAD(dst, src)` unchanged):

```cpp
TLOAD<TLoadL2Hint::NotAllocKeep>(dst, src);
```

Supported `TLoadL2Hint` values:

| Enumerator | Value | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | default allocate (no-op) | yes |
| NormalLastVictim | 1 | default allocate (no-op) | yes |
| NormalPersistent | 2 | default allocate (no-op) | yes |
| NotAllocKeep | 4 | non-allocate (GM addr += runtime `l2Cacheoffset`) | yes |
| NotAllocClean | 5 | non-allocate (same as Keep) | yes |
| NotAllocDrop | 6 | non-allocate (same as Keep) | yes |

On A2/A3 there are effectively **two options only**:

1. **Default allocate** — `NormalFirstVictim` (0), `NormalLastVictim` (1), `NormalPersistent` (2): no-ops on A2/A3 (no effect on VLU / different last/first/persist hints). They are not meaningful distinct modes; all follow the default allocate path.
2. **Non-allocate** — `NotAllocKeep` (4), `NotAllocClean` (5), `NotAllocDrop` (6): yes, via GM addr += runtime `l2Cacheoffset` (same behavior for Keep/Clean/Drop on A2/A3).

On A5 all listed values are passed through to DMA. CPU / costmodel accept the template and ignore it.

## Constraints

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - Destination tile location must be `TileType::Vec` or `TileType::Mat`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Runtime: all `src.GetShape(dim)` values and `dst.GetValidRow()/GetValidCol()` must be `> 0`.
    - `TileType::Vec` loads only support matching layouts: ND->ND, DN->DN, NZ->NZ.
    - `TileType::Mat` loads support: ND->ND, DN->DN, NZ->NZ, plus ND->NZ and DN->ZN.
    - For ND->NZ: `GlobalData::staticShape[0..1] == 1` and `TileData::SFractalSize == 512`.
    - For DN->ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
    - ND->NZ also accepts `GlobalData::staticShape[2] != 1`. `Shape2` is then the number of ND matrices
      a single instruction moves (the instruction's own `ndNum` operand), each matrix being
      `[Shape3, Shape4]`, and the matrices are stacked along the tile rows, so
      `dst.GetValidRow() == Shape2 * Shape3`. Because `srcNdMatrixStride` and `dstNzMatrixStride` are
      16-bit element counts on this platform, it also requires `Stride2 <= 65535` and
      `Shape3 * (32 / sizeof(DType)) <= 65535`, on top of `Shape2 * Shape3 <= TileData::Rows` and
      `1 <= Shape2 <= 65535`.
    - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
    - Vec tile (UB path): `1 <= TileData::Rows <= 4095`.
    - Mat tile (L1 path): `1 <= TileData::Rows <= 16384`.
- **Implementation checks (A5)**:
    - `sizeof(TileData::DType)` must be `1`, `2`, `4`, or `8` bytes, and must match `sizeof(GlobalData::DType)`.
    - For `int64_t/uint64_t`, `TileData::PadVal` must be `PadValue::Null` or `PadValue::Zero`.
    - `TileType::Vec` loads require one of the following layout pairs:
    - ND with row-major + `SLayout::NoneBox` (ND->ND),
    - DN with col-major + `SLayout::NoneBox` (DN->DN),
    - NZ with `SLayout::RowMajor` (NZ->NZ).
    - For row-major ND->ND with compile-time-known shapes, `TileData::ValidCol` must equal `GlobalData::staticShape[4]`, and `TileData::ValidRow` must equal the product of `GlobalData::staticShape[0..3]`.
    - `TileType::Mat` loads are additionally constrained by `TLoadCubeCheck` (e.g., only specific ND/DN/NZ conversions and L1-size limits).
    - For `TileType::Mat` ND->NZ and DN->NZ: `TileData::SFractalSize == 512`, `sizeof(TileData::DType) != 8`, and `GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1`.
    - ND->NZ additionally accepts `GlobalData::staticShape[2] != 1`. `Shape2` is then the number of ND matrices
      a single instruction moves (the hardware `ndNum` loop), each matrix being `[Shape3, Shape4]`. The matrices are
      stacked along the tile rows, so `dst.GetValidRow() == Shape2 * Shape3`, and it requires
      `Shape2 * Shape3 <= TileData::Rows`, `1 <= Shape2 <= 65535`, and a data type that is not fp4.
      DN->NZ still requires `Shape2 == 1`.
    - `TileType::Mat` loads also handle loads for mx format, which include `MX_A_ZZ/MX_A_ND/MX_A_DN` to ZZ for scalarA and `MX_B_NN/MX_B_ND/MX_B_DN` to NN for scalarB.
    - for `MX_A_ZZ/MX_B_NN`: `(GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - for `MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN`: `(GlobalData::staticShape[0] == 1 || GlobalData::staticShape[0] == -1)` and `(GlobalData::staticShape[1] == 1 || GlobalData::staticShape[1] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - for scaleA, `dst.GetValidCol() % 2 == 0`.
    - for scaleB, `dst.GetValidRow() % 2 == 0`

- **Valid region**:
    - The implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` as the transfer size.
    - On A2/A3, same-layout, block-aligned `TileType::Mat` loads write only this valid region. ND-to-NZ/DN-to-ZN loads (and the single-row/single-column Mat special paths) additionally zero-fill the final partial C0 block. Other data remains unchanged, including data owned by tile views that share the same backing storage.
    - On A5, same-layout `TileType::Mat` ND/DN loads fill only the final partial 32-byte block according to `PadVal`; ND/DN-to-fractal loads zero-fill the final partial C0 block. Full 32-byte gaps and inactive rows or columns remain unchanged.
    - On A2/A3 and A5, a `TileType::Vec` ND/DN load with a non-null `PadVal` fills only the sub-32-byte tail after each transferred burst. Full 32-byte gaps and inactive rows or columns remain unchanged; NZ loads and `PadValue::Null` do not add padding.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TASSIGN(t, 0x1000);
  TLOAD(t, gin);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
```

### PTO Assembly Form

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```
