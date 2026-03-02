# TLOAD


## Tile Operation Diagram

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## Introduction

Load data from a GlobalTensor (GM) into a Tile.

## Math Interpretation

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### IR Level 2 (DPS)

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
  - Destination tile location must be `TileType::Vec` or `TileType::Mat`.
  - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
  - Runtime: all `src.GetShape(dim)` values and `dst.GetValidRow()/GetValidCol()` must be `> 0`.
  - `TileType::Vec` loads only support matching layouts: ND->ND, DN->DN, NZ->NZ.
  - `TileType::Mat` loads support: ND->ND, DN->DN, NZ->NZ, plus ND->NZ and DN->ZN.
    - For ND->NZ or DN->ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
  - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
- **Implementation checks (A5)**:
  - `sizeof(TileData::DType)` must be `1`, `2`, `4`, or `8` bytes, and must match `sizeof(GlobalData::DType)`.
  - For `int64_t/uint64_t`, `TileData::PadVal` must be `PadValue::Null` or `PadValue::Zero`.
  - `TileType::Vec` loads require one of the following layout pairs:
    - ND with row-major + `SLayout::NoneBox` (ND->ND),
    - DN with col-major + `SLayout::NoneBox` (DN->DN),
    - NZ with `SLayout::RowMajor` (NZ->NZ).
  - For row-major ND->ND with compile-time-known shapes, `TileData::ValidCol` must equal `GlobalData::staticShape[4]`, and `TileData::ValidRow` must equal the product of `GlobalData::staticShape[0..3]`.
  - `TileType::Mat` loads are additionally constrained by `TLoadCubeCheck` (e.g., only specific ND/DN/NZ conversions and L1-size limits).
  - `TileType::Mat` loads also handle loads for mx format, which include `MX_A_ZZ/MX_A_ND/MX_A_DN` to ZZ for scalarA and `MX_B_NN/MX_B_ND/MX_B_DN` to NN for scalarB.
    - for `MX_A_ZZ/MX_B_NN`: `GlobalData::staticShape[3] == 16` and `GlobalData::staticShape[4] == 2`.
    - for `MX_A_ND/MX_ADN/MX_B_ND/MX_B_DN`: `GlobalData::staticShape[0] == 1` and `GlobalData::staticShape[1] == 1` and `GlobalData::staticShape[4] == 2`.
    - for scaleA, `dst.GetValidCol() % 2 == 0`.
    - for scaleB, `dst.GetValidRow() % 2 == 0`

- **Valid region**:
  - The implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` as the transfer size.

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
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
```

### PTO Assembly Form

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

