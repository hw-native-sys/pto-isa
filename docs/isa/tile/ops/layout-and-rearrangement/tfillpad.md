# pto.tfillpad

`pto.tfillpad` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Copy+pad a tile outside the valid region with a compile-time pad value.

## Mechanism

Copy a source tile into a destination tile and fill the remaining (padded) elements with a compile-time pad value
selected by `TileDataDst::PadVal` (e.g., `PadValue::Min`/`PadValue::Max`).

This is commonly used to materialize deterministic values outside the runtime valid region so that subsequent ops can
operate on a full static tile shape. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Let `VR = src.GetValidRow()` and `VC = src.GetValidCol()`. For each destination element `(i, j)`:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}
$$

`pad` is determined by `TileDataDst::PadVal` and the element type (e.g., `+inf/-inf` for floating types when available,
otherwise `std::numeric_limits<T>::max()/min()`).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form (conceptual):

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Implemented in the backend headers pulled in by `include/pto/common/pto_instr_impl.hpp`:

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `dst` names the destination tile. Must have same shape as `src`.
- `PadVal` is the compile-time pad value for elements outside the valid region.

## Expected Outputs

`dst` holds a copy of `src` with valid region copied and padded region filled with the specified pad value.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `TileDataDst::PadVal != PadValue::Null`.

    - `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)` and element size must be `1`, `2`, or `4` bytes.

    - `TFILLPAD`: `TileDataDst::Rows/Cols` must match `TileDataSrc::Rows/Cols`.

    - `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows` and `TileDataDst::Cols >= TileDataSrc::Cols`.

    - `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout only support (!TileData::isRowMajor && TileData::Slayout::RowMajor), and PadVal only support PadValue::Zero`

## Performance

### A2/A3 Cycle Count

`pto.tfillpad` writes the pad value into the destination's invalid region and copies `src` into the valid region. Cost is one row-wide vector store per row plus a fast-path scalar broadcast for the padding lanes.

**Cycle model**: `total ≈ startup + R × (per_row_store + per_row_pad)`.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.tfillpad` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

    - Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example1() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}

void example2() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256, BLayout::ColMajor, 1, 224, SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.tinsert](./tinsert.md)
- Next op in instruction set: [pto.tfillpad_inplace](./tfillpad-inplace.md)

