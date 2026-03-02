# TFILLPAD


## Tile Operation Diagram

![TFILLPAD tile operation](../figures/isa/TFILLPAD.svg)

## Introduction

Copy a source tile into a destination tile and fill the remaining (padded) elements with a compile-time pad value
selected by `TileDataDst::PadVal` (e.g., `PadValue::Min`/`PadValue::Max`).

This is commonly used to materialize deterministic values outside the runtime valid region so that subsequent ops can
operate on a full static tile shape.

## Math Interpretation

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

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form (conceptual):

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
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
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND(TileDataDst& dst, TileDataSrc& src);

template <typename TileData, PadValue PadVal = PadValue::Zero>
PTO_INTERNAL void TFILLPAD(TileData &dst, TileData &src);
```

## Constraints

- `TileDataDst::PadVal != PadValue::Null`.
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)` and element size must be `1`, `2`, or `4` bytes.
- `TFILLPAD`: `TileDataDst::Rows/Cols` must match `TileDataSrc::Rows/Cols`.
- `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows` and `TileDataDst::Cols >= TileDataSrc::Cols`.
- `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout only support (!TileData::isRowMajor && TileData::Slayout::RowMajor), and PadVal only support PadValue::Zero`

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

## ASM Form Examples

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
# IR Level 2 (DPS)
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

