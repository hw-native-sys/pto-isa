# TCOLARGMAX

## Tile Operation Diagram

![TCOLARGMAX tile operation](../figures/isa/TCOLARGMAX.svg)

## Introduction

Get the row index of the maximum element for each column.
A value+index variant is also available that returns both the maximum value and its row index.

## Math Interpretation

### Pure Index Mode

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dstIdx}_{0,j} = \underset{0 \le i < R}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

### Value + Index Mode

$$ \mathrm{dstVal}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j} $$

$$ \mathrm{dstIdx}_{0,j} = \underset{0 \le i < R}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

### Pure Index Mode

Synchronous form:

```text
%dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
```

IR Level 1 (SSA):

```text
%dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

IR Level 2 (DPS):

```text
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstIdx : !pto.tile_buf<...>)
```

### Value + Index Mode

Synchronous form:

```text
%dstVal, %dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>, !pto.tile<...>
```

IR Level 1 (SSA):

```text
%dstVal, %dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

IR Level 2 (DPS):

```text
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

### Pure Index Mode (3-argument)

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
```

### Value + Index Mode (4-argument)

```cpp
template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp,
                                WaitEvents&... events);
```

## Constraints

### General constraints / checks

- `dstIdx` and `src` must be `TileType::Vec`.
- `src` may use ND or DN non-fractal layout (`SLayout::NoneBox`).
- `dstIdx` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- Supported destination index element types: `uint32_t`, `int32_t`.
- Runtime checks:
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `dstIdx.GetValidRow() == 1`
    - `src.GetValidCol() == dstIdx.GetValidCol()`

### Pure Index Mode (3-argument)

#### A2A3 implementation checks

- Supported source element types: `half`, `float`, `uint16_t`, `uint32_t`.
- `tmp` must use the same element type as `src`.
- `tmp` is used as scratch storage for index tracking and current comparison values.

#### A5 implementation checks

- Supported source element sizes are 8-bit, 16-bit, or 32-bit; covers `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float`.
- `tmp` is accepted by the interface but not used by the implementation.

### Value + Index Mode (4-argument)

In addition to the general constraints:

- `dstVal` must be `TileType::Vec` with standard ND layout (row-major, non-fractal).
- `dstVal` element type must match the source element type `TileDataIn::DType`.
- 8-bit source types are **not** supported.
- Runtime checks:
    - `dstVal.GetValidRow() == 1`
    - `dstVal.GetValidCol() != 0`
    - `src.GetValidCol() == dstVal.GetValidCol()`
    - `dstVal.GetValidRow() == dstIdx.GetValidRow()`
    - `dstVal.GetValidCol() == dstIdx.GetValidCol()`

#### A2A3 implementation checks

- Supported source element types: `half`, `float`, `uint16_t`, `uint32_t`.
- When source element size is 2 bytes (`half`, `uint16_t`): `dstIdx` element type must be `uint16_t` or `int16_t`.
- When source element size is 4 bytes (`float`, `uint32_t`): `dstIdx` element type must be `uint32_t` or `int32_t`.
- `tmp` must use the same element type as `src`.
- `tmp` is used as scratch storage; for half input types an internal s16->f16->s32 conversion path is used for the index.

#### A5 implementation checks

- Source element size must be 16-bit or 32-bit (`sizeof(T) != 1`).
- When source element size is 2 bytes (`half`, `int16_t`, `uint16_t`): `dstIdx` element type must be `uint16_t` or `int16_t`.
- When source element size is 4 bytes (`float`, `int32_t`, `uint32_t`): `dstIdx` element type must be `uint32_t` or `int32_t`.
- `tmp` is accepted by the interface but not used by the implementation.

### About temporary tile `tmp` for A2A3

* `tmp` **is always used** in the A2A3 implementation as scratch space for intermediate results (current row index, argmax index, and current maximum elements).
* `tmp` tile's data type must be the same as `src`'s data type.
* `tmp` tile is organized into three regions within a single row:
  - Region 0 (`[0, tmpGapEles)`): current row index counter (incremented per row).
  - Region 1 (`[tmpGapEles, 2 * tmpGapEles)`): current maximum elements for comparison.
  - Region 2 (`[2 * tmpGapEles, 3 * tmpGapEles)`): argmax index result (before final conversion to `dstIdx`).
* `tmpGapEles` is determined as follows:
  - When `srcValidCol >= elemPerRpt`: `tmpGapEles = elemPerRpt`.
  - When `srcValidCol < elemPerRpt`: `tmpGapEles = ceil(srcValidCol / elemPerBlock) * elemPerBlock`.
* Simply set `tmp` tile size the same as `src` when `src` is small, or calculate using:

  ```text
  repeats = ceil(validCol / elementPerRepeat)
  stride = ceil(repeats * 2 / elementPerBlock) * elementPerBlock + ceil(repeats / elementPerBlock) * elementPerBlock
  ```

* In Value + Index mode with half input, `tmp` region 2 data undergoes s16->f16->s32 conversion before being stored to `dstIdx`.

### About temporary tile `tmp` for A5

* `tmp` temporary tile is **not used** in the A5 implementation for either mode. The A5 uses vector register-based computation (`__VEC_SCOPE__`) and does not require scratch tile storage.
* `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

### Pure Index Mode

#### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstT dst(1, 255);
  TmpT tmp(1, 32);
  TCOLARGMAX(dst, src, tmp);
}
```

#### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstT dst(1, 255);
  TmpT tmp(1, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dst, 0x1000);
  TASSIGN(tmp, 0x2000);
  TCOLARGMAX(dst, src, tmp);
}
```

### Value + Index Mode

#### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto_val_idx() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstValT = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1>;
  using DstIdxT = Tile<TileType::Vec, int32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstValT dstVal(1, 255);
  DstIdxT dstIdx(1, 255);
  TmpT tmp(1, 32);
  TCOLARGMAX(dstVal, dstIdx, src, tmp);
}
```

#### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual_val_idx() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstValT = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1>;
  using DstIdxT = Tile<TileType::Vec, int32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstValT dstVal(1, 255);
  DstIdxT dstIdx(1, 255);
  TmpT tmp(1, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dstVal, 0x1000);
  TASSIGN(dstIdx, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLARGMAX(dstVal, dstIdx, src, tmp);
}
```

## ASM Form Examples

### Pure Index Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Pure Index Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Value + Index Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dstVal, %dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### Value + Index Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dstVal, %dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### PTO Assembly Form

```text
# Pure index
%dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
# Value + index
%dstVal, %dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>, !pto.tile<...>

# IR Level 2 (DPS) - pure index
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstIdx : !pto.tile_buf<...>)

# IR Level 2 (DPS) - value + index
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```
