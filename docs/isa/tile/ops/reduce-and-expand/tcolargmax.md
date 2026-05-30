# pto.tcolargmax

`pto.tcolargmax` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Get the row index of the maximum element for each column. A value+index variant also returns the maximum
value for each column.

## Mechanism

Get the row index of the maximum element for each column. The 4-operand overload returns both the maximum
value and the row index for each column.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \underset{0 \le i < R}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

For value+index mode:

$$ \mathrm{dstVal}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j} $$

$$ \mathrm{dstIdx}_{0,j} = \underset{0 \le i < R}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
%dstVal, %dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>, !pto.tile<...>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dstVal, %dstIdx = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, !pto.tile<...>)
```

### AS Level 2 (DPS)

```text
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp,
                                WaitEvents&... events);
```

## Inputs

- `src` is the source tile.
- `tmp` is a temporary tile used for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.
- In value+index mode, `dstVal` names the value output tile and `dstIdx` names the index output tile.

## Expected Outputs

`dst` holds the row index of the column-wise maximum: for each column `j`, `dst[0,j]` = argmax of all elements in column `j` of `src`.
In value+index mode, `dstVal[0,j]` holds the maximum value and `dstIdx[0,j]` holds its row index.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    ### General constraints / checks

    - `dst` and `src` must be `TileType::Vec`.

    - Supported source element types: `half`, `float`, `int32_t`, `int16_t`.

    - Supported destination element types: `uint32_t`, `int32_t`.

    - `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).

    - `dst` and `src` must satisfy the shared column-reduce-index check path used by `TColArgMax`.

    - Temporary tile is not used when `srcValidRow <= ElementPerRepeat`, used when `srcValidRow > ElementPerRepeat`.

    - `tmp` tile's columns is the same as `src`.

    - Simply set `tmp` tile size the same as `src` when `src` is small.

    - `tmp` tile's stride can be calculated out based on `src`'s `validRow` using the following formula:

    ```text
    repeats = ceil(validRow / elementPerRepeat)
    stride = ceil(repeats * 2 / elementPerBlock) * elementPerBlock + ceil(repeats / elementPerBlock) * elementPerBlock
    ```

    ### Value+index mode

    - `dstVal` must be a `TileType::Vec` tile with standard ND layout.
    - `dstVal` element type must match `src`.
    - 8-bit source element types are not supported by the value+index overload.
    - `dstVal.GetValidRow() == 1`
    - `dstVal.GetValidCol() == dstIdx.GetValidCol()`
    - `dstVal.GetValidCol() == src.GetValidCol()`
    - For 16-bit source element types, `dstIdx` must use `uint16_t` or `int16_t`.
    - For 32-bit source element types, `dstIdx` must use `uint32_t` or `int32_t`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Runtime checks follow the shared column-reduce check path:
      - `src.GetValidRow() != 0`
      - `src.GetValidCol() != 0`
      - `src.GetValidCol() == dst.GetValidCol()`

    === "A2/A3"
        - `dst` is checked through the shared column-reduce-index path and may use either of these non-fractal layouts:
          - ND layout with one row (`BLayout::RowMajor`, `Rows == 1`), or
          - DN layout whose valid row count is 1.

    === "A5"
        - In the checked A5 implementation path, `tmp` is accepted by the interface but not used by `TCOLARGMAX_IMPL`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLARGMAX(dst, src, tmp);
}
```

### Auto Value + Index

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto_value_index() {
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

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLARGMAX(dst, src, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
%dstVal, %dstIdx = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>, !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dstVal, %dstIdx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.tcolmin](./tcolmin.md)
- Next op in instruction set: [pto.trowmax](./trowmax.md)
