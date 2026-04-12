<!-- Generated from `docs/isa/tile/ops/reduce-and-expand/trowargmin.md` -->

# pto.trowargmin

Standalone reference page for `pto.trowargmin`. This page belongs to the [Reduce And Expand](../../reduce-and-expand.md) family in the PTO ISA manual.

## Summary

Get the column index of the minimum element for each row.

## Mechanism

Get the column index of the minimum element for each row. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \underset{0 \le j < C}{\operatorname{argmin}} \; \mathrm{src}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trowargmin %src : !pto.tile<...> -> !pto.tile<...>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1 (SSA)

```text
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## Inputs

- `src` is the source tile.
- `tmp` is a temporary tile used for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the column index of the row-wise minimum: for each row `i`, `dst[i,0]` = argmin of elements in row `i` of `src`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `dst` and `src` must be `TileType::Vec`.

- Supported source element types: `half`, `float`.

- Supported destination element types: `uint32_t`, `int32_t`.

- `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).

- `dst` and `src` must satisfy the shared row-reduce-index check path used by `TRowArgMin`.

- Temporary tile is not used when `srcValidCol <= ElementPerRepeat`, used when `srcValidCol > ElementPerRepeat`.

- `tmp` tile's rows is the same as `src`.

- Simply set `tmp` tile size the same as `src` when `src` is small.

- `tmp` tile's stride can be calculated out based on `src`'s `validCol` using the following formula:

```text
repeats = ceil(validCol / elementPerRepeat)
stride = ceil(repeats * 2 / elementPerBlock) * elementPerBlock + ceil(repeats / elementPerBlock) * elementPerBlock
```

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- Runtime checks follow the shared row-reduce check path:
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

### A2A3 implementation checks

- `dst` is checked through the shared row-reduce-index path and may use either of these non-fractal layouts:
  - DN layout with one column (`BLayout::ColMajor`, `Cols == 1`), or
  - ND layout whose valid column count is 1.

### A5 implementation checks

- In the checked A5 implementation path, `tmp` is accepted by the interface but not used by `TROWARGMIN_IMPL`.

### About temporary tile `tmp` for A3

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWARGMIN(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWARGMIN(dst, src, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowargmin %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in family: [pto.trowargmax](./trowargmax.md)
- Next op in family: [pto.trowexpand](./trowexpand.md)
