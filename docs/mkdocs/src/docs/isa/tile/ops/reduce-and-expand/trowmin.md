<!-- Generated from `docs/isa/tile/ops/reduce-and-expand/trowmin.md` -->

# pto.trowmin

Standalone reference page for `pto.trowmin`. This page belongs to the [Reduce And Expand](../../reduce-and-expand.md) family in the PTO ISA manual.

## Summary

Reduce each row by taking the minimum across columns.

## Mechanism

Reduce each row by taking the minimum across columns. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \min_{0 \le j < C} \mathrm{src}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trowmin %src : !pto.tile<...> -> !pto.tile<...>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `tmp` is a temporary tile used for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the row-wise minimum: for each row `i`, `dst[i,0]` = min of all elements in row `i` of `src`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `dst` and `src` must both be `TileType::Vec`.

- `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).

- `dst` must use one of the following non-fractal layouts:
  - ND layout (`BLayout::RowMajor`, `SLayout::NoneBox`), or
  - DN layout with exactly one column (`BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`).

- `dst` and `src` must use the same element type.

- Runtime valid-region checks:
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

- Supported element types: `half`, `float`, `int32_t`, `int16_t`.

- The implementation accepts both ND output and DN output with `Cols == 1`.

- Runtime checks follow the shared row-reduce check path:
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

- The current implementation path passes `tmp` into the backend call, but this document does not add extra `tmp` shape/layout constraints beyond what is explicitly enforced by the checked implementation.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- The intrinsic signature requires an explicit `tmp` operand.

### A2A3 implementation checks

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMIN(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWMIN(dst, src, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowmin %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in family: [pto.trowmax](./trowmax.md)
- Next op in family: [pto.trowargmax](./trowargmax.md)
