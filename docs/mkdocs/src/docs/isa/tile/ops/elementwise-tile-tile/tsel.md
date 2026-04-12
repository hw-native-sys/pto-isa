<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/tsel.md` -->

# pto.tsel

Standalone reference page for `pto.tsel`. This page belongs to the [Elementwise Tile Tile](../../elementwise-tile-tile.md) family in the PTO ISA manual.

## Summary

Per-element conditional selection between two tiles using a predicate mask.

## Mechanism

For each element `(i, j)` in the destination's valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true (non-zero)} \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}
$$

The predicate mask tile uses a target-defined packed encoding. A temporary tile (`tmp`) is required as a working buffer for predicate unpacking.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0,
                          TileData &src1, TmpTile &tmp, WaitEvents &... events);
```

**Parameters:**
- `dst`: destination tile receiving the selected values.
- `selMask`: predicate mask tile. Lane `(i,j)` is true if non-zero; selects `src0[i,j]`.
- `src0`: source tile selected when mask lane is true.
- `src1`: source tile selected when mask lane is false.
- `tmp`: required temporary working tile for predicate unpacking. Must have compatible shape.

## Inputs

- `dst` names the destination tile receiving the selected values.
- `selMask` is the predicate mask tile. Lane `(i,j)` selects from `src0` if non-zero, otherwise from `src1`.
- `src0` is the source tile selected for mask-true lanes.
- `src1` is the source tile selected for mask-false lanes.
- `tmp` is a required temporary working tile for predicate unpacking.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile — `src0` where the mask is true, `src1` otherwise.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- `sizeof(TileData::DType)` MUST be `2` or `4` bytes.
- `dst`, `src0`, and `src1` MUST use the **same element type**.
- `dst`, `src0`, and `src1` MUST be row-major layout.
- `dst`, `src0`, and `src1` MUST have the same declared shape.
- `selMask` layout MUST be compatible with the target's predicate unpacking format.
- The iteration domain is `dst.GetValidRow()` × `dst.GetValidCol()`.
- `tmp` MUST have sufficient capacity to hold intermediate predicate bits; its exact requirements are target-defined.

## Cases That Are Not Allowed

- **MUST NOT** use non-row-major `dst`/`src0`/`src1` tiles.
- **MUST NOT** use `dst`/`src0`/`src1` with different declared shapes.

## Target-Profile Restrictions

| Check | A2/A3 | A5 |
|-------|:-----:|:--:|
| Supported dtypes | `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float` | Same |
| sizeof(dtype) | 2 or 4 bytes | Same |
| Row-major layout | Required | Required |
| Same shape (dst/src0/src1) | Required | Required |
| `tmp` tile required | Yes | Yes |

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TSEL(dst, mask, src0, src1, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TASSIGN(mask, 0x4000);
  TASSIGN(tmp,  0x5000);
  TSEL(dst, mask, src0, src1, tmp);
}
```

### PTO Assembly Form

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in family: [pto.tcvt](./tcvt.md)
- Next op in family: [pto.trsqrt](./trsqrt.md)
