<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/tcmp.md` -->

# pto.tcmp

Standalone reference page for `pto.tcmp`. This page belongs to the [Elementwise Tile Tile](../../elementwise-tile-tile.md) family in the PTO ISA manual.

## Summary

Compare two tiles element-wise and write a packed predicate mask tile.

## Mechanism

For each element `(i, j)` in the destination's valid region:

$$ \mathrm{dst}_{i,j} = \bigl(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\bigr) $$

where `cmpMode` is one of `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`.

The predicate mask is stored in `dst` using a target-defined packed encoding (not a boolean per lane).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1,
                          CmpMode cmpMode, WaitEvents &... events);
```

**Compare modes** (`pto::CmpMode`):

| Mode | Meaning |
|------|---------|
| `CmpMode::EQ` | Equal |
| `CmpMode::NE` | Not equal |
| `CmpMode::LT` | Less than |
| `CmpMode::LE` | Less than or equal |
| `CmpMode::GT` | Greater than |
| `CmpMode::GE` | Greater than or equal |

## Inputs

- `src0` is the first source tile (left-hand side of comparison).
- `src1` is the second source tile (right-hand side of comparison).
- `dst` names the destination predicate tile.
- `cmpMode` specifies the comparison predicate.
- The operation iterates over `dst`'s valid region; `src0` and `src1` are sampled at the same coordinates.

## Expected Outputs

`dst` carries the packed predicate mask tile. The mask encoding is target-defined; programs MUST NOT make assumptions about the bit layout.

## Side Effects

No architectural side effects beyond producing the predicate tile. Does not implicitly fence unrelated traffic.

## Constraints

- The iteration domain is `dst.GetValidRow()` × `dst.GetValidCol()`.
- `src0.GetValidRow()` and `src0.GetValidCol()` MUST match `dst`'s valid region.
- `src1`'s shape/validity is not verified by runtime assertions; out-of-region lanes read **implementation-defined values**.
- The output predicate tile uses a **packed encoding** (not one boolean per lane). Use `TSEL` with the predicate tile to apply it.

## Cases That Are Not Allowed

- **MUST NOT** assume any particular encoding of the predicate tile.
- **MUST NOT** use `dst` with a dtype other than the target-defined predicate dtype.

## Target-Profile Restrictions

| Check | A2/A3 | A5 |
|-------|:-----:|:--:|
| Supported input types | `int32_t`, `half`, `float` | `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`, `int8_t`, `float`, `half` |
| Output predicate dtype | `uint8_t` | `uint32_t` |
| Tile location | `TileType::Vec` | `TileType::Vec` |
| Layout | Row-major | Row-major |
| Static valid bounds | Required | Required |
| `src0` valid == `dst` valid | Required | Required |
| `src1` validity | Not verified | Not verified |
| `int32_t` with non-EQ mode | Ignores `cmpMode`, uses EQ | Full support |

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### PTO Assembly Form

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...>
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in family: [pto.tmax](./tmax.md)
- Next op in family: [pto.tdiv](./tdiv.md)
