<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/trem.md` -->

# pto.trem

Standalone reference page for `pto.trem`. This page belongs to the [Elementwise Tile Tile](../../elementwise-tile-tile.md) family in the PTO ISA manual.

## Summary

Elementwise remainder of two tiles.

## Mechanism

Elementwise remainder of two tiles. The result has the same sign as the divider. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \bmod \mathrm{src1}_{i,j}$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trem %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (left operand).
- `src1` is the second source tile (right operand).
- `dst` names the destination tile.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- **Division by Zero**:
    - Behavior is target-defined; the CPU simulator asserts in debug builds.

- **Valid Region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation Checks (A2A3)**:
    - `dst`, `src0`, and `src1` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst`, `src0`, and `src1` must be vector tiles.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow() > 0` and `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol() > 0`.
    - **tmp Buffer Requirements**:
      - `tmp.GetValidCol() >= dst.GetValidCol()` (at least as many columns as dst)
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - Data type must match `TileDataDst::DType`.

- **Implementation Checks (A5)**:
    - `dst`, `src0`, and `src1` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst`, `src0`, and `src1` must be vector tiles.
    - Static valid bounds: `ValidRow <= Rows` and `ValidCol <= Cols` for all tiles.
    - Runtime: `dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow()` and `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol()`.
    - Note: tmp parameter is accepted but not validated or used on A5.

- **For `int32_t` Inputs (A2A3 Only)**: Both `src0` and `src1` elements must be in the range `[-2^24, 2^24]` (i.e., `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  Tile<TileType::Vec, int32_t, 16, 16> tmp;
  TREM(out, a, b, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trem %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in family: [pto.tneg](./tneg.md)
- Next op in family: [pto.tfmod](./tfmod.md)
