<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/trems.md` -->

# pto.trems

Standalone reference page for `pto.trems`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Elementwise remainder with a scalar: `remainder(src, scalar)`.

## Mechanism

Elementwise remainder with a scalar: `%`. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1 (SSA)

```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `scalar` is the scalar value broadcast to all lanes.
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
    - `dst` and `src` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst` and `src` must be vector tiles.
    - `dst` and `src` must be row-major.
    - Runtime: `dst.GetValidRow() == src.GetValidRow() > 0` and `dst.GetValidCol() == src.GetValidCol() > 0`.
    - **tmp Buffer Requirements**:
      - `tmp.GetValidCol() >= dst.GetValidCol()` (at least as many columns as dst)
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - Data type must match `TileDataDst::DType`.

- **Implementation Checks (A5)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst` and `src` must be vector tiles.
    - Static valid bounds: `ValidRow <= Rows` and `ValidCol <= Cols` for both tiles.
    - Runtime: `dst.GetValidRow() == src.GetValidRow()` and `dst.GetValidCol() == src.GetValidCol()`.
    - Note: tmp parameter is accepted but not validated or used on A5.

- **For `int32_t` Inputs (A2A3 Only)**: Both `src` elements and `scalar` must be in the range `[-2^24, 2^24]` (i.e., `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
# AS Level 2 (DPS)
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in family: [pto.tfmods](./tfmods.md)
- Next op in family: [pto.tmaxs](./tmaxs.md)
