<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/tors.md` -->

# pto.tors

Standalone reference page for `pto.tors`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Elementwise bitwise OR of a tile and a scalar.

## Mechanism

Elementwise bitwise OR of a tile and a scalar. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;|\; \mathrm{scalar} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tors %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1 (SSA)

```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
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

- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - Intended for integral element types.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - In manual mode, setting the source tile and destination tile to the same memory is unsupported.

- **Implementation checks (A5)**:
    - Intended for integral element types supported by `TEXPANDS` and `TOR`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - In manual mode, setting the source tile and destination tile to the same memory is unsupported.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TORS(dst, src, 0xffu);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tors %src, %scalar : !pto.tile<...>, i32
# AS Level 2 (DPS)
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in family: [pto.tands](./tands.md)
- Next op in family: [pto.tshls](./tshls.md)
