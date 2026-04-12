<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/tmins.md` -->

# pto.tmins

Standalone reference page for `pto.tmins`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Elementwise minimum of a tile and a scalar.

## Mechanism

Elementwise minimum of a tile and a scalar. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tmins %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1 (SSA)

```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, WaitEvents &... events);
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

- **Common constraints**:
    - `dst` and `src` must use the same element type.
    - Scalar type must match the tile data type.
    - Tile location must be vector (`TileData::Loc == TileType::Vec`).

- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.

- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`, `bfloat16_t`.
    - Runtime: `src.GetValidCol() == dst.GetValidCol()`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMINS(dst, src, 0.0f);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TMINS(dst, src, 0.0f);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tmins %src, %scalar : !pto.tile<...>, f32
# AS Level 2 (DPS)
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in family: [pto.tsels](./tsels.md)
- Next op in family: [pto.tadds](./tadds.md)
