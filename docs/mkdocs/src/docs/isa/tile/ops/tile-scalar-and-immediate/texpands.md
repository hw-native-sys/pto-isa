<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/texpands.md` -->

# pto.texpands

Standalone reference page for `pto.texpands`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Broadcast a scalar into a destination tile.

## Mechanism

Broadcast a scalar into a destination tile. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = texpands %scalar : f32, !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events);
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
    - For `TileType::Vec` :
    - The op fills `dst` over `dst.GetValidRow()` / `dst.GetValidCol()`.
    - For  `TileType::Mat` :
    - For Tile : The op fills `dst` over `TileData::Rows` / `TileData::Cols`.
    - For ConvTile : The op fills `dst` over `ConvTileData`'s shape.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - For `TileType::Vec` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `bfloat16_t`, `float`.
      - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - For  `TileType::Mat` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `bfloat16_t`, `float`.
      - Static valid bounds: `The range of  TileData::Rows * TileData::Cols * sizeof(T) / 32 is [1, 32767]`.

- **Implementation checks (A5)**:
    - For `TileType::Vec` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`.
      - Tile layout must be row-major (`TileData::isRowMajor`).
      - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - For  `TileType::Mat` :
      - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`.
      - For`TileDataDst::layout == pto::Layout::NC1HWC0 || TileDataDst::layout == pto::Layout::FRACTAL_Z`:
        - `The range of convtile's (shape0 * shape1 * shape2 * shape3) is [1, 32767]`.
      - For`TileDataDst::layout == pto::Layout::NDC1HWC0 || TileDataDst::layout == pto::Layout::FRACTAL_Z_3D`:
        - `The range of convtile's (shape0 * shape1 * shape2 * shape3 * shape4) is [1, 32767]`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TEXPANDS(dst, 0.0f);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TEXPANDS(dst, 0.0f);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = texpands %scalar : f32, !pto.tile<...>
# AS Level 2 (DPS)
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Next op in family: [pto.tcmps](./tcmps.md)
