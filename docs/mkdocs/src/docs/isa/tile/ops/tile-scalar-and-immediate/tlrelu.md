<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/tlrelu.md` -->

# pto.tlrelu

Standalone reference page for `pto.tlrelu`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Leaky ReLU with a scalar slope.

## Mechanism

Leaky ReLU with a scalar slope. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} > 0) ? \mathrm{src}_{i,j} : (\mathrm{src}_{i,j} \cdot \mathrm{slope}) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tlrelu %src, %slope : !pto.tile<...>, f32
```

### AS Level 1 (SSA)

```text
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLRELU(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events);
```

## Inputs

- `src` is the source tile.
- `slope` is the scalar slope for negative values (broadcast to all lanes).
- `dst` names the destination tile.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- **Common constraints**:
    - Tile location must be vector (`TileData::Loc == TileType::Vec`).
    - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - Runtime: `dst` and `src` must have the same valid row/col.
    - Slope scalar type must match the Tile data type.

- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `half`, `float16_t`, `float`, `float32_t` (floating-point types only).
    - Tile layout must be row-major (`TileData::isRowMajor`).

- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `half`, `float` (floating-point types only).
    - Tile layout must be row-major (`TileData::isRowMajor`).

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLRELU(out, x, 0.1f);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tlrelu %src, %slope : !pto.tile<...>, f32
# AS Level 2 (DPS)
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in family: [pto.txors](./txors.md)
- Next op in family: [pto.taddsc](./taddsc.md)
