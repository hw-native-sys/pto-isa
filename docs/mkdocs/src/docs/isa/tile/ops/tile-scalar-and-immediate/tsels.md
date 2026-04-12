<!-- Generated from `docs/isa/tile/ops/tile-scalar-and-immediate/tsels.md` -->

# pto.tsels

Standalone reference page for `pto.tsels`. This page belongs to the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) family in the PTO ISA manual.

## Summary

Select one of two source tiles using a scalar `selectMode` (global select).

## Mechanism

Select between source tile and scalar using a mask tile (per-element selection for source tile). It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

For each element `(i, j)` in the valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true} \\
\mathrm{scalar} & \text{otherwise}
\end{cases}
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tsels %src0, %src1, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## Inputs

- `mask` is the predicate mask tile; lane `(i,j)` selects from `src` if true, otherwise from `scalar`.
- `src` is the source tile.
- `scalar` is the scalar fallback value broadcast to all lanes.
- `tmp` is a required temporary working tile for predicate unpacking.
- `dst` names the destination tile.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

- **Mask encoding**:
    - The mask tile is interpreted as packed predicate bits in a target-defined layout.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - `sizeof(TileDataDst::DType)` must be `2` or `4` bytes.
    - Supported data types are `half`, `float16_t`, `float`, and `float32_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be row-major.
    - Runtime: `src.GetValidRow()/GetValidCol()` must match `dst.GetValidRow()/GetValidCol()`.

- **Implementation checks (A5)**:
    - `sizeof(TileDataDst::DType)` may be `1`, `2`, or `4` bytes.
    - Supported data types are `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, and `float`.
    - `dst` and `src` must use the same element type.
    - `dst`, `mask`, and `src` must be row-major.
    - Runtime: `src.GetValidRow()/GetValidCol()` must match `dst.GetValidRow()/GetValidCol()`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TSELS(dst, mask, src, tmp, scalar);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TASSIGN(src, 0x1000);
  TASSIGN(tmp, 0x2000);
  TASSIGN(dst, 0x3000);
  TASSIGN(mask, 0x4000);
  TSELS(dst, mask, src, tmp, scalar);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
# AS Level 2 (DPS)
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in family: [pto.tcmps](./tcmps.md)
- Next op in family: [pto.tmins](./tmins.md)
