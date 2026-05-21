# pto.tscatter

`pto.tscatter` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.

## Summary

`pto.tscatter` supports two operation modes:

1. Index-based scatter: scatter source elements into destination positions using per-element flattened offsets.
2. Mask scatter: scatter source elements into a wider destination with a mask pattern, filling unselected positions with zero.

## Mechanism

### Index-Based Scatter

Scatter source elements into a destination tile using per-element flattened destination offsets. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For each source element `(i, j)`, let `k = idx[i,j]` and write:

$$ \mathrm{dst\_flat}_{k} = \mathrm{src}_{i,j} $$

Here `dst_flat` denotes the destination tile viewed as a single linear storage sequence. `TSCATTER` does **not** interpret `idx[i,j]` as a destination row selector. On the standard row-major tile layout, this is equivalent to writing the `k`-th flattened destination element.

If multiple elements map to the same destination location, the final value is undefined. On A2/A3 and A5, the last writer wins according to the hardware scheduling order; on the CPU simulator, the last writer wins according to the iteration order.

### Mask Scatter

Mask scatter is an A5-only overload that writes each source element into one selected lane of an expanded destination group and writes zero to the other lanes in the group.

For each source element `(i, j)` and mask pattern `P`, the destination column group is selected by the mask expansion factor:

$$ \mathrm{dst}_{i,\ F_P \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

All other columns in that expanded group are zero-filled. `F_P` is 1 for `P1111`, 2 for `P0101`/`P1010`, and 4 for `P0001`/`P0010`/`P0100`/`P1000`.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1 (SSA)

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents &... events);

template <MaskPattern maskPattern = MaskPattern::P1111, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

`MaskPattern` is defined in `include/pto/common/type.hpp`:

| Value | Pattern | Selected lane | Expansion |
|-------|---------|---------------|-----------|
| `P0101` | `0101...` | first lane in each 2-lane group | x2 |
| `P1010` | `1010...` | second lane in each 2-lane group | x2 |
| `P0001` | `0001...` | first lane in each 4-lane group | x4 |
| `P0010` | `0010...` | second lane in each 4-lane group | x4 |
| `P0100` | `0100...` | third lane in each 4-lane group | x4 |
| `P1000` | `1000...` | fourth lane in each 4-lane group | x4 |
| `P1111` | `1111...` | all lanes, equivalent to `TMOV` | x1 |

## Inputs

- `src` is the source tile.
- `indexes` is an index tile providing flattened destination offsets.
- `dst` names the destination tile. The operation iterates over src's valid region.

## Expected Outputs

Elements from `src` are scattered to positions in `dst` specified by `indexes`.

## Side Effects

No architectural side effects beyond producing the destination tile. Concurrent writes to the same location produce undefined results. On A2/A3 and A5, the final value is determined by hardware scheduling order; on the CPU simulator, the final value is determined by iteration order.

## Constraints

!!! warning "Constraints"
    - Operand shape, mode, and state tuples MUST match the documented contract of this operation and its instruction set overview.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (A2A3)**:
      - `TileDataD::Loc`, `TileDataS::Loc`, `TileDataI::Loc` must be `TileType::Vec`.
      - `TileDataD::DType`, `TileDataS::DType` must be one of: `int32_t`, `int16_t`, `int8_t`, `half`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
      - `TileDataI::DType` must be one of: `int16_t`, `int32_t`, `uint16_t` or `uint32_t`.
      - `indexes` values are interpreted as flattened destination element offsets in destination tile storage order.
      - No bounds checks are enforced on `indexes` values.
      - Static valid bounds: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
      - `TileDataD::DType` and `TileDataS::DType` must be the same.
      - When size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
      - When size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
      - When size of `TileDataD::DType` is 1 bytes, the size of `TileDataI::DType` must be 2 bytes.

    - **Implementation checks (A5)**:
      - `TileDataD::Loc`, `TileDataS::Loc`, `TileDataI::Loc` must be `TileType::Vec`.
      - `TileDataD::DType`, `TileDataS::DType` must be one of: `int32_t`, `int16_t`, `int8_t`, `half`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
      - `TileDataI::DType` must be one of: `int16_t`, `int32_t`, `uint16_t` or `uint32_t`.
      - `indexes` values are interpreted as flattened destination element offsets in destination tile storage order.
      - No bounds checks are enforced on `indexes` values.
      - Static valid bounds: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
      - `TileDataD::DType` and `TileDataS::DType` must be the same.
      - When size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
      - When size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
      - When size of `TileDataD::DType` is 1 bytes, the size of `TileDataI::DType` must be 2 bytes.

    - **Mask scatter checks (A5 only)**:
      - `DstTileData::Loc` and `SrcTileData::Loc` must be `TileType::Vec`.
      - `DstTileData::DType` and `SrcTileData::DType` must be the same supported TSCATTER data type.
      - `maskPattern` must be one of `P0101`, `P1010`, `P0001`, `P0010`, `P0100`, `P1000`, or `P1111`.
      - `SrcTileData::ValidRow` must equal `DstTileData::ValidRow`.
      - `DstTileData::ValidCol` must equal `SrcTileData::ValidCol * F_P`, where `F_P` is the pattern expansion factor.

!!! warning "Mask scatter zero fill"
    Before mask scatter writes source elements, the A5 implementation initializes the destination tile buffer to zero across the full tile allocation, not only the valid region. Code must not overlap that destination UB allocation with other live data.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### Mask Scatter

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_scatter() {
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010>(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in instruction set: [pto.tgatherb](./tgatherb.md)
