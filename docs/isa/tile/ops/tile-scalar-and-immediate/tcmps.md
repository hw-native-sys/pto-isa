# pto.tcmps

`pto.tcmps` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

Compare a tile against a scalar or against the first element of another tile and write per-element
comparison results.

## Mechanism

Compare a tile against a scalar or against the first element of another tile and write per-element
comparison results. It operates on tile payloads rather than scalar control state, and its legality is
constrained by tile shape, layout, valid-region, and target-profile support.

Scalar form, for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) $$

Tile form, for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{0,0}\right) $$

The encoding/type of `dst` is target-specific: on A2/A3 the predicate tile uses `uint8_t` with 1 bit per element (packed 8 elements per byte), and on A5 it uses `uint32_t` with 1 bit per element (packed 32 elements per DWORD).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename T, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc0& src0, T src1, CmpMode cmpMode, WaitEvents&... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1,
                           CmpMode cmpMode, WaitEvents&... events);
```

## Inputs

- `src0` is the source tile.
- `scalar` is the scalar value broadcast to all lanes; `cmpMode` selects comparison predicate.
- `src1` is the optional tile operand for the tile form; `src1[0,0]` is broadcast as the comparison scalar.
- `dst` names the destination predicate tile.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Common constraints**:
        - Tile location must be vector (`TileData::Loc == TileType::Vec`).
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
        - Runtime: `src0` and `dst` must have the same valid row/col.
        - Tile form: `src0` and `src1` must have the same data type.

    - **Valid region**:
        - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

    - **Comparison modes**:
        - Supports `CmpMode::EQ`, `CmpMode::NE`, `CmpMode::LT`, `CmpMode::GT`, `CmpMode::LE`, `CmpMode::GE`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (A2A3)**:
        - `TileData::DType` must be one of: `int32_t`, `float`, `half`, `uint16_t`, `int16_t`.
        - Tile layout must be row-major (`TileData::isRowMajor`).
        - For `int32_t` input, only `CmpMode::EQ` is supported; other comparison modes fall back to `EQ`.

    - **Implementation checks (A5)**:
        - `TileData::DType` must be one of: `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`,
          `uint8_t`, `int8_t`, `bfloat16_t`.
        - Tile layout must be row-major (`TileData::isRowMajor`).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Tile Form

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_tile() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  DstT dst(16, 2);
  TCMPS(dst, src0, src1, CmpMode::GE);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.texpands](./texpands.md)
- Next op in instruction set: [pto.tsels](./tsels.md)
