# pto.tgather

`pto.tgather` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.

## Summary

Gather/select elements using either an index tile or a compile-time mask pattern.

## Mechanism

Gather/select elements using either an index tile or a compile-time mask pattern. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Index-based gather (conceptual):

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] $$

Exact index interpretation and bounds behavior are as follows: On A2/A3 and A5, out-of-range indices produce undefined results (no explicit masking); on the CPU simulator, out-of-range indices wrap modulo the source extent.

Mask-pattern gather is a selection controlled by `pto::MaskPattern`. On A2/A3 and A5, the mask selects elements from the source in a pattern-defined order; on the CPU simulator, the same mask semantics apply.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Index-based gather:

```text
%dst = tgather %src0, %indices : !pto.tile<...> -> !pto.tile<...>
```

Mask-pattern gather:

```text
%dst = tgather %src {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## Inputs

- `src0` is the source tile.
- `indices` (index-based gather): index tile providing gather indices.
- `tmp` (optional): temporary tile for index-based gather.
- `maskPattern` (mask-pattern gather): compile-time mask pattern.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds gathered elements from `src0` at positions specified by `indices` or `maskPattern`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Bounds / validity**:
        - Index bounds are not validated by explicit runtime assertions; on A2/A3 and A5, out-of-range indices produce undefined results; on the CPU simulator, out-of-range indices are clamped to the valid range.

## Performance

### A2/A3 Cycle Count

`pto.tgather` issues an indirect vector-load through an index tile. Cost is dominated by the indirect-fetch throughput of the LSU/MTE2 path; uniform indices (sequential or broadcast) hit the fast path, while scattered indices serialize.

**Cycle model**: `total ≈ startup + N_idx × per_index + bank_conflict_penalty`.

### Layout and Shape Impact

| Index pattern | Effect |
|---|---|
| Sequential / contiguous | Fast path; coalesces to wide loads |
| Strided | Stride-bound throughput |
| Random | Worst-case; per-element issue |
| Out-of-range | Returns implementation-defined value (see Constraints) |

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Index-based gather: implementation checks (A2A3)**:
        - `sizeof(DstTileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float`.
        - `sizeof(Src1TileData::DType)` must be must be `int32_t`, `uint32_t`.
        - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
        - `src1.GetValidCol() == Src1TileData::Cols` and `dst.GetValidCol() == DstTileData::Cols`.

    - **Index-based gather: implementation checks (A5)**:
        - `sizeof(DstTileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float`.
        - `sizeof(Src1TileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.
        - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
        - `src1.GetValidCol() == Src1TileData::Cols` and `dst.GetValidCol() == DstTileData::Cols`.

    - **Mask-pattern gather: implementation checks (A2A3)**:
        - Source element size must be `2` or `4` bytes.
        - `SrcTileData::DType`/`DstTileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
        or `half` or `bfloat16_t` or `float`.
        - `dst` and `src` must both be `TileType::Vec` and row-major.
        - `sizeof(dst element) == sizeof(src element)` and `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).

    - **Mask-pattern gather: implementation checks (A5)**:
        - Source element size must be `1` or `2` or `4` bytes.
        - `dst` and `src` must both be `TileType::Vec` and row-major.
        - `SrcTileData::DType`/`DstTileData::DType` must be `int8_t` or `uint8_t` or `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
        or `half` or `bfloat16_t` or `float` or `float8_e4m3_t`or `float8_e5m2_t` or `hifloat8_t`.
        - Supported dtypes are restricted to a target-defined set (checked via `static_assert` in the implementation), and `sizeof(dst element) == sizeof(src element)`, `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src0;
  IdxT idx;
  DstT dst;
  TGATHER(dst, src0, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in instruction set: [pto.tsort32](./tsort32.md)
- Next op in instruction set: [pto.tgatherb](./tgatherb.md)

