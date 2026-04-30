# pto.trowexpandmul

`pto.trowexpandmul` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.

## Mechanism

Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{0,i} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (the tile to be modified).
- `src1` is the second source tile providing per-row scalar values.
- `tmp` (optional): temporary tile for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst[i,j]` = `src0[i,j]` * `src1[i,0]` (row-wise broadcast multiply of per-row scalar).

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - Source and destination shapes, layouts, and element types MUST satisfy the legality rules documented by the instruction set and target profile.

    - Programs must not assume implicit broadcasting, reshaping, or valid-region repair unless the operation documents it.

## Performance

### A2/A3 Cycle Count

`pto.trowexpandmul` lowers to a per-row `vmul` sequence in which a single per-row scalar from `src1` is broadcast across the working tile and combined with each element of `src0` via multiplication.

**Cycle model**:

```
total = startup + R × (per_row_vmul + interval)
```

where `R = dst.GetValidRow()` (row-direction) or `R = dst.GetValidCol()` (row-direction), and `per_row_vmul` scales with `C = dst.GetValidCol()` rounded up to the native vector width.

### Instruction Sequence by Shape (FP32)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|----------------------|------------------|
| 16×16 | `vmul`*16 → PIPE_V | ~O(64) |
| 32×32 | `vmul`*32 → PIPE_V | ~O(128) |
| 64×64 | `vmul`*64 → PIPE_V | ~O(256) |
| R×C   | `vmul`*R → PIPE_V  | ~O(R × ⌈C / vlen⌉) |

> Note: per-shape cycle counts are illustrative templates; populate with measured numbers from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

### Layout and Shape Impact

| Layout | validCol | Optimization |
|--------|----------|--------------|
| `RowMajor` | ≥ vlen (FP32) | Continuous fast path; one `vmul` per row |
| `RowMajor` | < vlen | General path with tail masking |
| Other | any | Not supported (compile-time `isRowMajor` constraint) |

The row-broadcast scalar is reused across all rows without re-fetching.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks**:
        - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType` (compile-time).
        - `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.
        - Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.
        - Mode 1: `src1` is expected to provide **one scalar per row** (i.e., its valid shape must cover `R` values).
        - Mode 2: `src1` is expected to provide **32 bytes data per row**.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDMUL(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TASSIGN(src0, 0x1000);
  TASSIGN(dst,  0x2000);
  TASSIGN(src1, 0x3000);
  TROWEXPANDMUL(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.trowexpandsub](./trowexpandsub.md)
- Next op in instruction set: [pto.trowexpanddiv](./trowexpanddiv.md)

