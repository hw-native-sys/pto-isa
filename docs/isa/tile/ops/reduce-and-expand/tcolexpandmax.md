# pto.tcolexpandmax

`pto.tcolexpandmax` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Column-wise broadcast max with per-column scalar vector.

## Mechanism

Column-wise broadcast max: take `max(src0, src1)` where `src1` provides one scalar per column.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_j)
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (the tile to be modified).
- `src1` is the second source tile providing per-column scalar values.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst[i,j]` = max(`src0[i,j]`, `src1[0,j]`) (column-wise broadcast max of per-column scalar).

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `TileDataDst::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.

    - Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.

    - `src1` is expected to provide **one scalar per column** (i.e., its valid shape must cover `C` values).

    - Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TColExpand*.hpp`.

## Performance

### A2/A3 Cycle Count

`pto.tcolexpandmax` lowers to a per-column `vmax` sequence in which a single per-column scalar from `src1` is broadcast across the working tile and combined with each element of `src0` via max.

**Cycle model**:

```
total = startup + R × (per_column_vmax + interval)
```

where `R = dst.GetValidRow()` (row-direction) or `R = dst.GetValidCol()` (column-direction), and `per_column_vmax` scales with `C = dst.GetValidCol()` rounded up to the native vector width.

### Instruction Sequence by Shape (FP32)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|----------------------|------------------|
| 16×16 | `vmax`*16 → PIPE_V | ~O(64) |
| 32×32 | `vmax`*32 → PIPE_V | ~O(128) |
| 64×64 | `vmax`*64 → PIPE_V | ~O(256) |
| R×C   | `vmax`*R → PIPE_V  | ~O(R × ⌈C / vlen⌉) |

> Note: per-shape cycle counts are illustrative templates; populate with measured numbers from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

### Layout and Shape Impact

| Layout | validCol | Optimization |
|--------|----------|--------------|
| `RowMajor` | ≥ vlen (FP32) | Continuous fast path; one `vmax` per column |
| `RowMajor` | < vlen | General path with tail masking |
| Other | any | Not supported (compile-time `isRowMajor` constraint) |

The column-broadcast scalar is reused across all columns without re-fetching.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.tcolexpandmax` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

    - Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using ColVecT = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor>;
  SrcT src0;
  DstT dst;
  ColVecT src1;
  // Col-expand-max: each column of dst = max(src0.col, src1.col_scalar)
  TCOLEXPANDMAX(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using ColVecT = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor>;
  SrcT src0;
  DstT dst;
  ColVecT src1;
  TASSIGN(src0, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(src1, 0x3000);
  // Col-expand-max in manual mode
  TCOLEXPANDMAX(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.tcolexpanddiv](./tcolexpanddiv.md)
- Next op in instruction set: [pto.tcolexpandmin](./tcolexpandmin.md)

