# pto.tcolexpandadd

`pto.tcolexpandadd` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Column-wise broadcast add with per-column scalar vector.

## Mechanism

Column-wise broadcast add: add each element of `src0` by a per-column scalar vector `src1`.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_j
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (the tile to be modified).
- `src1` is the second source tile providing per-column scalar values.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst[i,j]` = `src0[i,j]` + `src1[0,j]` (column-wise broadcast add of per-column scalar).

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

`TCOLEXPANDADD` lowers to a per-row vector-add sequence in which the per-column scalar vector `src1` is broadcast across all `R` rows of `src0`.

**Cycle model**:

```
total = startup + R × (per_row_vadd + interval)
```

where `R = dst.GetValidRow()` and `per_row_vadd` is the cost of one row-wide `vadd` over `C = dst.GetValidCol()` elements.

### Instruction Sequence by Shape (FP32)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|----------------------|------------------|
| 16×16 | `vadd`*16 → PIPE_V | ~O(64) |
| 32×32 | `vadd`*32 → PIPE_V | ~O(128) |
| 64×64 | `vadd`*64 → PIPE_V | ~O(256) |
| R×C | `vadd`*R → PIPE_V | ~O(R × ⌈C/vlen⌉) |

### Layout and Shape Impact

| Layout | validCol | Optimization |
|--------|----------|--------------|
| `RowMajor` | ≥ vlen (FP32) | Continuous fast path; one `vadd` per row |
| `RowMajor` | < vlen | General path with tail masking |
| Other | any | Not supported (compile-time `isRowMajor` constraint) |

Broadcast of `src1` is free per row: the per-column scalar vector is reused across all rows without re-fetching.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.tcolexpandadd` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

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
  // Col-expand-add: each column of dst = src0.col + src1.col_scalar
  TCOLEXPANDADD(dst, src0, src1);
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
  // Col-expand-add in manual mode
  TCOLEXPANDADD(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.tcolexpandmul](./tcolexpandmul.md)
- Next op in instruction set: [pto.tcolexpandmax](./tcolexpandmax.md)
