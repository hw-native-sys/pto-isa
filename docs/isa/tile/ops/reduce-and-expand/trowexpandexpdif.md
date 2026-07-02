# pto.trowexpandexpdif

`pto.trowexpandexpdif` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars.

## Mechanism

Row-wise exp-diff: compute `exp(src0 - src1)` where `src1` provides one scalar per row.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (the tile to be modified).
- `src1` is the second source tile providing per-row scalar values.
- `tmp` (optional): temporary tile for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst[i,j]` = exp(`src0[i,j]` - `src1[i,0]`) (row-wise exp-diff of per-row scalar).

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`

    - `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.

    - Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.

    - Mode 1: `src1` is expected to provide **one scalar per row** (i.e., its valid shape must cover `R` values).

    - Mode 2: `src1` is expected to provide **32 bytes data per row**.

    - Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TRowExpand*.hpp`.


### Temporary tile

The C++ API provides an overload with an explicit `TileDataTmp &tmp`. This overload only supports **Mode 1** (ColMajor expanded operand, scalar per row). Internally, `TROWEXPANDEXPDIF` is implemented as `TROWEXPANDSUB` followed by `TEXP`, so the tmp tile is used for the SUB step broadcast buffer.

- **A2A3**: The tmp tile is used as a broadcast buffer. The per-row scalar values from the ColMajor expanded operand are broadcast via the `vbrcb` instruction into the tmp buffer, creating a 32-byte block per row, which is then used as the expanded operand in the subtraction. The `vbrcb` instruction uses a repeat stride of 8 blocks (256 bytes) between repeat groups, processing 8 rows per repeat. Minimum tmp size calculation:
    - Common parameters: `R = dst.GetValidRow()`, `T = TileDataDst::DType`.
    - For `R < 256`:
      $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{ bytes} $$
    - For `R >= 256`, the operation is looped with at most 30 repeats (240 rows) per loop iteration. The tmp buffer is reused across loops, so the per-loop requirement is:
      $$ \text{tmpSize} = 30 \times 256 = 7680 \text{ bytes} $$
    - A compact shape-independent upper bound for any Mode 1 invocation is **8 KB** (8192 bytes).
    - The 3-arg overload without `tmp` supports both Mode 1 and Mode 2. For Mode 1, it uses an internal 8 KB buffer (`TMP_UB_OFFSET`). For Mode 2, no broadcast buffer is needed.
- **A5**: The `tmp` tile is accepted and ignored (`[[maybe_unused]]`). A5 hardware supports row-broadcast natively via the `vlds` instruction's broadcast modes, so no scratch buffer is required.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.trowexpandexpdif` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

    - Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using RowVecT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  SrcT src0;
  DstT dst;
  RowVecT src1;
  // Row-expand-expdif: each row of dst = exp(src0.row - src1.row_scalar)
  TROWEXPANDEXPDIF(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using RowVecT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  SrcT src0;
  DstT dst;
  RowVecT src1;
  TASSIGN(src0, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(src1, 0x3000);
  // Row-expand-expdif in manual mode
  TROWEXPANDEXPDIF(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.trowexpandmin](./trowexpandmin.md)
- Next op in instruction set: [pto.tcolmin](./tcolmin.md)
