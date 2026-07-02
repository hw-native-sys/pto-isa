# pto.trowexpandmul

`pto.trowexpandmul` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.

## Mechanism

Row-wise broadcast multiply: multiply each row of `src0` by a per-row broadcast operand.

### Mode 1 — Scalar per row (ColMajor expanded operand)

![TROWEXPANDMUL Mode 1 tile operation](../../../../figures/isa/TROWEXPANDMUL.svg)

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from the expanded operand.

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot s_i
$$

### Mode 2 — 32-byte block per row (RowMajor expanded operand)

![TROWEXPANDMUL Mode 2 tile operation](../../../../figures/isa/TROWEXPANDMUL_mode2.svg)

Let `b_i` be the 32-byte block for row `i` taken from the expanded operand. The block contains `32 / sizeof(T)` values and repeats across the row.

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,]
$$

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
    - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`.

    - `TileDataDst::DType`, `TileDataSrc0::DType`, and `TileDataSrc1::DType` must be one of: `half`, `float`, `int16`, `int32` for A2, A3, and A5; `uint16`, `uint32` for A5.

    - `TileDataDst` must be RowMajor.

    - Exactly one of `src0` or `src1` must have the same valid shape as `dst`. That operand is the full-sized operand; the other operand is the expanded row-broadcast operand.

    - The full-sized operand must be RowMajor.

    - Mode 1: the expanded operand is ColMajor, has valid column count 1, and has valid row count equal to `dst.GetValidRow()`.

    - Mode 2: the expanded operand is RowMajor, has valid row count equal to `dst.GetValidRow()`, and has valid column count `32 / sizeof(T)`: 16 for 16-bit types and 8 for 32-bit types.

    - Exact layout, fractal, and alignment constraints are target-specific; see backend headers under `include/pto/npu/*/TRowExpand*.hpp`.


### Temporary tile

The C++ API provides an overload with an explicit `TileDataTmp &tmp`. This overload only supports **Mode 1** (ColMajor expanded operand, scalar per row).

- **A2A3**: The tmp tile is used as a broadcast buffer. The per-row scalar values from the ColMajor expanded operand are broadcast via the `vbrcb` instruction into the tmp buffer, creating a 32-byte block per row, which is then used as the expanded operand in the binary operation. The `vbrcb` instruction uses a repeat stride of 8 blocks (256 bytes) between repeat groups, processing 8 rows per repeat. Minimum tmp size calculation:
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
    - **Implementation checks**:
        - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType` (compile-time).
        - `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`, `int16`, `int32` for A2, A3, and A5; `uint16`, `uint32` for A5.
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
# Manual mode: resources must be bound explicitly before issuing the instruction.
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

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in instruction set: [pto.trowexpanddiv](./trowexpanddiv.md)
- Next op in instruction set: [pto.trowexpandsub](./trowexpandsub.md)
