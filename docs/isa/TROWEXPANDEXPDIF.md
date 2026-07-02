# TROWEXPANDEXPDIF


## Tile Operation Diagram

![TROWEXPANDEXPDIF tile operation](../figures/isa/TROWEXPANDEXPDIF.svg)

## Introduction

Row-wise exp-diff: compute `exp(src0 - src1)` where `src1` provides one scalar per row.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)
$$

## Assembly Syntax

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

## Constraints

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.
- Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.
- Mode 1: `src1` is expected to provide **one scalar per row** (i.e., its valid shape must cover `R` values).
- Mode 2: `src1` is expected to provide **32 bytes data per row**.
- Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TRowExpand*.hpp`.

### Temporary tile

The C++ API provides an overload with an explicit `TileDataTmp &tmp`. This overload only supports **Mode 1** (ColMajor expanded operand, scalar per row). Internally, `TROWEXPANDEXPDIF` is implemented as `TROWEXPANDSUB` followed by `TEXP`, so the tmp tile is used for the SUB step's broadcast buffer.

- **A2A3**: The tmp tile is used as a broadcast buffer for the `TROWEXPANDSUB` step. The per-row scalar values from the ColMajor expanded operand are broadcast via the `vbrcb` instruction into the tmp buffer, creating a 32-byte block per row, which is then used as the expanded operand in the subtraction. The `vbrcb` instruction uses a repeat stride of 8 blocks (256 bytes) between repeat groups, processing 8 rows per repeat. Minimum tmp size calculation:
    - **Common parameters**:
        - `R = dst.GetValidRow()`, `T = TileDataDst::DType`.
    - For `R < 256`:
        $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{ bytes} $$
    - For `R >= 256`:
        - The operation is looped, with at most 30 repeats (240 rows) per loop iteration. The tmp buffer is reused across loops, so the per-loop requirement is:
        $$ \text{tmpSize} = 30 \times 256 = 7680 \text{ bytes} $$
    - A compact shape-independent upper bound for any Mode 1 invocation is **8 KB** (8192 bytes).
    - The 3-arg overload (without `tmp`) supports both Mode 1 and Mode 2. For Mode 1, it uses an internal 8 KB buffer (`TMP_UB_OFFSET`). For Mode 2, no broadcast buffer is needed.
- **A5**: The `tmp` tile is accepted and ignored (`[[maybe_unused]]`). A5 hardware supports row-broadcast natively via the `vlds` instruction's broadcast modes, so no scratch buffer is required.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
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

