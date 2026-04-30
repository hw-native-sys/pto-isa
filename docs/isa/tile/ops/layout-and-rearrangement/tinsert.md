# pto.tinsert

`pto.tinsert` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

`TINSERT` writes a source tile into a sub-region of a destination tile, starting at a configurable `(indexRow, indexCol)` offset. It is the natural counterpart to `TEXTRACT` — where `TEXTRACT` reads out of a source, `TINSERT` writes into a destination.

Two variants are documented here:

| Variant | Suffix | Description | Typical Use |
|---------|---------|-------------|-------------|
| Standard insert | *(none)* | Insert a sub-window with optional ReLU | `Acc→Mat` with row/col offsets |
| Fix-pipe insert | `_fp` | Insert through the fix-pipe quantization path | `Acc→Mat` with per-channel scaling |

## Mechanism

Conceptually copies the source tile's valid region into a window of `dst` starting at `(indexRow, indexCol)`. For `0 <= i < SrcRows` and `0 <= j < SrcCols`:

$$ \mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j} $$

**Fix-pipe variant** (`TINSERT_FP`): Routes through the hardware fix-pipe quantization pipeline. The `fp` tile carries per-channel quantization parameters:

$$ \mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Variants

### Variant 1: Standard Insert

`TINSERT(dst, src, indexRow, indexCol)` — plain sub-tile insertion.

### Variant 2: ReLU Insert

`TINSERT<..., reluMode>(dst, src, indexRow, indexCol)` — insert with ReLU pre-processing.

### Variant 3: Scalar-Quant Insert

`TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` — insert with a scalar quantization parameter.

### Variant 4: Fix-Pipe Insert (`TINSERT_FP`)

`TINSERT_FP(dst, src, fp, indexRow, indexCol)` — insert through the fix-pipe quantization path. The `_fp` suffix means **fix pipe**, not floating point.

### Variant 5: A5 Mode-Specific Insert

On A5 only, an additional template parameter selects specialized insertion modes:

```cpp
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint32_t indexRow = 0, uint32_t indexCol = 0, WaitEvents &... events);
```

`TInsertMode` values:

| Mode | Source Tile | Destination Tile | Notes |
|------|------------|-----------------|-------|
| `ND` | Row-major `Vec` | Matrix in ND layout | `Vec → Mat` with ND layout |
| `ND_VEC` | Row-major `Vec` | Row-major `Vec` | `Vec → Vec` in-row insertion |
| `NZ` | NZ-format `Vec` | Matrix | Inserts NZ vector into matrix tile |
| `NZ_PLUS_1` | NZ-format `Vec` | Matrix | Like `NZ` with +1 offset |
| `SPLIT2_NZ_PLUS_1` | NZ-format `Vec` | Matrix | Split-by-2 variant |
| `SPLIT4_NZ_PLUS_1` | NZ-format `Vec` | Matrix | Split-by-4 variant |

## Supported Tile-Type Pairs

### A2/A3

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix insertion |

### A5

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix insertion |
| `TileType::Vec` | `TileType::Mat` | Vector-to-matrix via `mode` parameter |
| `TileType::Vec` | `TileType::Vec` | Vector-to-vector via `mode` parameter |

## Supported Element Types

`int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`

## Syntax

### PTO Assembly Form

Standard insert:

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

Fix-pipe insert:

```text
%dst = tinsert.fp %src, %fp[%r0, %r1] : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
// Standard
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>

// Fix-pipe
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
// Standard
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)

// Fix-pipe
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Variant 1: Plain insert
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 2: ReLU insert
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 3: Scalar-quant insert
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 4: Fix-pipe insert (TINSERT_FP)
template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                               uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

// Variant 5: A5 mode-specific insert
#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint32_t indexRow = 0, uint32_t indexCol = 0, WaitEvents &... events);
#endif
```

## Inputs

- `src` — the source tile to insert.
- `indexRow` — starting row offset in `dst` where insertion begins.
- `indexCol` — starting column offset in `dst` where insertion begins.
- `dst` — the destination tile. The operation writes `src` into the `(indexRow, indexCol)` sub-region of `dst`.
- `fp` (fix-pipe variant only) — auxiliary fix-pipe tile. Must be `TileType::Scaling`.
- `reluMode` (optional) — `ReluPreMode::{NoRelu, NormalRelu}`.
- `preQuantScalar` (scalar-quant variant only) — scalar quantization factor.

## Expected Outputs

| Result | Type | Description |
|---|---|---|
| `RecordEvent` | token | Signals completion of the insertion. |
| `dst` | tile | The `(SrcRows, SrcCols)` sub-region of `dst` starting at `(indexRow, indexCol)` is overwritten with `src` (or with quantized `src` for `TINSERT_FP`); the rest of `dst` is unchanged. |

## Side Effects

- **Standard variants**: No architectural side effects beyond updating the destination region.
- **Fix-pipe variant (`TINSERT_FP`)**: Programs the FPC sideband state before the fix-pipe quantization. On the CPU simulator the `fp` parameter is ignored and the call falls back to standard `TINSERT`.
- Does not implicitly fence unrelated tile traffic.

## Constraints

!!! warning "Constraints"
    - **Runtime bounds**: `indexRow + SrcTileData::Rows <= DstTileData::Rows` and `indexCol + SrcTileData::Cols <= DstTileData::Cols`
    - **Fp tile location**: `FpTileData::Loc` must be `TileType::Scaling` (enforced on both A2/A3 and A5)
    - **Fix-pipe destination**: On A2/A3, destination must be `TileType::Mat` with fractal size 512 and column-width byte count divisible by 32
    - **A5 fix-pipe**: Destination must be `TileType::Mat` with `BLayout::ColMajor + SLayout::RowMajor`; source must be `float` or `int32_t` `Acc`
    - **Cpu simulator**: `TINSERT_FP` accepts the interface but ignores the `fp` parameter, falling back to standard `TINSERT`

## Performance

### A2/A3 Cycle Count

`pto.tinsert` is a layout-rearrangement op that writes a sub-window into a larger destination tile. Standard variants execute as a layout-converting move within UB / L0 buffers; the fix-pipe variant (`TINSERT_FP`) routes through the **FIXP** path before the destination write.

**Cycle model**:

```
# Standard / ReLU / scalar-quant
total ≈ startup + SrcRows × SrcCols / insert_throughput

# Fix-pipe
total ≈ startup + fixp_drain(SrcRows, SrcCols)
```

### Layout and Shape Impact

| Source → Dest | Path | Notes |
|---|---|---|
| `Vec → Mat` | Layout-converting | Common Vec→Cube tile-passing path (TPipe) |
| `Acc → Mat` (A5) | Layout-converting | Used by output-stationary GEMM postprocessing |
| `Acc → Mat` (`TINSERT_FP`) | FIXP | FIXP-bound; A5 has ~4× read/write bandwidth asymmetry |

The `(indexRow, indexCol)` offset is free at issue time. The CPU simulator ignores the `fp` parameter and routes through the standard path.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported tile-type pairs, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Out-of-range `(indexRow, indexCol)` (window exceeds `dst` bounds) is rejected at compile time when shapes are static, otherwise undefined at runtime.
    - `TINSERT_FP` with `FpTileData::Loc != TileType::Scaling` is rejected by `static_assert`.
    - On A5, `TINSERT_FP` requires destination `TileType::Mat` with `BLayout::ColMajor + SLayout::RowMajor`; other layouts are rejected.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

### Pattern 1: Accumulator Insert into Matrix

```cpp
// Insert a small accumulator tile into a larger matrix at a specific position
using AccT = TileAcc<float, 16, 16>;
using MatT = Tile<TileType::Mat, int8_t, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

AccT acc;
MatT mat;
TASSIGN(acc, 0x1000);
TASSIGN(mat, 0x2000);

// Insert the 16x16 accumulator tile into the matrix at row=8, col=16
TINSERT(mat, acc, /*indexRow=*/8, /*indexCol=*/16);
```

### Pattern 2: Fix-Pipe Quantized Insert

```cpp
// Accumulator tile quantized via fix-pipe and inserted into matrix at offset
using AccT = TileAcc<float, 16, 16>;
using MatT = Tile<TileType::Mat, int8_t, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
using FpT = Tile<TileType::Scaling, uint64_t, 1, 16>;

AccT acc;
MatT mat;
FpT fp(16);
TASSIGN(acc, 0x1000);
TASSIGN(mat, 0x2000);
TASSIGN(fp, 0x3000);

// Insert with fix-pipe quantization applied
TINSERT_FP(mat, acc, fp, /*indexRow=*/0, /*indexCol=*/0);
```

### Pattern 3: Accumulator Scatter via Staged Inserts

```cpp
// Scatter multiple small accumulator results into a large output matrix
using AccT = TileAcc<float, 16, 16>;
using MatT = Tile<TileType::Mat, float, 64, 64>;

MatT outMat;
TASSIGN(outMat, 0x4000);

// Fill 4 quadrants by repeated inserts from temporary accumulators
AccT accQ1, accQ2, accQ3, accQ4;
TASSIGN(accQ1, 0x1010);
TASSIGN(accQ2, 0x1020);
TASSIGN(accQ3, 0x1030);
TASSIGN(accQ4, 0x1040);

// Insert each 16x16 accumulator into the corresponding 32x32 quadrant
TINSERT(outMat, accQ1, 0,  0);   // Top-left
TINSERT(outMat, accQ2, 0,  32);  // Top-right
TINSERT(outMat, accQ3, 32, 0);   // Bottom-left
TINSERT(outMat, accQ4, 32, 32);  // Bottom-right
```

## See Also

- [Layout And Rearrangement](../../layout-and-rearrangement.md)
- [pto.textract](./textract.md) — the inverse operation (read from a source tile)
- [pto.tmov](./tmov.md) — full tile-to-tile movement including `TMOV_FP` fix-pipe variant
- [Assembly Spelling And Operands](../../../syntax-and-operands/assembly-model.md)
