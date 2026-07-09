# TINSERT


## Tile Operation Diagram

![TINSERT tile operation](../figures/isa/TINSERT.svg)

## Introduction

Insert a source sub-tile into a destination tile at `(indexRow, indexCol)`. Conceptually the inverse of `TEXTRACT`.

`TINSERT` is used for:

- Acc → Mat insertion (with optional relu, scalar-quant, or vector-quant)
- Acc → Vec insertion (with optional `AccToVecMode`, relu, scalar-quant, or vector-quant) *(A5)*
- Vec → Mat insertion (ND and NZ layouts) *(A5)*
- Vec → Vec insertion (ND and NZ layouts) *(A5)*
- NZ split insertion (`SPLIT2`, `SPLIT4`) *(A5)*

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src,
                                FpTileData &fp,
                                uint16_t indexRow, uint16_t indexCol,
                                WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &... events);
#endif
```

## Constraints

### General constraints / checks

- `TINSERT` has these overload families:
    - plain insert: `TINSERT(dst, src, indexRow, indexCol)`
    - relu form: `TINSERT<..., reluMode>(dst, src, indexRow, indexCol)`
    - accumulator-to-vector form: `TINSERT<..., mode, reluMode>(dst, src, indexRow, indexCol)`
    - scalar-quant form: `TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` and `TINSERT<..., mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol)`
    - vector-quant form: `TINSERT_FP<..., reluMode>(dst, src, fp, indexRow, indexCol)` and `TINSERT<..., FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol)`
    - NZ split form *(A5 only)*: `TINSERT<TInsertMode::SPLIT2>(dst, src, indexRow, indexCol)` or `TINSERT<TInsertMode::SPLIT4>(...)`
- `reluMode` is `ReluPreMode::{NoRelu, NormalRelu}`.
- `mode` is `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
- Runtime bounds: `indexRow + src.ValidRow <= dst.Rows` and `indexCol + src.ValidCol <= dst.Cols`.

### A2A3 implementation checks

- Supported tile-type pair: `TileType::Acc → TileType::Mat` only.
- Source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
- Destination layout must be `(BFractal: ColMajor, SFractal: RowMajor)` with `SFractalSize == 512`.
- `Dst.Cols * sizeof(DstDType)` must be a multiple of `32` bytes and non-zero.
- **Plain / relu** (non-quant) supported dtype pairs:
    - `float` Acc → `half`, `bfloat16_t`
- **Scalar-quant** supported dtype pairs:
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `int16_t`
- **Vector-quant** (`TINSERT_FP`) supported dtype pairs:
    - `float` Acc → `int8_t`, `uint8_t`
    - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `int16_t`
- Vector-quant requires an `FpTileData` scaling operand (`TileType::Scaling`).

### A5 implementation checks

- In addition to the Acc → Mat path, A5 supports Acc → Vec, Vec → Vec, Vec → Mat, and NZ split paths.

- **Acc → Mat** (`TileType::Acc → TileType::Mat`):
    - Source Acc type must be `float` or `int32_t`; source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
    - Destination layout must be `(!isRowMajor, SFractal: RowMajor)` (NZ format).
    - **Non-quant** (plain / relu) destination types:
        - `float` Acc → `half`, `bfloat16_t`, `float`
        - `int32_t` Acc → `int32_t`
    - **Scalar-quant** destination types:
        - `float` Acc → `int8_t`, `uint8_t`, `hifloat8_t`, `half`, `bfloat16_t`, `float8_e4m3_t`
        - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `bfloat16_t`
    - **Vector-quant** (`TINSERT_FP`) destination types: same as scalar-quant above.

- **Acc → Vec** (`TileType::Acc → TileType::Vec`):
    - Source Acc type must be `float` or `int32_t`; source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
    - **Non-quant** (plain / relu) destination types:
        - `float` Acc → `half`, `bfloat16_t`, `float`
        - `int32_t` Acc → `int32_t`
    - **Scalar-quant** destination types:
        - `float` Acc → `int8_t`, `uint8_t`, `hifloat8_t`, `half`, `bfloat16_t`, `float8_e4m3_t`
        - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `bfloat16_t`
    - **Vector-quant** (`TINSERT_FP` / `TINSERT` with `FpTileData`) destination types: same as scalar-quant above.
    - Destination layout must be one of: NZ-to-NZ (`!isRowMajor, SFractal: RowMajor`), NZ-to-ND (`isRowMajor, SFractal: NoneBox`), or NZ-to-DN (`!isRowMajor, SFractal: NoneBox`).
    - `AccToVecMode` selects `SingleModeVec0`, `SingleModeVec1`, `DualModeSplitM`, or `DualModeSplitN`.
    - Dual-destination modes (`DualModeSplitM`, `DualModeSplitN`) require `QuantMode_t::NoQuant` and do not support the NZ-to-DN path.
    - For 32-bit destination types (`float`/`int32_t`), when using `DualModeSplitN` the `ValidCol` (before the split) must be a multiple of `32`.
    - Destination stride must be non-zero and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes.

- **Vec → Vec** (`TileType::Vec → TileType::Vec`):
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - Source and destination layout must match (both ND or both NZ).
    - ND path: source valid region must fit within destination bounds. Dispatch selects `copy_ubuf_to_ubuf` (aligned), `vlds`/`vsts` (stride-aligned, unaligned validCol), `vlds`/`vstus` (unaligned strides or indexCol), or scalar copy (1×1 element).
    - NZ path: source cols must not exceed destination cols. Uses `ComputeNZBlockParams` for fractal-block `copy_ubuf_to_ubuf`.

- **Vec → Mat** (`TileType::Vec → TileType::Mat`, UB → L1):
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - ND path: source must be `isRowMajor`; uses `copy_ubuf_to_cbuf`. Data bytes per row must be aligned to `BLOCK_BYTE_SIZE` (32 bytes) for row-wise burst.
    - NZ path: source must be `(!isRowMajor, SFractal: RowMajor)`; uses `ComputeNZBlockParams` for fractal-block `copy_ubuf_to_cbuf`. For fp4 types (`float4_e2m1x2_t`, `float4_e1m2x2_t`), validCol and indexCol are halved for byte addressing.

- **NZ Split** (`TInsertMode::SPLIT2` / `TInsertMode::SPLIT4`, A5 only):
    - Destination must be `TileType::Mat`; source must be `TileType::Vec`.
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Source must be NZ format: `(!isRowMajor, SFractal: RowMajor)`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - `validRow` is aligned up to `FRACTAL_NZ_ROW` (16) for burst calculation.
    - Splits the `copy_ubuf_to_cbuf` total burst into 2 or 4 sub-transfers, each handling `totalBurstNum / SplitCount` column blocks (last sub-transfer takes the remainder).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat insertion (NZ layout)
void example_auto() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat insertion (NZ layout, manual buffer assignment)
void example_manual() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dst, 0x0);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```
