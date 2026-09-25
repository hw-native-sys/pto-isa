# TINSERT


## Tile Operation Diagram

![TINSERT tile operation](../figures/isa/TINSERT.svg)

## Introduction

Insert a source sub-tile into a destination tile at `(indexRow, indexCol)`. Conceptually the inverse of `TEXTRACT`.

`TINSERT` is used for:

- Acc → Mat insertion (with optional relu, scalar-quant, or vector-quant)
- Acc → Vec insertion (with optional `AccToVecMode`, relu, scalar-quant, or vector-quant) *(A5)*
- Vec → Mat insertion (ND, NZ, and ZN layouts) *(A5)*
- Vec → Vec insertion (ND and NZ layouts) *(A5)*
- NZ split insertion (`SPLIT2`, `SPLIT4`), also exposed in CPU simulation

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## Assembly Syntax

Synchronous form:

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
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

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
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

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90) || \
    defined(PTO_NPU_ARCH_KIRINDEV0000) || defined(__CPU_SIM)
template <TInsertMode mode, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &... events);
#endif

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);
```

The `STPhase` overloads set the unit flag on the L0C move-out instruction so it pairs with
`TMATMUL<AccPhase>` for Cube-to-Fixpipe hardware synchronization, removing the explicit
`set_flag`/`wait_flag` pair. They are exposed only on targets with matching backend support
(A2A3, Ascend 950PR/Ascend 950DT and the CPU simulator); the applicable paths and the pairing rule
are listed under the implementation checks below.

`TINSERT_FP(...)` is retained for source compatibility with the legacy fp-quantized form and maps directly
to the no-`mode` `TINSERT_IMPL(dst, src, fp, indexRow, indexCol)` path. The canonical
`TINSERT(..., fp, ...)` overload is selected only for `FpTileData::Loc == TileType::Scaling`.
`TINSERT_FP` also has an `STPhase` form, aligned with `TMOV_FP` / `TSTORE_FP`; the semantics match the
canonical overload.

## Constraints

### General constraints / checks

- The `STPhase` overloads (unit flag) apply only to the `TileType::Acc -> TileType::Mat` (L0C to L1)
  path; a `TileType::Vec` destination is rejected at compile time. They are exposed on A2A3,
  Ascend 950PR/Ascend 950DT and the CPU simulator.
- The move-out values do not mirror the accumulation values. The `TMATMUL` that produced the L0C result
  must already be `AccPhase::Final`, that is, the data is ready. `STPhase::Final` marks the last move-out
  and releases the unit flag. `STPhase::Partial` is only for the non-last move-outs when one L0C tile is
  drained more than once; it does not release the flag. Pairing `STPhase::Partial` with
  `AccPhase::Partial` makes the fixpipe wait on a flag that never arrives, which hangs; this was
  reproduced on the Ascend 950PR simulator.
- Acc-to-Mat move-out has passed A3 hardware tests covering `STPhase::Final` and
  `STPhase::Partial` followed by `STPhase::Final` in `tmov_acc2mat`.
  Ascend 950PR simulator tests have passed. On Ascend 950PR hardware with CANN 9.2.0, all 4 targeted
  `tinsert` tests (`TInsertTest.case_acc2mat_*`) passed with max diff 0, covering the baseline,
  `Final`, and `Partial` followed by `Final`.
- `TINSERT` has these overload families:
    - plain insert: `TINSERT(dst, src, indexRow, indexCol)`
    - relu form: `TINSERT<..., reluMode>(dst, src, indexRow, indexCol)`
    - accumulator-to-vector form: `TINSERT<..., mode, reluMode>(dst, src, indexRow, indexCol)`
    - scalar-quant form: `TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` and `TINSERT<..., mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol)`
    - vector-quant form: canonical `TINSERT<..., FpTileData, reluMode>(dst, src, fp, indexRow, indexCol)`,
      legacy `TINSERT_FP<..., reluMode>(dst, src, fp, indexRow, indexCol)`, and target-supported
      `TINSERT<..., FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol)` Acc-to-Vec routing
    - NZ split form *(A5, kirin9030, kirinX90, kirinDev0000, and CPU simulator)*: `TINSERT<TInsertMode::SPLIT2>(dst, src, indexRow, indexCol)` or `TINSERT<TInsertMode::SPLIT4>(...)`
- `reluMode` is `ReluPreMode::{NoRelu, NormalRelu}`.
- `mode` is `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
  The vector-quantized `fp + mode` overload is exposed only on targets with matching backend support
  (A5, kirin9030, kirinX90, and CPU simulator).
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
- **Vector-quant** (`TINSERT` with `FpTileData`; legacy `TINSERT_FP` alias) supported dtype pairs:
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `int16_t`
- The canonical vector-quant overload requires an `FpTileData` scaling operand (`TileType::Scaling`);
  `TINSERT_FP(...)` keeps legacy source compatibility and is checked by the selected backend implementation.

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
    - **Vector-quant** (`TINSERT` with `FpTileData`; legacy `TINSERT_FP` alias) destination types: same as scalar-quant above.

- **Acc → Vec** (`TileType::Acc → TileType::Vec`):
    - Source Acc type must be `float` or `int32_t`; source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
    - **Non-quant** (plain / relu) destination types:
        - `float` Acc → `half`, `bfloat16_t`, `float`
        - `int32_t` Acc → `int32_t`
    - **Scalar-quant** destination types:
        - `float` Acc → `int8_t`, `uint8_t`, `hifloat8_t`, `half`, `bfloat16_t`, `float8_e4m3_t`
        - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `bfloat16_t`
    - **Vector-quant** (`TINSERT` with `FpTileData`; legacy `TINSERT_FP` alias) destination types: same as scalar-quant above.
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
    - ZN path: source and destination must both be `(isRowMajor, SFractal: ColMajor)`; uses `ComputeZNBlockParams` for fractal-block `copy_ubuf_to_cbuf`. ZN is the transpose-dual of NZ: rows are the fractal dimension and columns are the free dimension. `indexRow` and `validRow` must be aligned to the fractal row size (`BLOCK_BYTE_SIZE / sizeof(T)`); `indexCol` and `validCol` are unconstrained (free dimension), matching the NZ row dimension. For fp4 types (`float4_e2m1x2_t`, `float4_e1m2x2_t`), validRow and indexRow are halved for byte addressing.

- **NZ Split** (`TInsertMode::SPLIT2` / `TInsertMode::SPLIT4`, A5 behavior, also modeled on CPU):
    - Destination must be `TileType::Mat`; source must be `TileType::Vec`.
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Source must be NZ format: `(!isRowMajor, SFractal: RowMajor)`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - `validRow` is aligned up to `FRACTAL_NZ_ROW` (16) for burst calculation.
    - Splits `copy_ubuf_to_cbuf` into 2 or 4 sub-transfers. The first `SplitCount - 1` each handle
      `totalBurstNum / SplitCount` column blocks; the last handles all remaining blocks.
    - The physical copy covers complete 32-byte column blocks and `ceil16(validRow)` rows, including
      row and column tails outside the logical valid window. These tails must be backed by source and destination storage.
    - Source block pitch in rows is `SrcTileData::Rows` for `CompactMode::Null`,
      `ceil16(validRow) + 1` for `RowPlusOne`, and `ceil16(validRow)` for the other modes
      (`Normal` and `RowAlignedPadding`).
      Destination block pitch is `DstTileData::Rows`; multiply either pitch by 32 to obtain bytes.
    - FP4 column counts and offsets are logical element counts; two elements share one byte.
    - CPU_SIM follows the same type, layout, and data-transfer rules described above.

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

Unit-flag L0C to L1 move-out (A2A3, Ascend 950PR/Ascend 950DT):

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_unit_flag() {
  TileLeft<half, 32, 32> a;
  TileRight<half, 32, 32> b;
  TileAcc<float, 32, 32> c;
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor> l1;
  TASSIGN(a, 0x0);
  TASSIGN(b, 0x0);
  TASSIGN(c, 0x0);
  TASSIGN(l1, 0x2000);
  // Move out once the data is ready; no explicit set_flag/wait_flag is needed between the two.
  TMATMUL<AccPhase::Final>(c, a, b);
  TINSERT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // Draining one L0C tile more than once: Partial on the non-last move-outs, Final on the last.
  // TINSERT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TINSERT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
