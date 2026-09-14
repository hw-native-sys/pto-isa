# TEXTRACT


## Tile Operation Diagram

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## Introduction

Extract a smaller sub-tile from a larger source tile.

## Math Interpretation

Conceptually copies a smaller window starting at `(indexRow, indexCol)` from the larger `src` tile into `dst`. Exact mapping depends on tile layouts.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## Assembly Syntax

Synchronous form:

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.textract ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                              uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename Dst0TileData, typename Dst1TileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(Dst0TileData &dst0, Dst1TileData &dst1, SrcTileData &src,
                              uint16_t indexRow0 = 0, uint16_t indexCol0 = 0,
                              uint16_t indexRow1 = 0, uint16_t indexCol1 = 0, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

The `STPhase` overloads set the unit flag on the L0C move-out instruction so it pairs with
`TMATMUL<AccPhase>` for Cube-to-Fixpipe hardware synchronization, removing the explicit
`set_flag`/`wait_flag` pair. They are exposed only on targets with matching backend support
(A2A3, Ascend 950PR/Ascend 950DT and the CPU simulator); the applicable paths and the pairing rule
are listed under the implementation checks below.

`TEXTRACT_FP(...)` is retained for source compatibility with the legacy fp-quantized form and maps directly
to the no-`mode` `TEXTRACT_IMPL(dst, src, fp, indexRow, indexCol)` path. The canonical
`TEXTRACT(..., fp, ...)` overload is selected only for `FpTileData::Loc == TileType::Scaling`.
The canonical interface also provides an explicit `AccToVecMode` form for target-supported Acc-to-Vec routing.
`TEXTRACT_FP` also has an `STPhase` form, aligned with `TMOV_FP` / `TSTORE_FP`; the semantics match the
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
  Ascend 950PR simulator tests have passed. On Ascend 950PR hardware with CANN 9.2.0, all 7 targeted
  `textract` tests (`case1` and `case21`–`case26`) passed with max diff 0, covering NZ512/NZ1024,
  `Final`, `Partial` followed by `Final`, `Unspecified`, and K-split accumulation.
- For same-dtype extraction/layout paths, `DstTileData::DType` must equal `SrcTileData::DType`.
  Acc conversion and quantized paths use the target-specific dtype pairs below.
- Runtime bounds checks:
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### A2A3 implementation checks

- Supported element types: `int8_t`, `half`, `bfloat16_t`, `float`.
- Source layout must satisfy one of the checked A2A3 extraction layouts:
    - `(SFractal == ColMajor && isRowMajor)`, or
    - `(SFractal == RowMajor && !isRowMajor)`.
- In GEMV scenarios targeting `TileType::Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- Destination must be `TileType::Left` or `TileType::Right` with a target-supported fractal configuration.

### A5 implementation checks

- Supported element types: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`.
- Source layout must satisfy one of the checked A5 extraction layouts:
    - for `Left` / `Right`: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`
    - for `ScaleLeft`: `(SFractal == RowMajor && isRowMajor)`
    - for `ScaleRight`: `(SFractal == ColMajor && !isRowMajor)`
- In GEMV scenarios targeting `Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- Destination supports `TileType::Mat -> TileType::Left/Right/Scale`, `TileType::Acc -> TileType::Mat` (including relu, scalar-quant, and vector-quantized forms), `TileType::Acc -> TileType::Vec`, and specific `TileType::Vec -> TileType::Mat` extraction paths.
- The canonical vector-quantized `TEXTRACT(..., fp, ...)` form additionally requires an `FpTileData`
  scaling operand. `TEXTRACT_FP(...)` remains available as a source-compatible legacy alias and is
  checked by the selected backend implementation.
- The vector-quantized Acc-to-Vec form is exposed only on targets with matching backend support
  (A5, kirin9030, kirinX90, and CPU simulator). It accepts
  `mode = AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
- For `TileType::Acc -> TileType::Vec` with a 32-bit destination type (`float`/`int32_t`), when using `DualModeSplitN` the `ValidCol` (before the split) must be a multiple of `32`.

### Vec → Vec extraction path

In addition to the `Mat/Acc -> ...` paths above, `TEXTRACT` supports a `TileType::Vec -> TileType::Vec` extraction path (ND and NZ layouts). A2A3 uses `CheckTExtractVecToVecCommon`; A5 checks this separately in `TEXTRACT_IMPL`:

- `DstTileData::DType` must equal `SrcTileData::DType`.
- A2A3 element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- A5 element types: `int8_t`, `int32_t`, `half`, `bfloat16_t`, `float`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`. A5 does not support `uint8_t`, `int16_t`, `uint16_t`, `uint32_t` or 64-bit integers on this path.
- A5 ND fp4 counts packed elements (one byte contains two fp4 values); row strides, static valid-column bytes and column-offset bytes must be 32-byte aligned.
- ND path: source/destination row strides must be 32-byte aligned; `Dst` rows/cols must not exceed `Src`.
- A5 ND Vec-to-Vec checks `indexRow + dst.GetValidRow() <= SrcTileData::Rows` and `indexCol + dst.GetValidCol() <= SrcTileData::Cols` first. After those checks, a zero destination valid row or column count returns without reading the source or writing the destination. This applies to aligned and unaligned column offsets; it does not extend dtype support to int64.

### ND → 2×NZ extraction path

The two-destination `TEXTRACT` overload extracts two independent ND sub-windows from a single ND source and writes each as a separate NZ destination in one call. It is implemented entirely with vector-frontend intrinsics (no MTE copy).

- Source must be a `TileType::Vec` ND tile (`BLayout::RowMajor`, `SLayout::NoneBox`); both destinations must be `TileType::Vec` NZ tiles (`BLayout::ColMajor`, `SLayout::RowMajor`).
- `DstTileData::DType` must equal `SrcTileData::DType`.
- Each window is placed by its own `(indexRow, indexCol)`. Runtime bounds checks per window `k`:
    - `indexRow_k + dst_k.GetValidRow() <= SrcTileData::Rows`
    - `indexCol_k + dst_k.GetValidCol() <= SrcTileData::Cols`
- Structural constraints (same as the Vec → Vec paths): destination `Cols` must be `c0`-aligned (NZ fractal width), and source row-stride bytes must be 32-byte aligned.
- Supported element types:
    - A5: `int8_t`, `half`, `bfloat16_t`, `float`, `int32_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - A2A3: `int8_t`, `half`, `bfloat16_t`, `float`, `int32_t`.
- Output compact mode:
    - A5 supports plain NZ (default) and the NZ+1 bank-conflict optimization (`CompactMode::RowPlusOne`).
    - A2A3 supports plain NZ only.

- Index alignment (a window's source base is `srcStart = src + indexRow*rowStride + indexCol`):
    - A5 (SIMD) handles a `c0`-unaligned `indexCol` (sub-`c0` column origin)
    via an element-exact unaligned load/store path; `c0`-aligned windows take
    the faster block path.
    - A2A3 (vec-core) vector engines require the operand base to be 32-byte
    aligned, and `dav-c220-vec` has no unaligned vector load (`vlds`/`vsts`
    are unavailable). A window therefore takes the vector path only when its
    source base is 32-byte aligned, i.e. `indexCol * sizeof(T)` is a multiple
    of 32. Windows whose `indexCol` does not satisfy this (and `1×1` windows)
    use an element-wise scalar copy, which has no alignment constraint.
- A2A3 vector paths (32-byte-aligned source base): `vcopy` reinterprets data
at 16-bit granularity (its smallest element width; there is no 8-bit
`vcopy`). 2-/4-byte types and `int8` with an even `validCol` map directly
through `vcopy`. `int8` with an **odd** `validCol` (odd byte count) uses a
fully vector widen path — `vconv_s82f16` (int8→half) into a scratch, the
ND→NZ reshape in `half`, then `vconv_f162s8` (half→int8) into the NZ
destination (all `int8` values round-trip losslessly through `half`).

| Arch | Mode | Implementation |
|------|------|----------------|
| A5 / A2A3 | `1×1` | scalar copy |
| A5 (SIMD) | `c0`-aligned `indexCol` | `vlds` + `vsstb` |
| A5 (SIMD) | `c0`-unaligned `indexCol` | `vldas` + `vldus` + `vsts` |
| A2A3 (vec-core) | unaligned source base not 32-byte aligned (`indexCol*sizeof(T) % 32 != 0`) | scalar copy |
| A2A3 (vec-core) | aligned base, 2-/4-byte or even-`validCol` `int8` | `vcopy` with 16-bit reinterpretation |
| A2A3 (vec-core) | aligned base, odd-`validCol` `int8` | `vconv_s82f16` + `vconv_f162s8` widen path |

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
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
  TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // Draining one L0C tile more than once: Partial on the non-last move-outs, Final on the last.
  // TEXTRACT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
