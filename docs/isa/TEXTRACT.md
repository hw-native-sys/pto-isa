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
- The legacy A2A3 Mat/Acc extraction paths check physical rectangular bounds
  (the small-M Mat-to-Left path below checks its valid window and full-fractal read extent separately):
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### A2A3 implementation checks

For Mat-to-Left/Right layout extraction:

- Supported element types: `int8_t`, `half`, `bfloat16_t`, `float`.
- Source layout must satisfy one of the checked A2A3 extraction layouts:
    - `(SFractal == ColMajor && isRowMajor)`, or
    - `(SFractal == RowMajor && !isRowMajor)`.
- In GEMV scenarios targeting `TileType::Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- Destination must be `TileType::Left` or `TileType::Right` with a target-supported fractal configuration.

### A5 implementation checks

- Supported element types: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`.
- Ordinary Acc-to-Mat extraction additionally accepts `int32_t -> int32_t`.
- Source layout must satisfy one of the checked A5 extraction layouts:
    - for `Left` / `Right`: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`
    - for `ScaleLeft`: `(SFractal == RowMajor && isRowMajor)`
    - for `ScaleRight`: `(SFractal == ColMajor && !isRowMajor)`
- In GEMV scenarios targeting `Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- Destination supports `TileType::Mat -> TileType::Left/Right/Scale`, `TileType::Acc -> TileType::Mat` (ReLU and quantized forms depend on dtype and layout; see the Acc-to-Mat NZ restrictions below), `TileType::Acc -> TileType::Vec`, and specific `TileType::Vec -> TileType::Mat` extraction paths.
- The canonical vector-quantized `TEXTRACT(..., fp, ...)` form additionally requires an `FpTileData`
  scaling operand. `TEXTRACT_FP(...)` remains available as a source-compatible legacy alias and is
  checked by the selected backend implementation.
- The vector-quantized Acc-to-Vec form is exposed only on targets with matching backend support
  (A5, kirin9030, kirinX90, and CPU simulator). It accepts
  `mode = AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
- For `TileType::Acc -> TileType::Vec` with a 32-bit destination type (`float`/`int32_t`), when using `DualModeSplitN` the `ValidCol` (before the split) must be a multiple of `32`.

### A5 Acc-to-Mat NZ layout conversion

The following paths use an NZ destination (`BLayout::ColMajor`, `SLayout::RowMajor`):

| Source → destination | Destination layout | Supported operation |
| --- | --- | --- |
| `float → half/bfloat16_t` | NZ1024, 32 columns per group | Plain conversion, with `NoRelu` or `NormalRelu`; scalar/vector quantization is rejected at compile time |
| `int32_t → int32_t` | NZ512, 8 columns per group | Bit-preserving move with `NoQuant` and `NoRelu`; ReLU is rejected at compile time |

Both the ordinary `TEXTRACT(dst, src, row, col)` and explicit `STPhase` forms reach these paths.
The ordinary form uses `STPhase::Unspecified` and requires explicit Cube-to-Fixpipe synchronization.
For ReLU, select the overload with an explicit `ReluPreMode` template argument.

The half/bfloat16 path emits Fixpipe NZ-to-ND instructions for groups of at most 32 columns.
Each group's destination row stride is 32 elements; its starting offset is
`group * DstTileData::Rows * 32` elements. A final 16-column group writes only those columns.
The int32 path uses the float instruction overload with channel splitting to produce 8-column groups;
it performs no numerical cast or floating-point arithmetic.

Callers must satisfy the following storage requirements:

- The source uses the ordinary 16-column L0C layout with `SrcTileData::Rows` as its physical stride;
  this path does not derive a compact stride from the source's valid row count.
- The L0C source window starts at a 64-byte-aligned address (`indexCol` is a multiple of 16 for
  an aligned ordinary L0C tile); the L1 destination address is 32-byte aligned.
- Valid rows and columns are positive and fit within the destination's physical shape.
  The complete source window must fit within allocated source storage.
- For half/bfloat16 NZ1024, valid columns must be a multiple of 16. The helper checks this and
  the source storage bounds with `PTO_ASSERT`, which is active only when `_DEBUG` is defined;
  these are not unconditional runtime checks, and source valid dimensions are not checked.
- For int32 NZ512, the instruction rounds valid columns up to a multiple of 8. The rounded window
  must fit within source and destination storage; trailing columns inside that block can be written.
  This branch does not add runtime bounds checks. Tests that require untouched column padding use
  valid column counts that are multiples of 8.

For half/bfloat16 NZ1024, the flag on each emitted instruction follows the outer phase:

| Outer phase | Non-last group | Last group |
| --- | --- | --- |
| `Final` | `Partial` | `Final` |
| `Partial` | `Partial` | `Partial` |
| `Unspecified` | `Unspecified` | `Unspecified` |

An outer `Partial` therefore requires a later `Final` extraction from the same accumulator.
The int32 path emits one instruction and forwards the outer phase unchanged.

`textract_acc2mat_layout` covers native NZ controls, phase sequencing, K accumulation, offsets,
partial valid shapes, padding and output guards, and representative special int32 bit patterns.
Cases 22–23 cover half/bfloat16 ReLU; cases 24–26 cover the ordinary int32/half/bfloat16 overloads.

For CPU simulation, when `TMATMUL` produces the source Acc tile from FP8/HIF8 inputs,
select A5 before binding Tiles. The input types and numerical constraints follow
[TMATMUL](TMATMUL.md#constraints); the source Acc tile contains `float`, not FP8/HIF8 elements.
See [Selecting the simulated architecture](../coding/cpu_sim.md#selecting-the-simulated-architecture).

### Small-M Mat-to-Left extraction (A2A3 and A5)

The ordinary `TEXTRACT(dst, src, indexRow, indexCol, events...)` overload automatically
selects `load_cbuf_to_ca` (L1 to L0A, also called load2d) for the following A2A3 and A5 cases.
No new mode or public overload is needed; the existing event interface is unchanged.
Use the overload without a `ReluPreMode` template argument. Explicitly specifying
`ReluPreMode::NoRelu` selects the Acc-to-Mat overload on A2A3, which does not support
this Mat-to-Left path.

| Property | Requirement |
| --- | --- |
| Element type | Source and destination both `half`, or both `bfloat16_t` |
| Source | `TileType::Mat`, NZ512: `BLayout::ColMajor`, `SLayout::RowMajor`, fractal size 512 |
| Source physical shape | Rows and columns are multiples of 16 |
| Destination on A2A3 | `TileType::Left`, ZZ512: `BLayout::RowMajor`, `SLayout::RowMajor`, fractal size 512 |
| Destination on A5 | `TileType::Left`, NZ512: `BLayout::ColMajor`, `SLayout::RowMajor`, fractal size 512 |
| Destination physical shape | Exactly 16 rows; columns are a multiple of 16 |
| Destination Compact mode | `CompactMode::Null` (ordinary `TileLeft`) or `CompactMode::Normal` (`TileLeftCompact`) |
| Valid shape | Static or dynamic; this load2d path handles 1–15 valid rows |

Let `M = dst.GetValidRow()`, `K = dst.GetValidCol()`, `r = indexRow`, and
`c = indexCol`. Both indices are unsigned (`uint16_t`). Set dynamic valid dimensions
before the call; `TEXTRACT` uses the existing valid shape and does not infer or update
it from the indices. The small-M path requires:

```text
1 <= M <= 15
1 <= K <= DstTileData::Cols
r + M <= src.GetValidRow()
c + K <= src.GetValidCol()
c % 16 == 0
```

`r` need not be aligned to 4 or 16. The valid window may cross an internal 16-row
source fractal boundary as long as these bounds and the storage bounds below hold.
Source valid dimensions may also be dynamic. A dynamic destination with runtime
`M = 16` uses the original ordinary/Compact implementation, including 16-aligned
row and column indices. A2A3 also retains the physical rectangular bounds above;
A5 retains its original Compact rules based on the rounded valid width. Static full-M,
transposed, Right, GEMV, other dtype and architectures other than A2A3/A5 keep their
existing behavior. This extension does not add support for empty or oversized valid shapes,
ReLU, quantization, or `STPhase` variants of Mat-to-Left extraction.

**Physical reads and writes.** A repeat still transfers a complete 512-byte fractal;
reducing valid M does not reduce the bytes transferred. Define `ceil16(K) = ((K + 15) / 16) * 16`
using integer division. The actual width is:

```text
copyCols = DstTileData::Cols         // ordinary TileLeft
copyCols = ceil16(K)                // TileLeftCompact
```

The following additional bounds apply, where `srcEndBytes` is the exclusive end
of the read in bytes from the source tile base:

```text
c + copyCols <= SrcTileData::Cols
copyCols <= DstTileData::Cols
srcEndBytes = (c + copyCols - 16) * SrcTileData::Rows * 2 + r * 32 + 512
srcEndBytes <= SrcTileData::Numel * 2
```

Consequently, a Compact destination may be physically wider than the source if its
rounded valid width fits; an ordinary destination must fit its full physical width.
On A2A3 this relaxation applies only to small M: dynamic `M = 16` still requires
`c + DstTileData::Cols <= SrcTileData::Cols`, including for Compact tiles. A5 dynamic
`M = 16` retains Compact's rounded valid width and does not add this A2A3 restriction.

Both backends offset the source by `(c / 16) * SrcTileData::Rows * 16 + r * 16`
elements. The load2d parameter formats differ:

| Backend | Small-M load parameters |
| --- | --- |
| A2A3 `pto_load_cbuf_to_ca` | `baseIdx = 0`, `repeat = copyCols / 16`, `srcStride = SrcTileData::Rows / 16`, destination gap 0 |
| A5 `load_cbuf_to_ca` | `mStart = 0`, `kStart = 0`, `mStep = 1`, `kStep = copyCols / 16`, `srcStride = SrcTileData::Rows / 16`, `dstStride = 1`, transpose 0 |

Successive source fractal starts are `srcStride * 512` bytes apart; destination
fractals are contiguous. The total payload read and written is `copyCols * 32` bytes,
which may be smaller than the source address span when `srcStride > 1`.
Repeat/kStep counts above 255 are split with matching source/destination pointer advances;
this does not relax the platform's L1/L0A capacity limits.

**Source padding and synchronization.** For a shared 16×K source with `c = 0` and K divisible
by 16, extracting four valid rows at offsets 4/8/12 through the final K fractal
reads 128/256/384 bytes beyond a tightly allocated 16×K tile. Declare physical
columns `K + 16` and valid columns `K` to own the extra 512 bytes. For tail K,
`ceil16(K) + 16` source columns and `ceil16(K)` destination columns give the same
safe arrangement. ND-to-NZ `TLOAD` can still load a single 16×K GlobalTensor with
the same NZ row stride. With a nonzero column origin, account for `c` in both
the source valid width (`c + K`) and the physical read-end formula. For other shapes, existing
capacity may already suffice, and adding 16 columns costs `SrcTileData::Rows * 32`
bytes, which is 512 bytes only for 16 physical rows.

The complete source allocation, including padding, must remain readable and must
not be overwritten or reused until its MTE1 reads finish. Padding need not be
initialized to zero. Destination elements outside the valid window are unspecified
and must not be consumed as valid results.
In manual mode, synchronize MTE2→MTE1 after `TLOAD`, MTE1→M before `TMATMUL`, and
M→MTE1 before overwriting a Left tile still consumed by Cube. Auto mode tracks
these dependencies through the tile operands. `STPhase` is an Acc-to-Mat move-out
feature and does not replace these dependencies. Allocate and synchronize the full
physical tiles; the valid shape defines the result, not the allocation size.
Runtime checks use `PTO_ASSERT`, enabled by `_DEBUG`; callers must satisfy the
constraints in every build.

The following fragments use `<pto/pto-inst.hpp>` and `using namespace pto;` in
an A2A3 or A5 device compilation context (`__CCE_AICORE__`); the Left aliases select
the corresponding architecture layout. They assume the full tiles
have been allocated, the source's 16×64 valid window has been loaded, and the
required synchronization is provided by the caller or auto mode:

```cpp
using Mat = Tile<TileType::Mat, half, 16, 80, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512>;
using Left = TileLeft<half, 16, 64, 4, 64>;
AICORE void ExtractFourRows(Left &left, Mat &mat)
{
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

An alternative destination with dynamic M/K and Compact storage can have 128
physical columns while reading only 64 when called with `m = 4, k = 63`. The helper
sets the valid shape explicitly. For this source and row offset, valid arguments
are `1 <= m <= 4` and `1 <= k <= 64`; the caller allocates the destination first:

```cpp
using CompactLeft = TileLeftCompact<half, 16, 128, DYNAMIC, DYNAMIC>;
AICORE void ExtractCompactRows(CompactLeft &left, Mat &mat, uint32_t m, uint32_t k)
{
    left.SetValidShape(m, k);
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

See the [A2A3 implementation](../../include/pto/npu/a2a3/TExtract.hpp) and
[A5 implementation](../../include/pto/npu/a5/TExtract.hpp), numerical ST with
TLOAD/TEXTRACT/TMATMUL for [A2A3](../../tests/npu/a2a3/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp)
and [A5](../../tests/npu/a5/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp),
and instruction/bounds tests for [A2A3](../../tests/costmodel/st/testcase/textract_small_m/main.cpp)
and [A5](../../tests/cpu/st/testcase/textract_small_m_a5/main.cpp).
The A5 host tests model the load primitive while executing the production A5 dispatcher and helper;
NPU ST verifies the complete instruction sequence numerically.

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
