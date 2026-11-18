# TLOAD

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->




## Tile Operation Diagram

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## Introduction

Load data from a GlobalTensor (GM) into a Tile.

## Math Interpretation

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);

template <
    TLoadL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

Existing `TLOAD(dst, src)` and `TLOAD<TileT, GTensor>(dst, src)` still work. The `TLoadL2Hint`-first form is an extra overload (no default on `l2Control`).

## L2 cache hint

Optional first-template overload (existing `TLOAD(dst, src)` unchanged):

```cpp
TLOAD<TLoadL2Hint::NotAllocKeep>(dst, src);
```

Supported `TLoadL2Hint` values:

| Enumerator | Value | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | default allocate (no-op) | yes |
| NormalLastVictim | 1 | default allocate (no-op) | yes |
| NormalPersistent | 2 | default allocate (no-op) | yes |
| NotAllocKeep | 4 | non-allocate (GM addr += runtime `l2Cacheoffset`) | yes |
| NotAllocClean | 5 | non-allocate (same as Keep) | yes |
| NotAllocDrop | 6 | non-allocate (same as Keep) | yes |

On A2/A3 there are effectively **two options only**:

1. **Default allocate** — `NormalFirstVictim` (0), `NormalLastVictim` (1), `NormalPersistent` (2): no-ops on A2/A3 (no effect on VLU / different last/first/persist hints). They are not meaningful distinct modes; all follow the default allocate path.
2. **Non-allocate** — `NotAllocKeep` (4), `NotAllocClean` (5), `NotAllocDrop` (6): yes, via GM addr += runtime `l2Cacheoffset` (same behavior for Keep/Clean/Drop on A2/A3).

On A5 all listed values are passed through to DMA. CPU / costmodel accept the template and ignore it.

## Constraints

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - Destination tile location must be `TileType::Vec` or `TileType::Mat`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Runtime: all `src.GetShape(dim)` values and `dst.GetValidRow()/GetValidCol()` must be `> 0`.
    - `TileType::Vec` loads only support matching layouts: ND->ND, DN->DN, NZ->NZ.
    - `TileType::Mat` loads support: ND->ND, DN->DN, NZ->NZ, plus ND->NZ and DN->ZN.
    - For ND->NZ: `GlobalData::staticShape[0..1] == 1` and `TileData::SFractalSize == 512`.
    - For DN->ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
    - ND->NZ also accepts `GlobalData::staticShape[2] != 1`. `Shape2` is then the number of ND matrices
      a single instruction moves (the instruction's own `ndNum` operand), each matrix being
      `[Shape3, Shape4]`, and the matrices are stacked along the tile rows, so
      `dst.GetValidRow() == Shape2 * Shape3`. Because `srcNdMatrixStride` and `dstNzMatrixStride` are
      16-bit element counts on this platform, runtime `Shape2 > 1` additionally requires
      `1 <= Stride2 <= 65535` and `Shape3 * (32 / sizeof(DType)) <= 65535`.
      A dynamic Shape2 whose runtime value is 1 does not use these matrix strides and is exempt from
      these two checks. `Shape2 * Shape3 <= TileData::Rows` and `1 <= Shape2 <= 65535` still apply.
    - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
    - Vec ND->ND requires `TileData::Rows < 4096`. For ordinary same-layout UB/L1 loads,
      the burst-count dimension must be less than 4096: `Shape3` for ND, `Shape4` for DN,
      and `Shape1` for NZ. The single-row/single-column Mat paths are described below.
    - Mat ND->NZ requires `1 <= Shape3 <= 16384`, `1 <= Shape4 <= 65535`,
      `1 <= Stride3 <= 65535`, and `TileData::Rows <= 16384`.
    - Mat DN->ZN requires `1 <= Shape4 <= 16384`, `1 <= Shape3 <= 65535`,
      `1 <= Stride4 <= 65535`, and `TileData::Cols <= 16384`.
    - Tile dimensions and placement must fit the target UB/L1 capacity; the limits above are path-specific.
- **Implementation checks (A5)**:
    - `sizeof(TileData::DType)` must be `1`, `2`, `4`, or `8` bytes, and must match `sizeof(GlobalData::DType)`.
    - For `int64_t/uint64_t`, `TileData::PadVal` must be `PadValue::Null` or `PadValue::Zero`.
    - `TileType::Vec` loads require one of the following layout pairs:
    - ND with row-major + `SLayout::NoneBox` (ND->ND),
    - DN with col-major + `SLayout::NoneBox` (DN->DN),
    - NZ with `SLayout::RowMajor` (NZ->NZ).
    - For row-major ND->ND with compile-time-known shapes, `TileData::ValidCol` must equal `GlobalData::staticShape[4]`, and `TileData::ValidRow` must equal the product of `GlobalData::staticShape[0..3]`.
    - A5 Vec NZ->NZ requires static inner dimensions `Shape3 == 16` and `Shape4 == 32 / sizeof(DType)`
      (`Shape4 == 64` for packed FP4). The source shape need not equal the tile's logical valid window.
      The same inner-shape constraints apply in CPU_SIM A5 mode.
    - `TileType::Mat` loads are additionally constrained by `TLoadCubeCheck` (e.g., only specific ND/DN/NZ conversions and L1-size limits).
    - For `TileType::Mat` ND->NZ and DN->NZ: `TileData::SFractalSize == 512`, `sizeof(TileData::DType) != 8`, and `GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1`.
    - ND->NZ additionally accepts `GlobalData::staticShape[2] != 1` (including dynamic Shape2).
      `Shape2` is the available number of ND matrices of shape `[Shape3, Shape4]`, stacked along the tile rows.
      Runtime: `1 <= Shape2 <= 65535`, `1 <= Shape3 <= 16384`, and
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape2 * Shape3)`. The data type must not be fp4.
    - DN->NZ requires `GlobalData::staticShape[2] == 1`.
    - For `TileType::Mat` DN->ZN, the ZN tile uses `BLayout::RowMajor` and `SLayout::ColMajor`.
      `GlobalData::staticShape[2] != 1` (including dynamic Shape2) selects the multi-matrix path.
      `Shape2` is the available number of DN matrices of shape `[Shape3, Shape4]`, merged along the tile columns;
      `Stride2` is the distance between consecutive matrices in elements. Rows within each matrix must be
      contiguous (`Stride3 == 1`); `Stride4` is the distance between consecutive columns in elements.
    - The multi-matrix DN->ZN path requires `GlobalData::staticShape[0..1] == 1`,
      `TileData::SFractalSize == 512`, `sizeof(TileData::DType)` of `1`, `2`, or `4` bytes (excluding fp4/hif4),
      and `TileData::Cols <= 65535`. Runtime: `1 <= Shape2 <= 65535`, `1 <= Shape4 <= 16384`,
      `1 <= dst.GetValidCol() <= min(TileData::Cols, Shape2 * Shape4)`, and
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape3)`.
    - A5 `TileType::Mat` multi-matrix ND->NZ and DN->ZN preserve 64-bit source strides for both static and dynamic
      `Stride` values.
      `Stride2` and the within-matrix stride (`Stride3` for ND, `Stride4` for DN) are `int64_t` element counts.
      For the supported b8/b16/b32 types, their byte distances are `stride * sizeof(DType)`, computed in 64 bits;
      element strides above `INT32_MAX` and byte distances of 4 GiB or more are not truncated to 32 bits.
      Form large stride expressions with 64-bit operands before passing them to `Stride`, and provide GM storage
      covering all accessed addresses.
    - A5 ConvTile `NCHW -> NC1HWC0` and `NCDHW -> NDC1HWC0` loads read the destination ConvTile `strideH` / `strideW`
      configuration fields (set by `SetStrideH` / `SetStrideW`). The configuration expresses input expansion for
      backward-convolution dx: the source GlobalTensor H/W still describes the compact input, while the destination
      ConvTile H/W describes expanded space, normally `(H - 1) * strideH + 1` and `(W - 1) * strideW + 1`. The caller
      must keep source shape, destination shape, and stride configuration consistent.
    - In these A5 expansion paths, `TLOAD` writes only positions covered by source data. Holes produced by stride expansion
      are not written and are not automatically zero-filled. Initialize the destination ConvTile backing storage first if a
      later computation requires the holes to be zero. When both `strideH` and `strideW` are 1, expansion placement is disabled
      and the existing load semantics are retained.
    - `TileType::Mat` loads also handle loads for mx format, which include `MX_A_ZZ/MX_A_ND/MX_A_DN` to ZZ for scalarA and `MX_B_NN/MX_B_ND/MX_B_DN` to NN for scalarB.
    - for `MX_A_ZZ/MX_B_NN`: `(GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - for `MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN`: `(GlobalData::staticShape[0] == 1 || GlobalData::staticShape[0] == -1)` and `(GlobalData::staticShape[1] == 1 || GlobalData::staticShape[1] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - for scaleA, `dst.GetValidCol() % 2 == 0`.
    - for scaleB, `dst.GetValidRow() % 2 == 0`

- **Valid region**:
    - Transfer extent depends on the selected layout path; it is not always the logical valid rectangle.
    - Source shape and `dst.GetValidRow()` / `dst.GetValidCol()` must agree with the selected path.
      They are not independent clipping bounds. In particular, A2/A3 ND->NZ and DN->ZN transfer the
      source matrix shape; A5 single-matrix ND->NZ transfers `Shape3` rows and `dst.GetValidCol()` columns.
    - A5 Vec NZ->NZ (also modeled in CPU A5 mode) transfers `Shape0` groups of `Shape1` column blocks.
      Each burst copies `dst.GetValidRow() * 32` bytes. Source group/block pitches are `Stride0`/`Stride1`
      in elements; destination block pitch is `TileData::Rows * 32` bytes and group pitch is
      `Shape1 * TileData::Rows * Shape4` elements. FP4 element pitches are divided by two for byte addressing.
      This path does not use `validCol` to trim column blocks or add `PadVal` padding. Rows beyond `validRow`
      and untransferred blocks retain their contents; GM and tile storage must cover every transferred block.
      For CPU_SIM, select A5 before assigning tiles; see
      [Selecting the simulated architecture](../coding/cpu_sim.md#selecting-the-simulated-architecture).
    - On A2/A3, same-layout, block-aligned `TileType::Mat` loads write only the logical valid region. ND-to-NZ/DN-to-ZN loads (and the single-row/single-column Mat special paths) additionally zero-fill the final partial C0 block. Other data remains unchanged, including data owned by tile views that share the same backing storage.
    - On A5, same-layout `TileType::Mat` ND/DN loads fill only the final partial 32-byte block according to `PadVal`; ND/DN-to-fractal loads zero-fill the final partial C0 block. Full 32-byte gaps and inactive rows or columns remain unchanged.
    - On A5, ND->NZ with `GlobalData::staticShape[2] != 1` loads the first `dst.GetValidRow()` merged rows.
      It transfers `dst.GetValidRow() / Shape3` complete matrices in one instruction, then transfers
      `dst.GetValidRow() % Shape3` tail rows separately if needed. With no complete matrices, only the tail
      transfer is issued. Both transfers retain the full tile's NZ column-block stride. This also applies
      when Shape2 is dynamic and its runtime value is 1. For example, `Shape3 = 3` and `dst.GetValidRow() = 17`
      load five complete matrices and the first two rows of the sixth matrix, requiring `Shape2 >= 6`.
    - On A5, DN->ZN with `GlobalData::staticShape[2] != 1` loads the first `dst.GetValidCol()` merged columns.
      It transfers `dst.GetValidCol() / Shape4` complete matrices in one instruction, then transfers
      `dst.GetValidCol() % Shape4` columns from the next matrix if needed. With no complete matrices, only
      the tail transfer is issued. This also applies when Shape2 is dynamic and its runtime value is 1.
      When a tail is present, with `n = dst.GetValidCol() / Shape4`, it starts at
      `src.data() + n * Stride2` (element offset). The source offset retains 64-bit precision even when
      the byte distance is 4 GiB or more.
      Both transfers retain `TileData::Cols` as the C0-block stride (in 32-byte units).
      Only the final partial C0 block along the valid rows is zero-filled; inactive columns and full
      C0 blocks beyond the valid rows retain their previous contents.
      A compile-time Shape2 of 1 uses the single-matrix path: the transfer covers `Shape4` columns and
      `dst.GetValidRow()` rows, so the source shape must describe the columns to load.
    - On A2/A3 and A5, a `TileType::Vec` ND/DN load with a non-null `PadVal` fills only the sub-32-byte tail after each transferred burst. Full 32-byte gaps and inactive rows or columns remain unchanged; NZ loads and `PadValue::Null` do not add padding.

### Stride and length boundaries

For A2/A3, A5, A6, KirinX90, Kirin9030 and Kirin9050, the affected TLOAD paths retain
source strides as `int64_t`. For b8/b16/b32/b64 data, element-to-byte conversion uses 64-bit
arithmetic. Construct large `Stride` expressions with 64-bit operands, provide GM storage covering
all accessed addresses, and keep the destination within its physical capacity. Byte distances and
computed offsets must remain representable; each path also imposes the constraints below.

A **gap** is the distance from the end of one burst to the start of the next; a **stride** is the
distance between burst starts. For the b8/b16/b32/b64 same-layout paths, with `E = sizeof(DType)`,
ND uses source byte stride `Stride3 * E`, DN uses `Stride4 * E`, and NZ uses `Stride1 * E`.
On the memory backends, the corresponding byte gaps are `(Stride3 - Shape4) * E`,
`(Stride4 - Shape3) * E`, and `(Stride1 - Shape2 * Shape3 * Shape4) * E`.
These gap-based paths require nonnegative gaps.
The following limits apply to different encoded fields:

| Path | Field and fallback |
| --- | --- |
| A2/A3, KirinX90 ordinary GM-to-UB | Source gap is in bytes. Above `UINT32_MAX`, issue one burst at a time using 64-bit source offsets. |
| A2/A3 ordinary GM-to-L1 | Source gap is in 32-byte blocks. Above `UINT16_MAX` blocks, issue one burst at a time. Existing block-alignment requirements still apply. |
| KirinX90 ordinary GM-to-L1 | Check length and gap in the units of the selected block or byte-aligned instruction. If they do not fit, split bursts and lengths before encoding. |
| A5/A6 ordinary GM-to-UB/L1 | Source stride is in bytes. Above `2^40 - 1`, issue one burst at a time. ND/DN Shape1/Shape2 loops also use explicit source addresses when their byte strides exceed this limit. |
| A5 NC1HWC0 and five-dimensional FRACTAL_Z (`[C1, H, W, N, C0]`) | Both use `TLoad5HD`: source loop byte strides from `Stride0`/`Stride1` use software loops above `2^40 - 1`; the burst byte stride from `Stride2` uses the shared per-burst fallback. |
| Kirin9030, Kirin9050 ordinary DMA | Preserve 64-bit source strides and byte conversion; calls through the shared register DMA helpers use the same burst-stride fallback. |
| KirinX90 b32 ND->NZ/DN->ZN | Check fields after conversion to b16 units; split the transfer when the converted values exceed 16 bits. The existing public within-matrix stride limit of 65535 elements remains unchanged. |

The limits are inclusive: `UINT32_MAX` bytes, `UINT16_MAX` blocks and `2^40 - 1` bytes
still fit their respective fields; `2^40` bytes already selects the register-helper fallback.
A5 four-dimensional FRACTAL_Z (`[C1HW, N/16, 16, C0]`) also uses the shared per-burst fallback,
with source byte stride derived from `Stride1`.
A2/A3 GM-to-L1 gap fallback also applies to same-layout convolution loads using the same helper
(NC1HWC0, both FRACTAL_Z representations and NDC1HWC0). It does not split an oversized burst length
or relax destination-gap/burst-count limits. Ordinary A2/A3 block transfers still require their lengths
and source/destination gaps to be multiples of 32 bytes, with length and destination gap fitting
16-bit block counts.

KirinX90 GM-to-L1 first uses the block instruction only when the destination gap is zero,
the length and source gap are multiples of 32 bytes, and both block counts fit 16 bits.
Otherwise, the byte-aligned instruction is used directly only when length and source gap are each
at most 65535 bytes and the destination gap is less than 32 bytes. The fallback computes each burst's
source and destination addresses independently, then splits lengths into 65504-byte chunks when
more than 65535 bytes remain. A remaining length of at most 65535 bytes is issued as the last chunk.
This preserves full destination gaps instead of narrowing them into padding fields.

For KirinX90 b32 ND->NZ/DN->ZN, the contiguous extent and within-matrix stride are doubled for the
b16 instruction. A value of 32768 b32 elements already exceeds the converted 16-bit field. The
fallback transfers rows separately and splits the contiguous extent at C0-aligned boundaries
(up to 65520 b16 elements per chunk). This conversion handling does not enable unsupported layouts
or multi-matrix forms.

On A2/A3 and KirinX90, same-layout Mat loads with `TileData::Rows == 1` (ND) or
`TileData::Cols == 1` (DN) copy complete 32-byte blocks, then copy and zero-pad any partial final block.
The element count is not narrowed to 16 bits, and b64 lengths do not pass through b16/b32 conversion.
Counts of 65536 elements and b64 counts of 16384/32768 elements do not impose
additional length limits. Source shape, valid dimensions and L1 capacity still constrain the load.
These paths require `GlobalData::staticShape[0..2] == 1`, contiguous elements in the loaded vector,
and matching valid dimensions: ND has `Shape3 == validRow == 1` and `Shape4 == validCol`;
DN has `Shape4 == validCol == 1` and `Shape3 == validRow`. They use source `Shape4` (ND) or `Shape3` (DN)
as the element count. Each complete-block transfer contains at most 65535 blocks; any final
1–31 bytes use a b8 conversion transfer that zero-fills the rest of the block, independently of `PadVal`.
An already block-aligned vector needs no tail transfer. Storage after the padded block is unchanged.

These fallbacks do not extend the shape/stride contracts of separate format-conversion instructions.
In particular, the ordinary DMA 40-bit handling is not a statement of the range supported by A5/A6
ND-to-fractal instructions. A 64-bit C++ parameter alone does not establish a hardware range.
Burst-length/count, destination-stride and hardware-loop count/destination-stride fields must
also satisfy their target-specific limits.

### Runtime dispatch and scalar overhead

The shared register DMA helper used by A5/A6 and the Kirin9030/Kirin9050 callers has a direct
native path: when the source byte stride is at most `2^40 - 1`, it issues a multi-burst
DMA and returns without entering the per-burst loop. Only an oversized stride enters that loop,
which issues one burst per iteration with zero encoded source/destination strides and full-width
software address offsets. On A5, when the compiler proves the burst count is constant and at most
`8`, the fallback uses bounded loop unrolling to reduce loop overhead. Larger or runtime-dependent
counts, and the other backends using this helper, keep unrolling disabled to bound code size.
The compile-time selection does not add a runtime burst-count check. L2-hint forwarding and padding
arguments are preserved, including the b64-to-b32 conversion of Mat padding counts.

The per-burst fallback adds software address calculations and DMA issue operations as the burst
count grows. Many short bursts can make this cost visible even without a scalar wait after every
load. Measurements of range dispatch on the native path do not characterize fallback performance;
evaluate the paths separately with the intended burst count, transfer size and synchronization.

The separate A5/A6 ND/DN outer-loop stride checks still apply. A5 `TLoad5HD` resets the hardware
loop count to one after either path. The A5/A6 MX-A vector paths use a loop count of one without
programming unused outer-loop strides. These setup and format-conversion operations also apply
when the helper takes the native path.

Known static `Shape`/`Stride` values can allow the compiler to eliminate range dispatch. Dynamic
values can retain comparisons and branches, so this is not a guarantee of zero scalar overhead.
An oversized static stride still requires the per-burst transfers even if the range comparison is eliminated.
The public `TLOAD` templates have no switch to disable stride fallback; `TLoadL2Hint` only selects
the cache hint. ptoas-generated C++ calls to these templates use the same behavior.

`PTO_ASSERT` is enabled when `_DEBUG` is defined, including `_DEBUG=0`, and is otherwise removed.
`NDEBUG` does not control it. These debug assertions are separate from the native/fallback dispatch,
which remains active without `_DEBUG`. Compile-time assertions also remain in effect. Callers must
satisfy the constraints in release builds too.

See the [stride ST coverage](../../tests/npu/a2a3/src/st/testcase/tload_large_stride/README.md)
for the registered cases on each backend.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TASSIGN(t, 0x1000);
  TLOAD(t, gin);
}
```

### A5 multi-matrix DN-to-ZN load (Manual)

The following Cube function loads 17 merged columns: five complete 3-column matrices and two columns
from the sixth. Physical GM storage is `[8, 9, 64]` (matrix, column, row); the DN view selects 3 columns
and 35 rows per matrix. Strides are in elements.

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn(__gm__ int16_t* in) {
  using SrcShape = Shape<1, 1, 8, 35, 3>;
  using SrcStride = pto::Stride<4608, 4608, 576, 1, 64>;
  using SrcGlobal = GlobalTensor<int16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, int16_t, 64, 32, BLayout::RowMajor,
                       35, 17, SLayout::ColMajor, 512>;

  SrcGlobal src(in);
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

For `0 <= r < 35` and `0 <= c < 17`, the result is
`dst[r, c] = in[(c / 3) * 576 + (c % 3) * 64 + r]`.

### A5 DN-to-ZN load with a large dynamic stride (Manual)

This Cube function loads three merged columns: both columns of the first matrix and the first column
of the second. `matrixStride` is the distance between matrix starts in `uint16_t` elements, supplied at runtime.
For example, `(int64_t{1} << 31) + 128` elements correspond to `4294967552` bytes (4 GiB + 256 bytes).
The GM allocation must cover all accessed elements, including the 32 rows at the second matrix's start.
With this example stride, the required span from `in` is `(matrixStride + 32) * sizeof(uint16_t)` bytes.
The `-1` in `SrcStride` makes only `Stride2` dynamic; `Shape2` remains the compile-time matrix count of 2.

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn_large_stride(__gm__ uint16_t* in, int64_t matrixStride) {
  using SrcShape = Shape<1, 1, 2, 32, 2>;
  using SrcStride = pto::Stride<1, 1, -1, 1, 64>;
  using SrcGlobal = GlobalTensor<uint16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, uint16_t, 64, 64, BLayout::RowMajor,
                       32, 3, SLayout::ColMajor, 512>;

  SrcGlobal src(in, SrcShape{}, SrcStride(1, 1, matrixStride, 1, 64));
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

For `0 <= r < 32` and `0 <= c < 3`, the result is
`dst[r, c] = in[(c / 2) * matrixStride + (c % 2) * 64 + r]`.
The last loaded column therefore comes from `in[matrixStride + r]`.
