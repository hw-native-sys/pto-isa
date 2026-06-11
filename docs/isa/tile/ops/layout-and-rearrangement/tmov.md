# pto.tmov

`pto.tmov` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.



## Tile Operation Diagram

![TMOV tile operation](../../../../figures/isa/TMOV.svg)

## Introduction

Move/copy between tiles, optionally applying implementation-defined conversion modes selected by template parameters and overloads.

`TMOV` is used for:

- Vec -> Vec moves
- Mat -> Left/Right/Bias/Scaling/Scale(Microscaling) moves (target-dependent)
- Acc -> Mat/Vec moves (target-dependent)

## Math Interpretation

Conceptually copies or transforms elements from `src` into `dst` over the valid region. Exact transformation depends on the selected mode and target.

For the pure copy case:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

### ND → NZ (data repack for the Cube Unit)

The PTO AS design recommends splitting `TMOV` into a family of ops:

```text
%left  = tmov.m2l %mat  : !pto.tile<...> -> !pto.tile<...>
%right = tmov.m2r %mat  : !pto.tile<...> -> !pto.tile<...>
%bias  = tmov.m2b %mat  : !pto.tile<...> -> !pto.tile<...>
%scale = tmov.m2s %mat  : !pto.tile<...> -> !pto.tile<...>
%vec   = tmov.a2v %acc  : !pto.tile<...> -> !pto.tile<...>
%v1    = tmov.v2v %v0   : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);
```

### ND → NZ / X → ZZ overloads

```cpp
// ND -> NZ (2-arg, no tmp)
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events);

// X -> ZZ (3-arg, with tmp). grp_axis=1 (default) = ND->ZZ; grp_axis=0 = DN->ZZ.
template <typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &...events);

template <int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &...events);
```

| Overload | `grp_axis` | Transform | `tmp` used? |
|----------|-----------|-----------|-------------|
| `TMOV(dst, src)` | — | ND → NZ | no |
| `TMOV(dst, src, tmp)` | 1 (default) | ND → ZZ | yes (vgather2 index buffer) |
| `TMOV<0>(dst, src, tmp)` | 0 | DN → ZZ | no (accepted for parity) |

## Constraints

### General constraints / checks

- `TMOV` has these overload families:
    - plain move: `TMOV(dst, src)`
    - relu form: `TMOV<..., reluMode>(dst, src)`
    - accumulator-to-vector form: `TMOV<..., mode, reluMode>(dst, src)`
    - vector-quant form: `TMOV<..., FpTileData, mode, reluMode>(dst, src, fp)`
    - scalar-quant form: `TMOV<..., reluMode>(dst, src, preQuantScalar)` and `TMOV<..., mode, reluMode>(dst, src, preQuantScalar)`
- `reluMode` is `ReluPreMode::{NoRelu, NormalRelu}`.
- `mode` is `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.

### A2A3 implementation checks

- Shape must match: `SrcTileData::Rows == DstTileData::Rows` and `SrcTileData::Cols == DstTileData::Cols`.
- Supported tile-type pairs are compile-time restricted to:
    - `TileType::Mat -> TileType::Left/Right/Bias/Scaling`
    - `TileType::Vec -> TileType::Vec`
    - `TileType::Acc -> TileType::Mat`
- For `TileType::Mat -> TileType::Bias`:
    - supported source/destination dtype pairs are `int32_t -> int32_t`, `float -> float`, and `half -> float`
    - source row must be `1`
    - `SrcTileData::Cols * sizeof(SrcType)` must be aligned to `64` bytes
- For `TileType::Mat -> TileType::Scaling`:
    - destination dtype must equal source dtype and must be `uint64_t`
    - source row must be `1`
    - `SrcTileData::Cols * sizeof(SrcType)` must be aligned to `128` bytes
- For `TileType::Acc -> TileType::Mat`:
    - additional `CheckTMovAccToMat<...>` compile-time checks are enforced
    - plain/relu forms use cast pre-quant mode derived by `GetCastPreQuantMode<SrcDType, DstDType>()`
    - scalar-quant forms use `GetScalarPreQuantMode<SrcDType, DstDType>()`
    - vector-quant forms require an `FpTileData` operand with `FpTileData::Loc == TileType::Scaling`, and use `GetVectorPreQuantMode<SrcDType, DstDType>()`

### A5 implementation checks

- `CommonCheck()` requires:
    - destination/source dtype must be identical
    - supported element types are `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`
    - source layout must satisfy one of:
        - `(SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor)`
        - `(SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor)`
        - `SrcTileData::isRowMajor`
- `CommonCheckMX()` for MX paths requires identical source/destination dtype and supports `float8_e8m0_t`.
- Supported paths include:
    - `TileType::Mat -> TileType::Left/Right/Bias/Scaling/ScaleLeft/ScaleRight`
    - `TileType::Vec -> TileType::Vec/TileType::Mat`
    - `TileType::Acc -> TileType::Vec/TileType::Mat`
    - specific `ND -> ZZ` and related internal path variants handled by the A5 implementation
- For `TileType::Mat -> TileType::Bias`:
    - supported dtype pairs are `int32_t -> int32_t`, `float -> float`, `half -> float`, `bfloat16_t -> float`
    - source row must be `1`
    - `DstTileData::Cols * sizeof(DstType)` must be aligned to `64` bytes
    - bias-table footprint `DstTileData::Cols * sizeof(DstType)` must not exceed `4096` bytes
- For `TileType::Mat -> TileType::Scaling`:
    - source row must be `1`
    - `DstTileData::Cols * sizeof(DstType)` must be aligned to `128` bytes
    - fixpipe-buffer footprint `DstTileData::Cols * sizeof(DstType)` must not exceed `4096` bytes
- For `TileType::Acc -> TileType::Vec`:
    - `mode` selects `SingleModeVec0`, `SingleModeVec1`, `DualModeSplitM`, or `DualModeSplitN`
    - dual-destination modes require `QuantMode_t::NoQuant`
    - dual-destination modes do not support the `nz2dn` path
    - for 32-bit destination types (`float`/`int32_t`), when using `DualModeSplitN` the `ValidCol` (before the split) must be a multiple of `32`
    - destination stride must be non-zero and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes
- For `TileType::Acc -> TileType::Mat`:
    - destination stride must be non-zero and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes
    - relu/scalar-quant/vector-quant forms are supported through the corresponding overloads


## Examples

### ND → NZ (data) — (128, 256) BF16

```cpp
// Source: 128 rows × 256 cols BF16 RowMajor Vec tile (ND).
// Destination: NZ fractal Mat tile for the Cube Unit (Left operand).
constexpr uint32_t R = 128, C = 256;
using SrcT = Tile<TileType::Vec, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::NoneBox>;
using DstT = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
SrcT src; DstT dst;
TMOV(dst, src);   // ND -> NZ, no tmp
```

**`tmp` for ND→NZ:** none — the 2-arg overload repacks in-place via `vsstb`.

### ND → ZZ (exponents) — `tmp` size derivation

Given a quantization input of shape $M \times N$ (group size $G = 32$), the ND-grouped
E8M0 exponent tile has shape:

| Quantity | Value |
|----------|-------|
| Exponent rows | $\mathrm{validRow} = M$ (one exponent row per input row) |
| Exponent cols | $\mathrm{validCol} = N/G = N/32$ (one exponent per 32-element column group) |
| Row blocks | $r_b = \lceil M/16 \rceil$ |
| Box-pair count | $P = \mathrm{validCol}/2 = N/64$ |

The `tmp` buffer holds the `vgather2` B16 index buffer used by `GenerateB8IndicesZZToUB`:

$$\boxed{\mathrm{tmpBytes} = \bigl(16 + r_b \cdot P + 16\bigr) \times 2 = \left(32 + \left\lceil\tfrac{M}{16}\right\rceil \cdot \tfrac{N}{64}\right) \times 2}$$

- 16 B16 lanes of headroom + $r_b \times P$ gather indices + 16 B16 lanes of tail.
- `tmp` dtype = `uint8_t` (E8M0), shape `1 × ⌈tmpBytes⌉`.

**Example:** $M = 128$, $N = 256$ → exponent tile $128 \times 8$, $r_b = 8$, $P = 4$:

$$\mathrm{tmpBytes} = (32 + 8 \times 4) \times 2 = 128\ \mathrm{B}$$

### DN → ZZ (exponents) — `tmp`

DN-grouped exponents have shape $\hat M \times N$ where $\hat M = M/32$. `TMOV<0>` accepts
a `tmp` operand for interface parity with ND→ZZ but **does not access it** (the `vsstb`
scatter needs no scratch). Any non-zero-sized tile satisfies the signature.

```cpp
// DN-grouped e8 exponents (M̂×N) -> ZZ fractal scale tile.
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);   // grp_axis=0 = DN->ZZ; tmp unused
```

### Usage in MX quantization (reference)

After `TQUANT` produces the quantized data + E8M0 exponents, two `TMOV`s repack them for
the Cube Unit. See `TQUANT.md` / `TQUANT_DN.md` for the full pipeline.

```cpp
// MXFP8 DN pipeline: quantize a 128×256 BF16 tile, then repack for the Cube.
constexpr uint32_t M = 128, N = 256, G = 32, Mhat = M / G;   // Mhat = 4
// Tiles
using SrcT   = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor>;
using Fp8T   = Tile<TileType::Vec, int8_t, M, N, BLayout::RowMajor>;
using E8DnT  = Tile<TileType::Vec, uint8_t, Mhat, N, BLayout::RowMajor>;        // 4×256
using E8ZzT  = Tile<TileType::Mat, uint8_t, N, Mhat, BLayout::ColMajor, N, Mhat, SLayout::RowMajor>;
using Fp8NzT = Tile<TileType::Mat, int8_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor>;
// 1. Quantize (DN grouping)
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);
// 2. Repack data ND->NZ (2-arg, no tmp)
TMOV(fp8NzTile, fp8Tile);
// 3. Repack exponents DN->ZZ (3-arg, tmp accepted but unused)
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);
```

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMOV(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT mat;
  DstT left;
  TASSIGN(mat, 0x1000);
  TASSIGN(left, 0x2000);
  TMOV(left, mat);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
