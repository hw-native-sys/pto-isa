# pto.tmov

`pto.tmov` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

`TMOV` copies or transforms tile data between tiles. It is the workhorse for tile-to-tile data movement, accumulator-to-vector conversion, and fix-pipe quantization paths.

Two variants are documented here:

| Variant | Suffix | Description | Typical Use |
|---------|---------|-------------|-------------|
| Standard move | *(none)* | Direct tile-to-tile copy or conversion | Vec→Vec, Mat→Left/Right, Acc→Mat |
| Fix-pipe move | `_fp` | Move through fix-pipe quantization path | Acc→Vec/int8_t with scaling |

## Mechanism

Conceptually copies or transforms elements from `src` into `dst` over the valid region. Exact transformation depends on the selected mode and variant.

**Standard move (pure copy case):**

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

**Fix-pipe variant** (`TMOV_FP`): Routes through the hardware fix-pipe quantization pipeline, applying conversion configured by the `fp` sideband tile:

$$ \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Variants

### Variant 1: Standard Move

`TMOV(dst, src)` — plain tile-to-tile copy with optional ReLU mode.

### Variant 2: ReLU Move

`TMOV<..., reluMode>(dst, src)` — copy with ReLU pre-processing.

### Variant 3: Accumulator-to-Vector

`TMOV<..., mode, reluMode>(dst, src)` — converts accumulator tile to vector tile with optional ReLU. The `mode` parameter selects the splitting strategy for multi-core configurations.

### Variant 4: Vector-Quant Move

`TMOV<..., FpTileData, mode, reluMode>(dst, src, fp)` — converts accumulator to vector through the fix-pipe quantization path. The `fp` tile carries scale factors.

### Variant 5: Scalar-Quant Move

`TMOV<..., reluMode>(dst, src, preQuantScalar)` — converts accumulator with a scalar quantization parameter.

### Variant 6: Fix-Pipe Move (`TMOV_FP`)

`TMOV_FP(dst, src, fp)` — explicit fix-pipe move. Same semantics as the vector-quant move but named explicitly for the assembly spelling.

## AccToVecMode Reference

The `AccToVecMode` parameter controls how accumulator tiles are split and transferred to vector tiles, especially in multi-core (dual-mode) configurations:

| Mode | Meaning | Used When |
|------|---------|----------|
| `SingleModeVec0` | Transfer to vector 0 only | 1-Cube, 1-Vec configurations |
| `SingleModeVec1` | Transfer to vector 1 only | Single-mode targeting Vec1 |
| `DualModeSplitM` | Split accumulator rows evenly across two vectors | 1-Cube, 2-Vec with row-wise split |
| `DualModeSplitN` | Split accumulator columns across two vectors | 1-Cube, 2-Vec with column-wise split |

## Supported Tile-Type Pairs

### A2/A3

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Mat` | `TileType::Left/Right/Bias/Scaling` | MX block-format extraction |
| `TileType::Vec` | `TileType::Vec` | Direct copy |
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix conversion |

**Mat→Bias restrictions:**
- Supported dtype pairs: `int32_t → int32_t`, `float → float`, `half → float`
- Source row must be `1`
- `SrcTileData::Cols * sizeof(SrcType)` must be aligned to 64 bytes

**Mat→Scaling restrictions:**
- Destination dtype must be `uint64_t`
- Source row must be `1`
- `SrcTileData::Cols * sizeof(SrcType)` must be aligned to 128 bytes

### A5

In addition to A2/A3 pairs:

| Source Type | Destination Type | Notes |
|------------|----------------|-------|
| `TileType::Mat` | `TileType::Left/Right/Bias/Scaling/ScaleLeft/ScaleRight` | Extended MX formats |
| `TileType::Vec` | `TileType::Mat` | Vector-to-matrix conversion |
| `TileType::Acc` | `TileType::Vec` | Accumulator-to-vector with mode selection |
| `TileType::Acc` | `TileType::Mat` | Accumulator-to-matrix |

**A5 Mat→Bias:**
- Supported dtype pairs: `int32_t → int32_t`, `float → float`, `half → float`, `bfloat16_t → float`
- `DstTileData::Cols * sizeof(DstType)` must be aligned to 64 bytes
- Bias-table footprint ≤ 4096 bytes

**A5 Mat→Scaling:**
- `DstTileData::Cols * sizeof(DstType)` must be aligned to 128 bytes
- Fix-pipe-buffer footprint ≤ 4096 bytes

**A5 Acc→Vec:**
- `mode` selects `SingleModeVec0`, `SingleModeVec1`, `DualModeSplitM`, `DualModeSplitN`
- Dual-mode requires `QuantMode_t::NoQuant`
- Dual-mode does not support the `nz2dn` path
- `dstStride * sizeof(dstType)` must be a multiple of 32 bytes

## Syntax

### PTO Assembly Form

Standard move:

```text
%dst = tmov.s2d %src : !pto.tile<...> -> !pto.tile<...>
```

The PTO AS design recommends splitting `TMOV` into a small set of instructions:

```text
%left  = tmov.m2l %mat  : !pto.tile<...> -> !pto.tile<...>
%right = tmov.m2r %mat  : !pto.tile<...> -> !pto.tile<...>
%bias  = tmov.m2b %mat  : !pto.tile<...> -> !pto.tile<...>
%scale = tmov.m2s %mat  : !pto.tile<...> -> !pto.tile<...>
%vec   = tmov.a2v %acc  : !pto.tile<...> -> !pto.tile<...>
%v1    = tmov.v2v %v0   : !pto.tile<...> -> !pto.tile<...>
```

Fix-pipe move:

```text
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
// Standard
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>

// Fix-pipe
%dst = pto.tmov.fp %src, %fp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
// Standard
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)

// Fix-pipe
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Variant 1: Plain move
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

// Variant 2: ReLU move
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

// Variant 3: Accumulator-to-vector with mode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

// Variant 4: Vector-quant (fix-pipe) move
template <typename DstTileData, typename SrcTileData, typename FpTileData,
          AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);

// Variant 5: Scalar-quant move
template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);

// Variant 5b: Scalar-quant with mode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);

// Variant 6: Explicit fix-pipe move
template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);
```

## Constraints

- **Shape**: `SrcTileData::Rows == DstTileData::Rows` and `SrcTileData::Cols == DstTileData::Cols`
- **`reluMode`**: `ReluPreMode::{NoRelu, NormalRelu}`
- **`mode`**: `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`
- **`FpTileData::Loc`**: Must be `TileType::Scaling` on both A2/A3 and A5 (verified by `static_assert`)
- **Vec→Vec**: Shape must match exactly
- **Mat→Left/Right/Bias/Scaling**: Compile-time restricted by tile type

## Common Patterns

### Pattern 1: Vec-to-Vec Tile Copy

```cpp
// Copy one vector tile to another (e.g., for double-buffering)
void tileCopy(TileT& dst, TileT& src) {
    TMOV(dst, src);  // Straight copy
}
```

### Pattern 2: MX Block Extraction (GEMM Setup)

```cpp
// Extract Left and Right block tiles from a matrix in NC1HWO layout
using MatT = Tile<TileType::Mat, float, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor>;
using LeftT = TileLeft<float, 64, 64>;
using RightT = TileRight<float, 64, 64>;

MatT mat;
LeftT left;
RightT right;
TASSIGN(mat, 0x1000);

TMOV(left, mat);   // Mat → Left
TMOV(right, mat);  // Mat → Right
```

### Pattern 3: Accumulator-to-Vector Conversion (Single Mode)

```cpp
// Convert accumulator to vector in single-mode (1 Cube, 1 Vec)
using AccT = TileAcc<float, 64, 128>;
using VecT = Tile<TileType::Vec, float, 64, 128>;

AccT acc;
VecT vec;
TASSIGN(acc, 0x1000);

TMOV<VecT, AccT, AccToVecMode::SingleModeVec0>(vec, acc);
```

### Pattern 4: Dual-Mode Accumulator-to-Vector (GEMM with 2 Vectors)

```cpp
// Accumulator split across two vector cores (1 Cube, 2 Vec)
using AccT = TileAcc<float, 64, 256>;
using VecT = Tile<TileType::Vec, float, 64, 128>;  // Half-width per vector

AccT acc;
VecT vec0, vec1;
TMOV<VecT, AccT, AccToVecMode::DualModeSplitN>(vec0, acc);  // Columns 0-127
TMOV<VecT, AccT, AccToVecMode::DualModeSplitN>(vec1, acc);  // Columns 128-255
```

### Pattern 5: Fix-Pipe Quantized Move (Production Inference)

```cpp
// Move accumulator through fix-pipe: float32 → int8_t with per-channel scaling
using AccT = TileAcc<float, 32, 32>;
using VecT = Tile<TileType::Vec, int8_t, 32, 32>;
using FpT = Tile<TileType::Scaling, uint64_t, 1, 32>;

AccT acc;
VecT vec;
FpT fp(32);  // 32 scale factors (one per output channel)
TASSIGN(acc, 0x1000);
TASSIGN(fp, 0x2000);

TMOV_FP(vec, acc, fp);  // Quantize through fix-pipe
```

### Pattern 6: Bias Tile Extraction

```cpp
// Extract a bias vector from a wider matrix (row=1 requirement)
using MatT = Tile<TileType::Mat, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::ColMajor>;
using BiasT = TileBias<float, 64>;

MatT mat;
BiasT bias;
TASSIGN(mat, 0x3000);

TMOV(bias, mat);  // Mat → Bias (row must be 1, width aligned to 64 bytes)
```

## See Also

- [Layout And Rearrangement](../../layout-and-rearrangement.md)
- [pto.tmov_fp](./tmov.md) — The fix-pipe variant (merged into this page; see Variant 6 above)
- [pto.treshape](./treshape.md)
- [pto.ttrans](./ttrans.md)
- [Assembly Spelling And Operands](../../../syntax-and-operands/assembly-model.md)
