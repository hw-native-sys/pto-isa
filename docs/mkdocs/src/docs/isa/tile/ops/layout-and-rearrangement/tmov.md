<!-- Generated from `docs/isa/tile/ops/layout-and-rearrangement/tmov.md` -->

# pto.tmov

Standalone reference page for `pto.tmov`. This page belongs to the [Layout And Rearrangement](../../layout-and-rearrangement.md) family in the PTO ISA manual.

## Summary

Move/copy between tiles, optionally applying implementation-defined conversion modes.

## Mechanism

Move/copy between tiles, optionally applying implementation-defined conversion modes selected by template parameters and overloads.

`TMOV` is used for:

- Vec -> Vec moves
- Mat -> Left/Right/Bias/Scaling/Scale(Microscaling) moves (target-dependent)
- Acc -> Mat/Vec moves (target-dependent) It belongs to the tile surface and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Conceptually copies or transforms elements from `src` into `dst` over the valid region. Exact transformation depends on the selected mode and target.

For the pure copy case:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

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

## Inputs

- `src` is the source tile.
- `dst` names the destination tile. The operation iterates over dst's valid region.
- `fp` (optional for TMOV_FP): scaling tile for vector quantization parameters.

## Expected Outputs

`dst` holds a copy or transformed version of `src`, with optional conversion applied (relu, quantization, etc.).

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `TMOV` has these overload families:
  - plain move: `TMOV(dst, src)`
  - relu form: `TMOV<..., reluMode>(dst, src)`
  - accumulator-to-vector form: `TMOV<..., mode, reluMode>(dst, src)`
  - vector-quant form: `TMOV<..., FpTileData, mode, reluMode>(dst, src, fp)`
  - scalar-quant form: `TMOV<..., reluMode>(dst, src, preQuantScalar)` and `TMOV<..., mode, reluMode>(dst, src, preQuantScalar)`

- `reluMode` is `ReluPreMode::{NoRelu, NormalRelu}`.

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

- `CommonCheck()` requires:
  - destination/source dtype must be identical
  - supported element types are `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`
  - source layout must satisfy one of:
    - `(SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor)`
    - `(SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor)`
    - `SrcTileData::isRowMajor`

- `CommonCheckMX()` for MX paths requires identical source/destination dtype and supports `float8_e8m0_t`.

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
  - destination stride must be non-zero and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes

- For `TileType::Acc -> TileType::Mat`:
  - destination stride must be non-zero and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes
  - relu/scalar-quant/vector-quant forms are supported through the corresponding overloads

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `mode` is `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.

### A2A3 implementation checks

- For `TileType::Acc -> TileType::Mat`:
  - additional `CheckTMovAccToMat<...>` compile-time checks are enforced
  - plain/relu forms use cast pre-quant mode derived by `GetCastPreQuantMode<SrcDType, DstDType>()`
  - scalar-quant forms use `GetScalarPreQuantMode<SrcDType, DstDType>()`
  - vector-quant forms require an `FpTileData` operand with `FpTileData::Loc == TileType::Scaling`, and use `GetVectorPreQuantMode<SrcDType, DstDType>()`

### A5 implementation checks

- Supported paths include:
  - `TileType::Mat -> TileType::Left/Right/Bias/Scaling/ScaleLeft/ScaleRight`
  - `TileType::Vec -> TileType::Vec/TileType::Mat`
  - `TileType::Acc -> TileType::Vec/TileType::Mat`
  - specific `ND -> ZZ` and related internal path variants handled by the A5 implementation

## Examples

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

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
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

## Related Ops / Family Links

- Family overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in family: [pto.tfillpad_expand](./tfillpad-expand.md)
- Next op in family: [pto.tmov_fp](./tmov-fp.md)
