# TSTORE

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

![TSTORE tile operation](../../figures/isa/TSTORE.svg)

## Introduction

Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters.

## Math Interpretation

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);
```

`TSTORE_FP(...)` is retained for source compatibility with the legacy fp-quantized form and maps directly to
`TSTORE_IMPL(dst, src, fp)`. The canonical `TSTORE(..., fp, ...)` overload is selected only for
`FpTileData::Loc == TileType::Scaling`; backend implementations may apply additional legality checks.

The move-out `STPhase` values are not symmetric with the accumulation side. The `TMATMUL` that produced
the L0C result must already be `AccPhase::Final`, that is, the data is ready. `STPhase::Final` marks the
last move-out and releases the unit flag. `STPhase::Partial` is only for the non-last move-outs when one
L0C tile is drained more than once; it does not release the flag. Pairing `STPhase::Partial` with
`AccPhase::Partial` makes the fixpipe wait on a flag that never arrives, which hangs.

This rule is confirmed by simulator tests on Ascend 950PR for all three move-out paths: L0C to L1, L0C to UB
and L0C to GM. `TSTORE` uses the L0C-to-GM `copy_matrix_cc_to_gm`, where the unit flag goes into an Xt
register rather than a positional argument; it is covered by `tstore_acc2gm`'s
`case_vector_quant_uf_multi_drain`.

On Ascend 950PR hardware with CANN 9.2.0, all 3 targeted `tstore_acc2gm` tests
(`case38`, `case_vector_quant_uf_final`, and `case_vector_quant_uf_multi_drain`) passed with max diff 0.
They cover the per-channel quantized baseline, `Final`, and `Partial` followed by `Final`.
Together with TEXTRACT, TINSERT, and TMOV, all 24 targeted move-out tests passed on hardware;
this result covers the selected cases, not the full A5 ST suite.

The vector-quantized `STPhase` form is exposed only on targets with matching backend support
(A2A3, A5, kirin9030, kirin9050, and CPU simulator).


## L2 cache hint

Optional first template `TStoreL2Hint l2Control` (default `NormalFirstVictim`):

```cpp
TSTORE(dst, src);
TSTORE<TStoreL2Hint::NotAllocClean>(dst, src);
TSTORE<TStoreL2Hint::NotAllocClean, AtomicType::AtomicAdd>(dst, src);
```

Put `TStoreL2Hint` first when combining with `AtomicType` / `STPhase` / `ReluPreMode`. Do not insert a defaulted hint before `TileData` on the legacy overload set.

Supported `TStoreL2Hint` values:

| Enumerator | Value | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | no-op | yes |
| NormalLastVictim | 1 | no-op | yes |
| NormalPersistent | 2 | no-op | yes |
| NotAllocClean | 4 | no-op | yes |

On A2/A3, L2 hints have **no effect** (all values are no-ops; no store L2 control). A5 passes listed values through to DMA. CPU / costmodel ignore it.


## Constraints

- **Implementation checks (A2A3)**:
    - Source tile location must be one of: `TileType::Vec`, `TileType::Mat`, `TileType::Acc`.
    - Runtime: all `dst.GetShape(dim)` values and `src.GetValidRow()/GetValidCol()` must be `> 0`.
    - For `TileType::Vec` / `TileType::Mat`:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
      In that special case the traversal follows the tile layout, not the `GlobalTensor` layout: the vector is
      written as one contiguous burst and the other axis' stride is not used. A ColMajor `[N, 1]` tile stored
      through an ND `GlobalTensor` therefore occupies `N` consecutive elements, not one element per row stride.
    - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
    - For `TileType::Acc`:
      - Supported layout conversions: NZ2ND, NZ2NZ, NZ2NC1HWC0, NZ2NDC1HWC0. NZ2DN is **not** supported.
      - Destination layout must be ND, NZ, NC1HWC0, or NDC1HWC0.
      - Source dtype must be `int32_t` or `float`.
      - When not using quantization, destination dtype must be `int32_t/float/half/bfloat16_t`.
      - ACC-to-GM dtype support:

        | Calling convention | Source dtype | Supported destination dtype |
        | --- | --- | --- |
        | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
        | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
        | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t` |
        | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half` |

        Other cross-type combinations are not supported.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ, NC1HWC0, or NDC1HWC0 then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
- **Implementation checks (A5)**:
    - Source tile location must be `TileType::Vec` or `TileType::Acc` (no `Mat` store on this target).
    - For `TileType::Vec`:
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float8_e8m0_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
      In that special case the traversal follows the tile layout, not the `GlobalTensor` layout: the vector is
      written as one contiguous burst and the other axis' stride is not used. A ColMajor `[N, 1]` tile stored
      through an ND `GlobalTensor` therefore occupies `N` consecutive elements, not one element per row stride.
    - Additional alignment constraints are enforced (e.g., for ND the row-major width in bytes must be a multiple of 32; for DN the column-major height in bytes must be a multiple of 32, with special-case exceptions).
    - For `TileType::Acc` / ACC source tiles:
      - Supported layout conversions: NZ2ND, NZ2NZ, NZ2NHWC, NZ2NCHW, NZ2NCDHW. NZ2DN is **not** supported.
      - Destination layout must be ND, NZ, NHWC, NCHW, or NCDHW; source dtype must be `int32_t` or `float`.
    - When not using quantization, destination dtype must be `int32_t/float/half/bfloat16_t`.
    - ACC-to-GM dtype support:

      | Calling convention | Source dtype | Supported destination dtype |
      | --- | --- | --- |
      | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
      | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
      | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t`, `half`, `bfloat16_t`, `hifloat8_t`, `float8_e4m3_t`, `float` |
      | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half`, `bfloat16_t` |

      Other cross-type combinations are not supported.
    - Static shape constraints match A2A3 for rows/cols; `AtomicAdd` additionally restricts destination dtype to supported atomic types.
- **Valid region**:
    - The implementation uses `src.GetValidRow()` / `src.GetValidCol()` as the transfer size.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```
