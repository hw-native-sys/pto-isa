# TDEQUANT

## Introduction

Dequantize a low-precision quantized tile (`S8` / `S16`) into a high-precision `FP32` tile. Each element undergoes **affine dequantization** — the inverse of `TQUANT` integer quantization (INT8 symmetric / asymmetric):

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} - \mathrm{offset}_{i}) \cdot \mathrm{scale}_{i} $$

`scale` and `offset` are **per-row** FP32 parameters (broadcast across the columns), so a single instruction performs the full "subtract zero-point + rescale" dequantization.

## Math Interpretation

Conceptually, for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \bigl(\mathrm{src}_{i,j} - \mathrm{offset}_{i}\bigr) \cdot \mathrm{scale}_{i} $$

- `src`: the quantized integer code (`S8` or `S16`).
- `scale`, `offset`: per-row FP32 dequantization parameters; `scale` valid-column count is `paraCols = max(1, scale.GetValidCol())` with parameter column index `paraCol = min(j, paraCols - 1)` — i.e. parameters broadcast along the column axis.
- Inverse of `TQUANT` integer affine quantization: `TQUANT` has $q = \mathrm{round}(x / \mathrm{scale}) + \mathrm{offset}$, hence $x = (q - \mathrm{offset}) \cdot \mathrm{scale}$.

> Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined. `scale` and `offset` are ISA-visible tile operands (not compiler scratch).

## C++ Intrinsics

Declared in `include/pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|-----------|-----------|---------|
| `dst` | output | Dequantized result tile, `FP32`, row-major |
| `src` | input | Quantized source tile, `S8` or `S16`, row-major, same valid shape as `dst` |
| `scale` | input | Per-row inverse scale, `FP32`, broadcast across columns |
| `offset` | input | Per-row zero-point, `FP32`, broadcast across columns |
| `events...` | input | Wait events (`WaitEvents`); an implicit `TSYNC` precedes the op |

`scale` and `offset` must share a type (`TileDataPara`) whose dtype matches `dst` (all `FP32`).

## Tile Sizes & Data Types

For a source/destination valid tile shape of $M \times N$:

| Tile | dtype | Valid shape | Layout | Notes |
|------|-------|-------------|--------|-------|
| `dst` | `float32_t` | $M \times N$ | RowMajor | Dequantized result |
| `src` | `int8_t` or `int16_t` | $M \times N$ | RowMajor | Quantized integer code; dtype selects the unpack path |
| `scale` | `float32_t` | $M \times 1$ (per row) | ColMajor / row broadcast | Broadcast across columns (`BRC_B32`) |
| `offset` | `float32_t` | $M \times 1$ (per row) | ColMajor / row broadcast | Broadcast across columns (`BRC_B32`) |

> `scale`/`offset` valid rows must equal `dst` valid rows; columns are broadcast in 32-byte blocks, so the typical form is one scalar per row (shape $M \times 1$).

## Supported Input Dtypes

| Source dtype | Destination dtype | scale/offset dtype | Notes |
|--------------|-------------------|--------------------|-------|
| `S8` (`int8_t`) | `FP32` | `FP32` | Unpacked via `UNPK4_B8`, then cast to FP32 |
| `S16` (`int16_t`) | `FP32` | `FP32` | Unpacked via `UNPK_B16`, then cast to FP32 |

> `dst`, `scale`, and `offset` must share a dtype and all be `FP32`; `src` supports only `S8` / `S16`. Other dtype combinations are illegal (rejected by an in-implementation `static_assert`).

## Implementation Notes

TDEQUANT runs on the vector pipeline (`PIPE_V`) and needs no `tmp` scratch tile (unlike the 5-stage cast chain of `TQUANT` on A2/A3):

1. **Load and unpack `src`**: `S8` via `UNPK4_B8`, `S16` via `UNPK_B16`, then `vcvt` to FP32 (on kirinX90, `S8` uses the `US_B8` + interleave path).
2. **Broadcast-load parameters**: `scale` and `offset` are loaded with `vlds ... BRC_B32`, broadcasting in 32-byte blocks across each row.
3. **Compute**: `vsub(dst, src, offset)` followed by `vmul(dst, dst, scale)` — subtract zero-point, then rescale.

## Encoding

TDEQUANT is a TEPL (Tile Elementwise Pipeline) complex-transform instruction:

```text
BSTART.TEPL TDEQUANT, DataType +
B.DATR(optional) +
B.DIM LB0 +
B.DIM (LB1/LB2 for 2D) +
B.IOT
```

| Field | Value |
|-------|-------|
| Mode | 3 (complex transform) |
| Function | 11 |
| TileOp | `0x6B` |
| Operands (`B.IOT`) | `dst, src, scale, offset` |

## Constraints

| Constraint | Applies to | Reason |
|------------|------------|--------|
| `dst` and `src` must be row-major | all targets | per-row parameter broadcast |
| `dst` and `src` share the valid shape ($M \times N$) | all targets | one-to-one element mapping |
| `scale` and `offset` valid rows == `dst` valid rows | all targets | one parameter group per row |
| `dst`/`scale`/`offset` dtypes must match and be `FP32` | all targets | dequantization output precision |
| `src` ∈ {`S8`, `S16`} | all targets | quantized integer code width |

## Examples

```cpp
// src: int8/int16 quantized codes; scale/offset: per-row FP32 parameters
TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
```

See `tests/npu/a5/src/st/testcase/tdequant/` (A5), `tests/npu/a2a3/src/st/testcase/tdequant/` (A2/A3), and `tests/cpu/st/testcase/tdequant/` (CPU reference) for complete ST examples.
