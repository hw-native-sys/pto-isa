# TQUANT

## Tile Operation Diagram

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## Introduction

Quantize a higher-precision tile (`FP32` / `BF16` / `FP16`) into a lower-precision format, producing the quantized data tile plus auxiliary per-group exponent / max / scaling tiles. The destination format, scale algorithm, and group axis are compile-time template parameters.

| Destination family | Formats | Grouping | Scale algorithm |
|--------------------|---------|----------|-----------------|
| **Microscaling (MX)** | MXFP8 (e4m3), MXFP4 (e2m1) | 32 elements per shared-exponent group | OCP, NV |
| **Integer** | INT8 (symmetric / asymmetric) | per-tile scale (+ optional offset) | affine |

## Quantization Process

### MX formats (3 stages, group size G = 32)

For a tile $x \in \mathbb{R}^{M \times N}$ grouped along axis `grp_axis` (ND: axis-1/columns, DN: axis-0/rows):

| Stage | Operation | Output |
|-------|-----------|--------|
| **1. Group max** | $m_g = \max_{i \in g} \|x_i\|$ | `max` (scratch, FP) |
| **2. Exponent + scale** | $s_g = \mathrm{biasedExp}(m_g) - e_{\max}$; $\alpha_g = 2^{254 - s_g}$ | `exp` (E8M0, 1 B/group), `scaling` (scratch, FP) |
| **3. Scale + cast** | $q_i = \mathrm{clip}_{[-V_{\max},V_{\max}]}(x_i \cdot \alpha_g) \to$ dest format | `dst` (FP8 / packed FP4) |

- $e_{\max}$ = max destination exponent (8 for e4m3, 1 for e2m1).
- $V_{\max}$ = destination MAX_NORM (448 for e4m3, 6 for e2m1).
- Stages 1–2 use exact IEEE-754 bit manipulation (no FP `log`/`floor`); stage 3 uses hardware cast with stochastic rounding (`SPR.CTRL[50]=1`).
- **ND** ("normal direction", `grp_axis=1`): groups of 32 consecutive **columns** — the default/standard grouping. **DN** (`grp_axis=0`): groups of 32 consecutive **rows** — a transposed-style grouping along axis-0; exponent tile shape `M̂×N`, `M̂ = M/32`.

### Integer INT8 (affine, 5-stage cast)

$$q_i = \mathrm{round}\!\left(\frac{x_i}{\mathrm{scale}}\right) + \mathrm{offset}, \qquad q_i \in [-128, 127]$$

No group structure; `scale` (and asymmetric `offset`) are per-tile FP32 scalars/vectors. To avoid double-rounding on A2/A3, the cast chain is `FP32 → S32 → FP32 → FP16 → INT8` (5 stages, via a `tmp` tile = src size); A5 uses native broadcast + cast (no `tmp`). Input must be **FP32**.

## C++ Intrinsics

Declared in `include/pto/common/pto_instr.hpp`.

### MX — grouped (`grp_axis` + `MxQuantAlg`) — recommended

```cpp
template <int grp_axis, auto mx_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events);
```

| Template param | Values | Meaning |
|----------------|--------|---------|
| `grp_axis` | `0` = DN (axis-0 groups), `1` = ND (axis-1 groups) | Quantization group axis |
| `mx_alg` | `MxQuantAlg::OcpMxFp8E4M3`, `NvMxFp8E4M3`, `OcpMxFp4E2M1`, `NvMxFp4E2M1` | Format + scale algorithm |

### MX — ND legacy (`QuantType` + `QuantScaleAlg`)

```cpp
template <auto quant_type, typename ...Tiles, auto scale_alg = QuantScaleAlg::OCP, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events);

// with explicit ZZ exponent store-mode
template <auto quant_type, auto store_mode, typename ...Tiles, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, TileDataExp *exp_zz, WaitEvents &...events);
```

| `quant_type` | Format | `scale_alg` |
|--------------|--------|-------------|
| `QuantType::MXFP8` | e4m3 + E8M0 | OCP / NV |
| `QuantType::MXFP4_E2M1` | e2m1 + E8M0 | OCP / NV |

### Integer INT8

```cpp
// symmetric
template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale,
                            TileDataPara *offset = nullptr, WaitEvents &...events);
// with explicit scratch (A2/A3)
template <auto quant_type, typename ...Tiles, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataTmp &tmp,
                            TileDataPara *offset = nullptr, WaitEvents &...events);
```

| `quant_type` | `offset` | `dst` dtype | Mode |
|--------------|----------|-----------|------|
| `QuantType::INT8_SYM` | `nullptr` | `int8_t` | symmetric ($q = \mathrm{round}(x/\mathrm{scale})$) |
| `QuantType::INT8_ASYM` | provided | `uint8_t` | asymmetric ($q = \mathrm{round}(x/\mathrm{scale}) + \mathrm{offset}$) |

> The `tmp`-aware overload (`dst, src, scale, tmp, offset`) exists for A2/A3 interface parity. On A5 `tmp` is unused; on A2/A3 it must be `M×N` FP32 (the S32 cast intermediate).

## Tile Sizes & Data Types

For an input tile of shape $M \times N$ elements (dtype $T \in \{\mathrm{FP32}, \mathrm{BF16}, \mathrm{FP16}\}$), group size $G = 32$:

### MXFP8 (e4m3)

| Tile | dtype | Shape (ND) | Shape (DN) | Bytes |
|------|-------|------------|------------|-------|
| `src` | $T$ | $M \times N$ | $M \times N$ | $M \cdot N \cdot \mathrm{sizeof}(T)$ |
| `dst` | `int8_t` (e4m3 alias) | $M \times N$ | $M \times N$ | $M \cdot N$ |
| `exp` | `uint8_t` (E8M0) | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32$ |
| `max` (scratch) | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |
| `scaling` (scratch) | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |

### MXFP4 (e2m1)

Same as MXFP8 except:
| Tile | dtype | Bytes |
|------|-------|-------|
| `dst` | `float4_e2m1x2_t` (2 e2m1 codes packed per byte) | $M \cdot N / 2$ |

> **Input restriction:** MXFP4 accepts **FP16/BF16 only** (not FP32).

### INT8

| Tile | dtype | Shape | Bytes |
|------|-------|-------|-------|
| `src` | `float32_t` | $M \times N$ | $M \cdot N \cdot 4$ |
| `dst` (SYM) | `int8_t` | $M \times N$ | $M \cdot N$ |
| `dst` (ASYM) | `uint8_t` | $M \times N$ | $M \cdot N$ |
| `scale` | FP32 scalar/vector | per-tile | — |
| `offset` (ASYM) | FP32 scalar/vector | per-tile | — |
| `tmp` (A2/A3 only) | FP32 | $M \times N$ | $M \cdot N \cdot 4$ |

> **`tmp` tile (A2/A3 only):** must be the **same size as `src`** ($M \times N$ FP32 = $4MN$ bytes). It holds the FP32→S32 intermediate (A3 has no in-place `tcvt`). A5 accepts the same `tmp` argument for interface parity but **does not use it** (A5 broadcasts scale/offset natively via `vlds BRC_B32`).

## Constraints

| Constraint | Applies to | Reason |
|------------|------------|--------|
| $M \bmod 16 = 0$ | ND MX (ZZ layout) | 16-row ZZ blocks |
| $M \bmod 32 = 0$ | DN MX | axis-0 group divisibility |
| $M \bmod 64 = 0$ | DN MX + ZZ conversion | δ-pairing ($\hat M / 2$ integer) |
| $N \bmod 32 = 0$ | all MX | group size $G = 32$ |
| $N \bmod 64 = 0$ | ND MX + ZZ conversion | even number of exponent groups |
| $R \cdot C \le 59461$ | MX (UB 256KB) | buffer budget with reuse |
| BF16/FP16: `validCols % 32 != 0` → zero-padded to `StaticCols` | MX B16 path | group alignment |

## Output Layout & Layout Conversion

TQUANT writes **ND** (row-major) output by default. The Cube Unit consumes two fractal layouts, produced by separate `TMOV` instructions:

| Output | Native (TQUANT) | Cube layout | Conversion |
|--------|-----------------|-------------|------------|
| FP8 / FP4 data | ND | NZ (ColMajor+RowMajor fractal) | `TMOV(dstNZ, dst)` (2-arg) |
| E8M0 exponents (ND groups) | ND | ZZ (zigzag, `[16,2]` boxes) | `TMOV(e8Zz, e8, tmp)` (3-arg) |
| E8M0 exponents (DN groups) | DN | ZZ | `TMOV<0>(e8Zz, e8Dn, tmp)` (3-arg, `grp_axis=0`) |

DN data FP8 mantissas share the same physical addresses as ND (the `(r,c)` element is identical), so the 2-arg `TMOV` ND→NZ is correct for DN data unchanged. Only the **exponent** path differs (DN→ZZ via `TMOV<0>`). See `TQUANT_DN.md` for the DN→ZZ transform.

## Supported Input Dtypes

| Format | Accepted input dtypes | Notes |
|--------|----------------------|-------|
| MXFP8 | FP32, BF16, FP16 | FP32: in-place FP8 output (4:1). BF16/FP16: source zero-padded to `StaticCols`; upcast to FP32 before cast (no direct b16→e4m3). |
| MXFP4 (e2m1) | **FP16, BF16 only** (not FP32) | output `float4_e2m1x2_t` (packed). |
| INT8 (sym/asym) | **FP32 only** | SYM→`int8_t`, ASYM→`uint8_t`. A2/A3 need `tmp` = src size. |

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

### AS Level 1 (SSA)

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Examples

```cpp
// MXFP8, DN grouping (axis-0 groups), OCP scale
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);

// MXFP8, ND grouping (legacy)
TQUANT<QuantType::MXFP8>(fp8Tile, srcTile, &e8NdTile, &maxTile, &scalingTile);

// MXFP4 E2M1, DN, NV scale
TQUANT<0, MxQuantAlg::NvMxFp4E2M1>(fp4Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);

// INT8 symmetric
TQUANT<QuantType::INT8_SYM>(int8Tile, srcTile, scale);

// Full MXFP8 DN pipeline: quantize + layout convert for cube
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);
TMOV(fp8NZTile, fp8Tile);                  // data  ND→NZ
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);      // exp   DN→ZZ
```

See `TQUANT_DN.md` for the DN→ZZ transform details and `tests/npu/a5/src/st/testcase/tquant_dn/` for a complete ST example.
