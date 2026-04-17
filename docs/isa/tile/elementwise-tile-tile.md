# Elementwise Tile-Tile Instruction Set

Elementwise tile-tile operations perform lane-wise binary and unary operations over tile valid regions. These are the most commonly used tile compute operations in PTO programs.

## Operations

| Operation | Description | Category | C++ Intrinsic |
|-----------|-------------|----------|----------------|
| [pto.tadd](./ops/elementwise-tile-tile/tadd.md) | Elementwise addition | Binary | `TADD(dst, src0, src1)` |
| [pto.tabs](./ops/elementwise-tile-tile/tabs.md) | Elementwise absolute value | Unary | `TABS(dst, src)` |
| [pto.tand](./ops/elementwise-tile-tile/tand.md) | Elementwise bitwise AND | Binary | `TAND(dst, src0, src1)` |
| [pto.tor](./ops/elementwise-tile-tile/tor.md) | Elementwise bitwise OR | Binary | `TOR(dst, src0, src1)` |
| [pto.tsub](./ops/elementwise-tile-tile/tsub.md) | Elementwise subtraction | Binary | `TSUB(dst, src0, src1)` |
| [pto.tmul](./ops/elementwise-tile-tile/tmul.md) | Elementwise multiplication | Binary | `TMUL(dst, src0, src1)` |
| [pto.tmin](./ops/elementwise-tile-tile/tmin.md) | Elementwise minimum | Binary | `TMIN(dst, src0, src1)` |
| [pto.tmax](./ops/elementwise-tile-tile/tmax.md) | Elementwise maximum | Binary | `TMAX(dst, src0, src1)` |
| [pto.tcmp](./ops/elementwise-tile-tile/tcmp.md) | Elementwise comparison | Binary | `TCMP(dst, src0, src1, cmp)` |
| [pto.tdiv](./ops/elementwise-tile-tile/tdiv.md) | Elementwise division | Binary | `TDIV(dst, src0, src1)` |
| [pto.tshl](./ops/elementwise-tile-tile/tshl.md) | Elementwise shift left | Binary | `TSHL(dst, src0, src1)` |
| [pto.tshr](./ops/elementwise-tile-tile/tshr.md) | Elementwise shift right | Binary | `TSHR(dst, src0, src1)` |
| [pto.txor](./ops/elementwise-tile-tile/txor.md) | Elementwise bitwise XOR | Binary | `TXOR(dst, src0, src1)` |
| [pto.tlog](./ops/elementwise-tile-tile/tlog.md) | Elementwise natural logarithm | Unary | `TLOG(dst, src)` |
| [pto.trecip](./ops/elementwise-tile-tile/trecip.md) | Elementwise reciprocal | Unary | `TRECIP(dst, src)` |
| [pto.tprelu](./ops/elementwise-tile-tile/tprelu.md) | Elementwise parameterized ReLU | Binary | `TPRELU(dst, src0, src1)` |
| [pto.taddc](./ops/elementwise-tile-tile/taddc.md) | Three-input fused addition | Ternary-like Binary | `TADDC(dst, src0, src1, src2)` |
| [pto.tsubc](./ops/elementwise-tile-tile/tsubc.md) | Three-input fused subtract/add | Ternary-like Binary | `TSUBC(dst, src0, src1, src2)` |
| [pto.tcvt](./ops/elementwise-tile-tile/tcvt.md) | Elementwise type conversion | Unary | `TCVT(dst, src)` |
| [pto.tsel](./ops/elementwise-tile-tile/tsel.md) | Elementwise conditional selection | Ternary | `TSEL(dst, src0, src1, cmp)` |
| [pto.trsqrt](./ops/elementwise-tile-tile/trsqrt.md) | Elementwise reciprocal square root | Unary | `TRSQRT(dst, src)` |
| [pto.tsqrt](./ops/elementwise-tile-tile/tsqrt.md) | Elementwise square root | Unary | `TSQRT(dst, src)` |
| [pto.texp](./ops/elementwise-tile-tile/texp.md) | Elementwise exponential | Unary | `TEXP(dst, src)` |
| [pto.tnot](./ops/elementwise-tile-tile/tnot.md) | Elementwise bitwise NOT | Unary | `TNOT(dst, src)` |
| [pto.trelu](./ops/elementwise-tile-tile/trelu.md) | Elementwise ReLU | Unary | `TRELU(dst, src)` |
| [pto.tneg](./ops/elementwise-tile-tile/tneg.md) | Elementwise negation | Unary | `TNEG(dst, src)` |
| [pto.trem](./ops/elementwise-tile-tile/trem.md) | Elementwise remainder | Binary | `TREM(dst, src0, src1)` |
| [pto.tfmod](./ops/elementwise-tile-tile/tfmod.md) | Elementwise floating-point modulo | Binary | `TFMOD(dst, src0, src1)` |

## Mechanism

Binary operations combine two source tiles lane-by-lane. Unary operations transform one source tile lane-by-lane. The iteration domain is the destination tile's valid region.

For each lane `(r, c)` in the destination's valid region:

$$ \mathrm{dst}_{r,c} = f(\mathrm{src0}_{r,c}, \mathrm{src1}_{r,c}) $$

For ternary selection (`TSEL`):

$$ \mathrm{dst}_{r,c} = (\mathrm{cmp}_{r,c} \neq 0) \; ?\; \mathrm{src0}_{r,c} \;:\; \mathrm{src1}_{r,c} $$

## Valid Region Compatibility

All elementwise tile-tile operations iterate over the **destination tile's valid region**. For each lane `(r, c)` in the destination's valid region:

- The corresponding lane `(r, c)` from each source tile is read, **regardless of whether that lane is within the source tile's own valid region**
- Source tiles whose valid region does not cover `(r, c)` read **implementation-defined values**
- Programs MUST NOT rely on any particular value being read from an out-of-region source lane unless the operation explicitly documents the behavior

## `_c` Variants

Within the current canonical per-op pages and intrinsic signatures, the `_c` suffix in this instruction family does **not** denote a generic saturating-arithmetic convention:

- `TADDC` is a three-input fused add: `src0 + src1 + src2`
- `TSUBC` is a three-input fused subtract/add: `src0 - src1 + src2`

Readers MUST NOT infer saturating semantics from the suffix alone; always treat the individual per-op page as the source of truth.

## Type Support by Target Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8 / u8 | Yes | Yes | Yes |
| i16 / u16 | Yes | Yes | Yes |
| i32 / u32 | Yes | Yes | Yes |
| i64 / u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |

## Constraints

- Tile layout, shape, and valid-region state affect legality.
- Type support varies by target profile (see per-op pages for exact restrictions).
- Comparison operations (`TCMP`) produce a **predicate tile**; arithmetic operations produce a **numeric tile**.
- Conversion operations (`TCVT`) may change element type between source and destination; dtype sizes may differ.
- All source and destination tiles MUST have the same physical shape `(Rows, Cols)`.
- Shift operations (`TSHL`, `TSHR`) interpret the second operand as an unsigned shift count; shift count MUST be `<` element bit-width.

## Cases That Are Not Allowed

- **MUST NOT** assume implicit broadcasting, reshaping, or valid-region repair.
- **MUST NOT** rely on a defined value from a source tile lane outside its valid region.
- **MUST NOT** infer a generic saturating-arithmetic meaning from the `_c` suffix alone.
- **MUST NOT** use a shift count `>=` element bit-width.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Binary elementwise
template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TADD(TileDst& dst, TileSrc0& src0, TileSrc1& src1);

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TMUL(TileDst& dst, TileSrc0& src0, TileSrc1& src1);

template <typename TileData, typename TileData0, typename TileData1, typename TileData2>
PTO_INST RecordEvent TADDC(TileData& dst, TileData0& src0, TileData1& src1, TileData2& src2);

// Unary elementwise
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TABS(TileDst& dst, TileSrc& src);

template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TEXP(TileDst& dst, TileSrc& src);

// Type conversion
template <typename TileDst, typename TileSrc>
PTO_INST RecordEvent TCVT(TileDst& dst, TileSrc& src);

// Comparison (produces predicate tile)
template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TCMP(TileDst& dst, TileSrc0& src0, TileSrc1& src1, CompareMode cmp);
```

## Throughput and Latency (A2/A3)

Tile elementwise operations use the Vector Core (PIPE_V) via the CCE instruction set. The performance model is defined in `include/pto/costmodel/a2a3/`.

### Cycle Model Formula

```
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

Where `repeats` is computed from tile layout and valid region shape.

### CCE Instruction Parameters

| Metric | Constant | Value (cycles) | Applies To |
|--------|-----------|---------------|------------|
| Startup latency | `A2A3_STARTUP_BINARY` | 14 | all arithmetic binary ops (`vadd`, `vmul`, `vsub`) |
| Startup latency | `A2A3_STARTUP_REDUCE` | 13 | transcendental/unary ops (`vexp`, `vsqrt`, `vabs`) |
| Completion: FP32 | `A2A3_COMPL_FP_BINOP` | 19 | `vadd`, `vsub` (f32), `vcadd`, `vcmax` |
| Completion: INT binary | `A2A3_COMPL_INT_BINOP` | 17 | `vadd`, `vsub` (int16) |
| Completion: INT mul | `A2A3_COMPL_INT_MUL` | 18 | `vmul` (int) |
| Completion: FP transcendental | `A2A3_COMPL_FP32_EXP` | 26 | `vexp` (f32) |
| Completion: FP transcendental | `A2A3_COMPL_FP32_SQRT` | 27 | `vsqrt` (f32) |
| Per-repeat throughput | `A2A3_RPT_1` | 1 | unary/scalar ops |
| Per-repeat throughput | `A2A3_RPT_2` | 2 | binary ops (`vadd`, `vmul`) |
| Per-repeat throughput | `A2A3_RPT_4` | 4 | transcendental ops (f16 exp/sqrt) |
| Pipeline interval | `A2A3_INTERVAL` | 18 | all vector ops |
| Pipeline interval (copy) | `A2A3_INTERVAL_VCOPY` | 13 | `vmov`, `copy_ubuf_to_ubuf` |

### Instruction Repeat Calculation

The `TBinOp.hpp` / `TBinSOp.hpp` / `TUnaryOp.hpp` headers compute `repeats` from tile geometry:

**Continuous (fast) path** (source stride == destination stride == 1):
```
repeats = validRow × validCol / elementsPerRepeat
```

**General path**: handles arbitrary stride combinations, including small-shape optimization (`Bin1LNormModeSmall`) where one repeat covers an entire row.

### Layout and Shape Impact

Tile layout (`RowMajor`, `ColMajor`, `Zigzag`, etc.) affects stride alignment and determines which optimization path is taken:

| Layout | Stride Pattern | Optimization |
|--------|---------------|-------------|
| `RowMajor` | src0/1: `(1, cols)`, dst: `(1, cols)` | Continuous fast path when col-aligned |
| `ColMajor` | src0/1: `(rows, 1)`, dst: `(rows, 1)` | General path |
| Mixed layouts | Mixed stride patterns | General path only |

**Shape-sensitive special cases** (FP32, hardcoded at compile time):

| Valid Shape | Instruction Sequence |
|-------------|---------------------|
| 64×128 (TROWSUM) | `vcgadd`*128 → PIPE_V → `vadd`*8 → PIPE_V → `vcgadd`*8 → PIPE_V |
| 32×256 (TROWSUM) | `vcgadd`*128 → PIPE_V → `vadd`*8 → PIPE_V → `vadd`*4 → PIPE_V → `vcgadd`*4 → PIPE_V |
| 16×512 (TROWSUM) | `vcgadd`*128 → PIPE_V → `vcgadd`*16 → PIPE_V → `vcgadd`*2 → PIPE_V |
| 8×1024 (TROWSUM) | `vcgadd`*128 → PIPE_V → `vcgadd`*16 → PIPE_V → `vadd`*8 → PIPE_V → `vcgadd`*8 → PIPE_V |

### Bandwidth Model for Tile Movements

| Transfer Path | Bandwidth (B/cycle) | Constant |
|---------------|---------------------|----------|
| GM → Vec Buffer (TLOAD) | 128 | `A2A3_BW_GM_VEC` |
| Vec → Vec (TMOV) | 128 | `A2A3_BW_VEC_VEC` |
| GM → Mat (TLOAD Mat) | 256 | `A2A3_BW_GM_MAT` |
| Mat → L0A (TMOV Left) | 256 | `A2A3_BW_MAT_LEFT` |
| Mat → L0B (TMOV Right) | 128 | `A2A3_BW_MAT_RIGHT` |
| Mat → Mat (TEXTRACT) | 32 | `A2A3_BW_MAT_MAT` |

Transfer cost: `ceil(bufferSize / bandwidth)` cycles.

### Accuracy and Testing

The cost model is validated against cycle-accurate profiling with ≥99% accuracy (error < 1%):
- Tests in `tests/costmodel/tadd_kernel.cpp` etc.
- Run via `tests/run_costmodel.py --testcase <name>`
- Build with `-D__COSTMODEL` preprocessor flag

---

## See Also

- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Tile instruction set](../instruction-surfaces/tile-instructions.md) — Instruction Set description
