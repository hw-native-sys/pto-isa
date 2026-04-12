<!-- Generated from `docs/isa/tile/elementwise-tile-tile.md` -->

# Elementwise Tile-Tile Family

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
| [pto.taddc](./ops/elementwise-tile-tile/taddc.md) | Saturating elementwise addition | Binary | `TADDC(dst, src0, src1)` |
| [pto.tsubc](./ops/elementwise-tile-tile/tsubc.md) | Saturating elementwise subtraction | Binary | `TSUBC(dst, src0, src1)` |
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

## Saturating Variants

Operations with the `_c` suffix perform saturating arithmetic instead of wrapping arithmetic:

| Base Op | Saturating Op | Overflow/Underflow Behavior |
|---------|--------------|--------------------------|
| `TADD` | `TADDC` | Clamp to type min/max |
| `TSUB` | `TSUBC` | Clamp to type min/max |

Programs MUST NOT assume that `TADDC` and `TADD` produce identical results when overflow does not occur; they MAY differ even for in-range values due to implementation precision choices.

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
- **MUST NOT** assume `TADDC`/`TSUBC` are bit-identical to `TADD`/`TSUB` for all inputs.
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

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TADDC(TileDst& dst, TileSrc0& src0, TileSrc1& src1);

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

## See Also

- [Tile families](../instruction-families/tile-families.md) — Family overview
- [Tile instruction surface](../instruction-surfaces/tile-instructions.md) — Surface description
