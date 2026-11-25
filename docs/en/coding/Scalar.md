# Scalar Parameters and Enums

Many PTO intrinsics take scalar parameters in addition to tiles (e.g., comparison modes, rounding modes, atomic modes, or literal constants).

This document summarizes the scalar/enumeration types that appear in the public intrinsics in `include/pto/common/pto_instr.hpp`.

## Scalar values

Some instructions take scalar values as plain C++ types:

- `TADDS/TMULS/TDIVS/TEXPANDS`: scalar is `TileData::DType`.
- `TMINS`: scalar is a template type `T` and must be convertible to the tile element type.
- `TCI`: scalar `S` is a template type `T` and must match `TileData::DType` (enforced by `static_assert` in the implementation).

## PTO ISA type mnemonics (reference)

ISA documentation uses short type mnemonics (e.g., `fp16`, `s8`) when describing instruction semantics. Backends may support only a subset at any given time; see `include/README.md` for implementation status.

### Integer types

| Kind | Mnemonics |
|---|---|
| Signed | `s4`, `s8`, `s16`, `s32`, `s64` |
| Unsigned | `u4`, `u8`, `u16`, `u32`, `u64` |

### Floating-point types

| Kind | Mnemonics |
|---|---|
| 4-bit float families | `fp4`, `hif4`, `mxfp4` |
| 8-bit float families | `fp8`, `hif8`, `mxfp8` |
| 16-bit float families | `bf16`, `fp16` |
| 32-bit float families | `tf32`, `hf32`, `fp32` |
| 64-bit float | `fp64` |

### Bit-width (typeless) values

| Kind | Mnemonics |
|---|---|
| Typeless bits | `b4`, `b8`, `b16`, `b32`, `b64` |

#### Compatibility rules (ISA convention)

Two mnemonics are considered compatible when they have the same bit-width, and either:

- they are the same kind, or
- they are signed vs unsigned integers of the same width, or
- one side is a typeless bits type (`b*`) of the same width.

These are documentation-level rules used to describe instruction legality. Individual instructions may further restrict types.

## Core enums

All enums below are available via `#include <pto/pto-inst.hpp>`.

### `pto::RoundMode`

Defined in `include/pto/common/type.hpp`. Used by `TCVT` to specify rounding behavior (e.g., `RoundMode::CAST_RINT`).
Supported modes and defaults depend on the conversion and backend; see [TCVT](../isa/TCVT.md).

### `pto::CmpMode`

Defined in `include/pto/common/type.hpp`. Used by `TCMPS` (and `TCMP`) for per-element comparisons (`EQ/NE/LT/GT/GE/LE`).

### `pto::MaskPattern`

Defined in `include/pto/common/type.hpp`. Used by the mask-pattern `TGATHER` variant to select a predefined 0/1 mask pattern.

### `pto::AtomicType`

Defined in `include/pto/common/constants.hpp`. Used as the template parameter to `TSTORE<..., AtomicType::AtomicAdd>` (or `AtomicNone`).

### `pto::AccToVecMode` and `pto::ReluPreMode`

Defined in `include/pto/common/constants.hpp`. Used by `TMOV` overloads when moving from accumulator tiles with optional quantization and/or ReLU behavior.

### `pto::PadValue`

Defined in `include/pto/common/constants.hpp`. Part of the `Tile<...>` template and used by some implementations to define how out-of-valid regions are treated (e.g., select/copy/pad paths).

## Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example(Tile<TileType::Vec, float, 16, 16>& dst,
             Tile<TileType::Vec, float, 16, 16>& src) {
  TCVT(dst, src, RoundMode::CAST_RINT);
  TMINS(dst, src, 0.0f);
}
```
