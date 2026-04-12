<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/tcvt.md` -->

# pto.tcvt

Standalone reference page for `pto.tcvt`. This page belongs to the [Elementwise Tile Tile](../../elementwise-tile-tile.md) family in the PTO ISA manual.

## Summary

Elementwise type conversion with a specified rounding mode and optional saturation mode.

## Mechanism

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode},\mathrm{satmode}}\!\left(\mathrm{src}_{i,j}\right) $$

where `rmode` is the rounding policy and `satmode` (if provided) controls saturation behavior.

## Rounding Modes

| Mode | Behavior |
|------|----------|
| `RoundMode::CAST_RINT` | Round to nearest, ties to even |
| `RoundMode::CAST_RZ` | Round toward zero |
| `RoundMode::CAST_RP` | Round toward +∞ |
| `RoundMode::CAST_RM` | Round toward -∞ |
| `RoundMode::CAST_RN` | Round to nearest, ties away from zero |

## Saturation Modes

When `SaturationMode` is provided (overload 2), saturation behavior is explicitly controlled:

| Mode | Behavior |
|------|----------|
| `SaturationMode::NONE` | No saturation; wraps on overflow |
| `SaturationMode::SAT` | Clamp to destination type's representable range |

When `SaturationMode` is omitted (overload 1), the implementation chooses a target-defined default for the specific source/destination type pair.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
// Overload 1: implementation-chosen default saturation
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);

// Overload 2: explicit saturation control (A2/A3, A5 only)
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);
```

Overload 2 (with explicit `SaturationMode`) is not currently implemented on the CPU simulator.

## Inputs

- `src` is the source tile whose elements are converted to the destination type.
- `dst` names the destination tile receiving the converted values.
- `mode` specifies the rounding mode.
- `satMode` (overload 2 only) specifies saturation behavior.
- The operation iterates over `dst`'s valid region.

## Expected Outputs

`dst` carries the result tile with converted element values. `src`'s element values are not modified.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- `src` and `dst` MUST have compatible shapes (declared shape and valid region).
- The source/destination type pair MUST be supported by the target profile.
- The rounding mode MUST be supported for the given type pair.
- The output tile `dst` MUST have a different element type from `src`.

## Cases That Are Not Allowed

- **MUST NOT** use a type pair not supported by the target profile.
- **MUST NOT** use a rounding mode not supported for the given type pair.

## Target-Profile Restrictions

| Feature | CPU Simulator | A2/A3 | A5 |
|---------|:-------------:|:------:|:--:|
| Overload 1 (default sat) | Yes | Yes | Yes |
| Overload 2 (explicit sat) | No | Yes | Yes |
| f32 → f16 | Yes | Yes | Yes |
| f16 → f32 | Yes | Yes | Yes |
| f32 → bf16 | Yes | Yes | Yes |
| bf16 → f32 | Yes | Yes | Yes |
| f32 → int32_t | Yes | Yes | Yes |
| int32_t → f32 | Yes | Yes | Yes |
| f16 → bf16 | No | Yes | Yes |
| FP8 types | No | No | Yes |

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### Explicit Saturation (A2/A3, A5)

```cpp
// A2/A3 and A5 only
TCVT(dst, src, RoundMode::CAST_RINT, SaturationMode::SAT);
```

### PTO Assembly Form

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in family: [pto.tsubc](./tsubc.md)
- Next op in family: [pto.tsel](./tsel.md)
