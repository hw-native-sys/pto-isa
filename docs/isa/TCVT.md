# TCVT


## Tile Operation Diagram

![TCVT tile operation](../figures/isa/TCVT.svg)

## Introduction

Elementwise type conversion with a specified rounding mode.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) $$

where `rmode` is a rounding policy (see `pto::RoundMode`).

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

## Constraints

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- The conversion `(src element type) -> (dst element type)` must be supported by the target for the given `RoundMode`.
- **Implementation notes (A2A3/A5)**:
    - One form accepts an explicit `SaturationMode`, and the specified saturation behavior is forwarded directly to the implementation.
    - The other form omits `SaturationMode`; in that case, the implementation chooses a target-defined default saturation behavior for the specific type pair.
    - On CPU, only the form without explicit `SaturationMode` is currently implemented.

## Supported Conversions (Side-by-Side: A2A3 vs A5)

| Source Type | A2A3 Destinations | A5 Destinations | Difference |
|---|---|---|---|
| FP32 | FP16, FP32 (round-only), BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | A5 adds FP8/H8 targets |
| FP16 | FP32, I32, I16, I8, U8, S4 (int4b_t) | FP32, I32, I16, I8, U8, H8 | A2A3 has S4 path; A5 has H8 path |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | A5 adds FP16/FP4 targets |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | A5 adds U8/U32/I32 targets |
| I32 | FP32, I16, I64, FP16 (deq path) | FP32, I16, U16, I64, U8 | A2A3 supports I32 -> FP16 (half, deq); A5 does not |
| I64 | FP32, I32 | FP32, I32 | Same |
| U8 | FP16 | FP16, U16 | A5 adds U16 target |
| I8 | FP16 | FP16, I16, I32 | A5 adds I16/I32 targets |
| S4 (int4b_t) | FP16 | N/A | A2A3-only |
| U32 | N/A | U8, U16, I16 | A5-only source type |
| FP8_E4M3 | N/A | FP32 | A5-only source type |
| FP8_E5M2 | N/A | FP32 | A5-only source type |
| H8 | N/A | FP32 | A5-only source type |
| FP4_E1M2X2 | N/A | BF16 | A5-only source type |
| FP4_E2M1X2 | N/A | BF16 | A5-only source type |

Notes:
- Key gap: A2A3 supports I32 -> FP16 (half) via deq path, while A5 has no I32 -> FP16 conversion.
- On A5, FP16 -> FP8_E4M3 and FP16 -> FP8_E5M2 are not supported.

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

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

