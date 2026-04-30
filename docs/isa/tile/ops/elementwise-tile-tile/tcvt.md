# pto.tcvt

`pto.tcvt` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

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
| `RoundMode::CAST_ROUND` | Round to nearest, ties away from zero |
| `RoundMode::CAST_FLOOR` | Round toward -∞ |
| `RoundMode::CAST_CEIL` | Round toward +∞ |
| `RoundMode::CAST_TRUNC` | Round toward zero |

## Saturation Modes

When `SaturationMode` is provided, saturation behavior is explicitly controlled:

| Mode | Behavior |
|------|----------|
| `SaturationMode::ON` | Saturation enabled |
| `SaturationMode::OFF` | Saturation disabled |

When `SaturationMode` is omitted, the implementation chooses the default behavior for the selected target/type path. Some conversion paths also expose a `tmp`-tile overload used for explicit scratch storage.

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

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

The `tmp`-tile overloads exist for conversion paths that need explicit scratch storage.

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src` | Source tile | Source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%dst` | Destination tile | Destination tile receiving the converted values |
| `mode` | Rounding mode | One of `CAST_RINT`, `CAST_ROUND`, `CAST_FLOOR`, `CAST_CEIL`, `CAST_TRUNC` |
| `satMode` | Saturation mode (optional) | `ON` or `OFF` |
| `tmp` | Temporary tile (optional) | Scratch tile for conversion paths that require explicit temporary storage |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain the converted element values after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `src` and `dst` MUST have compatible shapes (declared shape and valid region).
    - The source/destination type pair MUST be supported by the selected target profile.
    - The rounding mode MUST be supported for the given type pair.
    - When a conversion path requires explicit scratch storage, callers MUST use one of the `tmp`-tile overloads.
    - Disabling saturation may change overflow behavior for some backend/type paths, especially low-precision integer conversions.

## Performance

### A2/A3 Cycle Count

`pto.tcvt` is dispatched on the **vector pipe** for in-pipe type conversions and on the **fix-pipe (FIXP)** for accumulator-source paths (e.g. `Acc → Mat` quantization). The dominant cost depends on the source/dest type pair:

- **Same-width FP↔FP** (e.g. `half ↔ bfloat16`): single `vcvt`-class issue per row.
- **FP → low-precision int** (e.g. `fp16 → int8`): may take a sub-chunked path with row-aware tail handling and an explicit scratch tile; cost is roughly 2× a same-width conversion plus the tail overhead.
- **Acc → low-precision** (`Acc → int8/fp8`): runs through FIXP, which is bounded by the L0C → UB / OUT write asymmetry on A5.

**Cycle model**:

```
total ≈ startup + R × ⌈C / vlen⌉ × per_issue + tail_handling
```

### Instruction Sequence by Shape (FP32 → FP16)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|----------------------|------------------|
| 16×16 | `vcvt`*16 → PIPE_V | ~O(64) |
| 32×32 | `vcvt`*32 → PIPE_V | ~O(128) |
| 64×64 | `vcvt`*64 → PIPE_V | ~O(256) |
| R×C   | `vcvt`*R → PIPE_V  | ~O(R × ⌈C/vlen⌉) |

### Layout and Shape Impact

| Path | Effect |
|---|---|
| Same-width FP↔FP | Single-issue per row; fastest |
| FP → low-precision int | Sub-chunked + scratch tile; ~2× cost |
| Saturating vs non-saturating | Non-saturating may take a fix-up branch on overflow |
| `Acc → Mat` (FIXP) | FIXP-bound; overlap-friendly with PIPE_V |

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - **MUST NOT** use a type pair not supported by the target profile.
    - **MUST NOT** use a rounding mode not supported for the given type pair.
    - **MUST NOT** assume that disabling saturation still clamps overflow to the destination range.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    `pto.tcvt` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but the exact set of supported type pairs, scratch requirements, and saturation behavior is backend-specific.

    In this checkout, the fp16 → int8 non-saturating path is explicitly implemented through helper logic that may require temporary storage and row-aware sub-chunking.

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

### Explicit Saturation / Scratch

```cpp
using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
TmpT tmp;
TCVT(dst, src, tmp, RoundMode::CAST_TRUNC, SaturationMode::OFF);
```

### PTO Assembly Form

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tsubc](./tsubc.md)
- Next op in instruction set: [pto.tsel](./tsel.md)
