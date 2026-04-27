# pto.tpow

`pto.tpow` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise power with both base and exponent provided as tiles.

## Mechanism

For each lane `(r, c)` in the destination valid region:

$$ \mathrm{dst}_{r,c} = \mathrm{pow}(\mathrm{base}_{r,c}, \mathrm{exp}_{r,c}) $$

`pto.tpow` uses a temporary tile in the C++ intrinsic surface for backend scratch space. That temporary is an implementation aid, not a separate architecture-visible input in the textual assembly form.

## Syntax

### PTO Assembly Form

```text
%dst = tpow %base, %exp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpow %base, %exp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpow ins(%base, %exp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile,
          typename TmpTile, typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOW(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp, WaitEvents &... events);
```

`PrecisionType` accepts:

- `PowAlgorithm::DEFAULT`
- `PowAlgorithm::HIGH_PRECISION`

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `dst` | Destination tile | Result tile |
| `base` | Source tile | Elementwise base values |
| `exp` | Source tile | Elementwise exponent values |
| `tmp` | Temporary tile | Scratch tile required by the C++ intrinsic path |

## Expected Outputs

`dst` contains the elementwise power result for all lanes in its valid region.

## Side Effects

No architectural side effects beyond producing the destination tile. The `tmp` tile may be used as backend scratch storage.

## Constraints

- `dst`, `base`, and `exp` must use the same element type.
- `dst`, `base`, and `exp` must have matching valid-row and valid-column extents.
- In the current concrete implementation, all three tiles must be row-major `TileType::Vec`.
- `PowAlgorithm::DEFAULT` accepts `float`, `half`, `int32_t`, `uint32_t`, `int16_t`, `uint16_t`, `int8_t`, and `uint8_t`.
- `PowAlgorithm::HIGH_PRECISION` accepts `float`, `half`, and `bfloat16_t`.

## Exceptions

- Unsupported type/layout combinations and mismatched valid regions are rejected by backend legality checks.
- Programs must not assume CPU-simulator or A2/A3 support from the presence of the public template alone.

## Target-Profile Restrictions

The current authored tree documents concrete `tpow` implementation behavior for A5. This checkout exposes the public intrinsic template in `pto_instr.hpp`, but does not provide a stable documented CPU-simulator or A2/A3 implementation contract for `tpow`.

Portable code should treat `pto.tpow` as A5-specific unless a target-profile page explicitly widens that guarantee.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  TileT dst;
  TileT base;
  TileT exp;
  TileT tmp;

  TPOW(dst, base, exp, tmp);
  TPOW<PowAlgorithm::HIGH_PRECISION>(dst, base, exp, tmp);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.texp](./texp.md)
- Next op in instruction set: [pto.tnot](./tnot.md)
