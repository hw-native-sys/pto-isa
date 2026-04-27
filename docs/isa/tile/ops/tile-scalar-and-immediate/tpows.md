# pto.tpows

`pto.tpows` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

Elementwise power where the base comes from a tile and the exponent is a scalar.

## Mechanism

For each lane `(r, c)` in the destination valid region:

$$ \mathrm{dst}_{r,c} = \mathrm{pow}(\mathrm{base}_{r,c}, \mathrm{exp}) $$

Like `pto.tpow`, the C++ intrinsic requires a `tmp` scratch tile even though the scalar exponent itself is the only semantic non-tile input.

## Syntax

### PTO Assembly Form

```text
%dst = tpows %base, %exp : !pto.tile<...>, dtype -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpows %base, %exp : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpows ins(%base, %exp : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
          typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOWS(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp,
                               WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `dst` | Destination tile | Result tile |
| `base` | Source tile | Base tile |
| `exp` | Scalar | Scalar exponent broadcast across the destination valid region |
| `tmp` | Temporary tile | Scratch tile required by the C++ intrinsic path |

## Expected Outputs

`dst` contains the power result for each valid lane of `base`.

## Side Effects

No architectural side effects beyond producing the destination tile. The `tmp` tile may be used as backend scratch storage.

## Constraints

- `dst` and `base` must use the same element type.
- `dst` and `base` must have matching valid-row and valid-column extents.
- In the current concrete implementation, both tiles must be row-major `TileType::Vec`.
- The scalar exponent uses `typename DstTile::DType`.
- `PowAlgorithm::DEFAULT` accepts `float`, `half`, `int32_t`, `uint32_t`, `int16_t`, `uint16_t`, `int8_t`, and `uint8_t`.
- `PowAlgorithm::HIGH_PRECISION` accepts `float`, `half`, and `bfloat16_t`.

## Exceptions

- Unsupported type/layout combinations and mismatched valid regions are rejected by backend legality checks.
- Programs must not infer CPU-simulator or A2/A3 support from the public template alone.

## Target-Profile Restrictions

The current authored tree documents concrete `tpows` implementation behavior for A5. This checkout does not provide a stable documented CPU-simulator or A2/A3 implementation contract for `tpows`.

Portable code should treat `pto.tpows` as A5-specific unless a target-profile page explicitly widens that guarantee.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  TileT dst;
  TileT base;
  TileT tmp;

  TPOWS(dst, base, 2.0f, tmp);
  TPOWS<PowAlgorithm::HIGH_PRECISION>(dst, base, 2.0f, tmp);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tmuls](./tmuls.md)
- Next op in instruction set: [pto.tfmods](./tfmods.md)
