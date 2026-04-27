# pto.treshape

`pto.treshape` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Reinterpret a tile as another tile type/shape while preserving the underlying bytes.

## Mechanism

Reinterpret a tile as another tile type/shape while preserving the underlying bytes.

This is a *bitwise* reshape: it does not change values, it only changes how the same byte buffer is viewed. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5: TRESHAPE respects the valid region defined by GetValidRow/GetValidCol — only elements within the valid region are guaranteed to be preserved in the output reinterpretation; elements in the padding region of the source may or may not be accessible depending on the tile layout. On the CPU simulator: TRESHAPE is a direct memory reinterpretation that preserves all bytes in the underlying buffer without applying any valid-region filtering.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

```text
%dst = treshape %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `dst` names the destination tile. Must have same total byte size as `src`.

## Expected Outputs

`dst` holds the same byte data as `src`, reinterpreted with different tile type/shape.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

Enforced by `TRESHAPE_IMPL`:

- **Tile type must match**: `TileDataIn::Loc == TileDataOut::Loc`.

- **Total byte size must match**: `sizeof(InElem) * InNumel == sizeof(OutElem) * OutNumel`.

- **No boxed/non-boxed conversion**:
    - cannot reshape between `SLayout::NoneBox` and boxed layouts.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.treshape` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 16, 16>;
  using Dst = Tile<TileType::Vec, float, 8, 32>;
  static_assert(Src::Numel == Dst::Numel);

  Src src;
  Dst dst;
  TRESHAPE(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.tmov_fp](./tmov.md)
- Next op in instruction set: [pto.ttrans](./ttrans.md)
