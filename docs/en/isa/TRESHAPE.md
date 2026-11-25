# TRESHAPE

## Product Support<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->



## Tile Operation Diagram

![TRESHAPE tile operation](../../figures/isa/TRESHAPE.svg)

## Introduction

Reinterpret a tile as another tile type/shape while preserving the underlying bytes.

This is a *bitwise* reshape: it does not change values, it only changes how the same byte buffer is viewed.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## Constraints

Enforced by `TRESHAPE_IMPL`:

- **Tile type must match**: `TileDataIn::Loc == TileDataOut::Loc`.
- **Total byte size must match**. CPU simulation and manual NPU builds compare `TileDataIn::StorageBytes` and `TileDataOut::StorageBytes`, including element packing and panel padding.
- **No boxed/non-boxed conversion**:
    - cannot reshape between `SLayout::NoneBox` and boxed layouts.

## Notes

- **CPU simulation**: `dst` aliases the storage of `src`. Writes through either view are visible through the other.
- **A2/A3**: implemented as an alias (`TASSIGN_IMPL(dst, src.data())`), so `dst` and `src` refer to the same underlying storage.

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

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.
