# TGET_SCALE_ADDR

## Tile Operation Diagram

![TGET_SCALE_ADDR tile operation](../figures/isa/TGET_SCALE_ADDR.svg)

## Introduction

Bind the on-chip address of output tile as a scaled address of the input tile.

The scaling factor is defined by a right-shift amount `SHIFT_MX_ADDR` in `include/pto/npu/a5/utils.hpp`.

## Math Interpretation

Address(`dst`) = Address(`src`) >> `SHIFT_MX_ADDR`

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

### IR Level 1 (SSA)

TODO

### IR Level 2 (DPS)

TODO

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src, aitEvents&... events);
```

## Constraints

Enforced by `TGET_SCALE_ADDR_IMPL`:

- **Both `src` and `dst` must be Tile instances**
- **Currently only work in auto mode** (will support manual mode in the future)

## Examples

```cpp
#include <pto/pto-inst.hpp>

> wa
using namespace pto;

template <typename T, int ARows, int ACols, BRows, BCols> 
void example() {
    using LeftTile = TileLeft<T, ARows, ACols>;
    using RightTile = TileRight<T, BRows, BCols>;

    using LeftScaleTile = TileLeftScale<T, ARows, ACols>;
    using RightScaleTile = TileRightScale<T, BRows, BCols>;

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;

    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
}
```

## asm form examples

### Auto Mode

TODO

### Manual Mode

TODO

### PTO Assembly Form

TODO
