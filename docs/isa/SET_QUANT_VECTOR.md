# SET_QUANT_VECTOR

## Introduction

Set the vector quantization parameter for subsequent `TPUSH` operations by configuring the hardware FPC register from a Scaling-type tile's address. The tile address is converted to the quantization parameter address format and written to the hardware quantization configuration.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename FpTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_VECTOR(FpTileData &fpTile, WaitEvents &...events);
```

## Constraints

- `FpTileData::Loc` must be `TileType::Scaling`. Only Scaling-type tiles are supported as input.
- This instruction must be called before the `TPUSH` instruction that consumes this configuration.
- The tile address encoding into the hardware FPC register is implementation-defined.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_set_quant_vector()
{
    using ScalingTile = Tile<TileType::Scaling, T, 1, 128, BLayout::RowMajor, 1, 128>;
    ScalingTile fpTile;
    TASSIGN(fpTile, 0x0);

    SET_QUANT_VECTOR(fpTile);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `SET_QUANT_VECTOR`. Use the C++ intrinsic form for quantization configuration.
