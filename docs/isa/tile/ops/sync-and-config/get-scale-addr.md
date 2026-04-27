# pto.get_scale_addr

`pto.get_scale_addr` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Bind the on-chip address of an output tile as a scaled address of the input tile. The scaling factor is defined by a right-shift amount `SHIFT_MX_ADDR` in `include/pto/npu/a5/utils.hpp`.

On A2/A3, `pto.get_scale_addr` is unsupported — calling it is illegal. On A5, this instruction computes `dst_addr = src_addr >> SHIFT_MX_ADDR` and writes the scaled address into the destination tile's address register. On the CPU simulator, the operation updates the software tile descriptor's address field with the right-shifted value.

## Mechanism

Address(`dst`) = Address(`src`) >> `SHIFT_MX_ADDR`

The `SHIFT_MX_ADDR` value (currently 2) is defined in `include/pto/npu/a5/utils.hpp`.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = pto.get_scale_addr %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.get_scale_addr %src : (!pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.get_scale_addr ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent GET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | Source tile |
| `dst` | Destination tile that holds the scaled address |

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

- **A2/A3**: Unsupported — calling `pto.get_scale_addr` is illegal.
- **A5**: Computes `dst_addr = src_addr >> SHIFT_MX_ADDR` and writes the scaled address into the destination tile's address register within the tile register file. Consumed by subsequent MX (matrix-multiplication with scaling) operations.
- **CPU simulator**: Updates the software tile descriptor's address field with the right-shifted value.

## Constraints

- **Both `src` and `dst` must be Tile instances**.
- Currently only works in Auto mode (Manual mode support planned for future releases).

## Cases That Are Not Allowed

- Calling `pto.get_scale_addr` on A2/A3 profiles (unsupported).
- Using `pto.get_scale_addr` in Manual mode (not yet supported).

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int ARows, int ACols, int BRows, int BCols>
void example() {
    using LeftTile = TileLeft<T, ARows, ACols>;
    using RightTile = TileRight<T, BRows, BCols>;
    using LeftScaleTile = TileLeftScale<T, ARows, ACols>;
    using RightScaleTile = TileRightScale<T, BRows, BCols>;

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;

    GET_SCALE_ADDR(aScaleTile, aTile);
    GET_SCALE_ADDR(bScaleTile, bTile);
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.subview](./subview.md)
