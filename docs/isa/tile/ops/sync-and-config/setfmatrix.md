# pto.setfmatrix

`pto.setfmatrix` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Configure the FMATRIX (fast matrix) engine mode and address. This instruction programs the FMATRIX control registers and binds a tile to a specific FMATRIX slot for subsequent matrix multiply or convolution operations.

## Mechanism

`pto.setfmatrix` establishes the tile-to-FMATRIX-slot binding. On A2/A3 and A5, it writes to the FMATRIX control registers (`FMATRIX_CTRL_0`, `FMATRIX_SLICE_ADDR`) that determine which tile buffer is connected to the CCE's matrix multiply unit. On the CPU simulator, it is a functional no-op.

No direct tensor arithmetic is produced by this instruction. It updates tile-to-hardware resource bindings consumed by subsequent matrix operations.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
pto.setfmatrix %tile : !pto.tile<...>
```

### IR Level 1 (SSA)

```text
pto.setfmatrix %tile : !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.setfmatrix ins(%tile : !pto.tile_buf<...>) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(TileData &tile, WaitEvents &... events);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `tile` | `TileData` | The tile to bind to the FMATRIX slot |

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not produce a new payload tile.

## Side Effects

- **A2/A3 and A5**: Programs the FMATRIX control registers, binding the tile to the matrix multiply unit.
- **CPU simulator**: Functional no-op; no architectural state is affected.

## Constraints

- The tile must be a valid matrix tile type (`TileType::Mat`, `TileType::Left`, `TileType::Right`, or `TileType::Acc`) as required by the target profile.
- On A5, the tile's shape must be compatible with the FMATRIX slot dimensions.
- This instruction should be ordered before dependent matrix multiply operations (`pto.tmatmul`, `pto.tgemv`, etc.).

## Cases That Are Not Allowed

- Binding a non-matrix tile type to FMATRIX on backends that require specific tile types.
- Calling `pto.setfmatrix` twice for the same FMATRIX slot without an intervening operation that releases the binding.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  Tile<TileType::Mat, float, 16, 16> matTile;
  SETFMATRIX(matTile);
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Next op: [pto.set_img2col_rpt](./set-img2col-rpt.md)
