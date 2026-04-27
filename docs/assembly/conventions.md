# PTO ISA Conventions

This page defines shared conventions used across the per-instruction ISA reference pages under `docs/isa/`.

## Tiles and Shapes

- **Tile**: the core operand type for PTO instructions. Most instructions operate on a `Tile` and use its *valid region*.
- **Valid region**: the active sub-rectangle of a tile. Most operations iterate over `tile.GetValidRow()` and `tile.GetValidCol()`.
- **Layouts**: tile layouts are defined by template parameters such as `BLayout` (big-fractal) and `SLayout` (small-fractal).

## GlobalTensor (GM)

- **GlobalTensor** represents tensors stored in global memory (GM). `TLOAD`/`TSTORE` move data between GM and tiles.

## Events and Synchronization

PTO supports modeling dependencies between operations via events:

- **Producer**: an instruction may *record* an event when it completes.
- **Consumer**: an instruction may depend on one or more previously recorded events.

In the C++ intrinsics, this is represented by passing event objects as extra arguments.

## Assembly Syntax (PTO-AS)

Instruction docs show PTO-AS examples using MLIR-like conventions:

- SSA value names use `%name`.
- Types use MLIR-style spellings such as `!pto.tile<...>` and `!pto.memref<...>`.

See `docs/isa/syntax-and-operands/assembly-model.md` for the full syntax and grammar.
