# pto.tadd

`pto.tadd` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Lane-wise addition of two source tiles into a destination tile. The iteration domain is the destination tile's valid region.

## Mechanism

For each element `(i, j)` in the destination tile's valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} $$

Only the destination tile's valid region defines the iteration domain. Source tiles are read lane-by-lane at the same `(i, j)` coordinates; source tiles whose valid region does not cover `(i, j)` read all-one-bits on A2/A3 and all-one-bits on A5 (both platforms fill out-of-region lanes with 0xFF pattern in the vector register file before the ALU reads them).

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tadd %src0, %src1 : !pto.tile<...>
```

### AS Level 1 — SSA Form

PTO-AS at Level 1 uses SSA-style result binding:

```mlir
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 — DPS Form

PTO-AS at Level 2 uses the Def-Use-Style (DPS) explicit operand binding:

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

The `ins(...)` clause names operands in the input position; the `outs(...)` clause names the output. The tile buffer type `!pto.tile_buf<...>` is the in-memory storage form used at Level 2.

### Micro-Operation Mapping

The `pto.tadd` SSA operation maps to the following micro-operation sequence on the Tile Register File (TRF):

```
TRF_READ(src0, i, j)  →  A
TRF_READ(src1, i, j)  →  B
A + B                  →  C
TRF_WRITE(dst, i, j, C)
```

The micro-operation level is not exposed to the ISA author; it is the responsibility of the backend to schedule these steps subject to pipeline constraints.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src0` | Left tile | First source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%src1` | Right tile | Second source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

Both source tiles and the destination tile share the same element type. Layout and shape constraints are stated under Constraints.

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `src0[i,j] + src1[i,j]` after the operation |

## Side Effects

None beyond producing the destination tile. Does not implicitly fence unrelated tile traffic.

## Constraints

- **Type match**: All three tiles (`src0`, `src1`, `dst`) MUST have identical element types.
- **Layout**: Both source tiles and the destination tile MUST have compatible layouts. See the TileType–Layout compatibility table in [Tiles and Valid Regions](../../../programming-model/tiles-and-valid-regions.md).
- **Valid region**: The iteration domain is `dst.GetValidRow()` × `dst.GetValidCol()`. Source tiles with smaller valid regions read all-one-bits (0xFF) on A2/A3 and all-one-bits (0xFF) on A5 for lanes outside their valid region.
- **TileType**: The destination tile's TileType determines which pipelines execute the operation. See [Tiles and Valid Regions](../../../programming-model/tiles-and-valid-regions.md) for TileType constraints.

## Exceptions

- Verifier rejects type mismatches between source and destination tiles.
- Backend rejects unsupported element types, layouts, or shapes for the selected target profile.
- Programs that read values from destination lanes outside `dst`'s declared valid region observe undefined behavior.

## Target-Profile Restrictions

| | CPU Simulator | A2/A3 | A5 |
|-|--------------|-------|-----|
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
| `bf16` | Simulated | Supported | Supported |
| `i32` | Simulated | Supported | Supported |
| `i16` | Simulated | Supported | Supported |
| `i8` / `u8` | Simulated | No | Supported |
| `i64` / `u64` | Simulated | No | No |
| `f8e4m3` / `f8e5m2` | Simulated | No | Supported |
| Layout | Any | RowMajor only | RowMajor only |

A2/A3 requires `isRowMajor == true` for all operands. A5 additionally requires `isRowMajor == true` but supports more element types.

## Performance

### A2/A3 Throughput

Tile elementwise operations are compiled to CCE vector instructions. The `TBinOp.hpp` performance model computes cycles as follows:

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 cycles | `A2A3_STARTUP_BINARY` |
| Completion latency | 19 (FP) / 17 (INT) | `A2A3_COMPL_FP_BINOP` / `A2A3_COMPL_INT_BINOP` |
| Per-repeat throughput | 2 cycles | `A2A3_RPT_2` |
| Pipeline interval | 18 cycles | `A2A3_INTERVAL` |
| Cycle model | `14 + C + 2R + (R-1)×18` | R=repeats |

**Repeat calculation**: `R = validRow × validCol / 8` (assuming 8 elements per repeat, RowMajor layout).

### Shape-Dependent Optimizations

The performance model applies different instruction sequences based on valid region geometry:

| Shape | Optimization Path | Instruction Sequence |
|-------|-------------------|---------------------|
| Small tiles (few rows) | `Bin1LNormModeSmall` | 1 repeat covers entire row |
| Col-aligned (RowMajor) | Continuous fast path | `vadd(repeats, 1, 1, 1, 8, 8, 8)` |
| Misaligned | General path | stride-dependent repeats |

### Layout Impact on Throughput

| Layout | Stride Pattern | Cost Impact |
|--------|--------------|-----------|
| `RowMajor` | src: `(1, cols)`, dst: `(1, cols)` | Best: continuous fast path available |
| `ColMajor` | src: `(rows, 1)`, dst: `(rows, 1)` | General path: higher repeat count |
| `Zigzag` | non-linear strides | General path only |

### Example Throughput Estimate

For `TADD` on a 16×64 FP32 tile with `RowMajor` layout:

```
validRow = 16, validCol = 64, layout = RowMajor
repeats = 16 × 64 / 8 = 128
total = 14 + 19 + 256 + (128-1) × 18 = 14 + 19 + 256 + 2286 = 2575 cycles
```

---

## Examples

### C++ — Auto Mode

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles(Tile<Vec, float, 16, 16>& dst,
               Tile<Vec, float, 16, 16>& src0,
               Tile<Vec, float, 16, 16>& src1) {
    // Compiler inserts TASSIGN and TSYNC automatically in Auto mode.
    TADD(dst, src0, src1);
}
```

### C++ — Manual Mode

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles_manual(Tile<Vec, float, 16, 16>& dst,
                      Tile<Vec, float, 16, 16>& src0,
                      Tile<Vec, float, 16, 16>& src1) {
    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst,  0x3000);
    RecordEvent e0 = TLOAD(src0, ga);
    RecordEvent e1 = TLOAD(src1, gb);
    TSYNC(e0, e1);
    TADD(dst, src0, src1);
    TSYNC();
    TSTORE(gc, dst);
}
```

### MLIR — SSA Form

```mlir
%result = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

### MLIR — DPS Form

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%result : !pto.tile_buf<f32, 16, 16>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.tabs](./tabs.md)
- Instruction set: [Tile Instructions](../../../instruction-families/tile-families.md)
- Type system: [Type System](../../../state-and-types/type-system.md)
- Valid regions: [Tiles and Valid Regions](../../../programming-model/tiles-and-valid-regions.md)
