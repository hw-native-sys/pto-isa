# pto.tprint

`pto.tprint` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.

## Summary

Debug/print elements from a tile. On A2/A3/A5, output is emitted via `cce::printf` to the device-to-host debug channel; on the CPU simulator, output is printed to stdout.

## Mechanism

Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code.

The `TPRINT` instruction outputs the logical view of data stored in a Tile or GlobalTensor. It supports common data types (e.g., `float`, `half`, `int8`, `uint32`) and multiple memory layouts (`ND`, `DN`, `NZ` for GlobalTensor; vector tiles for on-chip buffers).

> **Important**:
> - This instruction is **for development and debugging ONLY**.
> - It incurs **significant runtime overhead** and **must not be used in production kernels**.
> - Output may be **truncated** if it exceeds the internal print buffer. The print buffer can be adjusted with `-DCCEBlockMaxSize=16384`; the default is 16 KiB.
> - **Requires CCE compilation option `-D_DEBUG --cce-enable-print`**. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5, output is emitted via `cce::printf` to the device-to-host debug channel; on the CPU simulator, output is printed to stdout.

- **Mandatory Compilation Flag**:

  On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the device-to-host debug channel. **You must enable the CCE option `-D_DEBUG --cce-enable-print`**.

- **Buffer Limitation:**

  The internal print buffer of `cce::printf` is limited in size. If the output exceeds this buffer, a warning message such as `"Warning: out of bound! try best to print"` may appear, and **only partial data will be printed**.

- **Synchronization**:

  Automatically inserts a `pipe_barrier(PIPE_ALL)` before printing to ensure all prior operations complete and data is consistent.

- **Formatting**:

    - Floating-point values: printed as `%6.2f`
    - Integer values: printed as `%6d`
    - For `GlobalTensor`, due to data size and buffer limitations, only elements within its logical shape (defined by `Shape`) are printed.
    - For `Tile`, invalid regions (beyond `validRows`/`validCols`) are still printed but marked with a `|` separator when partial validity is specified.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

### AS Level 1 (SSA)

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### AS Level 2 (DPS)

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

### IR Level 1 (SSA)

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### IR Level 2 (DPS)

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:
```cpp
// For printing GlobalTensor or Vec-type Tile
template <typename TileData>
PTO_INST void TPRINT(TileData &src);

// For printing Acc-type Tile and Mat-type Tile (Mat printing is currently A3-only)
template <typename TileData, typename GlobalData>
PTO_INTERNAL void TPRINT(TileData &src, GlobalData &tmp);
```

### Supported Types for T
- **Tile**: `TileType` may be `Vec`, `Acc`, or `Mat` (Mat printing is currently supported on A3 only).
- **GlobalTensor**: Must use layout `ND`, `DN`, or `NZ`, and have a supported element type.

## Inputs

- `src` is the Tile or GlobalTensor to print.

## Expected Outputs

Debug output is emitted to the device-to-host debug channel. The tile data is not modified.

## Side Effects

This operation emits debug output via `cce::printf`. It synchronizes by inserting a `pipe_barrier(PIPE_ALL)` before printing. Significant runtime overhead is expected.

## Constraints

- **Supported element type**:
    - Floating-point: `float`, `half`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`

- **For GlobalTensor**: Layout must be one of `Layout::ND`, `Layout::DN`, or `Layout::NZ`.

- **For temporary space**: Printing a `Tile` with `TileType::Mat` or `TileType::Acc` requires GM temporary space. The temporary buffer must be at least `TileData::Numel * sizeof(T)`.

- When `TileType` is `Mat`, the output is formatted according to `Layout::ND`; other layouts may appear misaligned.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- A5 does not yet support printing `TileType::Mat`.

## Examples

### Print a Tile

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugTile(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  using srcTileData = Tile<TileType::Vec, float, 16, 16>;
  srcTileData srcTile;
  TASSIGN(srcTile, 0x0);

  TLOAD(srcTile, srcGlobal);
  TPRINT(srcTile);
}
```

### Print a GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugGlobalTensor(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  TPRINT(srcGlobal);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### PTO Assembly Form

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
# AS Level 2 (DPS)
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Next op in instruction set: [pto.tmrgsort](./tmrgsort.md)
