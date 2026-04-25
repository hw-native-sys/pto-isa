# MSCATTER


## Tile Operation Diagram

![MSCATTER tile operation](../figures/isa/MSCATTER.svg)

## Introduction

Scatter-store elements from a tile into global memory using per-element indices.

## Math Interpretation

For each element `(i, j)` in the source valid region:

$$ \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (CPU simulator: last writer wins in row-major iteration order).

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
```

### AS Level 1 (SSA)

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2 (DPS)

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events);
```

## Constraints

- **Supported data types**:
    - `src`/`dst` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
    - On AICore targets, `float8_e4m3_t` and `float8_e5m2_t` are also supported.
    - `indexes` element type must be `int32_t` or `uint32_t`.
- **Tile and memory types**:
    - `src` must be a vector tile (`TileType::Vec`).
    - `indexes` must be a vector tile (`TileType::Vec`).
    - `src` and `indexes` must use row-major layout.
    - `dst` must be a `GlobalTensor` in GM memory.
    - `dst` must use `ND` layout.
- **Atomic operation constraints**:
    - Non-atomic scatter is supported for all supported element types.
    - `Add` atomic mode requires `int32_t`, `uint32_t`, `float`, or `half`.
    - `Max`/`Min` atomic mode requires `int32_t` or `float`.
- **Shape constraints**:
    - `src.Rows == indexes.Rows`.
    - `indexes` must be shaped as `[N, 1]` for row-indexed scatter or `[N, M]` for element-indexed scatter.
    - `src` row width must be 32-byte aligned, that is, `src.Cols * sizeof(DType)` must be a multiple of 32.
    - `dst` static shape must satisfy `Shape<1, 1, 1, TableRows, RowWidth>`.
- **Index interpretation**:
    - Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `dst.data()`.
    - The CPU simulator does not enforce bounds checks on `indexes`.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO Assembly Form

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
# AS Level 2 (DPS)
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

