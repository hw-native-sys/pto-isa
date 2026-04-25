# MGATHER


## Tile Operation Diagram

![MGATHER tile operation](../figures/isa/MGATHER.svg)

## Introduction

Gather-load elements from global memory into a tile using per-element indices.

## Math Interpretation

For each element `(i, j)` in the destination valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2 (DPS)

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

## Constraints

- **Supported data types**:
    - `dst`/`src` element type must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `bfloat16_t`, `float`.
    - On AICore targets, `float8_e4m3_t` and `float8_e5m2_t` are also supported.
    - `indexes` element type must be `int32_t` or `uint32_t`.
- **Tile and memory types**:
    - `dst` must be a vector tile (`TileType::Vec`).
    - `indexes` must be a vector tile (`TileType::Vec`).
    - `dst` and `indexes` must use row-major layout.
    - `src` must be a `GlobalTensor` in GM memory.
    - `src` must use `ND` layout.
- **Shape constraints**:
    - `dst.Rows == indexes.Rows`.
    - `indexes` must be shaped as `[N, 1]` for row-indexed gather or `[N, M]` for element-indexed gather.
    - `dst` row width must be 32-byte aligned, that is, `dst.Cols * sizeof(DType)` must be a multiple of 32.
    - `src` static shape must satisfy `Shape<1, 1, 1, TableRows, RowWidth>`.
- **Index interpretation**:
    - Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `src.data()`.
    - The CPU simulator does not enforce bounds checks on `indexes`.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### PTO Assembly Form

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

