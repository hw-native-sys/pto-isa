# pto.mgather

`pto.mgather` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

Gather-load elements from global memory into a tile using per-element indices.

## Mechanism

Gather-load elements from global memory into a tile using per-element indices. It is part of the tile memory/data-movement instruction set, so the visible behavior includes explicit transfer between GM-visible data and tile-visible state.

For each element `(i, j)` in the destination valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

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

## Inputs

- `src` is the source GlobalTensor.
- `indexes` is an index tile providing per-element indices into `src`.
- `dst` names the destination tile. The operation uses dst's valid region for the transfer shape.

## Expected Outputs

`dst` contains gathered elements from `src` at positions specified by `indexes`.

## Side Effects

This operation reads from global memory. Index bounds are target-defined.

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

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.mgather` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
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

## Related Ops / Instruction Set Links

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in instruction set: [pto.tstore_fp](./tstore.md)
- Next op in instruction set: [pto.mscatter](./mscatter.md)
