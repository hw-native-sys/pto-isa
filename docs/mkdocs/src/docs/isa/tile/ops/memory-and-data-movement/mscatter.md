<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/mscatter.md` -->

# pto.mscatter

Standalone reference page for `pto.mscatter`. This page belongs to the [Memory And Data Movement](../../memory-and-data-movement.md) family in the PTO ISA manual.

## Summary

Scatter-store elements from a tile into global memory using per-element indices.

## Mechanism

Scatter-store elements from a tile into global memory using per-element indices. It is part of the tile memory/data-movement surface, so the visible behavior includes explicit transfer between GM-visible data and tile-visible state.

For each element `(i, j)` in the source valid region:

$$ \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (CPU simulator: last writer wins in row-major iteration order).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

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

## Inputs

- `src` is the source tile.
- `indexes` is an index tile providing per-element indices into `dst`.
- `dst` is the destination GlobalTensor.

## Expected Outputs

Elements from `src` are scattered to positions in `dst` specified by `indexes`.

## Side Effects

This operation writes to global memory. Concurrent writes to the same location produce implementation-defined results.

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

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.mscatter` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
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

## Related Ops / Family Links

- Family overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in family: [pto.mgather](./mgather.md)
