# pto.tsort32

`pto.tsort32` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.

## Summary

Sort a fixed-size 32-element block and produce an index mapping.

## Mechanism

Sort each 32-element block of `src` together with the corresponding indices from `idx`, and write the sorted value-index pairs into `dst`. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For each row, `TSORT32` processes `src` in independent 32-element blocks. Let block `b` cover columns `32b ... 32b+31`, and let `n_b = min(32, C - 32b)` be the valid element count of that block.

For each valid element in the block, form a pair

$$
(v_k, i_k) = (\mathrm{src}_{r,32b+k}, \mathrm{idx}_{r,32b+k}), \quad 0 \le k < n_b
$$

Then sort the pairs by value and write the sorted value-index pairs to `dst`. The exact packing layout in `dst` is target-defined, but semantically the output of each block is the reordered sequence

$$
[(v_{\pi(0)}, i_{\pi(0)}), (v_{\pi(1)}, i_{\pi(1)}), \ldots, (v_{\pi(n_b-1)}, i_{\pi(n_b-1)})]
$$

where `π` is the permutation produced by the implementation for that 32-element block.

Notes:

- `idx` is an input tile, not an output tile.
- `dst` stores sorted value-index pairs, not just sorted values.
- The CPU simulation sorts in descending order by value, and for equal values keeps smaller indices first.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## Inputs

- `src` is the source tile containing values to sort.
- `idx` is the index tile providing initial indices.
- `tmp` (optional): temporary tile for non-32-aligned tails.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds sorted value-index pairs from `src` according to `idx` order.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `TSORT32` does not take `WaitEvents&...` and does not call `TSYNC(...)` internally; synchronize explicitly if needed.

    - `idx` is a required input operand in both overloads; it provides the indices that are permuted together with `src`.

    - **Valid region**:
        - The implementation uses `dst.GetValidRow()` as the row count.
        - The implementation uses `src.GetValidCol()` to determine how many elements participate in sorting in each row.
        - Sorting is performed independently per 32-element block; the 4-argument overload additionally supports non-32-aligned tails with `tmp`.

## Performance

### A2/A3 Cycle Count

`pto.tsort32` is dispatched on the **SFU** (special-function unit) as a 32-element bitonic sort. Each 32-element group sorts in a fixed number of SFU cycles; longer sequences are decomposed into groups of 32 and merged.

**Cycle model**: `total ≈ startup + ⌈N / 32⌉ × per_group_sort + merge`.

Throughput is SFU-bound, not vector-pipe-bound, so `tsort32` can overlap with PIPE_V issues in well-scheduled kernels.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (A2A3/A5)**:
        - `DstTileData::DType` must be `half` or `float`.
        - `SrcTileData::DType` must match `DstTileData::DType`.
        - `IdxTileData::DType` must be `uint32_t`.
        - `dst/src/idx` tile location must be `TileType::Vec`, and all must be row-major (`isRowMajor`).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 64>;
  SrcT src;
  IdxT idx;
  DstT dst;
  TSORT32(dst, src, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 64>;
  SrcT src;
  IdxT idx;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(idx, 0x2000);
  TASSIGN(dst, 0x3000);
  TSORT32(dst, src, idx);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in instruction set: [pto.tmrgsort](./tmrgsort.md)
- Next op in instruction set: [pto.tgather](./tgather.md)

