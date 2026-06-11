# TSORT32

## Tile Operation Diagram

![TSORT32 tile operation](../figures/isa/TSORT32.svg)

## Introduction

Sort each 32-element block of `src` together with the corresponding indices from `idx`, and write the sorted value-index pairs into `dst`.

## Math Interpretation

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

## Assembly Syntax

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

## Constraints

- `TSORT32` does not take `WaitEvents&...` and does not call `TSYNC(...)` internally; synchronize explicitly if needed.
- `idx` is a required input operand in both overloads; it provides the indices that are permuted together with `src`.
- **Implementation checks (A2A3/A5)**:
    - `DstTileData::DType` must be `half` or `float`.
    - `SrcTileData::DType` must match `DstTileData::DType`.
    - `IdxTileData::DType` must be `uint32_t`.
    - `dst/src/idx` tile location must be `TileType::Vec`, and all must be row-major (`isRowMajor`).
- **Valid region**:
    - The implementation uses `dst.GetValidRow()` as the row count.
    - The implementation uses `src.GetValidCol()` to determine how many elements participate in sorting in each row.
    - Sorting is performed independently per 32-element block; the 4-argument overload additionally supports non-32-aligned tails with `tmp`.

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

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
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

