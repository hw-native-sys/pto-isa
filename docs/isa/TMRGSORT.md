# TMRGSORT

## Tile Operation Diagram

![TMRGSORT tile operation](../figures/isa/TMRGSORT.svg)

## Introduction

Hardware-accelerated multi-way merge sort (`vmrgsort4`). Merges up to 4 pre-sorted lists into a single sorted output in **descending** order. Each element is a fixed 8-byte **value-index pair** structure.

## Data Format: Value-Index Pair

TMRGSORT operates on 8-byte structures where each element in the tile represents part of a value-index pair:

| Data type | Value field | Padding | Index field | Struct size | Tile elements per struct |
|-----------|-------------|---------|-------------|-------------|--------------------------|
| `float`   | 4 bytes     | 0       | 4 bytes (`uint32_t`) | 8 bytes | **2 elements** |
| `half`    | 2 bytes     | 2 bytes | 4 bytes (`uint32_t`) | 8 bytes | **4 elements** |

The number of sorted pairs in a tile is therefore:

- `float`: `numPairs = ValidCol / 2`
- `half`: `numPairs = ValidCol / 4`

The implementation converts `ValidCol` to pair count using `ELE_NUM_SHIFT`:

```cpp
// float: ELE_NUM_SHIFT = 1  →  numPairs = ValidCol >> 1
// half:  ELE_NUM_SHIFT = 2  →  numPairs = ValidCol >> 2
```

## Math Interpretation

Merges pre-sorted input lists into `dst` in descending order:

$$ \mathrm{dst} = \mathrm{merge\_desc}(\mathrm{src}_0, \mathrm{src}_1, \ldots) $$

## Two Variants

### Variant A: Single-list sort — `TMRGSORT(dst, src, blockLen)`

Treats `src` as **4 consecutive equal-length pre-sorted blocks** and performs a 4-way merge within a single tile.

```
src Tile (1 row):
┌── blockLen ──┬── blockLen ──┬── blockLen ──┬── blockLen ──┐
│   Block 0    │   Block 1    │   Block 2    │   Block 3    │
│ (pre-sorted) │ (pre-sorted) │ (pre-sorted) │ (pre-sorted) │
└──────────────┴──────────────┴──────────────┴──────────────┘
                        ↓ vmrgsort4
dst Tile (1 row):
┌──────────── merged result (descending) ────────────┐
└────────────────────────────────────────────────────┘
```

**Constraints:**

- `blockLen` must be a multiple of **64**.
- `src.GetValidCol()` must be an integer multiple of `blockLen * 4`.
- `repeatTimes = src.GetValidCol() / (blockLen * 4)` must be in `[1, 255]`.
- **No `tmp` required** — result is written directly to `dst`.
- No `exhausted` parameter (fixed to non-suspending mode).

**`blockLen` meaning in terms of sorted pairs:**

| blockLen | float pairs per block | half pairs per block |
|----------|-----------------------|----------------------|
| 64       | 32                    | 16                   |
| 128      | 64                    | 32                   |
| 256      | 128                   | 64                   |

### Variant B: Multi-list merge — `TMRGSORT<..., exhausted>(dst, executedNumList, tmp, src0, src1, [src2], [src3])`

Merges 2–4 **independent pre-sorted lists** into a single sorted output.

```
src0 Tile ──┐
src1 Tile ──┤
src2 Tile ──┼──→ vmrgsort4 ──→ tmp ──→ dst
src3 Tile ──┘
```

**Template parameter `exhausted`:**

- `exhausted = false`: Normal merge — processes all input data.
- `exhausted = true`: When any input list is exhausted, the hardware suspends and reports the number of elements processed from each list via `executedNumList`.

**`MrgSortExecutedNumList`:**

```cpp
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;  // elements processed from list 0
    uint16_t mrgSortList1;  // elements processed from list 1
    uint16_t mrgSortList2;  // elements processed from list 2
    uint16_t mrgSortList3;  // elements processed from list 3
};
```

Only meaningful when `exhausted = true`. Data is read from hardware register `VMS4_SR`.

**Mask configuration by list count:**

| List count | Xt[11:8] mask | Unused lists |
|------------|---------------|--------------|
| 2 lists    | `0b0011`      | src2, src3 (size=0) |
| 3 lists    | `0b0111`      | src3 (size=0) |
| 4 lists    | `0b1111`      | none |

## Assembly Syntax

Synchronous form (conceptual):

```text
%dst, %executed = tmrgsort %src0, %src1 {exhausted = false}
    : !pto.tile<...>, !pto.tile<...> -> (!pto.tile<...>, vector<4xi16>)
```

### AS Level 1 (SSA)

```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

### AS Level 2 (DPS)

```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

### Single-list variant

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen, WaitEvents &... events);
```

### Multi-list variants (2/3/4 lists)

```cpp
// 4 lists
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3,
                              WaitEvents &... events);

// 3 lists
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2,
                              WaitEvents &... events);

// 2 lists
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, WaitEvents &... events);
```

## Constraints

### General constraints (A2A3 and A5)

| Constraint | Requirement |
|------------|-------------|
| Tile type | All tiles must be `TileType::Vec` |
| Rows | All tiles must have `Rows == 1` |
| Layout | All tiles must be row-major (`BLayout::RowMajor`) |
| Data type | `half` or `float`, consistent across all tiles |
| UB memory | Total must not exceed 192KiB (`UB_SIZE`) |

### UB memory constraints by variant

| Variant | Constraint |
|---------|------------|
| Single-list | `(src.Cols + dst.Cols) * sizeof(T) < UB_SIZE` |
| 2-list | `(src0.Cols + src1.Cols + tmp.Cols) * sizeof(T) < UB_SIZE`, and `tmp.Cols + src0.Cols <= UB_SIZE / sizeof(T)` |
| 3-list | `(src0.Cols + src1.Cols + src2.Cols + tmp.Cols) * sizeof(T) < UB_SIZE` |
| 4-list | `(src0.Cols + src1.Cols + src2.Cols + src3.Cols + tmp.Cols) * sizeof(T) < UB_SIZE` |

### Single-list constraints

- `blockLen` must be a multiple of 64.
- `src.GetValidCol()` must be an integer multiple of `blockLen * 4`.
- `repeatTimes = src.GetValidCol() / (blockLen * 4)` must be in `[1, 255]`.

## Temporary Space

### Multi-list variants (2/3/4 lists)

`tmp` **is used** as intermediate output buffer for the `vmrgsort4` hardware instruction. The merge-sort result is first written to `tmp`, then copied to `dst` via `MovUb2Ub` (UB-to-UB memcpy).

- `tmp` must have the same element type as `dst` and all `src` tiles (`half` or `float`).
- `tmp` must have `Rows == 1` and be row-major.
- `tmp` Cols must be at least the sum of all input source Cols:
    - 2-list: `tmp.Cols >= src0.Cols + src1.Cols`
    - 3-list: `tmp.Cols >= src0.Cols + src1.Cols + src2.Cols`
    - 4-list: `tmp.Cols >= src0.Cols + src1.Cols + src2.Cols + src3.Cols`
- The helper function `GETMRGSORTTMPSIZE<...>()` returns the required `tmp` Cols:

```cpp
// 2 lists
GETMRGSORTTMPSIZE<Src0Tile, Src1Tile>() = Src0Tile::Cols + Src1Tile::Cols

// 3 lists
GETMRGSORTTMPSIZE<Src0Tile, Src1Tile, Src2Tile>() = Src0Tile::Cols + Src1Tile::Cols + Src2Tile::Cols

// 4 lists
GETMRGSORTTMPSIZE<Src0Tile, Src1Tile, Src2Tile, Src3Tile>() = sum of all 4 Cols
```

### Single-list variant

`tmp` is **not required**. The single-list variant writes directly to `dst`.

## Typical Usage: TopK

TMRGSORT is commonly used to implement TopK selection via iterative merge sort:

```
Phase 1: Single-list sort (progressively increasing blockLen)
  blockLen=64:  every 256 elements → 4-way merge → 256 sorted elements
  blockLen=256: every 1024 elements → 4-way merge → 1024 sorted elements
  ... until blockLen * 4 > totalCols

Phase 2: Tail merge (SortTailBlock)
  Use 2-list variant to merge remaining blocks, keeping top K elements
```

## A2A3 vs A5 Differences

Both implementations are nearly identical, both calling the `vmrgsort4` hardware instruction. Minor differences:

| Item | A2A3 | A5 |
|------|------|-----|
| `UB_SIZE` constant | Hardcoded `196608` (192×1024) | Uses `PTO_UBUF_SIZE_BYTES` |
| `TMRGSORT_BLOCK_LEN` | Defined as constant `64` | Not defined (uses literal) |
| Core logic | Identical | Identical |

## Examples

### Single-list sort (Auto)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_single() {
  using SrcT = Tile<TileType::Vec, float, 1, 256>;
  using DstT = Tile<TileType::Vec, float, 1, 256>;
  SrcT src;
  DstT dst;
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

### Multi-list merge (4 lists, non-exhausted)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_multi4() {
  using SrcT = Tile<TileType::Vec, float, 1, 128>;
  using DstT = Tile<TileType::Vec, float, 1, 512>;
  using TmpT = Tile<TileType::Vec, float, 1, 512>;
  SrcT src0, src1, src2, src3;
  DstT dst;
  TmpT tmp;
  MrgSortExecutedNumList executedNumList;
  TMRGSORT<DstT, TmpT, SrcT, SrcT, SrcT, SrcT, /*exhausted=*/false>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
}
```

### Multi-list merge (2 lists, exhausted)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_exhausted() {
  using SrcT = Tile<TileType::Vec, float, 1, 64>;
  using DstT = Tile<TileType::Vec, float, 1, 128>;
  using TmpT = Tile<TileType::Vec, float, 1, 128>;
  SrcT src0, src1;
  DstT dst;
  TmpT tmp;
  MrgSortExecutedNumList executedNumList;
  TMRGSORT<DstT, TmpT, SrcT, SrcT, /*exhausted=*/true>(
      dst, executedNumList, tmp, src0, src1);
  // After execution:
  // executedNumList.mrgSortList0 = elements processed from src0
  // executedNumList.mrgSortList1 = elements processed from src1
}
```

### Manual (single-list)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 1, 256>;
  using DstT = Tile<TileType::Vec, float, 1, 256>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
```
