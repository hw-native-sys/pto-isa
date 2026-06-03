# MSCATTER

## Tile Operation Diagram

![MSCATTER tile operation](../../../../../../../docs/figures/isa/MSCATTER.svg)

## Introduction

`MSCATTER` writes data from a UB source tile into a GM `GlobalTensor` using a UB index tile. On Ascend A5 it is implemented as a SIMT kernel dispatched through `cce::async_invoke` with up to `dim3{32, 32}` = 1024 threads.

The operating mode is selected by the `Coalesce` template parameter:

- **`Coalesce::Row`** (default): scatter whole rows — `table[idx[r], :] = src[r, :]`. The index tile is 1-D (`[R, 1]` or `[1, R]`). `R = 1` is allowed.
- **`Coalesce::Elem`**: scatter single elements — `table[idx[r, c]] = src[r, c]`. The index tile has the same shape as `src`. 1-D sources (`[1, N]` or `[N, 1]`) are also supported.

Three orthogonal template policies control write behavior:

- `ScatterAtomicOp` — `None` (plain store), `Add`, `Max`, `Min`.
- `ScatterOOB` — `Undefined`, `Skip`, `Clamp`, `Wrap`.
- `ScatterConflict` — `Last` (deterministic largest-index wins, only consulted when `Atomic == None`) or `Default` (warp-scheduler dependent).

## Math Interpretation

### Row Coalesce

Source `src[R, C]`, index `idx[R, 1]` or `idx[1, R]`:

$$ \mathrm{table}_{\mathrm{idx}_{r},\; j} = \mathrm{src}_{r, j},\quad 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce

Source `src[R, C]`, index `idx[R, C]` (same shape as `src`), flat table of length `TableSize = Shape[0]·Shape[1]·Shape[2]·Shape[3]·Shape[4]`:

$$ \mathrm{table}[\mathrm{idx}_{r, c}] = \mathrm{src}_{r, c},\quad 0 \le r < R,\; 0 \le c < C $$

### Atomic Accumulation

When `ScatterAtomicOp::Add` / `Max` / `Min` is selected, the write combines with the current table value:

$$ \mathrm{table}[\cdot] \mathrel{\oplus}= \mathrm{src}_{\cdot},\quad \oplus \in \{+,\; \max,\; \min\} $$

### Conflict Resolution

With `Atomic::None`:

- **`Conflict::Last`** — the source position with the **largest flat index** that targets a given destination slot is the one whose value is stored. Matches the sequential semantics of `for i in 0..N: table[idx[i]] = src[i]`. Implemented as a slot-centric reverse scan, race-free by construction.
- **`Conflict::Default`** — the surviving writer is warp-scheduler dependent. For collision-free index sets the result is identical to `Last`.

Atomic modes ignore `Conflict` because the GM atomic R-M-W serializes colliding writes by itself.

## Assembly Syntax

See [docs/assembly/PTO-AS.md](/docs/assembly/PTO-AS.md).

Row coalesce:

```text
mscatter.row %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<Rx1xi32>)
```

Element coalesce:

```text
mscatter.elem %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<RxCxi32>)
```

Atomic variants append the operation suffix, e.g. `mscatter.row.atomic_add`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/npu/a5/MScatter.hpp`:

```cpp
template <Coalesce         Mode     = Coalesce::Row,
          ScatterAtomicOp  Atomic   = ScatterAtomicOp::None,
          ScatterOOB       Oob      = ScatterOOB::Undefined,
          ScatterConflict  Conflict = ScatterConflict::Last,
          typename GlobalTable, typename TileSrc, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalTable& table, TileSrc& src, TileIdx& idx,
                              WaitEvents&... events);
```

The kernel iterates over `TileSrc::ValidRow × TileSrc::ValidCol` logical positions. Physical UB strides come from the same `Tile` types via `tile_offset_2d` (i.e. `TileSrc::Cols`, `TileIdx::Cols`).

For `Coalesce::Elem` with `TileSrc::ValidRow == 1 && TileSrc::ValidCol == 1` the implementation bypasses the SIMT launch and runs a scalar fallback (`MScatterScalarImpl`) on the AIV vector core.

**Parameters:**
- `table` — destination GM `GlobalTensor`. `GlobalTensor::DType` must be `__gm__ T` matching the source element type.
- `src` — UB source tile (`TileType::Vec`), shape `[R, C]`. Row-major and column-major storage are both accepted.
- `idx` — UB index tile (`TileType::Vec`). For `Row`: `[1, R]` or `[R, 1]`. For `Elem`: same shape as `src` (the index tile's `BLayout` may differ from the source's).
- `Mode` — `Coalesce` value (`Row` or `Elem`); first template parameter so the mode is always explicit at the call site.
- `Atomic`, `Oob`, `Conflict` — policies as described above.

## Enum Definitions

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // table[idx[r], :] = src[r, :]   (1-D index of length R)
    Elem = 1   // table[idx[i, j]] = src[i, j]   (idx shape == src shape)
};

enum class ScatterAtomicOp : uint8_t {
    None = 0,  // Plain store (collision-resolved by ScatterConflict)
    Add  = 1,  // Atomic addition
    Max  = 2,  // Atomic maximum
    Min  = 3   // Atomic minimum
};

enum class ScatterOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Skip      = 1,  // Drop the write
    Clamp     = 2,  // Clamp index to capacity - 1
    Wrap      = 3   // Index modulo capacity
};

enum class ScatterConflict : uint8_t {
    Last    = 0,  // Deterministic: largest source index wins
    Default = 1   // Warp-scheduler dependent (use only when collisions are impossible or order-insensitive)
};
```

`capacity` for OOB handling is `TableRows` in Row mode and the full flat table length in Elem mode.

### Atomic Type Support (A5)

| Atomic | Supported `T` |
|--------|---------------|
| `None` | all supported dtypes |
| `Add`  | `int32_t`, `uint32_t`, `float`, `half`, `bfloat16_t` |
| `Max`  | `int32_t`, `uint32_t`, `float` |
| `Min`  | `int32_t`, `uint32_t`, `float` |

## Constraints

### Data Types

`TileSrc::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.

### Index Types

`TileIdx::DType` must be `int32_t` or `uint32_t`.

### Tile Constraints

- Both tiles must live in UB (`TileSrc::Loc == TileType::Vec`, `TileIdx::Loc == TileType::Vec`).
- Source and table must share the same element type `T` (`GlobalTable::DType == __gm__ T`).
- For `Coalesce::Row`: the index tile's **valid shape** is `[1, R]` or `[R, 1]` matching `TileSrc::ValidRow`. The table's inner dim (`Shape[4]`) must equal `TileSrc::ValidCol`.
- For `Coalesce::Elem`: `TileIdx` must have the same valid shape as `TileSrc`. The `BLayout` of `TileIdx` is independent of the source layout — the kernel walks both via per-tile `tile_offset_2d`.

### Dynamic Runtime Shapes

`MSCATTER` accepts both compile-time and runtime-dynamic shapes. Any dimension declared as `-1` is resolved at runtime:

- `Tile<…, -1, -1>` stores the runtime valid extents in the tile object; `MSCATTER_IMPL` reads them via `src.GetValidRow()` / `src.GetValidCol()`.
- `Shape<…, -1, -1>` / `Stride<…, -1, -1>` are constructed with the runtime sizes; `MSCATTER_IMPL` reads them via `table.GetShape(…)` and folds them into `tableRows` or `tableSize`.

Static-asserts in `MScatterCheck` are gated on `if constexpr (DIM > 0)` so they fire only for compile-time dimensions. The padded `Tile::Rows` / `Tile::Cols` remain compile-time — they govern the UB DMA-burst alignment.

Example (mirrors `case_elem2d_dyn_user_float_1x9_in_1x16_3x10`):

```cpp
constexpr auto kPadCols = 16;
using SrcTileT    = Tile<TileType::Vec, float,   1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT    = Tile<TileType::Vec, int32_t, 1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t validCols = 9, tableR = 3, tableC = 10;
TableShape  tableShape(tableR, tableC);
TableStride tableStride(tableC, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(dstGm, tableShape, tableStride);

SrcTileT srcTile(1, validCols);
IdxTileT idxTile(1, validCols);
TASSIGN(srcTile, srcUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(tableGM, srcTile, idxTile);
```

### Layout Support

The SIMT kernel is layout-agnostic for UB tiles: every UB read goes through `tile_offset_2d<TileX>(r, c)`, which dispatches index arithmetic from the tile type's `BLayout`. The caller is responsible for staging the data into UB with the right `TLOAD` / `TMOV` / `TINSERT` configuration.

| Tile / Tensor | Supported layouts | Notes |
|---------------|-------------------|-------|
| `TileSrc` (UB) | `BLayout::RowMajor` or `ColMajor`, `SLayout::NoneBox` | Kernel reads via `tile_offset_2d<TileSrc>`. |
| `TileIdx` (UB) — Row mode | `[1, R]` `RowMajor` **or** `[R, 1]` `ColMajor` | Both produce a linear R-element layout in UB. |
| `TileIdx` (UB) — Elem mode | any `BLayout`, independent of `TileSrc` | Kernel reads via `tile_offset_2d<TileIdx>`. |
| `GlobalTable` (GM) | `Layout::ND` (linear contiguous) | Non-ND layouts must be pre-staged by the caller. |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles **any** `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. To express odd valid extents (e.g. `3×3`), pad the tile up to the nearest 32-byte aligned shape (e.g. `3×8` for int32) and let the kernel iterate over the valid sub-region only. The padding is purely an upstream `TLOAD` / `TSTORE` burst-alignment requirement.

Minimum padded inner dim per dtype (row-major):

| `T` | Min `Cols` (`RowMajor`) | Min `Rows` (`ColMajor`) |
|-----|-------------------------|-------------------------|
| `int8` / `uint8` / `float8_e4m3` / `float8_e5m2` / `hifloat8` | 32 | 32 |
| `int16` / `uint16` / `half` / `bfloat16` | 16 | 16 |
| `int32` / `uint32` / `float` | 8 | 8 |

### Mode Resolution

Mode is **explicit**, never auto-detected. Static asserts in `MScatterCheck` validate the tile shapes against the chosen `Coalesce` value:

```text
Coalesce::Row  : (Idx.Rows == 1 && Idx.Cols == Src.Rows) || (Idx.Rows == Src.Rows && Idx.Cols == 1)
Coalesce::Elem : (Idx.Rows == Src.Rows) && (Idx.Cols == Src.Cols)
```

## Examples

### Row Coalesce — Weight Update

```cpp
#include <pto/npu/a5/MScatter.hpp>

using namespace pto;

template <typename T, int R, int C, int TableRows>
AICORE void example_weight_update(__gm__ T* tablePtr)
{
    using SrcTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::None, ScatterOOB::Skip, ScatterConflict::Last>(
        tableGM, src, idx);
}
```

### Atomic Gradient Accumulation (Row Coalesce)

```cpp
AICORE void example_gradient_accumulation(__gm__ float* gradTable)
{
    constexpr int NumTokens = 16, D = 64, Vocab = 65536;

    using SrcTile     = Tile<TileType::Vec, float,   NumTokens, D, BLayout::RowMajor, NumTokens, D>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, NumTokens, BLayout::RowMajor, 1, NumTokens>;
    using TableShape  = Shape<1, 1, 1, Vocab, D>;
    using TableStride = Stride<1, 1, 1, D, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(gradTable);
    SrcTile grads; TASSIGN(grads, 0x0000);
    IdxTile idx;   TASSIGN(idx,   0x4000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::Add, ScatterOOB::Skip, ScatterConflict::Default>(
        tableGM, grads, idx);
}
```

### Element Coalesce — 2-D Sparse Update

```cpp
AICORE void example_sparse_update(__gm__ float* data)
{
    constexpr int R = 8, C = 32, TableRows = 64, TableCols = 64;

    using SrcTile     = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;
    using TableShape  = Shape<1, 1, 1, TableRows, TableCols>;
    using TableStride = Stride<1, 1, 1, TableCols, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor dataGM(data);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Wrap, ScatterConflict::Last>(
        dataGM, src, idx);
}
```

### Deterministic Last-Write-Wins Over Colliding Indices

```cpp
AICORE void example_last_deterministic(__gm__ half* tablePtr)
{
    constexpr int R = 8, C = 64, TableRows = 65536;

    using SrcTile     = Tile<TileType::Vec, half,    R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::None, ScatterOOB::Clamp, ScatterConflict::Last>(
        tableGM, src, idx);
}
```

## Performance Considerations

1. **Shape-adaptive launch.** The SIMT grid is sized as `dim3{32, kLaunchWarps}` from the resolved `validRows` / `validCols` / `tableSize`. Small tiles do not pay the cost of 1024 idle threads.
   - **Row, non-`Last`.** `kRowWarps = min(validRows, 32)` warps own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks.
   - **Elem, non-`Last`.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`.
   - **`Conflict::Last`.** Launch is sized by the destination instead of the source: `kLaunchWarps = min(ceil(tableSize / 32), 32)`. Each lane owns one slot and runs a reverse scan over the index tile.

2. **Conflict policy cost.**
   - `Last`: per-lane in-register reverse scan with early termination. No GM read-back, no atomic, no UB scratch. Worst-case `O(N)` per warp; uniformly random workloads average `O(tableSize / 32)`.
   - `Default`: zero extra work — the surviving lane is whatever the warp scheduler picked.
   - Atomic modes: serialized by the GM atomic R-M-W itself; no `cur` preload, no conflict gate.

3. **No thread divergence for mode / policy.** All policy decisions are `if constexpr`. In Row coalesce the `doWrite` predicate is warp-uniform (row-indexed) and hoisted out of the inner column loop. The slot-centric `Last` kernels compile their `found` predicate to a predicated store.

4. **Unrolled inner loops.** Inner column loop carries `#pragma unroll(4)`; outer scatter and reverse-scan loops use `#pragma unroll(1)` to keep code size bounded for large N.

5. **Row vs. Elem bandwidth.** Row coalesce achieves the best GM write bandwidth (32 consecutive lanes per coalesced store). Elem coalesce is a per-lane scalar GM store, non-coalesced in general.

6. **Register pressure.** Kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread) and use ≤ 16 live registers in the hot path. No spills are produced.

## UB Memory Budget

`MSCATTER` is a SIMT launch on the AIV vector core. All user tiles must fit inside the AIV's 256 KB Unified Buffer alongside two fixed runtime reservations: an 8 KB reserved region (AscendC / TBE bookkeeping) and the Data Cache (32 KB minimum, sized at launch time).

The UB layout is:

```text
+---------------------------+
| Static memory             |  Compile-time tile allocations (see note below)
+---------------------------+
| Dynamic memory            |  Sized at launch via dynUBufSize
+---------------------------+
| Reserved (8 KB)           |  Fixed compiler / AscendC reservation
+---------------------------+
| Data Cache (>= 32 KB)     |  Min 32 KB; grows when dynUBufSize is small
+---------------------------+
```

The configurable maximum is therefore:

```text
max dynUBufSize = 256 KB - 8 KB (reserved) - 32 KB (min DCache) - static_memory
                = 216 KB - static_memory
```

For `MSCATTER` kernels in this ST suite the source/index tiles are placed manually with `TASSIGN`, so the compiler sees `static_memory ≈ 0` and the full **216 KB** is available as `dynUBufSize`.

### Default Per-Call Budget (No `dynUBufSize`)

When the kernel is launched without an explicit `dynUBufSize` (i.e. `<<<numBlocks, nullptr, stream>>>`), the runtime keeps the default DCache size and reserves only the small default dynamic region. In practice, on Ascend A5 the safe `src + idx` working set is **≤ 128 KB**; beyond that, on-board execution may silently corrupt or zero out the result.

### Extending Per-Call UB Beyond 128 KB (`dynUBufSize`)

Callers that need a single-shot `src + idx` footprint **larger than 128 KB** must declare the dynamic-UB request explicitly via the second argument of the kernel launch:

```cpp
kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...);
```

`dynUBufSize` is the byte size of the dynamic-UB region the kernel will use. The bisheng/CCE compiler routes such launches through `__cce_rtKernelLaunchWithFlagV2`, setting `rtTaskCfgInfo_t::localMemorySize = dynUBufSize`. The runtime then shrinks the DCache toward its 32 KB minimum and hands the remaining space back to the kernel.

A few key points:

- **The simulator does not enforce this.** Passing `nullptr` (or `0`) still runs to completion in sim regardless of the actual UB footprint. Always set `dynUBufSize` explicitly when the workload exceeds 128 KB so the binary stays correct on real hardware.
- **Exceeding the ceiling is silent.** The compiler does not error and the simulator does not flag it. On-board, the first overflow byte corrupts the reserved region or DCache and the kernel returns undefined output.
- **Size to actual usage.** For an Elem-coalesce case with `R × C × sizeof(T)` source and `R × C × sizeof(int32_t)` index, the working set is `R * C * (sizeof(T) + 4)`. In the extended-UB ST cases below (`float` source + `int32_t` index, `C = 8`) the per-element footprint is `8 + 4 = 12 B`; we round up and pass `R * 8 * 8 = R * 64` as `dynUBufSize` to keep the math simple.

### Tiled-Iteration Pattern (2048×8 Cases - Example to use 128KB in chunked fashion for larger shapes)

The `case_elem2d_float_2048x8_*` ST cases predate the `dynUBufSize` path and use a **chunked** approach instead: a `2048 × 8` `float` source is split into 16 chunks of `128 × 8` (8 KB src + 8 KB idx per iteration), and `MSCATTER` is reissued per chunk into the same destination GM tensor. Semantics are preserved:

- `Conflict::Last` — each chunk writes its in-chunk last-writer to GM; later chunks overwrite earlier ones for any shared slot, so the surviving value is the global largest-index writer.
- `Conflict::Default` / atomic modes — writes from later chunks compose with earlier ones (overwrite, add, max, min) on the same GM table.

New large-shape cases (`2304×8` and above) use the `dynUBufSize` single-shot path described above instead of chunking.

### Cache-Coherence Flush

Every `runMSCATTER_*` wrapper finishes with:

```cpp
AICORE PTO_INLINE void FlushScatterOutput()
{
    dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
```

`dcci(0, ENTIRE_DATA_CACHE)` invalidates the AIV scalar D-cache so any buffered GM writes are pushed to HBM; `dsb(DSB_DDR)` blocks until the writes are observable at the DDR boundary. This guarantees committed output regardless of where the compiler inserts its own implicit flush.

## Runtime Dispatch Requirement

`MSCATTER` uses `cce::async_invoke<simt_mscatter_*_kernel>(cce::dim3{32, kLaunchWarps}, …)` internally to fan a per-warp/per-lane workload out across up to 1024 threads. `async_invoke` consumes runtime state (TID registers, warp / lane configuration, vector-pipe scheduling) that the **launch path** must install before the kernel function is entered. The standard CANN launch (`rtKernelLaunchWithHandleV2`, used by the `<<<numBlocks, dynUBufSize, stream>>>` syntax) installs this state correctly.

A runtime variant that dispatches kernels as a direct C function-pointer call is fine for SPMD ops (TLOAD, TSTORE, TADD, …) but skips the SIMT context init, so the first `async_invoke` inside `MSCATTER` has no warp scheduler to dispatch into and hangs. Use the standard launch syntax for any SIMT kernel.

## Related Instructions

- [`TSTORE`](/docs/isa/TSTORE.md): contiguous block transfer Tile → GM.
- [`TSCATTER`](/docs/isa/TSCATTER.md): index-based scatter within tiles (UB-to-UB).
- [`MGATHER`](../mgather/MGATHER.md): indexed gather GM → Tile (inverse operation).

## Test Cases

### Row Coalesce — `[1, R]` index form

| Case | Data Type | Src Size | TableRows | Atomic | OOB | Conflict | Idx Pattern |
|------|-----------|----------|-----------|--------|-----|----------|-------------|
| case_row_float_random_8x32_64rows   | float | 8×32  | 64 | None | Undefined | Last    | random |
| case_row_float_same_8x32_16rows     | float | 8×32  | 16 | None | Undefined | Last    | same   |
| case_row_half_random_16x64_64rows   | half  | 16×64 | 64 | None | Undefined | Last    | random |
| case_row_int32_random_8x16_32rows   | int32 | 8×16  | 32 | None | Undefined | Last    | random |
| case_row_uint8_random_8x32_32rows   | uint8 | 8×32  | 32 | None | Undefined | Last    | random |
| case_row_int16_random_8x16_32rows   | int16 | 8×16  | 32 | None | Undefined | Last    | random |
| case_row_float_atomicadd_8x32_8rows | float | 8×32  | 8  | Add  | Undefined | Default | random |
| case_row_float_skip_8x32_8rows      | float | 8×32  | 8  | None | Skip      | Last    | oob    |
| case_row_int32_clamp_8x16_8rows     | int32 | 8×16  | 8  | None | Clamp     | Last    | oob    |
| case_row_half_wrap_8x32_8rows       | half  | 8×32  | 8  | None | Wrap      | Last    | oob    |

### Row Coalesce — `[R, 1]` index form (ColMajor + DN)

| Case | Data Type | Src Size | TableRows | Atomic | OOB | Conflict |
|------|-----------|----------|-----------|--------|-----|----------|
| case_row_colidx_float_random_8x32_64rows | float | 8×32  | 64 | None | Undefined | Last |
| case_row_colidx_int32_clamp_8x16_8rows   | int32 | 8×16  | 8  | None | Clamp     | Last |
| case_row_colidx_half_random_16x64_64rows | half  | 16×64 | 64 | None | Undefined | Last |

### Element Coalesce — 1-D source `[1, N]`

| Case | Data Type | N / TableSize | Atomic | OOB | Conflict | Idx Pattern |
|------|-----------|---------------|--------|-----|----------|-------------|
| case_elem_float_random_64_128size          | float | 64 / 128 | None | Undefined | Last    | random |
| case_elem_float_same_64_8size              | float | 64 / 8   | None | Undefined | Last    | same   |
| case_elem_float_seq_32_32size              | float | 32 / 32  | None | Undefined | Last    | seq    |
| case_elem_half_random_64_128size           | half  | 64 / 128 | None | Undefined | Last    | random |
| case_elem_int32_random_32_64size           | int32 | 32 / 64  | None | Undefined | Last    | random |
| case_elem_uint8_random_64_128size          | uint8 | 64 / 128 | None | Undefined | Last    | random |
| case_elem_int16_random_32_64size           | int16 | 32 / 64  | None | Undefined | Last    | random |
| case_elem_float_atomicadd_32_32size        | float | 32 / 32  | Add  | Undefined | Default | random |
| case_elem_int32_atomicadd_skip_32_16size   | int32 | 32 / 16  | Add  | Skip      | Default | oob    |
| case_elem_float_skip_32_16size             | float | 32 / 16  | None | Skip      | Last    | oob    |
| case_elem_int32_clamp_32_16size            | int32 | 32 / 16  | None | Clamp     | Last    | oob    |
| case_elem_half_wrap_32_16size              | half  | 32 / 16  | None | Wrap      | Last    | oob    |
| case_elem_float_default_seq_32_32size      | float | 32 / 32  | None | Undefined | Default | seq    |
| case_elem_float_small_16_32size            | float | 16 / 32  | None | Undefined | Last    | random |
| case_elem_int32_atomicmax_random_32_32size | int32 | 32 / 32  | Max  | Undefined | Default | random |
| case_elem_float_atomicmin_random_32_32size | float | 32 / 32  | Min  | Undefined | Default | random |
| case_elem_float_last_same_32_8size         | float | 32 / 8   | None | Undefined | Last    | same   |
| case_elem_int32_last_seq_32_32size         | int32 | 32 / 32  | None | Undefined | Last    | seq    |
| case_elem_float_clamp_no_dup_32_16size     | float | 32 / 16  | None | Clamp     | Last    | random |
| case_elem_uint8_wrap_64_16size             | uint8 | 64 / 16  | None | Wrap      | Last    | random |
| case_elem_int16_clamp_32_16size            | int16 | 32 / 16  | None | Clamp     | Last    | oob    |

### Element Coalesce — 2-D source `[R, C]`

| Case | Data Type | Src Size | TableSize | Atomic | OOB | Conflict | Idx Pattern |
|------|-----------|----------|-----------|--------|-----|----------|-------------|
| case_elem2d_float_8x32_random_256size | float | 8×32 | 256 | None | Undefined | Last | random |
| case_elem2d_int32_8x16_random_256size | int32 | 8×16 | 256 | None | Undefined | Last | random |
| case_elem2d_half_4x32_random_256size  | half  | 4×32 | 256 | None | Undefined | Last | random |

### Element Coalesce — Tiled Iteration (Legacy, No `dynUBufSize`)

These cases pre-date the `dynUBufSize` path and chunk the input to keep each iteration under 128 KB. See [Tiled-Iteration Pattern](#tiled-iteration-pattern-legacy-2048x8-cases) above.

| Case | Data Type | Total Src | Chunk | UB per Chunk | TableSize | Atomic | OOB | Conflict | Idx Pattern |
|------|-----------|-----------|-------|--------------|-----------|--------|-----|----------|-------------|
| case_elem2d_float_2048x8_last_256size      | float | 2048×8 | 128×8 (16 iters) | 4 KB src + 4 KB idx | 256   | None | Undefined | Last    | random |
| case_elem2d_float_2048x8_default_16384size | float | 2048×8 | 128×8 (16 iters) | 4 KB src + 4 KB idx | 16384 | None | Undefined | Default | seq    |

### Element Coalesce — Extended UB (`dynUBufSize`) Single-Shot

These cases push the per-call `src + idx` footprint **beyond 128 KB** in a single launch by passing an explicit `dynUBufSize` (the second argument of `<<<numBlocks, dynUBufSize, stream>>>`). The runtime shrinks the DCache toward its 32 KB minimum and frees up to **216 KB** for the source/index tiles. See [Extending Per-Call UB Beyond 128 KB](#extending-per-call-ub-beyond-128-kb-dynubufsize). `dynUBufSize` is set to `R * 8 * 8` bytes per case (an upper bound on `R * C * (sizeof(float) + sizeof(int32_t))` with `C = 8`).

| Case | Data Type | Src Size | Total UB (src + idx) | dynUBufSize (bytes) | TableSize | Atomic | OOB | Conflict | Idx Pattern |
|------|-----------|----------|----------------------|---------------------|-----------|--------|-----|----------|-------------|
| case_elem2d_float_2304x8_last_256size      | float | 2304×8 | 72 KB + 72 KB = 144 KB | 147456 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_2304x8_default_18432size | float | 2304×8 | 72 KB + 72 KB = 144 KB | 147456 | 18432 | None | Undefined | Default | seq    |
| case_elem2d_float_2560x8_last_256size      | float | 2560×8 | 80 KB + 80 KB = 160 KB | 163840 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_2560x8_default_20480size | float | 2560×8 | 80 KB + 80 KB = 160 KB | 163840 | 20480 | None | Undefined | Default | seq    |
| case_elem2d_float_2816x8_last_256size      | float | 2816×8 | 88 KB + 88 KB = 176 KB | 180224 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_2816x8_default_22528size | float | 2816×8 | 88 KB + 88 KB = 176 KB | 180224 | 22528 | None | Undefined | Default | seq    |
| case_elem2d_float_3072x8_last_256size      | float | 3072×8 | 96 KB + 96 KB = 192 KB | 196608 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_3072x8_default_24576size | float | 3072×8 | 96 KB + 96 KB = 192 KB | 196608 | 24576 | None | Undefined | Default | seq    |
| case_elem2d_float_3200x8_last_256size      | float | 3200×8 | 100 KB + 100 KB = 200 KB | 204800 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_3200x8_default_25600size | float | 3200×8 | 100 KB + 100 KB = 200 KB | 204800 | 25600 | None | Undefined | Default | seq    |
| case_elem2d_float_3456x8_last_256size      | float | 3456×8 | 108 KB + 108 KB = 216 KB | 221184 | 256   | None | Undefined | Last    | random |
| case_elem2d_float_3456x8_default_27648size | float | 3456×8 | 108 KB + 108 KB = 216 KB | 221184 | 27648 | None | Undefined | Default | seq    |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles any `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. Express odd valid extents as a padded tile and let the kernel walk the valid sub-region only.

| Case | Mode | Data Type | Valid Src | Padded Src | Idx (Valid → Padded) | Table | Atomic | OOB | Conflict |
|------|------|-----------|-----------|------------|----------------------|-------|--------|-----|----------|
| case_elem2d_int32_unaligned_3x8_64size              | Elem          | int32 | 3×8  | 3×8  | 3×8  → 3×8   | 64 elems    | None | Undefined | Last |
| case_elem2d_uint8_unaligned_3x32_256size            | Elem          | uint8 | 3×32 | 3×32 | 3×32 → 3×32  | 256 elems   | None | Undefined | Last |
| case_elem2d_int32_unaligned_3x3_in_3x8_64size       | Elem          | int32 | 3×3  | 3×8  | 3×3  → 3×8   | 64 elems    | None | Undefined | Last |
| case_elem2d_int32_unaligned_9x9_in_9x16_256size     | Elem          | int32 | 9×9  | 9×16 | 9×9  → 9×16  | 256 elems   | None | Undefined | Last |
| case_elem2d_int32_scalar_1x1_in_1x8_8size           | Elem (scalar) | int32 | 1×1  | 1×8  | 1×1  → 1×8   | 8 elems     | None | Undefined | Last |
| case_row_int32_unaligned_3x8_8rows                  | Row           | int32 | 3×8  | 3×8  | 1×3  → 1×8   | 8 rows × 8  | None | Undefined | Last |
| case_row_int32_unaligned_9x16_16rows                | Row           | int32 | 9×16 | 9×16 | 1×9  → 1×16  | 16 rows × 16| None | Undefined | Last |

### Dynamic Runtime Shapes

`Tile<…, -1, -1>` paired with `GlobalTensor<…, Shape<…, -1, -1>, Stride<…, -1, -1>>`. The SIMT kernel sizes itself from `Tile::GetValidRow/Col()` and `GlobalTensor::GetShape()` at dispatch time; padded `Tile::Rows / Cols` remain compile-time so the UB layout stays statically known.

| Case | Mode | Data Type | Runtime Valid Src | Padded Src | Runtime Table | OOB |
|------|------|-----------|-------------------|------------|----------------|-----|
| case_elem2d_dyn_user_float_1x9_in_1x16_3x10 | Elem | float | 1×9  | 1×16 | 3×10              | Skip      |
| case_elem2d_dyn_int32_4x8_in_4x8_64size     | Elem | int32 | 4×8  | 4×8  | 8×8 (linear 64)   | Undefined |
| case_elem2d_dyn_float_3x3_in_3x8_64size     | Elem | float | 3×3  | 3×8  | 8×8 (linear 64)   | Undefined |
| case_elem2d_dyn_half_8x16_in_8x16_4x32      | Elem | half  | 8×16 | 8×16 | 4×32 (linear 128) | Undefined |
| case_row_dyn_int32_3x16_8rows               | Row  | int32 | 3×16 | 3×16 | 8 rows × 16       | Undefined |
| case_row_dyn_half_4x32_16rows               | Row  | half  | 4×32 | 4×32 | 16 rows × 32      | Undefined |
