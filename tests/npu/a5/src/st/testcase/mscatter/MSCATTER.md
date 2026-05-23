# MSCATTER


## Tile Operation Diagram

![MSCATTER tile operation](../../../../../../../docs/figures/isa/MSCATTER.svg)

## Introduction

Scatter data from a UB source tile into a GM `GlobalTensor` through a UB index tile. The A5 implementation is a SIMT kernel dispatched via `cce::async_invoke` with `dim3{32, 32}` (32 lanes × 32 warps = 1024 threads). The operating mode is selected explicitly by the `Coalesce` template parameter:

- **`Coalesce::Row`** (default) — scatter full rows from `src[R, C]` into `table[idx[r], :]`. Index tile is 1-D (`[R, 1]` or `[1, R]`). `R = 1` (single-row scatter) is allowed.
- **`Coalesce::Elem`** — element-wise scatter from `src[R, C]` into a linearized `table` using `idx[R, C]`. The source tile may be 2-D (`[R, C]`) or degenerate 1-D (`[1, N]` / `[N, 1]`); the index tile must have the same shape as the source.

Atomic, out-of-bounds, and write-conflict policies are selected by template parameters.

## Math Interpretation

### Row Coalesce (`Coalesce::Row`)

Source `src[R, C]` with `R > 1`, index `idx[R, 1]` or `idx[1, R]`:

$$ \mathrm{table}_{\mathrm{idx}_{r},\; j} = \mathrm{src}_{r, j} \quad\text{for } 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce (`Coalesce::Elem`)

Source `src[R, C]`, index `idx[R, C]` (same shape as src; `R = 1` and `C = 1` degenerate forms are accepted):

$$ \mathrm{table}[\mathrm{idx}_{r, c}] = \mathrm{src}_{r, c} \quad\text{for } 0 \le r < R,\; 0 \le c < C $$

The kernel iterates over the `R * C` elements as a flat row-major sequence; the table is treated as a linear region of `Shape[0]*Shape[1]*Shape[2]*TableRows*TableCols` elements.

### Atomic Accumulation Mode

When `ScatterAtomicOp::Add` / `Max` / `Min` is specified, the write combines with the current table value:

$$ \mathrm{table}[\cdot] \mathrel{\oplus}= \mathrm{src}_{\cdot} \quad \oplus \in \{+,\; \max,\; \min\} $$

### Conflict Resolution

With `Atomic::None`:

- **`Conflict::Last`** (default) — the **largest** source index targeting a given destination slot is the one whose value is stored, matching the sequential CPU loop `for i in 0..N: table[idx[i]] = src[i]`. Implemented as a **slot-centric reverse scan**: the SIMT launch is sized by the destination table (`min(ceil(TableSize / 32), 32)` warps); each lane owns a distinct destination slot, walks the index tile from `N-1` down to `0`, exits on the first match, and issues a single coalesced GM store for that slot. Race-free by construction — no two lanes target the same slot — so no UB workspace, GM atomics, or post-pass cleanup are required.
- **`Conflict::Default`** — surviving writer is **warp-scheduler dependent** (no extra computation). For collision-free index sets the result is identical to `Last`.

## Assembly Syntax

PTO-AS form: see [docs/assembly/PTO-AS.md](/docs/assembly/PTO-AS.md).

Row coalesce:

```text
mscatter.row %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<Rx1xi32>)
```

Element coalesce:

```text
mscatter.elem %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<RxCxi32>)
```

Atomic scatter (row-coalesce example):

```text
mscatter.row.atomic_add %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<Rx1xi32>)
```

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

The kernel iterates over `TileSrc::ValidRow * TileSrc::ValidCol` logical positions; physical UB strides come from the same `Tile` types via `tile_offset_2d` (i.e. `TileSrc::Cols`, `TileIdx::Cols`).

For `Coalesce::Elem` with `TileSrc::ValidRow == 1 && TileSrc::ValidCol == 1` the implementation **bypasses the SIMT launch entirely** and runs a real **scalar fallback** (`MScatterScalarImpl`) on the AIV vector core — `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` handshake, single indexed UB read, plain GM store, then `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` to release the vector pipe. Mirrors the `TInsertVecToVecNDScalarImpl` pattern in `TInsert.hpp`.

**Parameters:**
- `table`   : Destination GM `GlobalTensor`. The `GlobalTensor::DType` must be `__gm__ T` matching the source element type.
- `src`     : UB source tile (`TileType::Vec`); shape `[R, C]`. Both row-major and column-major storage are accepted in either mode.
- `idx`     : UB index tile (`TileType::Vec`). For `Coalesce::Row`: 1-D `[R, 1]` or `[1, R]`. For `Coalesce::Elem`: same shape as `src` (BLayout can be **independent** of `src` layout).
- `Mode`    : `Coalesce` — `Row` (default) or `Elem` — **first** template parameter so the operating mode is always explicit at the call site.
- `Atomic`  : `ScatterAtomicOp` — conflict-resolution operator.
- `Oob`     : `ScatterOOB` — out-of-bounds index handling.
- `Conflict`: `ScatterConflict` — collision policy for non-atomic writes.

## Coalesce Mode

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // dst[idx[r], :] = src[r, :]   (1-D idx of length R)
    Elem = 1   // dst[idx[i, j]] = src[i, j]   (idx shape == src shape)
};
```

## Atomic Types

```cpp
enum class ScatterAtomicOp : uint8_t {
    None = 0,  // Plain store (conflict-resolved by ScatterConflict)
    Add  = 1,  // Atomic addition
    Max  = 2,  // Atomic maximum
    Min  = 3   // Atomic minimum
};
```

### Atomic Type Constraints (A5 SIMT)

- `None`: available for **all** supported data types.
- `Add` : `int32_t`, `uint32_t`, `float`, `half`, `bfloat16_t`.
- `Max` : `int32_t`, `uint32_t`, `float`.
- `Min` : `int32_t`, `uint32_t`, `float`.

## Out-of-Bounds Handling

```cpp
enum class ScatterOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Skip      = 1,  // Drop the write
    Clamp     = 2,  // Clamp index to capacity - 1
    Wrap      = 3   // Index modulo capacity
};
```

`capacity` is `TableRows` in `Coalesce::Row`, and `TableRows * TableCols * Shape[0] * Shape[1] * Shape[2]` in `Coalesce::Elem`.

## Conflict Resolution

```cpp
enum class ScatterConflict : uint8_t {
    Last    = 0,  // Deterministic: largest source index wins (slot-centric reverse scan, no UB writes)
    Default = 1   // Warp-scheduler dependent (no extra computation, race semantics)
};
```

`ScatterConflict` is consulted only when `Atomic == None`. Atomic operations bypass the gate entirely since they already serialize colliding writes.

## Constraints

### Data Types

`TileSrc::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.

### Index Types

`TileIdx::DType` must be `int32_t` or `uint32_t`.

### Tile Constraints

- `TileSrc::Loc == TileType::Vec` (UB).
- `TileIdx::Loc == TileType::Vec` (UB).
- Source and table must share the same element type `T` (`GlobalTable::DType == __gm__ T`).
- For `Coalesce::Row`: `TileSrc::ValidRow >= 1` and `TileSrc::ValidCol >= 1`; the index tile's **valid shape** is `[1, R]` (`TileIdx::ValidRow == 1 && TileIdx::ValidCol == TileSrc::ValidRow`) or `[R, 1]` (`TileIdx::ValidRow == TileSrc::ValidRow && TileIdx::ValidCol == 1`). Source tile may be row-major or column-major. Table inner dim (`Shape[4]`) must equal `TileSrc::ValidCol` (i.e. the row width that gets scattered to GM is the **valid** width, not the padded one).
- For `Coalesce::Elem`: `TileIdx::ValidRow / ValidCol == TileSrc::ValidRow / ValidCol`. **No layout-pairing constraint** — `TileSrc` and `TileIdx` may independently be `RowMajor` or `ColMajor`; the kernel walks both via per-tile `tile_offset_2d`. Table size is the product of all five `Shape` dimensions.
- The `[R, 1]` index variant uses `BLayout::ColMajor` paired with a `Layout::DN` `GlobalTensor` for the upstream `TLOAD`; the `[1, R]` variant uses `BLayout::RowMajor` with the default `Layout::ND` `GlobalTensor`. Either way, the **upstream `Tile` itself must satisfy the 32-byte-burst alignment of its physical (padded) dim** — not the logical valid dim — so odd `R` index tiles are expressed as a padded shape with a smaller `ValidCol`/`ValidRow` (see [Minimum Tile Shape](#minimum-tile-shape) below).

### Layout Support

The SIMT kernel itself is **layout-agnostic** for every UB tile it touches: every UB read/write goes through `tile_offset_2d<TileX>(r, c)`, which dispatches the index arithmetic from the tile type's `BLayout`. The caller is responsible for getting the data into UB with whatever `TLOAD` / `TMOV` / `TINSERT` configuration matches their upstream layout (ND, DN, NZ, RowMajor, ColMajor, etc.).

| Tile / Tensor | Supported layouts | Notes |
|---------------|-------------------|-------|
| `TileSrc` (UB) | Any `BLayout` (`RowMajor` or `ColMajor`); `SLayout::NoneBox` | Kernel reads via `tile_offset_2d<TileSrc>`. |
| `TileIdx` (UB) – Row mode | `[1, R]` `BLayout::RowMajor` **or** `[R, 1]` `BLayout::ColMajor` | Both produce a linear `R`-element layout in UB; the kernel reads `indices[row]` directly. |
| `TileIdx` (UB) – Elem mode | Any `BLayout` — independent of `TileSrc::BLayout` | Kernel reads via `tile_offset_2d<TileIdx>`. Same logical `(r, c)` is consumed for both `src` and `indices`. |
| `GlobalTable` (GM) | `Layout::ND` (linear contiguous addressing); a non-ND-stride `GlobalTensor` is the caller's responsibility | The kernel writes `table[idx * RowWidth + col]` (Row mode) or `table[idx]` (Elem mode). For NZ-stored GM the caller should stage results back via `TMOV` ND→NZ post-`MSCATTER` — no static assert is produced for the layout. |

### Aligned vs Unaligned Tile Shapes

The kernel does **not** care whether the tile's logical shape is "aligned" — it walks all `Rows * Cols` positions of the tile via `tile_offset_2d`. The 32-byte alignment of `Cols * sizeof(T)` (for `BLayout::RowMajor`) is enforced by the upstream `Tile` declaration only. Callers handle "unaligned valid region" by:

1. Padding the tile up to the nearest 32-byte alignment (e.g. valid `[3, 3]` int32 → tile `[3, 8]`), and
2. Either zero-initializing the padding (`TASSIGN`-then-clear) or only inspecting the valid region post-scatter.

The kernel's job is to populate / consume every position of the declared tile shape correctly; partial-tile semantics live in the caller's pre/post-move ops, exactly as in `TLOAD` / `TSTORE` / `TINSERT` / `TMOV` ST suites.

### Minimum Tile Shape

`MScatterCheck` accepts any `(ValidRow, ValidCol)` with `ValidRow, ValidCol >= 1` (including the degenerate `(1, 1)` for both Row and Elem modes — the SIMT kernel and the scalar fallback handle them transparently).

The actual lower bound on the **padded** `Tile<…, Rows, Cols, BLayout, ValidRow, ValidCol>` shape is **not** an MSCATTER concern — it is enforced upstream by the `Tile` system because every `TLOAD` / `TSTORE` that brings data in/out of UB issues 32-byte GM↔UB **DMA bursts**. The contiguous-in-memory dim of the tile must therefore be a whole number of bursts:

- `BLayout::RowMajor` ⇒ `Cols * sizeof(T) % 32 == 0` (one row = N×32 B).
- `BLayout::ColMajor` ⇒ `Rows * sizeof(T) % 32 == 0` (one column = N×32 B).

`ValidRow` / `ValidCol` are not constrained by this rule — they only set the kernel's iteration bounds. So a logical `(1, 1)` int32 tile is expressed as `Tile<int32, 1, 8, RowMajor, 1, 1>` (one padded burst, one valid element), `(3, 3)` int32 as `Tile<int32, 3, 8, RowMajor, 3, 3>` (three padded 32 B rows, valid 3×3 sub-region), etc. Smallest padded `Cols` per dtype for a row-major tile:

| `T` | Min `Cols` (`BLayout::RowMajor`) | Min `Rows` (`BLayout::ColMajor`) |
|-----|---------------------------------|-----------------------------------|
| `int8` / `uint8` / `float8_e4m3` / `float8_e5m2` / `hifloat8` | 32 | 32 |
| `int16` / `uint16` / `half` / `bfloat16` | 16 | 16 |
| `int32` / `uint32` / `float` | 8 | 8 |

The padded dimension is purely a **TLOAD/TSTORE alignment artifact** — the SIMT kernel itself walks `ValidRow * ValidCol` positions and uses `tile_offset_2d<TileX>(r, c)` (which evaluates to `r * Tile::Cols + c` for RowMajor, etc.) to compute the right physical UB offset inside the padded tile.

### Mode Resolution

Mode is **explicit**, not auto-detected. The static-asserts in `MScatterCheck` validate that the supplied tile shapes match the chosen `Coalesce` value:

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
    using SrcTile    = Tile<TileType::Vec, T,        R,         C, BLayout::RowMajor, R, C>;
    using IdxTile    = Tile<TileType::Vec, int32_t,  1,         R, BLayout::RowMajor, 1, R>;

    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile    src;    TASSIGN(src,    0x0000);
    IdxTile    idx;    TASSIGN(idx,    0x1000);

    MSCATTER<ScatterAtomicOp::None, ScatterOOB::Skip, ScatterConflict::Last, Coalesce::Row>(
        tableGM, src, idx);
}
```

### Atomic Gradient Accumulation (Row Coalesce)

```cpp
#include <pto/npu/a5/MScatter.hpp>

using namespace pto;

AICORE void example_gradient_accumulation(__gm__ float* gradTable)
{
    constexpr int NumTokens = 16, D = 64, Vocab = 65536;

    using SrcTile    = Tile<TileType::Vec, float,   NumTokens,         D, BLayout::RowMajor, NumTokens, D>;
    using IdxTile    = Tile<TileType::Vec, int32_t,         1, NumTokens, BLayout::RowMajor,         1, NumTokens>;

    using TableShape  = Shape<1, 1, 1, Vocab, D>;
    using TableStride = Stride<1, 1, 1, D, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(gradTable);
    SrcTile    grads;  TASSIGN(grads,  0x0000);
    IdxTile    idx;    TASSIGN(idx,    0x4000);

    MSCATTER<ScatterAtomicOp::Add, ScatterOOB::Skip, ScatterConflict::Default, Coalesce::Row>(
        tableGM, grads, idx);
}
```

### Element Coalesce — 2-D Sparse Update

```cpp
#include <pto/npu/a5/MScatter.hpp>

using namespace pto;

AICORE void example_sparse_update(__gm__ float* data)
{
    constexpr int R = 8, C = 32, TableRows = 64, TableCols = 64;
    constexpr int TableSize = TableRows * TableCols;

    using SrcTile    = Tile<TileType::Vec, float,   R,         C, BLayout::RowMajor, R, C>;
    using IdxTile    = Tile<TileType::Vec, int32_t, R,         C, BLayout::RowMajor, R, C>;

    using TableShape  = Shape<1, 1, 1, TableRows, TableCols>;
    using TableStride = Stride<1, 1, 1, TableCols, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor dataGM(data);
    SrcTile    src;    TASSIGN(src,    0x0000);
    IdxTile    idx;    TASSIGN(idx,    0x0800);

    MSCATTER<ScatterAtomicOp::None, ScatterOOB::Wrap, ScatterConflict::Last, Coalesce::Elem>(
        dataGM, src, idx);
}
```

### Element Coalesce — 1-D Source

```cpp
#include <pto/npu/a5/MScatter.hpp>

using namespace pto;

AICORE void example_elem_1d(__gm__ float* data)
{
    constexpr int N = 64, TableSize = 128;

    using SrcTile    = Tile<TileType::Vec, float,   1,         N, BLayout::RowMajor, 1, N>;
    using IdxTile    = Tile<TileType::Vec, int32_t, 1,         N, BLayout::RowMajor, 1, N>;

    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor dataGM(data);
    SrcTile    src;    TASSIGN(src,    0x0000);
    IdxTile    idx;    TASSIGN(idx,    0x0200);

    MSCATTER<ScatterAtomicOp::None, ScatterOOB::Undefined, ScatterConflict::Last, Coalesce::Elem>(
        dataGM, src, idx);
}
```

### Deterministic Last-Write-Wins Over Colliding Indices

```cpp
#include <pto/npu/a5/MScatter.hpp>

using namespace pto;

AICORE void example_last_deterministic(__gm__ half* tablePtr)
{
    constexpr int R = 8, C = 64, TableRows = 65536;

    using SrcTile    = Tile<TileType::Vec, half,    R,         C, BLayout::RowMajor, R, C>;
    using IdxTile    = Tile<TileType::Vec, int32_t, R,         1, BLayout::ColMajor, R, 1>;

    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile    src;    TASSIGN(src,    0x0000);
    IdxTile    idx;    TASSIGN(idx,    0x1000);

    MSCATTER<ScatterAtomicOp::None, ScatterOOB::Clamp, ScatterConflict::Last, Coalesce::Row>(
        tableGM, src, idx);
}
```

## Performance Considerations

1. **Shape-adaptive launch.** `MScatterRowImpl` / `MScatterElemImpl` size the SIMT grid as `dim3{WARP_SIZE, kLaunchWarps}` from the resolved `validRows` / `validCols` / `tableSize` (compile-time constants for static tiles, runtime values for dynamic tiles). Small tiles do not pay the cost of launching 1024 idle threads.
   - **Row, non-`Last`.** `kRowWarps = min(validRows, 32)` own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks. `kLaunchWarps = kRowWarps * kWarpsPerRow`. Lane writes form 128-byte coalesced stores; `kColStride = kWarpsPerRow * 32`.
   - **Elem, non-`Last`.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`. Threads with `tid >= totalElems` skip the loop body (no garbage access). For `totalElems > 1024` the strided loop walks `launchThreads` at a time.
   - **`Conflict::Last` (Row and Elem).** Launch is sized by the **destination** instead of the source: `kLaunchWarps = min(ceil(TableSize / 32), 32)`. Each lane owns one slot per outer iteration, so partitioning by `tableSize` keeps the slot-centric kernels balanced even when `N >> TableSize`.

2. **Linear thread mapping (non-`Last`).** Each thread handles one element / row via stride-`kLaunchThreads` iteration over the flat work set. Lanes 0..31 of each warp read/write 32 consecutive UB / GM addresses per iteration (UB reads are bank-conflict-free; GM writes are scatter-stride-driven by the `idx` values).

3. **No thread divergence for mode / policy control.** All mode / atomic / OOB / conflict decisions are `if constexpr`. In Row coalesce the `doWrite` predicate depends only on `row` (warp-uniform) and is hoisted outside the inner 32-lane column loop, so the whole warp takes the branch together. The slot-centric `Last` kernels have a per-lane `found` predicate that compiles to a predicated store (no control-flow divergence).

4. **Conflict policy cost.**
   - `Last`: each lane owns a distinct destination slot and runs an in-register reverse scan over the index tile, terminating on the first matching source position. The race is removed by construction — no two lanes write the same slot — so the kernel never reads back the GM table, never issues an atomic, and never allocates UB scratch. Worst-case work per warp is `O(N)` (uncoalesced index space), but a uniformly random workload averages `O(TableSize / 32)` lockstep iterations per warp.
   - `Default`: zero extra work — the surviving lane is whatever the warp scheduler picked. Use only when collisions are impossible (unique indices) or the result is order-insensitive.
   - **Atomic modes (`Add` / `Max` / `Min`) skip the conflict gate entirely** and serialize via the GM atomic instruction itself; no `cur` preload is performed (the atomic R-M-W binds the destination region naturally).

5. **Unrolled inner loops.** Inner column loop in Row coalesce carries `#pragma unroll(4)` so the compiler unrolls for small compile-time trip counts (e.g. `RowWidth=32, kColStride=32` ⇒ 1 iter, fully unrolled). The outer scatter loop and the slot-centric reverse-scan loop are `#pragma unroll(1)` to keep code size bounded for large `N`.

6. **Out-of-bounds mode.** `ScatterOOB::Undefined` is fastest but requires valid indices. Use `Skip`/`Clamp`/`Wrap` when indices may exceed bounds.

7. **Row vs. Elem.** Row coalesce achieves the best GM write bandwidth (32 consecutive lanes per coalesced store). Elem coalesce performs one scalar GM store per active lane — non-coalesced at GM in general.

8. **Register pressure / MRF.** The kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread budget) and use ≤ 16 live registers per thread in the hot path. No spills are produced; the compile flag `-mllvm -cce-aicore-record-overflow=true` reports no overflow events for any of the instantiations.

## SIMT Usage Restrictions

`MSCATTER` is a SIMT launch on the AIV vector core. Every byte the runtime, the compiler, and the user store in UB must coexist inside the single 256 KB Unified Buffer that the AIV exposes. The on-board ceiling is fixed by two top-of-UB reservations the toolchain installs before any user tile is allocated:

| Region                  | Size on a5 (V310) | Source                                                                                              |
|-------------------------|-------------------|-----------------------------------------------------------------------------------------------------|
| Physical UB             | 256 KB            | Hardware                                                                                            |
| Hardware D-cache        | 32 KB             | Top of UB, scalar/SIMT D-cache working set                                                          |
| Compiler stack (scalar + VF + SIMT) | 8 KB              | All scalar, vector-fragment, and per-thread SIMT spill traffic for the `MSCATTER` call chain        |

The remaining `256 − 32 − 8 = 216 KB` is what user tiles can address. In practice the **safe per-call budget for `src + idx` in UB is `≤ 128 KB`** — at that point both tiles are already at the 64 KB-each comfort line and any further growth starts to push the SIMT spill region into the user-tile band, which the compiler does not flag and which surfaces on-board as a silent all-zero output (the CPU simulator does not model the spill, so it still passes).

When sizing a workload, account for both the **source** tile (`R * C * sizeof(T)`, padded up to the 32-byte burst alignment) and the **index** tile (`R * C * sizeof(TIdx)`, same padding rule). For `Conflict::Last` the destination side adds no extra UB pressure — the slot-centric scan operates directly out of the same `src` / `idx` UB tiles and stores straight to GM.

### Tiled-Iteration Pattern for Large Inputs

Once a single `src + idx` footprint exceeds the safe budget, the caller must process the input in **chunks** that each fit comfortably under the ceiling. Each chunk does its own `TLOAD → MSCATTER` round-trip into the same destination GM tensor; semantics are preserved because:

- **`Conflict::Last`**: each chunk writes its in-chunk last-writer to GM; later chunks overwrite earlier ones for any shared slot, so the surviving value is the global largest source index targeting that slot.
- **`Conflict::Default`** / atomic modes: writes from later chunks naturally compose with writes from earlier chunks (overwrite, add, max, min) into the same GM table.

The `case_elem2d_float_2048x8_*` ST cases use this pattern: a `2048 × 8` `float` source plus matching `int32_t` index would total `128 KB` in a single shot, so the wrapper splits it into **16 chunks of `128 × 8`** (`4 KB src + 4 KB idx = 8 KB UB per iteration`) and re-issues `MSCATTER` per chunk. The same shape with no chunking failed silently on-board while passing the CPU simulator — a textbook example of the over-budget mode described above.

### Cache-Coherence Flush

Every `runMSCATTER_*` wrapper finishes with a `FlushScatterOutput()` helper:

```cpp
AICORE PTO_INLINE void FlushScatterOutput()
{
    dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
```

`dcci(0, ENTIRE_DATA_CACHE)` invalidates the AIV scalar D-cache so any GM writes still buffered in the cache are forced down to HBM, and `dsb(DSB_DDR)` waits until the writes are observable at the DDR boundary. On a5/V310 the compiler default `--cce-no-dcache-flush=0` already emits a similar flush before kernel exit, but `MSCATTER` issues its GM writes from inside an `async_invoke` SIMT VF call, so adding the explicit flush guarantees the writes are committed regardless of where the compiler decides to insert the implicit one.

## Runtime Dispatch Requirement

`MSCATTER` (like every SIMT kernel in PTO and CANN) uses `cce::async_invoke<simt_mscatter_*_kernel>(cce::dim3{WARP_SIZE, kLaunchWarps}, …)` internally to fan a per-warp/per-lane workload out across up to `32 × 32 = 1024` threads. `cce::async_invoke` consumes hardware/runtime state — TID registers (`__cce_simt_get_TID_X/Y`), warp/lane configuration, vector-pipe scheduling — that the **launch path** has to install **before** the kernel function is entered. The standard CANN launch (`rtKernelLaunchWithHandleV2`, used by the `<<<1, nullptr, stream>>>` syntax in every ST in this suite) installs that state correctly.

`pto-isa#96` describes a runtime variant (`tensormap_and_ringbuffer/aicore/aicore_executor.cpp`) that dispatches kernels as a **direct C function-pointer call**:

```cpp
UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
```

This is fine for SPMD ops (TLOAD, TSTORE, TADD …) but skips the SIMT-context init step, so the first `cce::async_invoke` inside `MSCATTER` has no warp scheduler to dispatch into and hangs.

A compile-time-only "use the full launch pipeline" override has to live in the **dispatcher**, not the kernel — by the time the kernel function is entered, the dispatch decision has already been made. Specifically:

- `__simt_vf__`, `LAUNCH_BOUND(1024)`, `__simt_callee__`, `__cce_simt_get_TID_X/Y`, `cce::async_invoke`, `cce::dim3` are **all toolchain (bisheng/CCE) intrinsics**; PTO does not define them. Whatever metadata the compiler attaches to the resulting `.o` is what the dispatcher needs to inspect.
- There is no in-tree `cce::is_simt_active()` guard or `cce::simt_init()` device-side bootstrap intrinsic — a regular AIV function cannot self-promote itself into SIMT context.

`MSCATTER` already carries every signal a smart dispatcher could use to pick the right path: the `cce::async_invoke<…>(dim3{…}, …)` call site itself, the `__simt_vf__ LAUNCH_BOUND(1024)` attributes on `simt_mscatter_*_kernel`, and the templated SIMT entry that the toolchain tags in the `.o`.

In dependency order (cheapest first): Note - We will try to resolve this issue as soon as possible

1. **Mark SIMT kernels in `PTO2DispatchPayload` with a flag, branch the dispatch path** in `aicore_executor.cpp` — direct fn-pointer for SPMD payloads, `rtKernelLaunch` for SIMT payloads. Smallest runtime change; isolates the hot path for SPMD ops.
2. **Detect SIMT kernels at compile time** (toolchain-emitted attribute / ELF note from `__simt_vf__` / `LAUNCH_BOUND(1024)`) and unconditionally route them through `rtKernelLaunch`. Works automatically for every SIMT kernel (existing and future), but requires the dispatcher to read the toolchain's metadata.
3. **Initialize the SIMT context** (TID/warp/pipe regs) inside `aicore_executor.cpp` immediately before the function-pointer call. Most invasive — duplicates work the standard CANN launch path already does — but works without per-payload flagging.

## Related Instructions

- [`TSTORE`](/docs/isa/TSTORE.md): Contiguous block transfer from Tile to GM
- [`TSCATTER`](/docs/isa/TSCATTER.md): Index-based scatter within tiles (UB-to-UB)
- [`MGATHER`](../mgather/MGATHER.md): Indexed gather from GM to Tile (inverse operation)

## Test Cases

### Row Coalesce — `[1, R]` index form

| Case | Data Type | Src Size | TableRows | Atomic | OOB Mode | Conflict | Idx Pattern |
|------|-----------|----------|-----------|--------|----------|----------|-------------|
| case_row_float_random_8x32_64rows       | float | 8×32  | 64 | None | Undefined | Last  | random |
| case_row_float_same_8x32_16rows         | float | 8×32  | 16 | None | Undefined | Last  | same   |
| case_row_half_random_16x64_64rows       | half  | 16×64 | 64 | None | Undefined | Last  | random |
| case_row_int32_random_8x16_32rows       | int32 | 8×16  | 32 | None | Undefined | Last  | random |
| case_row_uint8_random_8x32_32rows       | uint8 | 8×32  | 32 | None | Undefined | Last  | random |
| case_row_int16_random_8x16_32rows       | int16 | 8×16  | 32 | None | Undefined | Last  | random |
| case_row_float_atomicadd_8x32_8rows     | float | 8×32  | 8  | Add  | Undefined | Default | random |
| case_row_float_skip_8x32_8rows          | float | 8×32  | 8  | None | Skip      | Last  | oob    |
| case_row_int32_clamp_8x16_8rows         | int32 | 8×16  | 8  | None | Clamp     | Last  | oob    |
| case_row_half_wrap_8x32_8rows           | half  | 8×32  | 8  | None | Wrap      | Last  | oob    |

### Row Coalesce — `[R, 1]` index form (ColMajor + DN)

| Case | Data Type | Src Size | TableRows | Atomic | OOB Mode | Conflict |
|------|-----------|----------|-----------|--------|----------|----------|
| case_row_colidx_float_random_8x32_64rows  | float | 8×32  | 64 | None | Undefined | Last |
| case_row_colidx_int32_clamp_8x16_8rows    | int32 | 8×16  | 8  | None | Clamp     | Last |
| case_row_colidx_half_random_16x64_64rows  | half  | 16×64 | 64 | None | Undefined | Last |

### Element Coalesce — 1-D source `[1, N]`

| Case | Data Type | N / TableSize | Atomic | OOB Mode | Conflict | Idx Pattern |
|------|-----------|---------------|--------|----------|----------|-------------|
| case_elem_float_random_64_128size          | float | 64  / 128  | None | Undefined | Last  | random |
| case_elem_float_same_64_8size              | float | 64  / 8    | None | Undefined | Last  | same   |
| case_elem_float_seq_32_32size              | float | 32  / 32   | None | Undefined | Last  | seq    |
| case_elem_half_random_64_128size           | half  | 64  / 128  | None | Undefined | Last  | random |
| case_elem_int32_random_32_64size           | int32 | 32  / 64   | None | Undefined | Last  | random |
| case_elem_uint8_random_64_128size          | uint8 | 64  / 128  | None | Undefined | Last  | random |
| case_elem_int16_random_32_64size           | int16 | 32  / 64   | None | Undefined | Last  | random |
| case_elem_float_atomicadd_32_32size        | float | 32  / 32   | Add  | Undefined | Default | random |
| case_elem_int32_atomicadd_skip_32_16size   | int32 | 32  / 16   | Add  | Skip      | Default | oob    |
| case_elem_float_skip_32_16size             | float | 32  / 16   | None | Skip      | Last    | oob    |
| case_elem_int32_clamp_32_16size            | int32 | 32  / 16   | None | Clamp     | Last    | oob    |
| case_elem_half_wrap_32_16size              | half  | 32  / 16   | None | Wrap      | Last    | oob    |
| case_elem_float_default_seq_32_32size      | float | 32  / 32   | None | Undefined | Default | seq    |
| case_elem_float_small_16_32size            | float | 16  / 32   | None | Undefined | Last    | random |
| case_elem_int32_atomicmax_random_32_32size | int32 | 32  / 32   | Max  | Undefined | Default | random |
| case_elem_float_atomicmin_random_32_32size | float | 32  / 32   | Min  | Undefined | Default | random |
| case_elem_float_last_same_32_8size         | float | 32  / 8    | None | Undefined | Last  | same   |
| case_elem_int32_last_seq_32_32size         | int32 | 32  / 32   | None | Undefined | Last  | seq    |
| case_elem_float_clamp_no_dup_32_16size     | float | 32  / 16   | None | Clamp     | Last  | random |
| case_elem_uint8_wrap_64_16size             | uint8 | 64  / 16   | None | Wrap      | Last  | random |
| case_elem_int16_clamp_32_16size            | int16 | 32  / 16   | None | Clamp     | Last  | oob    |

### Element Coalesce — 2-D source `[R, C]`

| Case | Data Type | Src Size | TableSize | Atomic | OOB Mode | Conflict | Idx Pattern |
|------|-----------|----------|-----------|--------|----------|----------|-------------|
| case_elem2d_float_8x32_random_256size       | float | 8×32     | 256   | None | Undefined | Last    | random |
| case_elem2d_int32_8x16_random_256size       | int32 | 8×16     | 256   | None | Undefined | Last    | random |
| case_elem2d_half_4x32_random_256size        | half  | 4×32     | 256   | None | Undefined | Last    | random |

### Element Coalesce — Tiled Iteration

These cases would exceed the safe `src + idx` UB budget if loaded in one shot, so the wrapper drives `MSCATTER` per-chunk and lets later chunks overwrite earlier ones to compose the final result (see [Tiled-Iteration Pattern for Large Inputs](#tiled-iteration-pattern-for-large-inputs)).

| Case | Data Type | Total Src | Chunk | UB per Chunk | TableSize | Atomic | OOB Mode | Conflict | Idx Pattern |
|------|-----------|-----------|-------|--------------|-----------|--------|----------|----------|-------------|
| case_elem2d_float_2048x8_last_256size       | float | 2048×8 | 128×8 (16 iters) | 4 KB src + 4 KB idx | 256   | None | Undefined | Last    | random |
| case_elem2d_float_2048x8_default_16384size  | float | 2048×8 | 128×8 (16 iters) | 4 KB src + 4 KB idx | 16384 | None | Undefined | Default | seq    |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles **any** `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. There is no SIMD-style batch / predicate path — the kernel walks `ValidRow * ValidCol` positions one element per thread, and `tile_offset_2d<TileX>` resolves the physical UB offset inside the padded tile. So "unaligned" reduces to one mechanism: declare the `Tile` with the **padded** shape that satisfies the `Tile` system's burst-alignment rule (see [Minimum Tile Shape](#minimum-tile-shape)) and the **valid** shape carrying the logical `(R, C)`.

Three categories are covered by ST:

| Category | Pattern | What it exercises |
|----------|---------|-------------------|
| Odd row count, valid == padded | `Tile<T, R, C, RowMajor, R, C>` with odd `R` (3, 9, …) and `C` already 32-B aligned | Shape-adaptive launch — `kRowWarps = R`, `kWarpsPerRow = 32 / R`, etc. |
| True valid-in-padded | `Tile<T, PadR, PadC, RowMajor, ValidR, ValidC>` with `ValidR/ValidC < PadR/PadC` | Per-thread iteration over `ValidR × ValidC` only, padding bytes are never touched by the SIMT body. |
| Scalar `(1, 1)` | `Tile<T, 1, MinAlignedC, RowMajor, 1, 1>` | Compile-time fast-path to `MScatterScalarImpl` — no `cce::async_invoke`. |

For `Coalesce::Row` with odd `R`, the index tile must also be expressed as valid-in-padded: e.g. `Tile<int32, 1, 8, RowMajor, 1, 3>` for 3 row-indices, since the upstream `Tile` system needs the padded inner dim aligned. `MSCATTER` itself only checks the **valid** equality `TileIdx::ValidCol == TileSrc::ValidRow` (or the `[R, 1]` analog).

| Case | Mode | Data Type | Valid Src | Padded Src | Idx (Valid → Padded) | Table | Atomic | OOB Mode | Conflict |
|------|------|-----------|-----------|------------|----------------------|-------|--------|----------|----------|
| case_elem2d_int32_unaligned_3x8_64size              | Elem | int32 | 3×8 | 3×8 | 3×8 → 3×8 | 64 elems | None | Undefined | Last |
| case_elem2d_uint8_unaligned_3x32_256size            | Elem | uint8 | 3×32 | 3×32 | 3×32 → 3×32 | 256 elems | None | Undefined | Last |
| case_elem2d_int32_unaligned_3x3_in_3x8_64size       | Elem | int32 | 3×3 | 3×8 | 3×3 → 3×8 | 64 elems | None | Undefined | Last |
| case_elem2d_int32_unaligned_9x9_in_9x16_256size     | Elem | int32 | 9×9 | 9×16 | 9×9 → 9×16 | 256 elems | None | Undefined | Last |
| case_elem2d_int32_scalar_1x1_in_1x8_8size           | Elem (scalar) | int32 | 1×1 | 1×8 | 1×1 → 1×8 | 8 elems | None | Undefined | Last |
| case_row_int32_unaligned_3x8_8rows                  | Row | int32 | 3×8 | 3×8 | 1×3 → 1×8 | 8 rows × 8 | None | Undefined | Last |
| case_row_int32_unaligned_9x16_16rows                | Row | int32 | 9×16 | 9×16 | 1×9 → 1×16 | 16 rows × 16 | None | Undefined | Last |
