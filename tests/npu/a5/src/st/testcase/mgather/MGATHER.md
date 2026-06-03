# MGATHER

## Tile Operation Diagram

![MGATHER tile operation](../../../../../../../docs/figures/isa/MGATHER.svg)

## Introduction

`MGATHER` reads data from a GM `GlobalTensor` into a UB destination tile using a UB index tile. On Ascend A5 it is implemented as a SIMT kernel dispatched through `cce::async_invoke` with up to `dim3{32, 32}` = 1024 threads.

The operating mode is selected by the `Coalesce` template parameter:

- **`Coalesce::Row`** (default): gather whole rows — `dst[r, :] = table[idx[r], :]`. The index tile is 1-D (`[R, 1]` or `[1, R]`). `R = 1` is allowed.
- **`Coalesce::Elem`**: gather single elements — `dst[r, c] = table[idx[r, c]]`. The index tile has the same shape as `dst`. 1-D destinations (`[1, N]` or `[N, 1]`) are also supported.

Out-of-bounds index handling is controlled by the `GatherOOB` template parameter. Gather has no atomic or conflict policy — every destination slot has exactly one source index, so collisions cannot occur.

## Math Interpretation

### Row Coalesce

Destination `dst[R, C]`, index `idx[R, 1]` or `idx[1, R]`, table `table[TableRows, C]`:

$$ \mathrm{dst}_{r, j} = \mathrm{table}_{\mathrm{idx}_{r},\; j},\quad 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce

Destination `dst[R, C]`, index `idx[R, C]` (same shape as `dst`), flat table of length `TableSize = Shape[0]·Shape[1]·Shape[2]·Shape[3]·Shape[4]`:

$$ \mathrm{dst}_{r, c} = \mathrm{table}[\mathrm{idx}_{r, c}],\quad 0 \le r < R,\; 0 \le c < C $$

## Assembly Syntax

See [docs/assembly/PTO-AS.md](/docs/assembly/PTO-AS.md).

Row coalesce:

```text
mgather.row %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<Rx1xi32>)
```

Element coalesce:

```text
mgather.elem %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<RxCxi32>)
```

OOB-aware variants append the mode suffix, e.g. `mgather.row.clamp`, `mgather.elem.zero`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/npu/a5/MGather.hpp`:

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Mode  = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& table, TileInd& idx,
                             WaitEvents&... events);
```

The kernel iterates over `TileDst::ValidRow × TileDst::ValidCol` logical positions. Physical UB strides come from the same `Tile` types via `tile_offset_2d` (i.e. `TileDst::Cols`, `TileIdx::Cols`).

For `Coalesce::Elem` with `TileDst::ValidRow == 1 && TileDst::ValidCol == 1` the implementation bypasses the SIMT launch and runs a scalar fallback (`MGatherScalarImpl`) on the AIV vector core.

**Parameters:**
- `dst` — UB destination tile (`TileType::Vec`), shape `[R, C]`. Row-major and column-major storage are both accepted.
- `table` — source GM `GlobalTensor`. `GlobalTensor::DType` must be `__gm__ T` matching the destination element type.
- `idx` — UB index tile (`TileType::Vec`). For `Row`: `[1, R]` or `[R, 1]`. For `Elem`: same shape as `dst` (the index tile's `BLayout` may differ from the destination's).
- `CMode` — `Coalesce` value (`Row` or `Elem`); first template parameter so the mode is always explicit at the call site.
- `Mode` — `GatherOOB` value for out-of-bounds handling.

## Enum Definitions

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // dst[r, :] = table[idx[r], :]   (1-D index of length R)
    Elem = 1   // dst[i, j] = table[idx[i, j]]   (idx shape == dst shape)
};

enum class GatherOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Clamp     = 1,  // Clamp index to capacity - 1
    Wrap      = 2,  // Index modulo capacity
    Zero      = 3   // Return 0 for OOB; in-bounds indices loaded normally
};
```

`capacity` for OOB handling is `TableRows` in Row mode and the full flat table length in Elem mode.

## Constraints

### Data Types

`TileDst::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.

### Index Types

`TileIdx::DType` must be `int32_t` or `uint32_t`.

### Tile Constraints

- Both tiles must live in UB (`TileDst::Loc == TileType::Vec`, `TileIdx::Loc == TileType::Vec`).
- Destination and table must share the same element type `T` (`GlobalTable::DType == __gm__ T`).
- For `Coalesce::Row`: the index tile's **valid shape** is `[1, R]` or `[R, 1]` matching `TileDst::ValidRow`. The table's inner dim (`Shape[4]`) must equal `TileDst::ValidCol`.
- For `Coalesce::Elem`: `TileIdx` must have the same valid shape as `TileDst`. The `BLayout` of `TileIdx` is independent of the destination layout — the kernel walks both via per-tile `tile_offset_2d`.

### Dynamic Runtime Shapes

`MGATHER` accepts both compile-time and runtime-dynamic shapes. Any dimension declared as `-1` is resolved at runtime:

- `Tile<…, -1, -1>` stores the runtime valid extents in the tile object; `MGATHER_IMPL` reads them via `dst.GetValidRow()` / `dst.GetValidCol()`.
- `Shape<…, -1, -1>` / `Stride<…, -1, -1>` are constructed with the runtime sizes; `MGATHER_IMPL` reads them via `table.GetShape(…)` and folds them into `tableRows` or `tableSize`.

Static-asserts in `MGatherCheck` are gated on `if constexpr (DIM > 0)` so they fire only for compile-time dimensions. The padded `Tile::Rows` / `Tile::Cols` remain compile-time — they govern the UB DMA-burst alignment.

Example (mirrors `case_elem2d_dyn_user_float_1x9_in_1x16_3x10`):

```cpp
constexpr auto kPadCols = 16;
using DstTileT    = Tile<TileType::Vec, float,   1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT    = Tile<TileType::Vec, int32_t, 1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t validCols = 9, tableR = 3, tableC = 10;
TableShape  tableShape(tableR, tableC);
TableStride tableStride(tableC, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(srcGm, tableShape, tableStride);

DstTileT dstTile(1, validCols);
IdxTileT idxTile(1, validCols);
TASSIGN(dstTile, dstUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MGATHER<Coalesce::Elem, GatherOOB::Undefined>(dstTile, tableGM, idxTile);
```

### Layout Support

The SIMT kernel is layout-agnostic for UB tiles: every UB read / write goes through `tile_offset_2d<TileX>(r, c)`, which dispatches index arithmetic from the tile type's `BLayout`. The caller is responsible for staging the data into UB with the right `TLOAD` configuration.

| Tile / Tensor | Supported layouts | Notes |
|---------------|-------------------|-------|
| `TileDst` (UB) | `BLayout::RowMajor` or `ColMajor`, `SLayout::NoneBox` | Kernel writes via `tile_offset_2d<TileDst>`. |
| `TileIdx` (UB) — Row mode | `[1, R]` `RowMajor` **or** `[R, 1]` `ColMajor` | Both produce a linear R-element layout in UB. |
| `TileIdx` (UB) — Elem mode | any `BLayout`, independent of `TileDst` | Kernel reads via `tile_offset_2d<TileIdx>`. |
| `GlobalTable` (GM) | `Layout::ND` (linear contiguous) | Non-ND layouts must be pre-staged by the caller. |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles **any** `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. To express odd valid extents (e.g. `3×3`), pad the tile up to the nearest 32-byte aligned shape (e.g. `3×8` for int32) and let the kernel iterate over the valid sub-region only. The padding is purely an upstream `TLOAD` burst-alignment requirement.

Minimum padded inner dim per dtype (row-major):

| `T` | Min `Cols` (`RowMajor`) | Min `Rows` (`ColMajor`) |
|-----|-------------------------|-------------------------|
| `int8` / `uint8` / `float8_e4m3` / `float8_e5m2` / `hifloat8` | 32 | 32 |
| `int16` / `uint16` / `half` / `bfloat16` | 16 | 16 |
| `int32` / `uint32` / `float` | 8 | 8 |

### Mode Resolution

Mode is **explicit**, never auto-detected. Static asserts in `MGatherCheck` validate the tile shapes against the chosen `Coalesce` value:

```text
Coalesce::Row  : (Idx.Rows == 1 && Idx.Cols == Dst.Rows) || (Idx.Rows == Dst.Rows && Idx.Cols == 1)
Coalesce::Elem : (Idx.Rows == Dst.Rows) && (Idx.Cols == Dst.Cols)
```

## Examples

### Row Coalesce — Embedding Lookup

```cpp
#include <pto/npu/a5/MGather.hpp>

using namespace pto;

template <typename T, int R, int C, int TableRows>
AICORE void example_embedding_lookup(__gm__ T* tablePtr, __gm__ int32_t* idxPtr, __gm__ T* outPtr)
{
    using DstTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;
    using IdxShape    = Shape<1, 1, 1, 1, R>;
    using IdxStride   = Stride<1, 1, 1, R, 1>;
    using IdxTensor   = GlobalTensor<int32_t, IdxShape, IdxStride>;

    TableTensor tableGM(tablePtr);
    IdxTensor   idxGM(idxPtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    TLOAD(idx, idxGM);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    MGATHER<Coalesce::Row, GatherOOB::Clamp>(dst, tableGM, idx);
}
```

### Element Coalesce — 2-D Random Access

```cpp
AICORE void example_elem_2d(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 32, TableSize = 256;

    using DstTile     = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MGATHER<Coalesce::Elem, GatherOOB::Wrap>(dst, tableGM, idx);
}
```

### Row Coalesce — `[R, 1]` ColMajor Index

```cpp
AICORE void example_row_colidx(__gm__ half* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 64, TableRows = 64;

    using DstTile     = Tile<TileType::Vec, half,    R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;
    using IdxShape    = Shape<1, 1, 1, R, 1>;
    using IdxStride   = Stride<1, 1, 1, 1, 1>;
    using IdxTensor   = GlobalTensor<int32_t, IdxShape, IdxStride, Layout::DN>;

    TableTensor tableGM(tablePtr);
    IdxTensor   idxGM(idxPtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    TLOAD(idx, idxGM);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    MGATHER<Coalesce::Row, GatherOOB::Undefined>(dst, tableGM, idx);
}
```

## Performance Considerations

1. **Shape-adaptive launch.** The SIMT grid is sized as `dim3{32, kLaunchWarps}` from the resolved `validRows` / `validCols`. Small tiles do not pay the cost of 1024 idle threads.
   - **Row.** `kRowWarps = min(validRows, 32)` warps own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks. Lane reads form 128-byte coalesced GM bursts; lane writes form 128-byte coalesced UB stores.
   - **Elem.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`. Threads with `tid >= totalElems` skip the loop body. For `totalElems > 1024` the strided loop walks `launchThreads` at a time.

2. **OOB policy cost.**
   - `Undefined`: zero overhead — caller guarantees valid indices.
   - `Clamp` / `Wrap`: a single arithmetic remap per lane (`min` / `mod`).
   - `Zero`: one extra compare-and-select per lane to substitute `static_cast<T>(0)` for OOB lanes.

3. **No thread divergence for mode / OOB.** All decisions are `if constexpr`. The `gather_remap` lookup compiles to a small data-dependent transform with no control-flow split.

4. **Unrolled inner loops.** Inner column loop in Row coalesce carries `#pragma unroll(4)`; outer per-row and elem flat loops are `#pragma unroll(1)` to keep code size bounded.

5. **Row vs. Elem bandwidth.** Row coalesce achieves the best aggregate bandwidth (32 consecutive lanes per coalesced GM burst). Elem coalesce performs one scalar GM load per active lane — non-coalesced in general; UB writes remain coalesced.

6. **Register pressure.** Kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread) and use ≤ 12 live registers in the hot path. No spills are produced.

## UB Memory Budget

`MGATHER` is a SIMT launch on the AIV vector core. All user tiles must fit inside the AIV's 256 KB Unified Buffer alongside two fixed runtime reservations: an 8 KB reserved region (AscendC / TBE bookkeeping) and the Data Cache (32 KB minimum, sized at launch time).

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

For `MGATHER` kernels in this ST suite the destination and index tiles are placed manually with `TASSIGN`, so the compiler sees `static_memory ≈ 0` and the full **216 KB** is available as `dynUBufSize`.

`MGATHER` itself does not allocate any UB scratch — every read flows GM → register → UB. The only UB consumers from the kernel side are the destination tile (`R * C * sizeof(T)`, padded up to the 32-byte burst alignment) and the index tile (`R * C * sizeof(TIdx)`, same padding rule).

### Default Per-Call Budget (No `dynUBufSize`)

When the kernel is launched without an explicit `dynUBufSize` (i.e. `<<<numBlocks, nullptr, stream>>>`), the runtime keeps the default DCache size and reserves only the small default dynamic region. In practice, on Ascend A5 the safe `dst + idx` working set is **≤ 128 KB**; beyond that, on-board execution may silently corrupt or zero out the result while still passing the CPU simulator (which does not model these reservations).

### Extending Per-Call UB Beyond 128 KB (`dynUBufSize`)

Callers that need a single-shot `dst + idx` footprint **larger than 128 KB** must declare the dynamic-UB request explicitly via the second argument of the kernel launch:

```cpp
kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...);
```

`dynUBufSize` is the byte size of the dynamic-UB region the kernel will use. The bisheng/CCE compiler routes such launches through `__cce_rtKernelLaunchWithFlagV2`, setting `rtTaskCfgInfo_t::localMemorySize = dynUBufSize`. The runtime then shrinks the DCache toward its 32 KB minimum and hands the remaining space back to the kernel.

Key points:

- **The simulator does not enforce this.** Passing `nullptr` (or `0`) still runs to completion in sim regardless of the actual UB footprint. Always set `dynUBufSize` explicitly when the workload exceeds 128 KB so the binary stays correct on real hardware.
- **Exceeding the ceiling is silent.** The compiler does not error and the simulator does not flag it. On-board, the first overflow byte corrupts the reserved region or DCache and the kernel returns undefined output.
- **Size to actual usage.** For an Elem-coalesce case with `R × C × sizeof(T)` destination and `R × C × sizeof(int32_t)` index, the working set is `R * C * (sizeof(T) + 4)`; round up to a comfortable margin when passing `dynUBufSize`.

Example launch wrapper for an extended-UB gather (mirrors the pattern used by the MSCATTER extended-UB cases):

```cpp
// Round dst + idx up to a safe dynUBufSize (here T = float, idx = int32_t, so
// per-element footprint = 4 + 4 = 8 B; we pass 8 * R * C with some headroom).
constexpr uint32_t kDynUbBytes = (uint32_t)(R * C * (sizeof(T) + sizeof(int32_t)));
runMGATHER_kernel<<<numBlocks, kDynUbBytes, stream>>>(out, table, indices);
```

The current MGATHER ST suite does not include extended-UB cases — every existing case fits in the default budget — but workloads that exceed 128 KB should follow this pattern.

## Runtime Dispatch Requirement

`MGATHER` uses `cce::async_invoke<simt_mgather_*_kernel>(cce::dim3{32, kLaunchWarps}, …)` internally to fan a per-warp/per-lane workload out across up to 1024 threads. `async_invoke` consumes runtime state (TID registers, warp / lane configuration, vector-pipe scheduling) that the **launch path** must install before the kernel function is entered. The standard CANN launch (`rtKernelLaunchWithHandleV2`, used by the `<<<numBlocks, dynUBufSize, stream>>>` syntax) installs this state correctly.

A runtime variant that dispatches kernels as a direct C function-pointer call is fine for SPMD ops (TLOAD, TSTORE, TADD, …) but skips the SIMT context init, so the first `async_invoke` inside `MGATHER` has no warp scheduler to dispatch into and hangs. Use the standard launch syntax for any SIMT kernel.

## Related Instructions

- [`TLOAD`](/docs/isa/TLOAD.md): contiguous block transfer GM → Tile.
- [`MSCATTER`](../mscatter/MSCATTER.md): indexed scatter Tile → GM (inverse operation).
- [`TGATHER`](/docs/isa/TGATHER.md): index-based gather within tiles (UB-to-UB).

## Test Cases

### Row Coalesce — `[1, R]` index form

| Case | Data Type | Dst Size | TableRows | OOB | Idx Pattern |
|------|-----------|----------|-----------|-----|-------------|
| case_row_float_8x32_64rows      | float | 8×32  | 64 | Undefined | random |
| case_row_half_16x64_64rows      | half  | 16×64 | 64 | Undefined | random |
| case_row_int32_8x16_32rows      | int32 | 8×16  | 32 | Undefined | random |
| case_row_uint8_8x32_32rows      | uint8 | 8×32  | 32 | Undefined | random |
| case_row_int16_8x16_32rows      | int16 | 8×16  | 32 | Undefined | random |
| case_row_float_clamp_8x32_8rows | float | 8×32  | 8  | Clamp     | oob    |
| case_row_int32_wrap_8x16_8rows  | int32 | 8×16  | 8  | Wrap      | oob    |
| case_row_half_zero_8x32_8rows   | half  | 8×32  | 8  | Zero      | oob    |

### Row Coalesce — `[R, 1]` index form (ColMajor + DN)

| Case | Data Type | Dst Size | TableRows | OOB |
|------|-----------|----------|-----------|-----|
| case_row_colidx_float_8x32_64rows      | float | 8×32  | 64 | Undefined |
| case_row_colidx_int32_clamp_8x16_8rows | int32 | 8×16  | 8  | Clamp     |
| case_row_colidx_half_16x64_64rows      | half  | 16×64 | 64 | Undefined |

### Element Coalesce — 1-D destination `[1, N]`

| Case | Data Type | N / TableSize | OOB | Idx Pattern |
|------|-----------|---------------|-----|-------------|
| case_elem_float_64_128size      | float | 64 / 128 | Undefined | random |
| case_elem_half_64_128size       | half  | 64 / 128 | Undefined | random |
| case_elem_int32_32_64size       | int32 | 32 / 64  | Undefined | random |
| case_elem_uint8_64_128size      | uint8 | 64 / 128 | Undefined | random |
| case_elem_int16_32_64size       | int16 | 32 / 64  | Undefined | random |
| case_elem_float_clamp_32_16size | float | 32 / 16  | Clamp     | oob    |
| case_elem_int32_wrap_32_16size  | int32 | 32 / 16  | Wrap      | oob    |
| case_elem_half_zero_32_16size   | half  | 32 / 16  | Zero      | oob    |

### Element Coalesce — 2-D destination `[R, C]`

| Case | Data Type | Dst Size | TableSize | OOB | Idx Pattern |
|------|-----------|----------|-----------|-----|-------------|
| case_elem2d_float_8x32_256size | float | 8×32 | 256 | Undefined | random |
| case_elem2d_int32_8x16_256size | int32 | 8×16 | 256 | Undefined | random |
| case_elem2d_half_4x32_256size  | half  | 4×32 | 256 | Undefined | random |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles any `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. Express odd valid extents as a padded tile and let the kernel walk the valid sub-region only.

| Case | Mode | Data Type | Valid Dst | Padded Dst | Idx (Valid → Padded) | Table | OOB |
|------|------|-----------|-----------|------------|----------------------|-------|-----|
| case_elem2d_int32_unaligned_3x8_64size              | Elem          | int32 | 3×8  | 3×8  | 3×8  → 3×8   | 64 elems    | Undefined |
| case_elem2d_uint8_unaligned_3x32_256size            | Elem          | uint8 | 3×32 | 3×32 | 3×32 → 3×32  | 256 elems   | Undefined |
| case_elem2d_int32_unaligned_3x3_in_3x8_64size       | Elem          | int32 | 3×3  | 3×8  | 3×3  → 3×8   | 64 elems    | Undefined |
| case_elem2d_int32_unaligned_9x9_in_9x16_256size     | Elem          | int32 | 9×9  | 9×16 | 9×9  → 9×16  | 256 elems   | Undefined |
| case_elem2d_int32_scalar_1x1_in_1x8_8size           | Elem (scalar) | int32 | 1×1  | 1×8  | 1×1  → 1×8   | 8 elems     | Undefined |
| case_row_int32_unaligned_3x8_8rows                  | Row           | int32 | 3×8  | 3×8  | 1×3  → 1×8   | 8 rows × 8  | Undefined |
| case_row_int32_unaligned_9x16_16rows                | Row           | int32 | 9×16 | 9×16 | 1×9  → 1×16  | 16 rows × 16| Undefined |

### Dynamic Runtime Shapes

`Tile<…, -1, -1>` paired with `GlobalTensor<…, Shape<…, -1, -1>, Stride<…, -1, -1>>`. The SIMT kernel sizes itself from `Tile::GetValidRow/Col()` and `GlobalTensor::GetShape()` at dispatch time; padded `Tile::Rows / Cols` remain compile-time so the UB layout stays statically known.

| Case | Mode | Data Type | Runtime Valid Dst | Padded Dst | Runtime Table | OOB |
|------|------|-----------|-------------------|------------|----------------|-----|
| case_elem2d_dyn_user_float_1x9_in_1x16_3x10 | Elem | float | 1×9  | 1×16 | 3×10           | Undefined |
| case_elem2d_dyn_int32_4x8_in_4x8_64size     | Elem | int32 | 4×8  | 4×8  | 1×64           | Undefined |
| case_elem2d_dyn_float_3x3_in_3x8_64size     | Elem | float | 3×3  | 3×8  | 1×64           | Undefined |
| case_elem2d_dyn_half_8x16_in_8x16_4x32      | Elem | half  | 8×16 | 8×16 | 4×32           | Undefined |
| case_row_dyn_int32_3x16_8rows               | Row  | int32 | 3×16 | 3×16 | 8 rows × 16    | Undefined |
| case_row_dyn_half_4x32_16rows               | Row  | half  | 4×32 | 4×32 | 16 rows × 32   | Undefined |
