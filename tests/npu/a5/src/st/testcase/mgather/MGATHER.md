# MGATHER

## Tile Operation Diagram

![MGATHER tile operation](../../../../../../../docs/figures/isa/MGATHER.svg)

## Introduction

Indexed gather from a GM `GlobalTensor` into a UB destination tile through a UB index tile. The A5 implementation is a SIMT kernel dispatched via `cce::async_invoke` with a shape-adaptive `dim3{32, kLaunchWarps}` (up to 32 lanes × 32 warps = 1024 threads). The operating mode is selected explicitly by the `Coalesce` template parameter:

- **`Coalesce::Row`** (default) — gather full rows from `table[idx[r], :]` into `dst[r, :]`. Index tile is 1-D (`[R, 1]` or `[1, R]`). `R = 1` (single-row gather) is allowed.
- **`Coalesce::Elem`** — element-wise gather from a linearized `table` using `idx[R, C]` into `dst[R, C]`. The destination tile may be 2-D (`[R, C]`) or degenerate 1-D (`[1, N]` / `[N, 1]`); the index tile must have the same shape as the destination.

Out-of-bounds index handling is selected via the `GatherOOB` template parameter. Gather has no atomic or conflict policy: every destination slot has exactly one defined source index, so collisions cannot occur.

## Math Interpretation

### Row Coalesce (`Coalesce::Row`)

Destination `dst[R, C]`, index `idx[R, 1]` or `idx[1, R]`, table `table[TableRows, C]`:

$$ \mathrm{dst}_{r, j} = \mathrm{table}_{\mathrm{idx}_{r},\; j} \quad\text{for } 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce (`Coalesce::Elem`)

Destination `dst[R, C]`, index `idx[R, C]` (same shape as dst), flat table of length `TableSize`:

$$ \mathrm{dst}_{r, c} = \mathrm{table}[\mathrm{idx}_{r, c}] \quad\text{for } 0 \le r < R,\; 0 \le c < C $$

The kernel iterates over the `R * C` logical elements as a flat sequence while physical UB offsets follow each tile's own `BLayout`; the table is treated as a linear region of `Shape[0]*Shape[1]*Shape[2]*Shape[3]*Shape[4]` elements.

### Out-of-Bounds Behaviour

```cpp
enum class GatherOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Clamp     = 1,  // Clamp index to capacity - 1
    Wrap      = 2,  // Index modulo capacity
    Zero      = 3   // Return zero for OOB; in-bounds indices loaded normally
};
```

`capacity` is `TableRows` in `Coalesce::Row` and `Shape[0]*Shape[1]*Shape[2]*Shape[3]*Shape[4]` in `Coalesce::Elem`.

## Assembly Syntax

PTO-AS form: see [Assembly Spelling And Operands](../../../../../../../docs/isa/syntax-and-operands/assembly-model.md).

Row coalesce:

```text
mgather.row %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<Rx1xi32>)
```

Element coalesce:

```text
mgather.elem %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<RxCxi32>)
```

OOB-aware variants append the mode suffix (`mgather.row.clamp`, `mgather.elem.zero`, etc.).

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/npu/a5/MGather.hpp`:

```cpp
template <Coalesce  CMode    = Coalesce::Row,
          GatherOOB Mode     = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& table, TileInd& idx,
                             WaitEvents&... events);
```

The kernel iterates over `TileDst::ValidRow * TileDst::ValidCol` logical positions; physical UB strides come from the same `Tile` types via `tile_offset_2d` (i.e. `TileDst::Cols`, `TileIdx::Cols`). The `Tile` type is the **single source of truth** for both — no separate `ValidRows` / `ValidCols` API knobs.

For `Coalesce::Elem` with `TileDst::ValidRow == 1 && TileDst::ValidCol == 1` the implementation **bypasses the SIMT launch entirely** and runs a real **scalar fallback** (`MGatherScalarImpl`) on the AIV vector core — `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` handshake, single GM read, single UB write, then `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` to release the vector pipe. Mirrors the `TInsertVecToVecNDScalarImpl` pattern in `TInsert.hpp`.

**Parameters:**
- `dst`     : UB destination tile (`TileType::Vec`); shape `[R, C]`. Both row-major and column-major storage are accepted.
- `table`   : Source GM `GlobalTensor`. The `GlobalTensor::DType` must be `__gm__ T` matching the destination element type.
- `idx`     : UB index tile (`TileType::Vec`). For `Coalesce::Row`: 1-D `[R, 1]` or `[1, R]`. For `Coalesce::Elem`: same shape as `dst` (BLayout can be **independent** of `dst` layout).
- `CMode`   : `Coalesce` — `Row` (default) or `Elem` — **first** template parameter so the operating mode is always explicit at the call site.
- `Mode`    : `GatherOOB` — out-of-bounds index handling.

## Coalesce Mode

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // dst[r, :] = table[idx[r], :]   (1-D idx of length R)
    Elem = 1   // dst[i, j] = table[idx[i, j]]   (idx shape == dst shape)
};
```

## Constraints

### Data Types

`TileDst::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.

### Index Types

`TileIdx::DType` must be `int32_t` or `uint32_t`.

### Tile Constraints

- `TileDst::Loc == TileType::Vec` (UB).
- `TileIdx::Loc == TileType::Vec` (UB).
- Destination and table must share the same element type `T` (`GlobalTable::DType == __gm__ T`).
- For `Coalesce::Row`: `TileDst::ValidRow >= 1` and `TileDst::ValidCol >= 1`; the index tile's **valid shape** is `[1, R]` (`TileIdx::ValidRow == 1 && TileIdx::ValidCol == TileDst::ValidRow`) or `[R, 1]` (`TileIdx::ValidRow == TileDst::ValidRow && TileIdx::ValidCol == 1`). Destination tile may be row-major or column-major. Table inner dim (`Shape[4]`) must equal `TileDst::ValidCol`.
- For `Coalesce::Elem`: `TileIdx::ValidRow / ValidCol == TileDst::ValidRow / ValidCol`. **No layout-pairing constraint** — `TileDst` and `TileIdx` may independently be `RowMajor` or `ColMajor`; the kernel walks both via per-tile `tile_offset_2d`.
- The `[R, 1]` index variant uses `BLayout::ColMajor` paired with a `Layout::DN` `GlobalTensor` for the upstream `TLOAD`; the `[1, R]` variant uses `BLayout::RowMajor` with the default `Layout::ND` `GlobalTensor`. Either way, the **upstream `Tile` itself must satisfy the 32-byte-burst alignment of its physical (padded) dim** — not the logical valid dim — so odd `R` index tiles are expressed as a padded shape with a smaller `ValidCol`/`ValidRow` (see [Minimum Tile Shape](#minimum-tile-shape) below).

### Dynamic Runtime Shapes

`MGATHER` supports both compile-time fixed shapes and **runtime-dynamic** shapes for the source `GlobalTensor` and the destination / index `Tile`s. Any dimension declared as `DYNAMIC` (`-1`) at template-instantiation time is resolved at runtime through the standard PTO accessors:

- `Tile<…, RowMask, ColMask>` with `RowMask == -1` and/or `ColMask == -1` stores the runtime valid extents in the tile object; `MGATHER_IMPL` reads them via `dst.GetValidRow()` / `dst.GetValidCol()` and forwards them to the SIMT kernel as `validRows` / `validCols` arguments.
- `Shape<S0, S1, S2, S3, S4>` / `Stride<…>` with one or more `-1` entries are constructed with the runtime sizes; `MGATHER_IMPL` reads them via `table.GetShape(GlobalTensorDim::DIM_X)` and folds them into `tableRows` (Row mode) or `tableSize = ∏ shape[0..4]` (Elem mode).

Static-asserts in `MGatherCheck` are gated on `if constexpr (DIM > 0)` so they fire only for compile-time-known dimensions; mixed static/dynamic combinations check exactly the static dims and defer the dynamic ones to runtime arithmetic. Padded `Tile::Rows` / `Tile::Cols` are always compile-time (they govern the UB DMA-burst alignment); only the **valid** sub-region and the GM table extents may be dynamic.

Example (mirrors `case_elem2d_dyn_user_float_1x9_in_1x16_3x10`):

```cpp
constexpr auto kPadCols = 16;
using DstTileT = Tile<TileType::Vec, float,    1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT = Tile<TileType::Vec, int32_t,  1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t idxShape4 = 9, d3 = 3, d4 = 10, srcStride3 = 10;
TableShape  tableShape(d3, d4);
TableStride tableStride(srcStride3, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(srcGm, tableShape, tableStride);

DstTileT dstTile(1, idxShape4);
IdxTileT idxTile(1, idxShape4);
TASSIGN(dstTile, dstUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MGATHER<Coalesce::Elem, GatherOOB::Undefined>(dstTile, tableGM, idxTile);
```

At dispatch time `MGATHER_IMPL` resolves `validRows = 1`, `validCols = 9`, and `tableSize = 1·1·1·3·10 = 30`; the SIMT launch sizes itself by `ceil(validRows*validCols / WARP_SIZE) = 1` warp. The padded UB `Tile::Cols = 16` is purely a `TLOAD` / `TSTORE` burst-alignment artifact — the SIMT body only walks the valid 9 elements.

### Layout Support

The SIMT kernel itself is **layout-agnostic** for every UB tile it touches: every UB read/write goes through `tile_offset_2d<TileX>(r, c)`, which dispatches the index arithmetic from the tile type's `BLayout`. The caller is responsible for getting the data into UB with whatever `TLOAD` / `TMOV` / `TINSERT` configuration matches their upstream layout (ND, DN, NZ, RowMajor, ColMajor, etc.).

| Tile / Tensor | Supported layouts | Notes |
|---------------|-------------------|-------|
| `TileDst` (UB) | Any `BLayout` (`RowMajor` or `ColMajor`); `SLayout::NoneBox` | Kernel writes via `tile_offset_2d<TileDst>`. |
| `TileIdx` (UB) – Row mode | `[1, R]` `BLayout::RowMajor` **or** `[R, 1]` `BLayout::ColMajor` | Both produce a linear `R`-element layout in UB; the kernel reads `indices[row]` directly. |
| `TileIdx` (UB) – Elem mode | Any `BLayout` — independent of `TileDst::BLayout` | Kernel reads via `tile_offset_2d<TileIdx>`. Same logical `(r, c)` is consumed for both `dst` and `indices`. |
| `GlobalTable` (GM) | `Layout::ND` (linear contiguous addressing); a non-ND-stride `GlobalTensor` is the caller's responsibility | The kernel addresses `table[idx * RowWidth + col]` (Row mode) or `table[idx]` (Elem mode). For NZ-stored GM the caller should pre-stage the data into a contiguous ND view (e.g. via `TMOV` NZ→ND) before calling `MGATHER` — no static assert is produced for the layout. |

### Aligned vs Unaligned Tile Shapes

The kernel does **not** care whether the tile's logical shape is "aligned" — it walks all `Rows * Cols` positions of the tile via `tile_offset_2d`. The 32-byte alignment of `Cols * sizeof(T)` (for `BLayout::RowMajor`) is enforced by the upstream `Tile` declaration only. Callers handle "unaligned valid region" by:

1. Padding the tile up to the nearest 32-byte alignment (e.g. valid `[3, 3]` int32 → tile `[3, 8]`), and
2. Either zero-initializing the padding (`TASSIGN`-then-clear) or only inspecting the valid region post-gather.

The kernel's job is to populate / consume every position of the declared tile shape correctly; partial-tile semantics live in the caller's pre/post-move ops, exactly as in `TLOAD` / `TSTORE` / `TINSERT` / `TMOV` ST suites.

### Minimum Tile Shape

`MGatherCheck` accepts any `(ValidRow, ValidCol)` with `ValidRow, ValidCol >= 1` (including the degenerate `(1, 1)` for both Row and Elem modes — the SIMT kernel and the scalar fallback handle them transparently).

The actual lower bound on the **padded** `Tile<…, Rows, Cols, BLayout, ValidRow, ValidCol>` shape is **not** an MGATHER concern — it is enforced upstream by the `Tile` system because every `TLOAD` / `TSTORE` that brings data in/out of UB issues 32-byte GM↔UB **DMA bursts**. The contiguous-in-memory dim of the tile must therefore be a whole number of bursts:

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

Mode is **explicit**, not auto-detected. The static-asserts in `MGatherCheck` validate that the supplied tile shapes match the chosen `Coalesce` value:

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
    using DstTile = Tile<TileType::Vec, T,        R,         C, BLayout::RowMajor, R, C>;
    using IdxTile = Tile<TileType::Vec, int32_t,  1,         R, BLayout::RowMajor, 1, R>;

    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    using IdxShape  = Shape<1, 1, 1, 1, R>;
    using IdxStride = Stride<1, 1, 1, R, 1>;
    using IdxTensor = GlobalTensor<int32_t, IdxShape, IdxStride>;

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
#include <pto/npu/a5/MGather.hpp>

using namespace pto;

AICORE void example_elem_2d(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 32, TableSize = 256;

    using DstTile = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;

    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MGATHER<Coalesce::Elem, GatherOOB::Wrap>(dst, tableGM, idx);
}
```

### Element Coalesce — 1-D Source

```cpp
#include <pto/npu/a5/MGather.hpp>

using namespace pto;

AICORE void example_elem_1d(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int N = 64, TableSize = 128;

    using DstTile = Tile<TileType::Vec, float,   1, N, BLayout::RowMajor, 1, N>;
    using IdxTile = Tile<TileType::Vec, int32_t, 1, N, BLayout::RowMajor, 1, N>;

    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0200);

    MGATHER<Coalesce::Elem, GatherOOB::Undefined>(dst, tableGM, idx);
}
```

### Row Coalesce — `[R, 1]` ColMajor Index

```cpp
#include <pto/npu/a5/MGather.hpp>

using namespace pto;

AICORE void example_row_colidx(__gm__ half* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 64, TableRows = 64;

    using DstTile = Tile<TileType::Vec, half,    R, C, BLayout::RowMajor, R, C>;
    using IdxTile = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;

    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;

    using IdxShape  = Shape<1, 1, 1, R, 1>;
    using IdxStride = Stride<1, 1, 1, 1, 1>;
    using IdxTensor = GlobalTensor<int32_t, IdxShape, IdxStride, Layout::DN>;

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

1. **Shape-adaptive launch.** `MGatherRowImpl` / `MGatherElemImpl` size the SIMT grid as `dim3{WARP_SIZE, kLaunchWarps}` from the resolved `validRows` / `validCols` (compile-time constants for static tiles, runtime values for dynamic tiles). Small tiles do not pay the cost of launching 1024 idle threads.
   - **Row.** `kRowWarps = min(validRows, 32)` own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks. `kLaunchWarps = kRowWarps * kWarpsPerRow`. Lane reads form 128-byte coalesced GM bursts; lane writes form 128-byte coalesced UB stores. `kColStride = kWarpsPerRow * 32`.
   - **Elem.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`. Threads with `tid >= totalElems` skip the loop body (no garbage access). For `totalElems > 1024` the strided loop walks `launchThreads` at a time.

2. **Linear thread mapping (Elem).** Each thread handles one element via stride-`kLaunchThreads` iteration over the flat `[0, R*C)` range. Lanes 0..31 of each warp read/write 32 consecutive UB / GM addresses per iteration — UB writes are bank-conflict-free, GM reads are gather-stride-driven by the `idx` values.

3. **No thread divergence for mode / OOB control.** All mode / OOB decisions are `if constexpr`. The `gather_remap` lookup compiles to a small data-dependent transform with no control-flow split. The `Zero` mode's "load 0 for OOB" is implemented as a per-lane select on the loaded value, again without divergence.

4. **OOB policy cost.**
   - `Undefined`: zero overhead — caller guarantees valid indices.
   - `Clamp` / `Wrap`: a single arithmetic remap per lane (`min`/`mod`).
   - `Zero`: one extra compare-and-select per lane to substitute `static_cast<T>(0)` for OOB lanes.

5. **Unrolled inner loops.** Inner column loop in Row coalesce carries `#pragma unroll(4)` so the compiler unrolls for small compile-time trip counts (e.g. `RowWidth=32, kColStride=32` ⇒ 1 iter, fully unrolled). The outer per-row loop and the elem flat loop are `#pragma unroll(1)` to keep code size bounded for large shapes.

6. **Row vs. Elem.** Row coalesce achieves the best aggregate bandwidth (32 consecutive lanes per coalesced GM burst). Elem coalesce performs one scalar GM load per active lane — non-coalesced at GM in general, but UB writes remain coalesced.

7. **Register pressure / MRF.** The kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread budget) and use ≤ 12 live registers per thread in the hot path. No spills are produced; the compile flag `-mllvm -cce-aicore-record-overflow=true` reports no overflow events for any of the instantiations.

## SIMT Usage Restrictions

`MGATHER` is a SIMT launch on the AIV vector core. Every byte the runtime, the compiler, and the user store in UB must coexist inside the single 256 KB Unified Buffer that the AIV exposes. The following table itemises every region the toolchain carves out before user tiles are allocated, with the source of each value so the budget is reproducible:

| Region                         | Size on a5 (V310)               | Source                                                                                                       |
|--------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------|
| Physical UB                    | 256 KB                           | Hardware                                                                                                     |
| Hardware D-cache scratch       | 8 KB (top of UB)                 | `TOTAL_UB_SIZE = 248 * 1024` for `__NPU_ARCH__ == 3510` in `kernel_utils_constants.h` (256 − 248)             |
| AscendC / TBE reserved         | 2 KB                             | `--user-reserved-ub-size` default (`2048` bytes from `ccec -mllvm --print-all-options`)                      |
| Scalar main stack              | 32 KB                            | `--cce-aicore-stack-size=0x8000` set in `tests/npu/a5/src/st/CMakeLists.txt`                                  |
| Per-call-depth scalar stack    | 32 KB × depth                    | `--cce-aicore-function-stack-size=0x8000` set in `tests/npu/a5/src/st/CMakeLists.txt`                        |
| Vector-fragment (VF) stack     | 8 KB                             | `--cce-vf-stack-size` default (`8192` bytes from `ccec -mllvm --print-all-options`)                          |
| Per-thread SIMT stack          | 4 KB                             | `--cce-simt-stack-size` default (`4096` bytes from `ccec -mllvm --print-all-options`)                        |

The `MGATHER` call chain is `runMGATHER_* → MGATHER<...> → MGather{Row,Elem}Impl → cce::async_invoke<simt_mgather_*_kernel>`. The `Impl` layer is marked `__tf__ + PTO_INLINE`, but on-board testing of the matching `MSCATTER` SIMT path has shown the compiler retains it as a separate scalar frame in this configuration, so the effective scalar call-depth is **2**. The SIMT-VF kernel launched by `async_invoke` runs in its own VF + per-thread context whose stacks are counted separately and do not extend the scalar chain.

Plugging depth = 2 into the table:

```
256 KB  physical UB
−  8 KB  D-cache scratch
−  2 KB  AscendC / TBE reservation
− 32 KB  scalar main stack
− 64 KB  function stack  (2 frames × 32 KB)
−  8 KB  VF stack
−  4 KB  per-thread SIMT stack
= 138 KB  user-addressable UB
```

A depth-1 path would lift the budget to 170 KB, but on a5 with the current flag set the inliner does not deliver it (empirically confirmed on the matching `MSCATTER` cases: tile footprints of 170 KB fall back to silent zeroed output on-board even though they pass the CPU simulator). All designs should therefore size against the **138 KB depth-2 ceiling**.

When sizing a workload, account for both the **destination** tile (`R * C * sizeof(T)`, padded up to the 32-byte burst alignment) and the **index** tile (`R * C * sizeof(TIdx)`, same padding rule). `MGATHER` itself does not allocate any UB scratch — every read flows GM → register → UB.

Overflowing the ceiling is **silent**. The compiler does not error, the simulator does not flag it, and small overruns may even appear to work on hardware. Once the overflow reaches the stack region, however, the first spilled value from any SIMT thread corrupts a tile byte and the kernel returns all-zero (or otherwise undefined) output on-board while still passing the CPU simulator.

### Large-Workload Tiling

A single `TSTORE` after the gather is bounded by the same UB window. When the destination + index tile pair approaches the 138 KB on-board ceiling, split the work across multiple iterations: invoke `MGATHER` for a slice of indices, `TSTORE` that slice to its GM region, then advance to the next slice. A combined footprint of **≤ 128 KB** per iteration (for example `2048 × 8` with `float` and `int32_t`) keeps a 10 KB safety margin against the ceiling. `MGATHER` has no cross-element ordering semantics, so slice order is unconstrained.

## Runtime Dispatch Requirement

`MGATHER` (like every SIMT kernel in PTO and CANN) uses `cce::async_invoke<simt_mgather_*_kernel>(cce::dim3{WARP_SIZE, kLaunchWarps}, …)` internally to fan a per-warp/per-lane workload out across up to `32 × 32 = 1024` threads. `cce::async_invoke` consumes hardware/runtime state — TID registers (`__cce_simt_get_TID_X/Y`), warp/lane configuration, vector-pipe scheduling — that the **launch path** has to install **before** the kernel function is entered. The standard CANN launch (`rtKernelLaunch`, used by the `<<<1, nullptr, stream>>>` syntax in every ST in this suite) installs that state correctly.

Dispatch kernels as a **direct C function-pointer call**:

```cpp
UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
```

This is fine for SPMD ops (TLOAD, TSTORE, TADD …) but skips the SIMT-context init step, so the first `cce::async_invoke` inside `MGATHER` has no warp scheduler to dispatch into and hangs.

A compile-time-only "use the full launch pipeline" override has to live in the **dispatcher**, not the kernel — by the time the kernel function is entered, the dispatch decision has already been made. Specifically:

- `__simt_vf__`, `LAUNCH_BOUND(1024)`, `__simt_callee__`, `__cce_simt_get_TID_X/Y`, `cce::async_invoke`, `cce::dim3` are **all toolchain (bisheng/CCE) intrinsics**; PTO does not define them. Whatever metadata the compiler attaches to the resulting `.o` is what the dispatcher needs to inspect.
- There is no in-tree `cce::is_simt_active()` guard or `cce::simt_init()` device-side bootstrap intrinsic — a regular AIV function cannot self-promote itself into SIMT context.
- The only kernel-side "always works" alternative is to **stop using `cce::async_invoke` entirely** and re-implement `MGATHER` as a single-thread sequential AIV loop, which would discard the entire 1024-thread parallel design to work around a runtime bug. That is not a fix; it is a regression.

`MGATHER` already carries every signal a smart dispatcher could use to pick the right path: the `cce::async_invoke<…>(dim3{…}, …)` call site itself, the `__simt_vf__ LAUNCH_BOUND(1024)` attributes on `simt_mgather_*_kernel`, and the templated SIMT entry that the toolchain tags in the `.o`.

In dependency order (cheapest first): Note : We will resolve this issue as soon as possible.

1. **Mark SIMT kernels in `PTO2DispatchPayload` with a flag, branch the dispatch path** in `aicore_executor.cpp` — direct fn-pointer for SPMD payloads, `rtKernelLaunch` for SIMT payloads. Smallest runtime change; isolates the hot path for SPMD ops.
2. **Detect SIMT kernels at compile time** (toolchain-emitted attribute / ELF note from `__simt_vf__` / `LAUNCH_BOUND(1024)`) and unconditionally route them through `rtKernelLaunch`. Works automatically for every SIMT kernel (existing and future), but requires the dispatcher to read the toolchain's metadata.
3. **Initialize the SIMT context** (TID/warp/pipe regs) inside `aicore_executor.cpp` immediately before the function-pointer call. Most invasive — duplicates work the standard CANN launch path already does — but works without per-payload flagging.

## Related Instructions

- [`TLOAD`](/docs/isa/TLOAD.md): Contiguous block transfer from GM to Tile.
- [`MSCATTER`](../mscatter/MSCATTER.md): Indexed scatter from Tile to GM (inverse operation).
- [`TGATHER`](/docs/isa/TGATHER.md): Index-based gather within tiles (UB-to-UB).

## Test Cases

### Row Coalesce — `[1, R]` index form

| Case | Data Type | Dst Size | TableRows | OOB Mode | Idx Pattern |
|------|-----------|----------|-----------|----------|-------------|
| case_row_float_8x32_64rows         | float | 8×32  | 64 | Undefined | random |
| case_row_half_16x64_64rows         | half  | 16×64 | 64 | Undefined | random |
| case_row_int32_8x16_32rows         | int32 | 8×16  | 32 | Undefined | random |
| case_row_uint8_8x32_32rows         | uint8 | 8×32  | 32 | Undefined | random |
| case_row_int16_8x16_32rows         | int16 | 8×16  | 32 | Undefined | random |
| case_row_float_clamp_8x32_8rows    | float | 8×32  | 8  | Clamp     | oob    |
| case_row_int32_wrap_8x16_8rows     | int32 | 8×16  | 8  | Wrap      | oob    |
| case_row_half_zero_8x32_8rows      | half  | 8×32  | 8  | Zero      | oob    |

### Row Coalesce — `[R, 1]` index form (ColMajor + DN)

| Case | Data Type | Dst Size | TableRows | OOB Mode |
|------|-----------|----------|-----------|----------|
| case_row_colidx_float_8x32_64rows         | float | 8×32  | 64 | Undefined |
| case_row_colidx_int32_clamp_8x16_8rows    | int32 | 8×16  | 8  | Clamp     |
| case_row_colidx_half_16x64_64rows         | half  | 16×64 | 64 | Undefined |

### Element Coalesce — 1-D destination `[1, N]`

| Case | Data Type | N / TableSize | OOB Mode | Idx Pattern |
|------|-----------|---------------|----------|-------------|
| case_elem_float_64_128size       | float | 64 / 128 | Undefined | random |
| case_elem_half_64_128size        | half  | 64 / 128 | Undefined | random |
| case_elem_int32_32_64size        | int32 | 32 / 64  | Undefined | random |
| case_elem_uint8_64_128size       | uint8 | 64 / 128 | Undefined | random |
| case_elem_int16_32_64size        | int16 | 32 / 64  | Undefined | random |
| case_elem_float_clamp_32_16size  | float | 32 / 16  | Clamp     | oob    |
| case_elem_int32_wrap_32_16size   | int32 | 32 / 16  | Wrap      | oob    |
| case_elem_half_zero_32_16size    | half  | 32 / 16  | Zero      | oob    |

### Element Coalesce — 2-D destination `[R, C]`

| Case | Data Type | Dst Size | TableSize | OOB Mode | Idx Pattern |
|------|-----------|----------|-----------|----------|-------------|
| case_elem2d_float_8x32_256size | float | 8×32 | 256 | Undefined | random |
| case_elem2d_int32_8x16_256size | int32 | 8×16 | 256 | Undefined | random |
| case_elem2d_half_4x32_256size  | half  | 4×32 | 256 | Undefined | random |

### Unaligned / Odd-Dimension Tiles

The SIMT kernel handles **any** `(ValidRow, ValidCol)` with `1 <= Valid <= Padded`. There is no SIMD-style batch / predicate path — the kernel walks `ValidRow * ValidCol` positions one element per thread, and `tile_offset_2d<TileX>` resolves the physical UB offset inside the padded tile. So "unaligned" reduces to one mechanism: declare the `Tile` with the **padded** shape that satisfies the `Tile` system's burst-alignment rule (see [Minimum Tile Shape](#minimum-tile-shape)) and the **valid** shape carrying the logical `(R, C)`.

Three categories are covered by ST:

| Category | Pattern | What it exercises |
|----------|---------|-------------------|
| Odd row count, valid == padded | `Tile<T, R, C, RowMajor, R, C>` with odd `R` (3, 9, …) and `C` already 32-B aligned | Shape-adaptive launch — `kRowWarps = R`, `kWarpsPerRow = 32 / R`, etc. |
| True valid-in-padded | `Tile<T, PadR, PadC, RowMajor, ValidR, ValidC>` with `ValidR/ValidC < PadR/PadC` | Per-thread iteration over `ValidR × ValidC` only, padding bytes are never touched by the SIMT body. |
| Scalar `(1, 1)` | `Tile<T, 1, MinAlignedC, RowMajor, 1, 1>` | Compile-time fast-path to `MGatherScalarImpl` — no `cce::async_invoke`. |

For `Coalesce::Row` with odd `R`, the index tile must also be expressed as valid-in-padded: e.g. `Tile<int32, 1, 8, RowMajor, 1, 3>` for 3 row-indices. `MGATHER` itself only checks the **valid** equality `TileIdx::ValidCol == TileDst::ValidRow` (or the `[R, 1]` analog).

| Case | Mode | Data Type | Valid Dst | Padded Dst | Idx (Valid → Padded) | Table | OOB Mode |
|------|------|-----------|-----------|------------|----------------------|-------|----------|
| case_elem2d_int32_unaligned_3x8_64size            | Elem | int32 | 3×8 | 3×8 | 3×8 → 3×8 | 64 elems | Undefined |
| case_elem2d_uint8_unaligned_3x32_256size          | Elem | uint8 | 3×32 | 3×32 | 3×32 → 3×32 | 256 elems | Undefined |
| case_elem2d_int32_unaligned_3x3_in_3x8_64size     | Elem | int32 | 3×3 | 3×8 | 3×3 → 3×8 | 64 elems | Undefined |
| case_elem2d_int32_unaligned_9x9_in_9x16_256size   | Elem | int32 | 9×9 | 9×16 | 9×9 → 9×16 | 256 elems | Undefined |
| case_elem2d_int32_scalar_1x1_in_1x8_8size         | Elem (scalar) | int32 | 1×1 | 1×8 | 1×1 → 1×8 | 8 elems | Undefined |
| case_row_int32_unaligned_3x8_8rows                | Row | int32 | 3×8 | 3×8 | 1×3 → 1×8 | 8 rows × 8 | Undefined |
| case_row_int32_unaligned_9x16_16rows              | Row | int32 | 9×16 | 9×16 | 1×9 → 1×16 | 16 rows × 16 | Undefined |

### Dynamic Runtime Shapes

`Tile<…, -1, -1>` (runtime valid extents) paired with `GlobalTensor<…, Shape<1,1,1,-1,-1>, Stride<1,1,1,-1,-1>>` (runtime table shape/stride). The SIMT kernel sizes itself from `Tile::GetValidRow/Col()` and `GlobalTensor::GetShape(DIM_X)` at dispatch time; padded `Tile::Rows / Cols` remain compile-time so the UB layout / DMA bursts stay statically known.

| Case | Mode | Data Type | Runtime Valid Dst | Padded Dst | Runtime Table | OOB Mode |
|------|------|-----------|-------------------|------------|----------------|----------|
| case_elem2d_dyn_user_float_1x9_in_1x16_3x10 | Elem | float | 1×9 | 1×16 | 3×10 | Undefined |
| case_elem2d_dyn_int32_4x8_in_4x8_64size     | Elem | int32 | 4×8 | 4×8  | 1×64 | Undefined |
| case_elem2d_dyn_float_3x3_in_3x8_64size     | Elem | float | 3×3 | 3×8  | 1×64 | Undefined |
| case_elem2d_dyn_half_8x16_in_8x16_4x32      | Elem | half  | 8×16| 8×16 | 4×32 | Undefined |
| case_row_dyn_int32_3x16_8rows               | Row  | int32 | 3×16| 3×16 | 8 rows × 16 | Undefined |
| case_row_dyn_half_4x32_16rows               | Row  | half  | 4×32| 4×32 | 16 rows × 32 | Undefined |
