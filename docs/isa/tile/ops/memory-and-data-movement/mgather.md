# pto.mgather

## Tile Operation Diagram

![MGATHER tile operation](../../../../figures/isa/MGATHER.svg)

## Introduction

`MGATHER` reads data from a GM `GlobalTensor` into a UB destination tile through a UB index tile. The operating mode is selected explicitly through the `Coalesce` template parameter:

- **`Coalesce::Row`** (default) — gather full rows from `table[idx[r], :]` into `dst[r, :]`. The index tile is 1-D (`[1, R]` row-major; on A5 also `[R, 1]` column-major). `R = 1` is allowed.
- **`Coalesce::Elem`** — element-wise gather from a linearized `table` into `dst[R, C]` through `idx[R, C]`. The index tile must have the same valid shape as the destination. The degenerate `(1, 1)` case is allowed.

Out-of-bounds handling is selected through the `GatherOOB` template parameter. `MGATHER` has no atomic or conflict policy: every destination slot has exactly one defined source index, so collisions cannot occur.

Per-target dispatch summary:

- **CPU Simulator** — pure C++ reference. Templates on the same `Coalesce` / `GatherOOB` parameters as A5, with the same `Coalesce::Row` default, so a non-templated `MGATHER(dst, src, idx)` means `Coalesce::Row` on sim exactly as on hardware. Row mode reads `table[idx[r], :]` into `dst[r, :]` (table row stride `Shape[4]`, `tableRows = Shape[3]`); Elem mode walks `validRow * validCol` and reads `table[idx[i, j]]`. `GatherOOB` is modeled (`Clamp` / `Wrap` remap the index, `Zero` writes a zero element on out-of-range; `Undefined` reads unchecked). Row iteration uses `pto::cpu::parallel_for_rows`, which by default runs sequentially because `PTO_CPU_MAX_THREADS` defaults to `1u`.
- **A2/A3 VEC-CORE** — single-threaded scalar / MTE2 walk driven from the scalar pipe. Row mode issues one wide `copy_gm_to_ubuf_align_b*` DMA per row through `tablePtr + safeIdx * tableRowStride` (where `tableRowStride = table.GetStride(DIM_3)` and `tableRows = ∏ Shape[0..3]`); Elem mode performs a scalar GM→UB copy per element (DMA bursts cannot satisfy per-element UB-destination 32-byte alignment). Supports ND-GM with ND-UB **and** NZ-GM with NZ-UB tile pairs.
- **A5 SIMT** — SIMT launch through `cce::async_invoke` with up to `dim3{32, 32}` (1024 threads). Row mode uses warp-parallel lane reads that the SIMT hardware coalesces into 128 B GM bursts when consecutive; Elem mode maps one lane to one element with per-lane scalar GM loads. The Row kernel computes `srcRow = table + safeIdx * validCols`, so the GM table is treated as **packed ND with row stride = `validCols`** — `MGatherCheck` enforces `GlobalTable::staticShape[4] == TileDst::ValidCol` at compile time, and `tableRows = Shape[3]`. NZ block-stride layouts are not implemented on A5. The `(1, 1)` Elem case bypasses the SIMT launch and runs `MGatherScalarImpl` on the AIV vector core.

## Math Interpretation

### Row Coalesce (`Coalesce::Row`)

Destination `dst[R, C]`, index `idx[1, R]` (or `idx[R, 1]` on A5), table `table[TableRows, C]`:

$$ \mathrm{dst}_{r, j} = \mathrm{table}_{\mathrm{idx}_{r},\; j} \quad\text{for } 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce (`Coalesce::Elem`)

Destination `dst[R, C]`, index `idx[R, C]` (same valid shape as `dst`), flat table of length `TableSize`:

$$ \mathrm{dst}_{r, c} = \mathrm{table}[\mathrm{idx}_{r, c}] \quad\text{for } 0 \le r < R,\; 0 \le c < C $$

`TableSize = Shape[0] * Shape[1] * Shape[2] * Shape[3] * Shape[4]` of the source `GlobalTensor` (5-D, any combination of static and dynamic dims). For NZ tables (A2/A3) the scalar `idx` is decomposed into `(logicalRow = idx / nLogicalCols, logicalCol = idx % nLogicalCols)` and then translated to the NZ block-stride GM offset.

### Out-of-Bounds Behaviour

```cpp
enum class GatherOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Clamp     = 1,  // Clamp index to capacity - 1
    Wrap      = 2,  // Index modulo capacity
    Zero      = 3   // Return zero for OOB; in-bounds indices are loaded normally
};
```

`capacity` is `TableRows` in Row mode and the full flat table length in Elem mode.

- `Undefined`: caller guarantees `idx < capacity`; no remap is applied.
- `Clamp`: `idx = min(idx, capacity - 1)` before access.
- `Wrap`: `idx = idx % capacity` before access.
- `Zero`: out-of-bounds destinations receive `static_cast<T>(0)`. In Row mode the OOB row is filled with `T(0)` (A2/A3 ND fills inline on the scalar pipe; A2/A3 NZ pre-zeros the whole tile once before the DMA loop; A5 SIMT does the substitution inline per lane). In Elem mode the OOB lane writes `T(0)` inline through the same store. All dtypes are supported under every `GatherOOB` value.
- CPU simulator: models `GatherOOB` the same way as A5 — `Clamp`/`Wrap` remap the (uint32-cast) index, `Zero` writes `T(0)` for out-of-range, and only `Undefined` (the default) reads `table.data()[idx]` unchecked (target-defined).

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../../../syntax-and-operands/assembly-model.md).

Synchronous form:

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

Row coalesce:

```text
mgather.row %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<1xRxi32>)
```

Element coalesce:

```text
mgather.elem %dst, %table, %idx : (!pto.tile<RxCxT>, !pto.memref<...>, !pto.tile<RxCxi32>)
```

OOB-aware variants append the mode suffix (`mgather.row.clamp`, `mgather.elem.zero`, etc.).

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

Declared in `include/pto/common/pto_instr.hpp` (the shared dispatcher) and the per-target implementation headers (`include/pto/cpu/MGatherScatter.hpp`, `include/pto/npu/a2a3/MGather.hpp`, `include/pto/npu/a5/MGather.hpp`).

### CPU Reference Form

```cpp
template <Coalesce  Mode = Coalesce::Row,
          GatherOOB Oob  = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

The CPU form mirrors the A5 signature and defaults, so a non-templated `MGATHER(dst, src, idx)` resolves to `Coalesce::Row` on both sim and hardware and one kernel source validates against one golden. Row mode reads `src[idx[r], :]` into `dst[r, :]` with `tableRows = Shape[3]` and row stride `Shape[4]`; Elem mode walks `validRow × validCol` and reads `dst[i, j] = src[idx[i, j]]`. `GatherOOB` is modeled the same way as hardware: `Clamp`/`Wrap` remap the (uint32-cast) index, `Zero` writes a zero element for out-of-range, and `Undefined` reads unchecked.

### A2/A3 Form

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Oob   = GatherOOB::Undefined,
          typename TileDst, typename GlobalTable, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalTable& table, TileIdx& idx,
                             WaitEvents&... events);
```

The kernel iterates over `TileDst::ValidRow * TileDst::ValidCol` logical positions. Physical UB strides come from each tile's `RowStride` (which equals padded `Cols` for `BLayout::RowMajor`). Dispatched as a regular AIV function call — no async-launch or cross-core orchestration.

### A5 Form

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Mode  = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& table, TileInd& idx,
                             WaitEvents&... events);
```

The implementation launches `simt_mgather_row_kernel` or `simt_mgather_elem_kernel` through `cce::async_invoke<…>(cce::dim3{32, kLaunchWarps}, …)`. UB addressing inside the SIMT kernel goes through `tile_offset_2d<TileX>(r, c)`, which makes the kernel layout-agnostic for UB tiles. The degenerate Elem `(1, 1)` case bypasses the SIMT launch and runs `MGatherScalarImpl` on the AIV vector core.

### Parameters (NPU forms)

- `dst` — UB destination tile (`TileType::Vec`); shape `[R, C]`.
- `table` — source GM `GlobalTensor`. `GlobalTensor::DType` must be `__gm__ T` matching the destination element type.
- `idx` — UB index tile (`TileType::Vec`).
- `CMode` — `Coalesce` value (`Row` or `Elem`). First template parameter, so the operating mode is always explicit at the call site.
- `Oob` / `Mode` — `GatherOOB` value for out-of-bounds handling.

### Enums

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // dst[r, :] = table[idx[r], :]   (1-D index of length R)
    Elem = 1   // dst[i, j] = table[idx[i, j]]   (idx shape == dst shape)
};

enum class GatherOOB : uint8_t {
    Undefined = 0,
    Clamp     = 1,
    Wrap      = 2,
    Zero      = 3
};
```

## Constraints

The constraints below are split into target-specific sections so each backend lists exactly what its implementation enforces. Symbols that are common across targets (`T = TileDst::DType`, `TIdx = TileIdx::DType`) are reused.

### Tile Constraints (CPU)

**Supported data types:**

- `dst` / `src` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- On AICore targets (CPU simulator compiled with `__CCE_AICORE__`), `float8_e4m3_t` and `float8_e5m2_t` are also supported.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `dst` must be a vector tile (`TileType::Vec`).
- `indexes` must be a vector tile (`TileType::Vec`).
- `dst` and `indexes` must use row-major layout (`BLayout::RowMajor + SLayout::NoneBox`).
- `src` must be a `GlobalTensor` in GM memory.
- `src` must use `Layout::ND`.

**Shape constraints:**

- `dst.Rows == indexes.Rows` (the index tile and destination tile share the row count).
- `indexes` must be shaped as `[N, 1]` for row-indexed gather or `[N, M]` for element-indexed gather.
- `dst` row width must be 32-byte aligned, that is, `dst.Cols * sizeof(TileDst::DType)` must be a multiple of 32.
- `src` static shape must satisfy `Shape<1, 1, 1, TableRows, RowWidth>`.

**Index interpretation:**

- Index interpretation follows the `Coalesce` mode, with the same `Coalesce::Row` default as hardware. `Row` treats `idx[r]` as a logical row index into a `[TableRows, Shape[4]]` table; `Elem` treats `idx[i, j]` as a **linear element index into `src.data()`**.
- The CPU simulator models `GatherOOB`: `Clamp`/`Wrap` remap the index, `Zero` writes `T(0)` for out-of-range, and `Undefined` (the default) reads `src.data()[idx]` unchecked (target-defined). Indices are cast to `uint32_t` first, so negative indices resolve as large values.

**Header-level enforcement.** The CPU header (`include/pto/cpu/MGatherScatter.hpp`) static-asserts only the minimum closure rules: `std::is_integral_v<TileInd::DType>` for the index dtype and `sizeof(TileDst::DType) == sizeof(GlobalData::DType)` for byte-wise compatibility. The dtype / shape / layout constraints above are the **PTO ABI contract** — callers are expected to honor them so the same kernel source compiles and runs unmodified against the A2/A3 and A5 backends.

### Tile Constraints (A2/A3)

**Supported data types:**

- `dst` / `src` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. No `float8_e4m3_t` / `float8_e5m2_t` / `hifloat8_t` on the A2/A3 vec-core.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `dst` must be a vector tile (`TileDst::Loc == TileType::Vec`).
- `indexes` must be a vector tile (`TileIdx::Loc == TileType::Vec`).
- The index tile is **always** `BLayout::RowMajor + SLayout::NoneBox` (ND), regardless of the table layout.
- `src` must be a `GlobalTensor` in GM memory; `GlobalTable::DType == __gm__ T`.
- The destination tile's bulk + sub layout must be paired with the table layout exactly:
    - `GlobalTable::layout == Layout::ND` ⇒ `TileDst` is `BLayout::RowMajor + SLayout::NoneBox`.
    - `GlobalTable::layout == Layout::NZ` ⇒ `TileDst` is `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == TileConfig::fractalABSize` (= 512 B).

**Shape constraints:**

- Padded `TileDst::Cols * sizeof(T)` must be 32-byte aligned in both layouts (the same DMA-burst rule that `TLOAD` / `TSTORE` enforce). `ValidRow` / `ValidCol` are not constrained by this rule.
- For `Coalesce::Row`: `TileIdx::ValidRow == 1` and `TileIdx::ValidCol == TileDst::ValidRow`.
- For `Coalesce::Elem`: `TileIdx::ValidRow == TileDst::ValidRow` and `TileIdx::ValidCol == TileDst::ValidCol`.
- Both modes require `TileDst::ValidRow >= 1` and `TileDst::ValidCol >= 1`.
- NZ tables additionally require `GlobalTable::staticShape[3] == FRACTAL_NZ_ROW` (= 16), `GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T)` (= 32 B / element width), `TileDst::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0`, and `TileDst::Rows % FRACTAL_NZ_ROW == 0`.

**Index interpretation:**

- For `Coalesce::Row`, each index is treated as a **logical row index** into the `[TableRows, RowWidth]` ND table (or as a logical row across `gShape2 * FRACTAL_NZ_ROW` for NZ).
- For `Coalesce::Elem`, each index is treated as a **linear element index** into the flat 5-D table (`tableSize = ∏ shape[0..4]`). For NZ the kernel splits the linear index into `(logicalRow = idx / nLogicalCols, logicalCol = idx % nLogicalCols)` and applies the NZ block-stride translation through `MGatherNZGmOffset`.
- OOB handling follows the `GatherOOB` template parameter (`Undefined` / `Clamp` / `Wrap` / `Zero`); see the Out-of-Bounds subsection above.

### Tile Constraints (A5)

**Supported data types:**

- `dst` / `src` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `dst` must be a vector tile (`TileDst::Loc == TileType::Vec`).
- `indexes` must be a vector tile (`TileIdx::Loc == TileType::Vec`).
- The SIMT kernel is layout-agnostic for UB tiles: every UB read/write goes through `tile_offset_2d<TileX>(r, c)`, so `TileDst` may be `BLayout::RowMajor` or `BLayout::ColMajor` (with `SLayout::NoneBox`).
- `src` must be a `GlobalTensor` in GM memory; `GlobalTable::DType == __gm__ T`.
- **GM table layout: `Layout::ND` only.** A5 SIMT kernels address GM as a flat row-major buffer with row stride hard-wired to `validCols` (`srcRow = table + safeIdx * validCols`); `MGatherCheck` enforces `GlobalTable::staticShape[4] == TileDst::ValidCol` so the table cannot have any inter-row padding. The kernel does not honor any other `GlobalTensor::layout` value — `MGatherCheck` does not static-assert the GM `layout` field, but using `Layout::NZ` (or anything other than packed ND with row width = `validCols`) produces wrong output because the kernel never performs block-stride translation. Callers that need NZ data must pre-stage into ND.

**Shape constraints:**

- Padded `TileDst::Cols * sizeof(T)` (RowMajor) or `TileDst::Rows * sizeof(T)` (ColMajor) must be 32-byte aligned — this is an upstream `TLOAD` / `TSTORE` requirement, not a `MGATHER` constraint per se. `ValidRow` / `ValidCol` are not constrained by this rule.
- For `Coalesce::Row`: the index tile's valid shape is `[1, R]` (`BLayout::RowMajor`) **or** `[R, 1]` (`BLayout::ColMajor`). Both forms produce a linear R-element layout in UB and the kernel reads `idx[row]` directly. The choice of `BLayout` for the index tile is independent of the GM table layout — `MGatherCheck` does not constrain the GM `layout` field for the table.
- For `Coalesce::Elem`: `TileIdx::ValidRow == TileDst::ValidRow` and `TileIdx::ValidCol == TileDst::ValidCol`. The `BLayout` of `TileIdx` is independent of `TileDst`'s — the kernel walks both through per-tile `tile_offset_2d`.
- Both modes require `TileDst::ValidRow >= 1` and `TileDst::ValidCol >= 1`. The degenerate `(1, 1)` shape in Elem mode bypasses the SIMT launch and runs `MGatherScalarImpl` on the AIV vector core.

**Index interpretation:**

- For `Coalesce::Row`, each index is treated as a **logical row index** into the `[TableRows, validCols]` ND table; the kernel computes `srcRow = table + safeIdx * validCols`.
- For `Coalesce::Elem`, each index is treated as a **linear element index** into the flat 5-D table (`tableSize = ∏ shape[0..4]`); the kernel reads `table[safeIdx]`.
- OOB handling follows the `GatherOOB` template parameter (`Undefined` / `Clamp` / `Wrap` / `Zero`); see the Out-of-Bounds subsection above.

### Dynamic Runtime Shapes (A2/A3 and A5)

`MGATHER` accepts both compile-time and runtime-dynamic shapes:

- `Tile<…, RowMask, ColMask>` with `RowMask == -1` and/or `ColMask == -1` stores the runtime valid extents in the tile object; the implementation reads them through `dst.GetValidRow()` / `dst.GetValidCol()`.
- `Shape<S0, S1, S2, S3, S4>` / `Stride<…>` with one or more `-1` entries are constructed with the runtime sizes; the implementation reads them through `table.GetShape(GlobalTensorDim::DIM_*)` and folds them into `tableRows` (Row mode) or `tableSize = ∏ shape[0..4]` (Elem mode).

Static-asserts in `MGatherCheck` are gated on `if constexpr (DIM > 0)`, so they fire only for compile-time-known dimensions. Padded `Tile::Rows / Cols` are always compile-time — they govern the UB DMA-burst alignment and the SIMT lane addressing.

Example (mirrors `case_elem2d_dyn_user_float_1x9_in_1x16_3x10`):

```cpp
constexpr auto kPadCols = 16;
using DstTileT    = Tile<TileType::Vec, float,    1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT    = Tile<TileType::Vec, int32_t,  1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t validCols = 9, d3 = 3, d4 = 10, srcStride3 = 10;
TableShape  tableShape(d3, d4);
TableStride tableStride(srcStride3, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(srcGm, tableShape, tableStride);

DstTileT dstTile(1, validCols);
IdxTileT idxTile(1, validCols);
TASSIGN(dstTile, dstUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MGATHER<Coalesce::Elem, GatherOOB::Undefined>(dstTile, tableGM, idxTile);
```

At dispatch the implementation resolves `validRows = 1`, `validCols = 9`, and `tableSize = 1·1·1·3·10 = 30`. The padded UB `Tile::Cols = 16` is purely a `TLOAD` burst-alignment artifact — the gather loop walks only the valid 9 elements.

## Mode Resolution

Mode is **explicit** on A2/A3 and A5, never auto-detected. Static asserts in `MGatherCheck` validate the supplied tile shapes against the chosen `Coalesce` value:

```text
A2/A3:
  Coalesce::Row  : Idx.ValidRow == 1 && Idx.ValidCol == Dst.ValidRow
  Coalesce::Elem : Idx.ValidRow == Dst.ValidRow && Idx.ValidCol == Dst.ValidCol

A5:
  Coalesce::Row  : (Idx.ValidRow == 1 && Idx.ValidCol == Dst.ValidRow) ||
                   (Idx.ValidRow == Dst.ValidRow && Idx.ValidCol == 1)
  Coalesce::Elem : (Idx.ValidRow == Dst.ValidRow) && (Idx.ValidCol == Dst.ValidCol)
```

The CPU reference path enforces the same A5 Row rule (`[1, R]` or `[R, 1]` index valid shape) when valid shapes are statically known; Elem walks element-wise over the destination valid region.

## Layout Support

UB addressing is computed from each tile's `Rows` / `Cols` plus (on A2/A3 NZ) an optional fractal block-col stride; GM addressing is driven from `GlobalTensor::GetStride(DIM_*)`. The matrix below summarises what each backend accepts.

| Tile / Tensor | CPU | A2/A3 | A5 |
|---------------|-----|-------|----|
| `TileDst` (UB) — ND | `BLayout::RowMajor + SLayout::NoneBox` only | `BLayout::RowMajor + SLayout::NoneBox` | `BLayout::RowMajor` or `ColMajor`, `SLayout::NoneBox` (Elem mode walks both via `tile_offset_2d`) |
| `TileDst` (UB) — NZ | not supported | `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512` (paired with NZ GM table) | **not supported** |
| `TileIdx` (UB) — Row | `[1, R]` or `[R, 1]` row-major (same valid-shape rule as A5) | `[1, R]` `BLayout::RowMajor + SLayout::NoneBox` (always ND, regardless of table layout) | `[1, R]` `RowMajor` **or** `[R, 1]` `ColMajor` (independent of GM table layout) |
| `TileIdx` (UB) — Elem | `BLayout::RowMajor + SLayout::NoneBox` | `[R, C]` `BLayout::RowMajor + SLayout::NoneBox` | any `BLayout`, independent of `TileDst` (kernel reads through `tile_offset_2d<TileIdx>`) |
| `GlobalTable` (GM) — ND | `Layout::ND` only | `Layout::ND` (linear contiguous addressing); 5-D `Shape<…, R, C>`; kernel uses `tableRowStride = GetStride(DIM_3)`, so stride-padded ND tables are supported | `Layout::ND` only; Row mode hard-wires `tableRowStride = validCols` and requires `Shape[4] == validCols` |
| `GlobalTable` (GM) — NZ | not supported | `Layout::NZ`; 5-D `Shape<B, BCols, BRows, 16, C0>` with `staticShape[3] == 16` and `staticShape[4] == 32 / sizeof(T)` (`B` may be > 1; the kernel loops `for i in 0..gShape0`) | **not supported** |

### NZ Layout (A2/A3)

> **NZ paths exist only on A2/A3.** The A5 SIMT kernel addresses GM as a flat ND buffer and has no NZ block-stride translation — there is no "inherent SIMT NZ support" to be enabled. Callers that need NZ data on A5 must transpose / repack into ND first.

When `GlobalTable::layout == Layout::NZ` and `TileDst` is the matching `BLayout::ColMajor + SLayout::RowMajor + SFractalSize = 512` tile, `MGATHER` (A2/A3) runs the dedicated NZ paths (`MGatherRowNzImpl`, `MGatherElemNzImpl`).

- **Constants.** `kC0 = C0_SIZE_BYTE / sizeof(T) = 32 / sizeof(T)`; `kFRow = FRACTAL_NZ_ROW = 16`. Each fractal block is `kFRow × kC0` elements (= 32 B × 16 = 512 B).
- **Logical shape.** Logical rows = `gShape2 * kFRow`. Logical cols = `gShape0 * gShape1 * kC0`. Row-mode `mgather_remap` clamps / wraps against the logical row count; Elem-mode against the total element count `(gShape2 * kFRow) * (gShape0 * gShape1 * kC0)`.
- **Row mode.** For each logical row `r`, the kernel maps `idx[r]` to `(srcBlockRow, srcRowInBlock)` and `r` to `(dstBlockRow, dstRowInBlock)`, then issues **one multi-burst MTE2 transfer per outer batch** (`nBurst = gShape1`, `lenBurst = kC0 * sizeof(T) = 32 B`, `gmGap = (gStride1 - kC0) * sizeof(T)`, `ubGap = TileDst::Rows - 1` blocks). When `Oob == GatherOOB::Zero` the kernel pre-fills the whole tile with `T(0)` once before the DMA loop and simply skips DMAs for OOB rows.
- **Elem mode.** For each `(r, c)` the kernel maps `idx` to `(logicalRow, logicalCol) = (idx / nLogicalCols, idx % nLogicalCols)`, then to NZ physical offsets through `MGatherNZGmOffset`. The destination offset is `(c / kC0) * (TileDst::Rows * kC0) + r * kC0 + (c % kC0)`. The walk order is **block-col → row → col-in-block** so consecutive writes always target consecutive 32 B UB blocks. Out-of-bounds lanes write `T(0)` inline when `Oob == GatherOOB::Zero`.
- **Stride vs. valid-shape.** `MGather*NzImpl` reads strides from the `GlobalTensor` runtime (`GetStride(DIM_*)`), so packed (`gStride1 == gShape2 * gShape3 * gShape4`) and stride-padded NZ tensors both work without caller-side adjustment.

### Why Elem Mode Uses Scalar GM Reads (A2/A3)

`copy_gm_to_ubuf_align_b8/b16/b32` requires the **UB destination address** to be 32-byte aligned, and the destination must be a whole number of 32-byte burst chunks. A per-element MTE2 burst of `lenBurst = sizeof(T)` to `dstPtr + r * RowStride + c` does not satisfy that rule whenever `(c * sizeof(T)) % 32 != 0`, which covers almost every elem-mode lane. On the simulator the runtime accepts the misaligned burst; on real A2/A3 hardware the transfer silently drops, leaving the destination lane at its initial value (typically zero). Row mode does not hit this problem because each row write starts at `r * RowStride`, and `RowStride * sizeof(T)` is always a multiple of 32 bytes.

A2/A3 Elem mode therefore uses scalar GM→UB copies, which have element-level addressing granularity and place no alignment requirement on the destination. Atomic semantics are not needed for gather (no destination is written from multiple sources), and `OOB::Zero` collapses into a direct scalar zero-write. The scalar GM↔UB path is the same `(1, 1)` fallback already validated on hardware, extended to all elem shapes.

A5 does not face this constraint: the SIMT lane issues per-thread loads instead of a UB-aligned DMA burst, so Elem mode on A5 is naturally per-element with no extra alignment workaround.

## Aligned vs Unaligned Tile Shapes

The kernel does **not** care whether the tile's logical shape is "aligned" — it walks all `ValidRow * ValidCol` positions. The 32-byte alignment of the contiguous dim is enforced upstream by the `Tile` system because every `TLOAD` / `TSTORE` issues 32 B GM↔UB bursts.

- **Row mode (A2/A3 ND).** per-row DMA `lenBurst = validCol * sizeof(T)`; `Tile::RowStride * sizeof(T)` is forced 32-byte aligned by the `Tile` system, so subsequent rows always start on a 32 B burst boundary.
- **Row mode (A2/A3 NZ).** one multi-burst transfer per logical row × outer-batch; `lenBurst = kC0 * sizeof(T) = 32 B` is fixed by the fractal layout, so per-row alignment is automatic. `validRow` does not have to be a multiple of `kFRow`.
- **Elem mode (A2/A3).** one scalar GM→UB copy per element. The scalar pipe has element-level addressing granularity, so any `(ValidRow, ValidCol)` inside the padded tile works.
- **Row + Elem mode (A5).** the SIMT kernel walks through `tile_offset_2d<TileX>(r, c)` so any `(ValidRow, ValidCol)` with `1 ≤ Valid ≤ Padded` is accepted, including the degenerate `(1, 1)` which routes to `MGatherScalarImpl`.

Callers handle "unaligned valid region" by padding the tile up to the nearest 32-byte alignment (for example valid `[3, 3]` int32 → tile `[3, 8]`), and either zero-initializing the padding (`TASSIGN`-then-clear) or only inspecting the valid region post-gather.

### Minimum Tile Shape

The minimum padded inner dim is set by the upstream `TLOAD` / `TSTORE` burst-alignment rule; `MGATHER` itself accepts any `(ValidRow, ValidCol) >= (1, 1)`.

**A2/A3** — row-major tile only (NZ paths use fractal `[16, kC0]` blocks instead):

| `T` | Min `Cols` (`BLayout::RowMajor`) |
|-----|---------------------------------|
| `int8` / `uint8` | 32 |
| `int16` / `uint16` / `half` / `bfloat16` | 16 |
| `int32` / `uint32` / `float` | 8 |

**A5** — row-major or column-major tile (the contiguous dim must satisfy the 32 B rule):

| `T` | Min `Cols` (`RowMajor`) | Min `Rows` (`ColMajor`) |
|-----|-------------------------|-------------------------|
| `int8` / `uint8` / `float8_e4m3` / `float8_e5m2` / `hifloat8` | 32 | 32 |
| `int16` / `uint16` / `half` / `bfloat16` | 16 | 16 |
| `int32` / `uint32` / `float` | 8 | 8 |

**CPU** — same row-major min rule as A2/A3 (any `Cols * sizeof(T)` that the test's `TLOAD` path requires).

## Pipe / Synchronisation Model

The two NPU targets use very different mechanisms; the CPU reference path has no pipes to synchronise.

### A2/A3 — explicit pipe handshakes

The implementation centralises every pipe handshake the kernel needs. **Callers do not need to insert any extra barriers** beyond the standard `TLOAD` post-load `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` pair that brings the index tile into a clean state on the vector pipe before `MGATHER`. The kernel never uses `pipe_barrier(PIPE_ALL)` — every wait is a specific producer→consumer pair.

| Phase | Pipe transition | What it guards |
|-------|-----------------|----------------|
| Pre-amble (Row ND + Row NZ) | `V→S`, `MTE3→S` flag chain | Make the index tile visible to scalar reads (V→S transitively waits for MTE2 through the caller's `TLOAD` post-load `MTE2→V` flag) and flush any pending MTE3 writes that might overlap UB before the scalar loop starts. |
| Pre-amble (Elem ND + Elem NZ) | `V→S`, `MTE3→S`, `MTE2→S` flag chain | Same as Row, with an additional `MTE2→S` flag that flushes any in-flight MTE2 burst before the scalar loop reads `idxPtr[r * IdxRowStride + c]`. |
| Body (Row, ND) | `copy_gm_to_ubuf_align_b*` per row | One DMA per row, `lenBurst = validCol * sizeof(T)`; trip count = `validRow`. |
| Body (Row, NZ) | `copy_gm_to_ubuf_align_b*` multi-burst per logical row × batch | `nBurst = gShape1`, `lenBurst = C0 * sizeof(T) = 32 B`, `gmGap = (gStride1 - C0) * sizeof(T)`, `ubGap = Tile::Rows - 1` blocks; trip count = `validRow * gShape0`. |
| Body (Elem, ND and NZ) | Scalar `dstUb[r, c] = tableGm[gmOff]` per element | Per-element scalar GM→UB copy; trip count = `validRow * validCol`. NZ walks block-col-major to keep each 32 B UB block written contiguously in time. `OOB::Zero` lanes write `T(0)` inline. |
| Row post-amble (ND + NZ) | `S→MTE2`, `MTE2→V`, `MTE2→MTE3`, `S→V`, `S→MTE3` flag chain (each pair is a `set_flag` + `wait_flag`) | First close the scalar→MTE2 race so any in-flight DMA observes the loop's final scalar state; then drain the MTE2 DMAs before V or MTE3 consumers touch the destination tile; finally release the scalar pipe to V and MTE3 for downstream operators (e.g. a follow-up vector op or `TSTORE`). |
| Elem post-amble (ND + NZ) | `S→V`, `S→MTE2`, `S→MTE3` flag chain | Make the scalar UB writes visible to V (for downstream vector ops), MTE2 (for follow-up gathers / loads), and MTE3 (bridges to the caller's `TSTORE`). |

### A5 — SIMT launch with V↔S handshake

The A5 implementation hides almost the entire pipe model behind `cce::async_invoke`, which establishes the warp-scheduler context and orchestrates per-lane GM and UB accesses. The only explicit kernel-side handshake is in the scalar fallback path (`MGatherScalarImpl`, used for Elem `(1, 1)`):

| Phase | Pipe transition | What it guards |
|-------|-----------------|----------------|
| Pre-amble (scalar fallback) | `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` | Make the index tile visible to the scalar pipe before the single-element gather. |
| Body (SIMT Row / Elem) | `cce::async_invoke<simt_mgather_*_kernel>(dim3{32, kLaunchWarps}, …)` | The SIMT launch handles all per-lane GM loads and UB stores internally; no caller-visible flags. |
| Post-amble (scalar fallback) | `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` | Release the scalar UB write to downstream vector ops. |

**Caller responsibility on A5.** The same `TLOAD` post-load `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` pair is required before `MGATHER` so the index tile is observable on the vector pipe by the time the SIMT launch begins. Downstream consumers (`TSTORE`, vector ops on the destination) only need their normal V↔MTE3 / V↔V handshake — the SIMT launch is treated as a vector-pipe producer.

## UB Memory Budget

The unified buffer is shared between user tiles, the runtime reserved region, and the Data Cache. Budget rules differ per target.

### CPU simulator

No UB — the implementation runs in host memory. Tile sizes are bounded by the test harness only.

### A2/A3

The AIV vector core has the standard CANN 192 KB UB layout. `MGATHER` does not allocate any UB scratch from inside the kernel — the only UB consumers are the caller-allocated destination tile (`R * C * sizeof(T)`, padded up to the 32-byte burst alignment) and index tile (`R * C * sizeof(TIdx)`, same padding rule). A2/A3 has no `dynUBufSize` knob; the working set must fit in the caller's static UB budget.

### A5

A5 SIMT kernels run on the AIV vector core. All user tiles must fit inside the AIV's 256 KB Unified Buffer alongside two fixed runtime reservations: an 8 KB reserved region (AscendC / TBE bookkeeping) and the Data Cache (32 KB minimum, sized at launch time). The UB layout is:

```text
+---------------------------+
| Static memory             |  Compile-time tile allocations
+---------------------------+
| Dynamic memory            |  Sized at launch through dynUBufSize
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

When tiles are placed manually with `TASSIGN` (as in the A5 ST suite), the compiler sees `static_memory ≈ 0` and the full **216 KB** is available as `dynUBufSize`. `MGATHER` itself does not allocate any UB scratch — every read flows GM → register → UB. The only UB consumers are the destination and index tiles.

#### Default Per-Call Budget (No `dynUBufSize`)

When the kernel is launched without an explicit `dynUBufSize` (`<<<numBlocks, nullptr, stream>>>`), the runtime keeps the default DCache size and reserves only a small default dynamic region. In practice the safe `dst + idx` working set is **≤ 128 KB**; beyond that, on-board execution may silently corrupt or zero out the result while still passing the CPU simulator (which does not model these reservations).

#### Extending Per-Call UB Beyond 128 KB

Callers that need a single-shot `dst + idx` footprint larger than 128 KB must declare the dynamic-UB request explicitly through the second argument of the kernel launch:

```cpp
kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...);
```

`dynUBufSize` is the byte size of the dynamic-UB region the kernel will use. The bisheng/CCE compiler routes such launches through `__cce_rtKernelLaunchWithFlagV2`, setting `rtTaskCfgInfo_t::localMemorySize = dynUBufSize`. The runtime then shrinks the DCache toward its 32 KB minimum and hands the remaining space back to the kernel.

Key points:

- **The simulator does not enforce this.** Passing `nullptr` (or `0`) still runs to completion in sim regardless of the actual UB footprint. Always set `dynUBufSize` explicitly when the workload exceeds 128 KB so the binary stays correct on real hardware.
- **Exceeding the ceiling is silent.** The compiler does not error and the simulator does not flag it. On-board, the first overflow byte corrupts the reserved region or DCache and the kernel returns undefined output.
- **Size to actual usage.** For Elem coalesce with `R × C × sizeof(T)` destination and `R × C × sizeof(int32_t)` index, the working set is `R * C * (sizeof(T) + 4)`. Round up to a comfortable margin when passing `dynUBufSize`.

Example launch wrapper for an extended-UB gather:

```cpp
// Round dst + idx up to a safe dynUBufSize (here T = float, idx = int32_t, so
// per-element footprint = 4 + 4 = 8 B; pass R * C * 8 with some headroom).
constexpr uint32_t kDynUbBytes = (uint32_t)(R * C * (sizeof(T) + sizeof(int32_t)));
runMGATHER_kernel<<<numBlocks, kDynUbBytes, stream>>>(out, table, indices);
```

The current MGATHER ST suite does not include extended-UB cases — every existing case fits in the default budget — but workloads that exceed 128 KB should follow this pattern (see `MSCATTER.md` for worked examples that push the destination + index footprint up to 216 KB).

## Runtime Dispatch Requirement (A5)

`MGATHER` on A5 uses `cce::async_invoke<simt_mgather_*_kernel>(cce::dim3{32, kLaunchWarps}, …)` internally to fan a per-warp / per-lane workload out across up to 1024 threads. `async_invoke` consumes runtime state (TID registers, warp / lane configuration, vector-pipe scheduling) that the **launch path** must install before the kernel function is entered. The standard CANN launch (`rtKernelLaunchWithHandleV2`, used by the `<<<numBlocks, dynUBufSize, stream>>>` syntax) installs this state correctly.

A runtime variant that dispatches kernels as a direct C function-pointer call is fine for SPMD ops (`TLOAD`, `TSTORE`, `TADD`, …) but skips the SIMT context init, so the first `async_invoke` inside `MGATHER` has no warp scheduler to dispatch into and hangs. Use the standard launch syntax for any SIMT kernel.

## Performance Considerations

### A2/A3

1. **Row vs. Elem.** Row coalesce achieves the best aggregate bandwidth — one wide DMA per logical row (ND) or one multi-burst DMA per logical row × batch (NZ). Elem coalesce issues one scalar GM read + UB write per active lane: no DMA-engine pipelining, throughput bound by scalar GM access latency. Prefer Row whenever the indexing structure permits.
2. **Sequential scalar loop (Elem).** A2/A3 dispatches `MGATHER` as a single-thread sequential walk of the `validRow * validCol` lanes. Trip counts of `ValidCol ≤ 32 / sizeof(T)` rows are the sweet spot; large flat tiles are bound by scalar GM read latency. The block-col-major walk used for NZ keeps consecutive writes spatially-local in UB.
3. **Why not per-element MTE2 in Elem mode.** See "Why Elem mode uses scalar GM reads" above — the UB-destination 32 B alignment rule rules out per-element DMA bursts.
4. **DMA cost (Row).** ND: each row is one `copy_gm_to_ubuf_align_b*` call with `nBurst = 1`, `lenBurst = validCol * sizeof(T)`. NZ: each (logical row, batch) pair is one call with `nBurst = gShape1`, `lenBurst = C0 * sizeof(T) = 32 B`, `gmGap = (gStride1 - C0) * sizeof(T)`, `ubGap = Tile::Rows - 1` blocks. Back-pressure is bounded by `MAX_OUTSTANDING_MTE2`; no row chunking.
5. **OOB cost.** `Undefined` is free; `Clamp` / `Wrap` add a single arithmetic remap per lane; `Zero` writes `T(0)` inline (Elem) or pre-zeros the tile and skips DMAs for OOB rows (Row).
6. **Single-pass dispatch.** `MGATHER` is a regular AIV function call from the kernel (no async-launch or cross-core orchestration). Concurrency comes from DMA engine pipelining behind the scalar issue loop, not from multiple worker threads.

### A5

1. **Shape-adaptive launch.** The SIMT grid is sized as `dim3{32, kLaunchWarps}` from the resolved `validRows` / `validCols`. Small tiles do not pay the cost of 1024 idle threads.
    - **Row.** `kRowWarps = min(validRows, 32)` warps own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks. When consecutive lanes in a warp read consecutive `col` values from the same `srcRow`, the SIMT hardware coalesces them into a 128 B GM burst.
    - **Elem.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`. Threads with `tid >= totalElems` skip the loop body. For `totalElems > 1024` the strided loop walks `launchThreads` at a time.
2. **OOB policy cost.** `Undefined`: zero overhead. `Clamp` / `Wrap`: a single arithmetic remap per lane. `Zero`: one extra compare-and-select per lane to substitute `static_cast<T>(0)` for OOB lanes.
3. **No thread divergence for mode / OOB.** All decisions are `if constexpr`. The `gather_remap` lookup compiles to a small data-dependent transform with no control-flow split.
4. **Unrolled inner loops.** Inner column loop in Row coalesce carries `#pragma unroll(4)`; outer per-row and elem flat loops are `#pragma unroll(1)` to keep code size bounded.
5. **Row vs. Elem bandwidth.** Row coalesce achieves the best aggregate bandwidth (per-warp 32 lanes coalesce a single 128 B GM burst when `idx[r]` is the same across the warp). Elem coalesce performs one scalar GM load per lane — non-coalesced for random indices; UB writes remain coalesced because consecutive `tid` map to consecutive UB offsets.
6. **Register pressure.** Kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread) and use ≤ 12 live registers in the hot path. No spills are produced.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  DstT dst;
  IdxT idx;
  // src is a GlobalTensor in GM
  MGATHER(dst, src, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  DstT dst;
  IdxT idx;
  TASSIGN(dst, 0x1000);
  TASSIGN(idx, 0x2000);
  MGATHER(dst, src, idx);
}
```

### Row Coalesce — Embedding Lookup (A2/A3 or A5)

```cpp
#include <pto/pto-inst.hpp>

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

### Row Coalesce — `[R, 1]` ColMajor Index (A5 only)

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

### Element Coalesce — `(1, 1)` Degenerate Case

```cpp
AICORE void example_scalar(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int TableSize = 32;

    using DstTile     = Tile<TileType::Vec, float,   1, 8, BLayout::RowMajor, 1, 1>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0080);

    MGATHER<Coalesce::Elem>(dst, tableGM, idx);
}
```

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

## Related Instructions

- [`TLOAD`](TLOAD.md): contiguous block transfer GM → Tile.
- [`MSCATTER`](./mscatter.md): indexed scatter Tile → GM (inverse operation).
- [`TGATHER`](TGATHER.md): index-based gather within tiles (UB-to-UB on the same vec-core).
