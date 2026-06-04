# pto.mscatter

## Tile Operation Diagram

![MSCATTER tile operation](../../../../figures/isa/MSCATTER.svg)

## Introduction

`MSCATTER` writes data from a UB source tile into a GM `GlobalTensor` through a UB index tile. The operating mode is selected explicitly through the `Coalesce` template parameter:

- **`Coalesce::Row`** (default) — scatter full rows `src[r, :]` into `table[idx[r], :]`. The index tile is 1-D (`[1, R]` row-major; on A5 also `[R, 1]` column-major). `R = 1` is allowed.
- **`Coalesce::Elem`** — element-wise scatter from `src[R, C]` (or `src[1, N]`) into a linearized `table` through `idx[R, C]`. The index tile must have the same valid shape as the source. The degenerate `(1, 1)` case is allowed.

Write behaviour is controlled by orthogonal template policies:

- `ScatterAtomicOp` — `None` (plain store), `Add`, `Max`, `Min`. Atomic dtype support varies per target (see the table below).
- `ScatterOOB` — `Undefined`, `Skip`, `Clamp`, `Wrap`. There is no `Zero` option (the operation writes into an existing table; out-of-bounds indices have no real destination to zero).
- `ScatterConflict` (**A5 only**) — `Last` (deterministic largest-index wins, only consulted when `Atomic == None`) or `Default` (warp-scheduler dependent). A2/A3 has no `ScatterConflict` parameter because the kernel is strictly sequential and collisions are resolved by "last write wins".

Per-target dispatch summary:

- **CPU Simulator** — pure C++ reference. The implementation walks `validRow * validCol` with a plain sequential `for` loop in row-major order and writes `table[idx[i, j]] = src[i, j]` (Elem semantics). When multiple sources map to the same destination, the last writer in row-major iteration order wins. No atomic mode is exposed at the intrinsic surface.
- **A2/A3 VEC-CORE** — single-threaded scalar / MTE3 walk driven from the scalar pipe. Row mode issues one wide `copy_ubuf_to_gm_align_b*` DMA per row through `tablePtr + safeIdx * tableRowStride` (where `tableRowStride = table.GetStride(DIM_3)` and `tableRows = ∏ Shape[0..3]`); Elem mode performs a scalar UB→GM store per element (DMA bursts cannot satisfy per-element UB-source 32-byte alignment). Supports ND-GM with ND-UB **and** NZ-GM with NZ-UB tile pairs. Always "last write wins" for `ScatterAtomicOp::None`.
- **A5 SIMT** — SIMT launch through `cce::async_invoke` with up to `dim3{32, 32}` (1024 threads). Row mode uses warp-parallel lane writes that the SIMT hardware coalesces into 128 B GM bursts when consecutive; Elem mode maps one lane to one element with per-lane scalar GM stores. The Row kernel computes `dstRow = table + safeIdx * validCols`, so the GM table is treated as **packed ND with row stride = `validCols`** — `MScatterCheck` enforces `GlobalTable::staticShape[4] == TileSrc::ValidCol` at compile time, and `tableRows = Shape[3]`. `Conflict::Last` is implemented as a slot-centric reverse scan (`last_owner_find_*`) so the result is deterministic and race-free. NZ block-stride layouts are not implemented on A5. The `(1, 1)` Elem case bypasses the SIMT launch and runs `MScatterScalarImpl` on the AIV vector core.

## Math Interpretation

### Row Coalesce (`Coalesce::Row`)

Source `src[R, C]`, index `idx[1, R]` (or `idx[R, 1]` on A5), table `table[TableRows, C]`. For each row `r`:

$$ \mathrm{table}_{\mathrm{idx}_{r},\; j} \;\leftarrow\; \mathrm{atom}\!\left(\mathrm{table}_{\mathrm{idx}_{r},\; j},\; \mathrm{src}_{r, j}\right) \quad\text{for } 0 \le j < C $$

where `atom` is the identity (replace) for `ScatterAtomicOp::None` or the corresponding atomic accumulation otherwise.

### Element Coalesce (`Coalesce::Elem`)

Source `src[R, C]`, index `idx[R, C]` (same valid shape as `src`), flat table of length `TableSize`:

$$ \mathrm{table}[\mathrm{idx}_{r, c}] \;\leftarrow\; \mathrm{atom}\!\left(\mathrm{table}[\mathrm{idx}_{r, c}],\; \mathrm{src}_{r, c}\right) $$

`TableSize = Shape[0] * Shape[1] * Shape[2] * Shape[3] * Shape[4]` of the destination `GlobalTensor`. For NZ tables (A2/A3) the scalar `idx` is decomposed into `(logicalRow, logicalCol)` and then translated to the NZ block-stride GM offset through `MScatterNZGmOffset`.

### Atomic Accumulation

When `ScatterAtomicOp::Add` / `Max` / `Min` is selected:

$$ \mathrm{table}[\cdot] \mathrel{\oplus}= \mathrm{src}_{\cdot},\quad \oplus \in \{+,\; \max,\; \min\} $$

### Conflict Resolution

- **A2/A3.** The kernel is strictly sequential in increasing `(r)` (Row) or `(r, c)` (Elem) order. For `ScatterAtomicOp::None`, **the later write always wins** ("last write wins"). For `ScatterAtomicOp::Add` every Row-mode store goes through the MTE3 atomic-add unit and every Elem-mode store does a scalar read-add-write; duplicate destination indices accumulate.
- **A5.** With `ScatterAtomicOp::None`:
    - **`Conflict::Last`** — the source position with the **largest flat index** that targets a given destination slot is the one whose value is stored. Matches the sequential semantics of `for i in 0..N: table[idx[i]] = src[i]`. Implemented as a slot-centric reverse scan (per-warp `last_owner_find_*`), race-free by construction.
    - **`Conflict::Default`** — the surviving writer is warp-scheduler dependent. For collision-free index sets the result is identical to `Last`.
- **CPU simulator.** Last writer in row-major iteration order wins (matches `Conflict::Last`).

Atomic modes ignore `ScatterConflict` because the GM atomic R-M-W serialises colliding writes by itself.

### Out-of-Bounds Behaviour

```cpp
enum class ScatterOOB : uint8_t {
    Undefined = 0,  // No bounds check; caller guarantees valid indices
    Skip      = 1,  // Drop the write (preserve original table value)
    Clamp     = 2,  // Clamp index to capacity - 1
    Wrap      = 3   // Index modulo capacity
};
```

`capacity` is `TableRows` (Row mode) or `TableSize` (Elem mode):

- `Undefined`: caller guarantees `idx < capacity`; no remap is applied.
- `Skip`: out-of-bounds rows / elements are simply not written (no DMA issued, no scalar store performed). The original table value at that GM address is preserved.
- `Clamp`: `idx = min(idx, capacity - 1)` before access.
- `Wrap`: `idx = idx % capacity` before access.

There is no `Zero` option — an OOB index never identifies a real destination slot, so `Skip` is the natural "do nothing on OOB" policy.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../../../syntax-and-operands/assembly-model.md).

Synchronous form:

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
```

Row coalesce:

```text
mscatter.row %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<1xRxi32>)
```

Element coalesce:

```text
mscatter.elem %table, %src, %idx : (!pto.memref<...>, !pto.tile<RxCxT>, !pto.tile<RxCxi32>)
```

OOB and atomic variants append the mode suffix (`mscatter.row.clamp.atomic_add`, `mscatter.elem.skip`, etc.).

### AS Level 1 (SSA)

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2 (DPS)

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` (the shared dispatcher) and the per-target implementation headers (`include/pto/cpu/MGatherScatter.hpp`, `include/pto/npu/a2a3/MScatter.hpp`, `include/pto/npu/a5/MScatter.hpp`).

### CPU Reference Form

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events);
```

The CPU form has no `Coalesce`, `ScatterAtomicOp`, `ScatterOOB`, or `ScatterConflict` template parameter. The implementation always walks `validRow × validCol` and writes `dst.data()[indexes.data()[idxOff]] = src.data()[srcOff]` — Elem semantics regardless of the index tile shape. Atomic is always plain replace; the last writer in row-major iteration order wins. Bounds are not enforced.

### A2/A3 Form

```cpp
template <Coalesce        CMode  = Coalesce::Row,
          ScatterAtomicOp AtomOp = ScatterAtomicOp::None,
          ScatterOOB      Oob    = ScatterOOB::Undefined,
          typename GlobalTable, typename TileSrc, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalTable& table, TileSrc& src, TileIdx& idx,
                              WaitEvents&... events);
```

A2/A3 has no `ScatterConflict` template parameter (the kernel is always sequential, so collisions are deterministic "last write wins"). Dispatched as a regular AIV function call — no async-launch or cross-core orchestration.

### A5 Form

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

The implementation launches one of `simt_mscatter_row_kernel`, `simt_mscatter_row_last_kernel`, `simt_mscatter_elem_kernel`, or `simt_mscatter_elem_last_kernel` through `cce::async_invoke<…>(cce::dim3{32, kLaunchWarps}, …)`. UB addressing inside the SIMT kernel goes through `tile_offset_2d<TileX>(r, c)`, which makes the kernel layout-agnostic for UB tiles. The degenerate Elem `(1, 1)` case bypasses the SIMT launch and runs `MScatterScalarImpl` on the AIV vector core.

### Parameters (NPU forms)

- `table` — destination GM `GlobalTensor`. `GlobalTensor::DType` must be `__gm__ T` matching the source element type.
- `src` — UB source tile (`TileType::Vec`); shape `[R, C]`.
- `idx` — UB index tile (`TileType::Vec`).
- `CMode` / `Mode` — `Coalesce` value (`Row` or `Elem`). First template parameter, so the operating mode is always explicit at the call site.
- `AtomOp` / `Atomic` — `ScatterAtomicOp` value. `Max` / `Min` are not supported on A2/A3.
- `Oob` — `ScatterOOB` value for out-of-bounds handling.
- `Conflict` (**A5 only**) — `ScatterConflict` value. Consulted only when `Atomic == None`.

### Enums

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // table[idx[r], :] = src[r, :]   (1-D index of length R)
    Elem = 1   // table[idx[i, j]] = src[i, j]   (idx shape == src shape)
};

enum class ScatterAtomicOp : uint8_t {
    None = 0,  // Plain store (collision-resolved by ScatterConflict on A5)
    Add  = 1,  // Atomic addition
    Max  = 2,  // Atomic maximum (A5 only)
    Min  = 3   // Atomic minimum (A5 only)
};

enum class ScatterOOB : uint8_t {
    Undefined = 0,
    Skip      = 1,
    Clamp     = 2,
    Wrap      = 3
};

enum class ScatterConflict : uint8_t {  // A5 only
    Last    = 0,  // Deterministic: largest source index wins
    Default = 1   // Warp-scheduler dependent
};
```

### Atomic Type Support

| Atomic | CPU (ABI contract / simulator behavior) | A2/A3 | A5 |
|--------|-----------------------------------------|-------|----|
| `None` | all dtypes | all dtypes | all dtypes |
| `Add`  | ABI contract: `int32_t`, `uint32_t`, `float`, `half` (simulator ignores the template parameter and runs plain replace with last-writer-wins) | `int8_t`, `int16_t`, `int32_t`, `half`, `bfloat16_t`, `float` (signed integers only — no `uint*`; MTE3 atomic-add unit in Row, scalar read-modify-write in Elem) | `int32_t`, `uint32_t`, `float`, `half`, `bfloat16_t` (SIMT `atomicAdd`) |
| `Max`  | ABI contract: `int32_t` or `float` (simulator runs plain replace) | unsupported (rejected at compile time by `MScatterCheck`) | `int32_t`, `uint32_t`, `float` (SIMT `atomicMax`) |
| `Min`  | ABI contract: `int32_t` or `float` (simulator runs plain replace) | unsupported (rejected at compile time by `MScatterCheck`) | `int32_t`, `uint32_t`, `float` (SIMT `atomicMin`) |

A2/A3 `Max` / `Min` would need a hardware atomic-max/min unit on the MTE3 path that the SoC does not provide; `MScatterCheck` static-asserts reject them. The CPU simulator's `MSCATTER` does not template on `ScatterAtomicOp` — the contract column above is what callers must honor at the source level so the same kernel still compiles and behaves correctly when the same source is built against A2/A3 or A5.

## Constraints

The constraints below are split into target-specific sections so each backend lists exactly what its implementation enforces. Symbols that are common across targets (`T = TileSrc::DType`, `TIdx = TileIdx::DType`) are reused.

### Tile Constraints (CPU)

**Supported data types:**

- `src` / `dst` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- On AICore targets (CPU simulator compiled with `__CCE_AICORE__`), `float8_e4m3_t` and `float8_e5m2_t` are also supported.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `src` must be a vector tile (`TileType::Vec`).
- `indexes` must be a vector tile (`TileType::Vec`).
- `src` and `indexes` must use row-major layout (`BLayout::RowMajor + SLayout::NoneBox`).
- `dst` must be a `GlobalTensor` in GM memory.
- `dst` must use `Layout::ND`.

**Atomic operation constraints:**

- Non-atomic scatter is supported for all supported element types.
- `ScatterAtomicOp::Add` mode requires `int32_t`, `uint32_t`, `float`, or `half`.
- `ScatterAtomicOp::Max` / `Min` mode requires `int32_t` or `float`.

**Shape constraints:**

- `src.Rows == indexes.Rows` (the index tile and source tile share the row count).
- `indexes` must be shaped as `[N, 1]` for row-indexed scatter or `[N, M]` for element-indexed scatter.
- `src` row width must be 32-byte aligned, that is, `src.Cols * sizeof(TileSrc::DType)` must be a multiple of 32.
- `dst` static shape must satisfy `Shape<1, 1, 1, TableRows, RowWidth>`.

**Index interpretation:**

- Index interpretation is target-defined. The CPU simulator treats indices as **linear element indices into `dst.data()`** and writes `dst.data()[idx] = src.data()[srcOff]` for every `(r, c)` in the source valid region — equivalent to `Coalesce::Elem` semantics, regardless of the index tile shape.
- The CPU simulator does not enforce bounds checks on `indexes`. Out-of-range indices write to whatever GM address `dst.data() + idx` resolves to and are target-defined.
- The simulator does **not** model `ScatterAtomicOp`, `ScatterOOB`, or `ScatterConflict` template parameters; the implementation always does plain replace, with "last writer in row-major iteration order wins" as the conflict policy.

**Header-level enforcement.** The CPU header (`include/pto/cpu/MGatherScatter.hpp`) static-asserts only the minimum closure rules: `std::is_integral_v<TileInd::DType>` for the index dtype and `sizeof(TileSrc::DType) == sizeof(GlobalData::DType)` for byte-wise compatibility. The dtype / shape / atomic / layout constraints above are the **PTO ABI contract** — callers are expected to honor them so the same kernel source compiles and runs unmodified against the A2/A3 and A5 backends.

### Tile Constraints (A2/A3)

**Supported data types:**

- `src` / `dst` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. No `float8_e4m3_t` / `float8_e5m2_t` / `hifloat8_t` on the A2/A3 vec-core.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `src` must be a vector tile (`TileSrc::Loc == TileType::Vec`).
- `indexes` must be a vector tile (`TileIdx::Loc == TileType::Vec`).
- The index tile is **always** `BLayout::RowMajor + SLayout::NoneBox` (ND), regardless of the table layout.
- `dst` must be a `GlobalTensor` in GM memory; `GlobalTable::DType == __gm__ T`.
- The source tile's bulk + sub layout must be paired with the table layout exactly:
    - `GlobalTable::layout == Layout::ND` ⇒ `TileSrc` is `BLayout::RowMajor + SLayout::NoneBox`.
    - `GlobalTable::layout == Layout::NZ` ⇒ `TileSrc` is `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == TileConfig::fractalABSize` (= 512 B).

**Atomic operation constraints:**

- `ScatterAtomicOp::None` is supported for all of the dtypes above.
- `ScatterAtomicOp::Add` requires `int8_t`, `int16_t`, `int32_t`, `half`, `bfloat16_t`, or `float`. Unsigned-integer atomic-add is not supported on A2/A3 (no `uint8_t` / `uint16_t` / `uint32_t`).
- `ScatterAtomicOp::Max` and `Min` are **not supported on A2/A3** — `MScatterCheck` rejects them at compile time.

**Shape constraints:**

- Padded `TileSrc::Cols * sizeof(T)` must be 32-byte aligned in both layouts (the same DMA-burst rule that `TLOAD` / `TSTORE` enforce). `ValidRow` / `ValidCol` are not constrained by this rule.
- For `Coalesce::Row`: `TileIdx::ValidRow == 1` and `TileIdx::ValidCol == TileSrc::ValidRow`.
- For `Coalesce::Elem`: `TileIdx::ValidRow == TileSrc::ValidRow` and `TileIdx::ValidCol == TileSrc::ValidCol`.
- Both modes require `TileSrc::ValidRow >= 1` and `TileSrc::ValidCol >= 1`.
- NZ tables additionally require `GlobalTable::staticShape[3] == FRACTAL_NZ_ROW` (= 16), `GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T)` (= 32 B / element width), `TileSrc::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0`, and `TileSrc::Rows % FRACTAL_NZ_ROW == 0`.

**Index interpretation:**

- For `Coalesce::Row`, each index is treated as a **logical row index** into the `[TableRows, RowWidth]` ND table (or as a logical row across `gShape2 * FRACTAL_NZ_ROW` for NZ).
- For `Coalesce::Elem`, each index is treated as a **linear element index** into the flat 5-D table (`tableSize = ∏ shape[0..4]`). For NZ the kernel splits the linear index into `(logicalRow = idx / nLogicalCols, logicalCol = idx % nLogicalCols)` and applies the NZ block-stride translation.
- OOB handling follows the `ScatterOOB` template parameter (`Undefined` / `Skip` / `Clamp` / `Wrap`); see the Out-of-Bounds subsection above.

### Tile Constraints (A5)

**Supported data types:**

- `src` / `dst` element type must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`. On `__CCE_AICORE__` builds the list also includes `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`.
- `indexes` element type must be `int32_t` or `uint32_t`.

**Tile and memory types:**

- `src` must be a vector tile (`TileSrc::Loc == TileType::Vec`).
- `indexes` must be a vector tile (`TileIdx::Loc == TileType::Vec`).
- The SIMT kernel is layout-agnostic for UB tiles: every UB read/write goes through `tile_offset_2d<TileX>(r, c)`, so `TileSrc` may be `BLayout::RowMajor` or `BLayout::ColMajor` (with `SLayout::NoneBox`).
- `dst` must be a `GlobalTensor` in GM memory; `GlobalTable::DType == __gm__ T`.
- **GM table layout: `Layout::ND` only.** A5 SIMT kernels address GM as a flat row-major buffer with row stride hard-wired to `validCols` (`dstRow = table + safeIdx * validCols`); `MScatterCheck` enforces `GlobalTable::staticShape[4] == TileSrc::ValidCol` so the table cannot have any inter-row padding. The kernel does not honor any other `GlobalTensor::layout` value — `MScatterCheck` does not static-assert the GM `layout` field, but using `Layout::NZ` (or anything other than packed ND with row width = `validCols`) produces wrong output because the kernel never performs block-stride translation. Callers that need NZ data must pre-stage into ND.

**Atomic operation constraints:**

- `ScatterAtomicOp::None` is supported for all of the dtypes above.
- `ScatterAtomicOp::Add` requires `int32_t`, `uint32_t`, `float`, `half`, or `bfloat16_t` (uses SIMT `atomicAdd`).
- `ScatterAtomicOp::Max` requires `int32_t`, `uint32_t`, or `float` (uses SIMT `atomicMax`).
- `ScatterAtomicOp::Min` requires `int32_t`, `uint32_t`, or `float` (uses SIMT `atomicMin`).
- `IsValidScatterAtomic<T, Atomic>::value` rejects unsupported combinations at compile time.

**Shape constraints:**

- Padded `TileSrc::Cols * sizeof(T)` (RowMajor) or `TileSrc::Rows * sizeof(T)` (ColMajor) must be 32-byte aligned — this is an upstream `TLOAD` / `TSTORE` requirement, not a `MSCATTER` constraint per se. `ValidRow` / `ValidCol` are not constrained by this rule.
- For `Coalesce::Row`: the index tile's valid shape is `[1, R]` (`BLayout::RowMajor`) **or** `[R, 1]` (`BLayout::ColMajor`). Both forms produce a linear R-element layout in UB and the kernel reads `idx[row]` directly. The choice of `BLayout` for the index tile is independent of the GM table layout — `MScatterCheck` does not constrain the GM `layout` field for the table.
- For `Coalesce::Elem`: `TileIdx::ValidRow == TileSrc::ValidRow` and `TileIdx::ValidCol == TileSrc::ValidCol`. The `BLayout` of `TileIdx` is independent of `TileSrc`'s — the kernel walks both through per-tile `tile_offset_2d`.
- Both modes require `TileSrc::ValidRow >= 1` and `TileSrc::ValidCol >= 1`. The degenerate `(1, 1)` shape in Elem mode bypasses the SIMT launch and runs `MScatterScalarImpl` on the AIV vector core.

**Index interpretation:**

- For `Coalesce::Row`, each index is treated as a **logical row index** into the `[TableRows, validCols]` ND table; the kernel computes `dstRow = table + safeIdx * validCols`.
- For `Coalesce::Elem`, each index is treated as a **linear element index** into the flat 5-D table (`tableSize = ∏ shape[0..4]`); the kernel writes `table[safeIdx] = src[srcOff]`.
- OOB handling follows the `ScatterOOB` template parameter (`Undefined` / `Skip` / `Clamp` / `Wrap`); see the Out-of-Bounds subsection above.
- Conflict handling follows the `ScatterConflict` template parameter (`Last` / `Default`) when `Atomic == None`; see the Conflict Resolution subsection above.

### Dynamic Runtime Shapes (A2/A3 and A5)

`MSCATTER` accepts both compile-time and runtime-dynamic shapes:

- `Tile<…, RowMask, ColMask>` with `RowMask == -1` and/or `ColMask == -1` stores the runtime valid extents in the tile object; the implementation reads them through `src.GetValidRow()` / `src.GetValidCol()`.
- `Shape<S0, S1, S2, S3, S4>` / `Stride<…>` with one or more `-1` entries are constructed with the runtime sizes; the implementation reads them through `table.GetShape(GlobalTensorDim::DIM_*)` and folds them into `tableRows` (Row mode) or `tableSize = ∏ shape[0..4]` (Elem mode).

Static-asserts in `MScatterCheck` are gated on `if constexpr (DIM > 0)`, so they fire only for compile-time-known dimensions. Padded `Tile::Rows / Cols` are always compile-time — they govern the UB DMA-burst alignment and the SIMT lane addressing.

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

## Mode Resolution

Mode is **explicit** on A2/A3 and A5, never auto-detected. Static asserts in `MScatterCheck` validate the supplied tile shapes against the chosen `Coalesce` value:

```text
A2/A3:
  Coalesce::Row  : Idx.ValidRow == 1 && Idx.ValidCol == Src.ValidRow
  Coalesce::Elem : Idx.ValidRow == Src.ValidRow && Idx.ValidCol == Src.ValidCol

A5:
  Coalesce::Row  : (Idx.ValidRow == 1 && Idx.ValidCol == Src.ValidRow) ||
                   (Idx.ValidRow == Src.ValidRow && Idx.ValidCol == 1)
  Coalesce::Elem : (Idx.ValidRow == Src.ValidRow) && (Idx.ValidCol == Src.ValidCol)
```

On the CPU reference path, no mode parameter exists; the implementation always walks element-wise.

## Layout Support

UB addressing is computed from each tile's `Rows` / `Cols` plus (on A2/A3 NZ) an optional fractal block-col stride; GM addressing is driven from `GlobalTensor::GetStride(DIM_*)`. The matrix below summarises what each backend accepts.

| Tile / Tensor | CPU | A2/A3 | A5 |
|---------------|-----|-------|----|
| `TileSrc` (UB) — ND | `BLayout::RowMajor + SLayout::NoneBox` only | `BLayout::RowMajor + SLayout::NoneBox` | `BLayout::RowMajor` or `ColMajor`, `SLayout::NoneBox` (Elem mode walks both via `tile_offset_2d`) |
| `TileSrc` (UB) — NZ | not supported | `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512` (paired with NZ GM table) | **not supported** |
| `TileIdx` (UB) — Row | row-major (Cols must equal 1 to match CPU's elementwise indexing) | `[1, R]` `BLayout::RowMajor + SLayout::NoneBox` (always ND, regardless of table layout) | `[1, R]` `RowMajor` **or** `[R, 1]` `ColMajor` (independent of GM table layout) |
| `TileIdx` (UB) — Elem | `BLayout::RowMajor + SLayout::NoneBox` | `[R, C]` `BLayout::RowMajor + SLayout::NoneBox` | any `BLayout`, independent of `TileSrc` (kernel reads through `tile_offset_2d<TileIdx>`) |
| `GlobalTable` (GM) — ND | `Layout::ND` only | `Layout::ND` (linear contiguous addressing); 5-D `Shape<…, R, C>`; kernel uses `tableRowStride = GetStride(DIM_3)`, so stride-padded ND tables are supported | `Layout::ND` only; Row mode hard-wires `tableRowStride = validCols` and requires `Shape[4] == validCols` |
| `GlobalTable` (GM) — NZ | not supported | `Layout::NZ`; 5-D `Shape<B, BCols, BRows, 16, C0>` with `staticShape[3] == 16` and `staticShape[4] == 32 / sizeof(T)` (`B` may be > 1; the kernel loops `for i in 0..gShape0`) | **not supported** |

### NZ Layout (A2/A3)

> **NZ paths exist only on A2/A3.** The A5 SIMT kernel addresses GM as a flat ND buffer and has no NZ block-stride translation — there is no "inherent SIMT NZ support" to be enabled. Callers that need NZ data on A5 must transpose / repack into ND first.

When `GlobalTable::layout == Layout::NZ` and `TileSrc` is the matching `BLayout::ColMajor + SLayout::RowMajor + SFractalSize = 512` tile, `MSCATTER` (A2/A3) runs the dedicated NZ paths (`MScatterRowNzImpl`, `MScatterElemNzImpl`).

- **Constants.** `kC0 = C0_SIZE_BYTE / sizeof(T) = 32 / sizeof(T)`; `kFRow = FRACTAL_NZ_ROW = 16`. Each fractal block is `kFRow × kC0` elements (= 32 B × 16 = 512 B).
- **Logical shape.** Logical rows = `gShape2 * kFRow`. Logical cols = `gShape0 * gShape1 * kC0`. Row-mode `mscatter_remap` clamps / wraps / skips against the logical row count; Elem-mode against the total element count.
- **Row mode.** For each logical source row `r`, the kernel maps `idx[r]` to `(dstBlockRow, dstRowInBlock)` and `r` to `(srcBlockRow, srcRowInBlock)`, then issues **one multi-burst MTE3 transfer per outer batch** (`nBurst = gShape1`, `lenBurst = kC0 * sizeof(T) = 32 B`, `ubGap = TileSrc::Rows - 1` blocks, `gmGap = (gStride1 - kC0) * sizeof(T)`). Atomic-add wraps the loop the same way as the ND path.
- **Elem mode.** For each `(r, c)` the kernel maps `idx` to `(logicalRow, logicalCol)` and through `MScatterNZGmOffset` to the NZ block-stride GM offset; the source UB offset is `(c / kC0) * (TileSrc::Rows * kC0) + r * kC0 + (c % kC0)`. The walk order is **block-col → row → col-in-block** so consecutive scalar reads always come from consecutive 32 B UB blocks. Atomic-add is implemented through scalar read-modify-write on the GM destination.
- **Stride vs. valid-shape.** `MScatter*NzImpl` reads strides from the `GlobalTensor` runtime, so packed and stride-padded NZ tensors both work without any caller-side adjustment.

### Why Elem Mode Uses Scalar GM Writes (A2/A3)

`copy_ubuf_to_gm_align_b8/b16/b32` requires the **UB source address** to be 32-byte aligned, and the source must be a whole number of 32-byte burst chunks. A per-element MTE3 burst of `lenBurst = sizeof(T)` from `srcPtr + r * RowStride + c` does not satisfy that rule whenever `(c * sizeof(T)) % 32 != 0`, which covers almost every elem-mode lane. Row mode does not hit this problem because each row read starts at `r * RowStride`, and `RowStride * sizeof(T)` is always a multiple of 32 bytes.

A2/A3 Elem mode therefore uses scalar UB→GM stores, which have element-level addressing granularity and place no alignment requirement on the source. Atomic add is implemented through scalar read-modify-write (`tableGm[idx] = tableGm[idx] + srcUb[r, c]`), which preserves the "last write wins" / "all writes accumulate" semantics on a single AICORE.

A5 does not face this constraint: the SIMT lane issues per-thread stores instead of a UB-source-aligned DMA burst, so Elem mode on A5 is naturally per-element with no extra alignment workaround.

## Aligned vs Unaligned Tile Shapes

The kernel does **not** care whether the tile's logical shape is "aligned" — it walks all `ValidRow * ValidCol` positions. The 32-byte alignment of the contiguous dim is enforced upstream by the `Tile` system because every `TLOAD` / `TSTORE` issues 32 B GM↔UB bursts.

- **Row mode (A2/A3 ND).** per-row DMA `lenBurst = validCol * sizeof(T)`; `Tile::RowStride * sizeof(T)` is forced 32-byte aligned by the `Tile` system, so subsequent rows always start on a 32 B burst boundary.
- **Row mode (A2/A3 NZ).** one multi-burst transfer per logical row × outer-batch; `lenBurst = kC0 * sizeof(T) = 32 B` is fixed by the fractal layout, so per-row alignment is automatic. `validRow` does not have to be a multiple of `kFRow`.
- **Elem mode (A2/A3).** one scalar UB→GM copy per element (replace) or scalar read-modify-write (atomic add). The scalar pipe has element-level addressing granularity, so any `(ValidRow, ValidCol)` inside the padded tile works.
- **Row + Elem mode (A5).** the SIMT kernel walks through `tile_offset_2d<TileX>(r, c)` so any `(ValidRow, ValidCol)` with `1 ≤ Valid ≤ Padded` is accepted, including the degenerate `(1, 1)` which routes to `MScatterScalarImpl`.

Callers handle "unaligned valid region" by padding the tile up to the nearest 32-byte alignment (for example valid `[3, 3]` int32 → tile `[3, 8]`), and either zero-initializing the padding or only inspecting the valid region after the scatter.

### Minimum Tile Shape

The minimum padded inner dim is set by the upstream `TLOAD` / `TSTORE` burst-alignment rule; `MSCATTER` itself accepts any `(ValidRow, ValidCol) >= (1, 1)`.

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

The implementation centralises every pipe handshake the kernel needs. **Callers do not need to insert any extra barriers** beyond the standard `TLOAD` post-load `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` pair that brings the source and index tiles into a clean state on the vector pipe before `MSCATTER`. The kernel never uses `pipe_barrier(PIPE_ALL)`.

| Phase | Pipe transition | What it guards |
|-------|-----------------|----------------|
| Pre-amble (Row, ND + NZ) | `V→S`, `MTE2→S`; if `Atomic == Add`, then `MScatterAtomicAddSet<T>()` (which calls one of `set_atomic_f32 / f16 / bf16 / s32 / s16 / s8()` followed by `set_atomic_add()`); finally `S→MTE3` | Make the source and index tiles visible to scalar reads; for atomic-add, switch the MTE3 unit into per-dtype atomic-add mode before any DMA is issued; the trailing `S→MTE3` makes the new atomic mode visible to the first MTE3 burst. |
| Pre-amble (Elem, ND + NZ) | `V→S`, `MTE3→S`, `MTE2→S` flag chain | Make the source and index tiles visible to scalar reads; the `MTE3→S` flush also makes any prior MTE3 writes visible before the scalar loop starts the per-element UB→GM (or read-modify-write) stores. |
| Body (Row, ND) | `copy_ubuf_to_gm_align_b*` per row | One DMA per row through `tablePtr + safeIdx * tableRowStride`, `lenBurst = validCol * sizeof(T)`; trip count = `validRow`. For non-atomic mode, a per-iteration `MTE3→S` flag is interleaved after each DMA so the next iteration sees the burst completion. |
| Body (Row, NZ) | `copy_ubuf_to_gm_align_b*` multi-burst per logical row × batch | `nBurst = gShape1`, `lenBurst = C0 * sizeof(T) = 32 B`, `ubGap = Tile::Rows - 1` blocks, `gmGap = (gStride1 - C0) * sizeof(T)`; trip count = `validRow * gShape0`. Same per-iteration `MTE3→S` flag for non-atomic mode. |
| Body (Elem, ND and NZ) | Scalar `tableGm[gmOff] = srcUb[r, c]` (or `+=`) per element | Per-element scalar UB→GM copy or read-modify-write; trip count = `validRow * validCol`. NZ walks block-col-major to keep each 32 B UB block read contiguously in time. |
| Atomic-add reset (Row, `Add` only) | `MTE3→S`, then `set_atomic_none()`, then `S→V`, `S→MTE2` | After the final atomic-add MTE3 burst drains, restore normal store semantics for downstream operators and publish the clean atomic state to V and MTE2. |
| Row post-amble (ND + NZ, all atomic modes) | `MTE3→V`, `MTE3→MTE2` flag chain | Drain the MTE3 DMAs before V or MTE2 consumers touch GM. |
| Elem post-amble (ND + NZ) | `S→V`, `S→MTE2`, `S→MTE3` flag chain | Make scalar GM writes visible to V (for downstream vector ops), MTE2 (for follow-up loads from the same table), and MTE3 (for follow-up stores or row-mode scatters). |

### A5 — SIMT launch with V↔S handshake

The A5 implementation hides almost the entire pipe model behind `cce::async_invoke`, which establishes the warp-scheduler context and orchestrates per-lane GM and UB accesses. The only explicit kernel-side handshake is in the scalar fallback path (`MScatterScalarImpl`, used for Elem `(1, 1)`):

| Phase | Pipe transition | What it guards |
|-------|-----------------|----------------|
| Pre-amble (scalar fallback) | `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` | Make the source and index tiles visible to the scalar pipe before the single-element scatter. |
| Body (SIMT Row / Elem) | `cce::async_invoke<simt_mscatter_*_kernel>(dim3{32, kLaunchWarps}, …)` | The SIMT launch handles all per-lane GM stores (and `atomicAdd / atomicMax / atomicMin` for atomic modes) internally; no caller-visible flags. |
| Post-amble (scalar fallback) | `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` | Release the scalar pipe back to vector ops. |

**Caller responsibility on A5.** The same `TLOAD` post-load `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` pair is required before `MSCATTER` so the source and index tiles are observable on the vector pipe by the time the SIMT launch begins.

### Cache-Coherence Flush Pattern (A5 wrappers)

`MSCATTER` does not insert a GM cache flush. Test wrappers in the A5 ST suite end their per-call function with:

```cpp
AICORE PTO_INLINE void FlushScatterOutput()
{
    dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
```

`dcci(0, ENTIRE_DATA_CACHE)` invalidates the AIV scalar D-cache so any buffered GM writes are pushed to HBM; `dsb(DSB_DDR)` blocks until the writes are observable at the DDR boundary. This is a **wrapper-level pattern** (not part of the `MSCATTER` intrinsic) — kernel authors should append it to their scatter wrapper when the host code reads back the GM table immediately after the kernel returns.

## UB Memory Budget

The unified buffer is shared between user tiles, the runtime reserved region, and the Data Cache. Budget rules differ per target.

### CPU simulator

No UB — the implementation runs in host memory. Tile sizes are bounded by the test harness only.

### A2/A3

The AIV vector core has the standard CANN 192 KB UB layout. `MSCATTER` does not allocate any UB scratch from inside the kernel — the only UB consumers are the caller-allocated source tile (`R * C * sizeof(T)`, padded up to the 32-byte burst alignment) and index tile (`R * C * sizeof(TIdx)`, same padding rule). A2/A3 has no `dynUBufSize` knob; the working set must fit in the caller's static UB budget.

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

When tiles are placed manually with `TASSIGN` (as in the A5 ST suite), the compiler sees `static_memory ≈ 0` and the full **216 KB** is available as `dynUBufSize`.

#### Default Per-Call Budget (No `dynUBufSize`)

When the kernel is launched without an explicit `dynUBufSize` (`<<<numBlocks, nullptr, stream>>>`), the runtime keeps the default DCache size and reserves only a small default dynamic region. In practice the safe `src + idx` working set is **≤ 128 KB**; beyond that, on-board execution may silently corrupt or zero out the result while still passing the CPU simulator (which does not model these reservations).

#### Extending Per-Call UB Beyond 128 KB

Callers that need a single-shot `src + idx` footprint larger than 128 KB must declare the dynamic-UB request explicitly through the second argument of the kernel launch:

```cpp
kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...);
```

`dynUBufSize` is the byte size of the dynamic-UB region the kernel will use. The bisheng/CCE compiler routes such launches through `__cce_rtKernelLaunchWithFlagV2`, setting `rtTaskCfgInfo_t::localMemorySize = dynUBufSize`. The runtime then shrinks the DCache toward its 32 KB minimum and hands the remaining space back to the kernel.

Key points:

- **The simulator does not enforce this.** Passing `nullptr` (or `0`) still runs to completion in sim regardless of the actual UB footprint. Always set `dynUBufSize` explicitly when the workload exceeds 128 KB so the binary stays correct on real hardware.
- **Exceeding the ceiling is silent.** The compiler does not error and the simulator does not flag it. On-board, the first overflow byte corrupts the reserved region or DCache and the kernel returns undefined output.
- **Size to actual usage.** For Elem coalesce with `R × C × sizeof(T)` source and `R × C × sizeof(int32_t)` index, the working set is `R * C * (sizeof(T) + 4)`. Round up to a comfortable margin when passing `dynUBufSize`. In the extended-UB ST cases (`float` source + `int32_t` index, `C = 8`) the per-element footprint is `8 + 4 = 12 B`; the suite rounds up and passes `R * 8 * 8 = R * 64` as `dynUBufSize` to keep the math simple.

#### Tiled-Iteration Pattern (Legacy 2048×8 Cases)

The `case_elem2d_float_2048x8_*` ST cases predate the `dynUBufSize` path and use a chunked approach instead: a `2048 × 8` `float` source is split into 16 chunks of `128 × 8` (8 KB src + 8 KB idx per iteration), and `MSCATTER` is reissued per chunk into the same destination GM tensor. Semantics are preserved:

- `Conflict::Last` — each chunk writes its in-chunk last-writer to GM; later chunks overwrite earlier ones for any shared slot, so the surviving value is the global largest-index writer.
- `Conflict::Default` / atomic modes — writes from later chunks compose with earlier ones (overwrite, add, max, min) on the same GM table.

New large-shape cases (`2304×8` and above) use the `dynUBufSize` single-shot path instead of chunking.

## Runtime Dispatch Requirement (A5)

`MSCATTER` on A5 uses `cce::async_invoke<simt_mscatter_*_kernel>(cce::dim3{32, kLaunchWarps}, …)` internally to fan a per-warp / per-lane workload out across up to 1024 threads. `async_invoke` consumes runtime state (TID registers, warp / lane configuration, vector-pipe scheduling) that the **launch path** must install before the kernel function is entered. The standard CANN launch (`rtKernelLaunchWithHandleV2`, used by the `<<<numBlocks, dynUBufSize, stream>>>` syntax) installs this state correctly.

A runtime variant that dispatches kernels as a direct C function-pointer call is fine for SPMD ops (`TLOAD`, `TSTORE`, `TADD`, …) but skips the SIMT context init, so the first `async_invoke` inside `MSCATTER` has no warp scheduler to dispatch into and hangs. Use the standard launch syntax for any SIMT kernel.

## Performance Considerations

### A2/A3

1. **Row vs. Elem.** Row coalesce achieves the best aggregate bandwidth — one wide DMA per logical row (ND) or one multi-burst DMA per logical row × batch (NZ). Elem coalesce issues one scalar UB read + GM write per active lane (plus an extra GM read for atomic add): no DMA-engine pipelining, throughput bound by scalar GM access latency. Prefer Row whenever the indexing structure permits.
2. **Sequential scalar loop (Elem).** A2/A3 dispatches `MSCATTER` as a single-thread sequential walk of the `validRow * validCol` lanes. The block-col-major walk used for NZ keeps consecutive reads spatially-local in UB.
3. **Why not per-element MTE3 in Elem mode.** See "Why Elem mode uses scalar GM writes" above — the UB-source 32 B alignment rule rules out per-element DMA bursts.
4. **DMA cost (Row).** ND: each row is one `copy_ubuf_to_gm_align_b*` call with `nBurst = 1`, `lenBurst = validCol * sizeof(T)`. NZ: each (logical row, batch) pair is one call with `nBurst = gShape1`, `lenBurst = C0 * sizeof(T) = 32 B`, `ubGap = Tile::Rows - 1` blocks, `gmGap = (gStride1 - C0) * sizeof(T)`. Back-pressure is bounded by `MAX_OUTSTANDING_MTE3`; no row chunking.
5. **OOB cost.** `Undefined` is free; `Skip` adds one branch per row / element; `Clamp` / `Wrap` add a single arithmetic remap per row / element.
6. **Atomic-add cost (Row).** One `set_atomic_add()` / `set_atomic_none()` pair per kernel invocation (~ 2 cycles each); the MTE3 atomic-add unit handles the accumulation on every burst. The atomic-add unit serialises same-address bursts across cores, so heavy hashing collisions degrade throughput predictably.
7. **Atomic-add cost (Elem).** One scalar `+=` per active lane (read GM → add → write GM). Same-core semantics match the MTE3 atomic-add unit on a single AICORE.
8. **Single-pass dispatch.** `MSCATTER` is a regular AIV function call from the kernel (no async-launch or cross-core orchestration). Concurrency comes from the DMA engine pipelining row DMAs behind the scalar issue loop, not from multiple worker threads.

### A5

1. **Shape-adaptive launch.** The SIMT grid is sized as `dim3{32, kLaunchWarps}` from the resolved `validRows` / `validCols` / `tableSize`. Small tiles do not pay the cost of 1024 idle threads.
    - **Row, non-`Last`.** `kRowWarps = min(validRows, 32)` warps own rows; `kWarpsPerRow = min(32 / kRowWarps, ceil(validCols / 32))` cooperate on each row's column chunks. When consecutive lanes in a warp write consecutive `col` values to the same `dstRow`, the SIMT hardware coalesces them into a 128 B GM burst.
    - **Elem, non-`Last`.** `kLaunchWarps = min(ceil(validRows*validCols / 32), 32)`.
    - **`Conflict::Last`.** Launch is sized by the destination instead of the source: for Row, by `tableRows`; for Elem, by `tableSize`. Each lane owns one slot and runs a reverse scan over the index tile (`last_owner_find_row` / `last_owner_find_elem`) to find the largest-flat-index writer for that slot.
2. **Conflict policy cost.**
    - `Last`: per-lane in-register reverse scan with early termination. No GM read-back, no atomic, no UB scratch. Worst-case `O(N)` per warp; uniformly random workloads average `O(tableSize / 32)`.
    - `Default`: zero extra work — the surviving lane is whatever the warp scheduler picked.
    - Atomic modes: serialised by the GM atomic R-M-W itself; no `cur` preload, no conflict gate.
3. **No thread divergence for mode / policy.** All policy decisions are `if constexpr`. In Row coalesce the `doWrite` predicate is warp-uniform (row-indexed) and hoisted out of the inner column loop. The slot-centric `Last` kernels compile their `found` predicate to a predicated store.
4. **Unrolled inner loops.** Inner column loop carries `#pragma unroll(4)`; outer scatter and reverse-scan loops use `#pragma unroll(1)` to keep code size bounded for large N.
5. **Row vs. Elem bandwidth.** Row coalesce achieves the best GM write bandwidth (per-warp 32 consecutive lanes per coalesced store). Elem coalesce is a per-lane scalar GM store, non-coalesced for random indices; UB reads remain coalesced because consecutive `tid` map to consecutive UB offsets.
6. **Register pressure.** Kernels carry `LAUNCH_BOUND(1024)` (32 regs/thread) and use ≤ 16 live registers in the hot path. No spills are produced.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src;
  IdxT idx;
  // dst is a GlobalTensor in GM
  MSCATTER(dst, src, idx);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(idx, 0x2000);
  MSCATTER(dst, src, idx);
}
```

### Row Coalesce — Embedding Scatter (A2/A3 or A5)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int R, int C, int TableRows>
AICORE void example_embedding_scatter(__gm__ T* tablePtr, __gm__ T* srcPtr, __gm__ int32_t* idxPtr)
{
    using SrcTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    // ... TLOAD src and idx, then handshake ...
    MSCATTER<Coalesce::Row, ScatterAtomicOp::None, ScatterOOB::Clamp>(tableGM, src, idx);
}
```

### Row Coalesce — Atomic-Add Aggregation

```cpp
template <typename T, int R, int C, int TableRows>
AICORE void example_row_atomic_add(__gm__ T* tablePtr, __gm__ T* srcPtr, __gm__ int32_t* idxPtr)
{
    using SrcTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::Add, ScatterOOB::Wrap>(tableGM, src, idx);
}
```

### Element Coalesce — Sparse Update

```cpp
AICORE void example_elem_sparse(__gm__ float* tablePtr, __gm__ float* srcPtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 32, TableSize = 256;

    using SrcTile     = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(tableGM, src, idx);
}
```

### Deterministic Last-Write-Wins (A5 only)

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

### Element Coalesce — `(1, 1)` Degenerate Case

```cpp
AICORE void example_scalar(__gm__ float* tablePtr, __gm__ float* srcPtr, __gm__ int32_t* idxPtr)
{
    constexpr int TableSize = 32;

    using SrcTile     = Tile<TileType::Vec, float,   1, 8, BLayout::RowMajor, 1, 1>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0080);

    MSCATTER<Coalesce::Elem>(tableGM, src, idx);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
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

## Related Instructions

- [`TSTORE`](TSTORE.md): contiguous block transfer Tile → GM.
- [`MGATHER`](./mgather.md): indexed gather GM → Tile (inverse operation).
- [`TSCATTER`](TSCATTER.md): index-based scatter within tiles (UB-to-UB on the same vec-core).
