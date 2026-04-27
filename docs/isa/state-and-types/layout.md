# Layout Reference

**BLayout**, **SLayout**, **Fractal Layout**, **GlobalTensor Layout**, and **Compact Mode** are the canonical PTO layout references. For programming-model context and valid-region semantics, see [Tiles and Valid Regions](../programming-model/tiles-and-valid-regions.md).

## Two Layout Dimensions

PTO layouts operate at two levels:

1. **GlobalTensor Layout** — how the `GlobalTensor` (GM view) is laid out in off-chip memory. This is the `Layout::ND` / `Layout::DN` / `Layout::NZ` template parameter on `GlobalTensor`.
2. **Tile Layout** — how the tile buffer (UB or TRF) is organized internally. This is the combination of `BLayout`, `SLayout`, and `Fractal` on a `Tile<...>`.

These two levels must be compatible when a `GlobalTensor` is loaded into a tile via `TLOAD`, and when a tile is stored to a `GlobalTensor` via `TSTORE`.

## GlobalTensor Layout (GM View)

The `GlobalTensor` is a view over off-chip GM. Its layout parameter determines the stride pattern in GM:

| Layout | Stride Pattern | Description | Use Case |
|--------|---------------|-------------|----------|
| `Layout::ND` | Row-major, C-contiguous | `stride[R] = Cols, stride[W] = Cols*Width, ...` | Standard row-major tensors |
| `Layout::DN` | Column-major, Fortran-contiguous | `stride[C] = Rows, stride[R] = Rows*Col, ...` | Column-major tensors |
| `Layout::NZ` | Row-major fractal (Z-order) | GM data is stored in Z-order for fractal tile compatibility | A5 matmul LHS with NZ layout |

The GM layout must be compatible with the tile's internal layout during `TLOAD`/`TSTORE`. The compatibility rules are documented on the [TLOAD](../tile/ops/memory-and-data-movement/tload.md) and [TSTORE](../tile/ops/memory-and-data-movement/tstore.md) instruction pages.

## Block Layout (BLayout)

`BLayout` describes the in-memory stride between adjacent elements along the row and column axes within a tile buffer. It is the tile's **storage order** — the order in which element data is laid out in the local tile buffer selected by the tile role. For `TileType::Vec`, that local tile buffer is the hardware Unified Buffer.

### Values

| BLayout | Row-Direction Stride | Col-Direction Stride | Mental Model |
|---------|---------------------|---------------------|--------------|
| `RowMajor` | `Cols` (elements per row) | `1` (contiguous) | C/C++/PyTorch convention |
| `ColMajor` | `1` (strided) | `Rows` (elements per column) | Fortran/Julia convention |

For a `RowMajor` tile of shape `(R, C)`, element `(r, c)` is at byte offset:

$$ \mathrm{offset}(r, c) = (r \times C + c) \times \mathrm{sizeof(DType)} $$

For a `ColMajor` tile of shape `(R, C)`:

$$ \mathrm{offset}(r, c) = (c \times R + r) \times \mathrm{sizeof(DType)} $$

### Usage

`RowMajor` is the default for most operations. `ColMajor` is accepted by a subset of operations on A5; consult per-op Target-Profile Restrictions.

## Stripe Layout (SLayout)

`SLayout` describes whether the tile's sub-elements use a **uniform rectangular layout** or a **fractal/strided layout**. It controls whether the tile is addressed as a flat 2D rectangle or with a strided access pattern.

### Values

| SLayout | Description | Requires |
|---------|-------------|----------|
| `NoneBox` | Uniform rectangular tile: all `(Rows, Cols)` elements are equally spaced | Default for most operations |
| `RowMajor` | Strided row layout: addresses elements with a row-major stride pattern | `Fractal ∈ {NZ, FR}` |
| `ColMajor` | Strided column layout: addresses elements with a column-major stride pattern | `Fractal ∈ {ZN, RN}` |

When `SLayout = NoneBox`, the tile behaves as a standard rectangular buffer. When `SLayout ∈ {RowMajor, ColMajor}`, the `Fractal` parameter further specifies the stride formula.

## Fractal Layout

When `SLayout ≠ NoneBox`, the `Fractal` parameter encodes the precise striding pattern for matrix multiplication or other strided-access patterns. Fractal layouts are designed to match the CUBE engine's internal dataflow for high-performance matmul.

### Fractal Address Formula

Fractal layouts use a **Z-order (Morton code)** stride pattern. Elements are not stored in simple row-major or column-major order; instead, they follow a space-filling curve that improves data reuse in the CUBE engine.

For `Fractal = NZ` with `SLayout = RowMajor`:

$$ \mathrm{offset}(r, c) = \bigl(\mathrm{zigzag\_index}(r, c)\bigr) \times \mathrm{sizeof(DType)} $$

The zigzag index maps 2D coordinates to a 1D Z-order sequence. The mapping is hardware-defined; PTO authors should not compute fractal offsets manually — rely on the frontend to handle address generation via `TASSIGN`.

### Fractal Layout Values

| Fractal | SLayout | BLayout | Stride Pattern | Typical Use |
|---------|---------|---------|----------------|-------------|
| `None` | `NoneBox` | Any | Standard rectangular | Elementwise ops, general compute |
| `NZ` | `RowMajor` | `ColMajor` | Z-order row-major fractal | LHS matmul operand on A5 |
| `ZN` | `ColMajor` | `RowMajor` | Z-order col-major fractal | Symmetric variant of `NZ` |
| `FR` | `RowMajor` | `ColMajor` | Row-fractal (fixed-stride variant) | CUBE-specific pattern |
| `RN` | `ColMajor` | `RowMajor` | Row-N-fractal | CUBE-specific pattern |

> **Note for A5/A2/A3:** The exact fractal block dimensions are `FRACTAL_NZ_ROW = 16` (elements per fractal row) and `CUBE_BLOCK_SIZE = 512` (bytes per fractal block). These affect address generation in hardware but are not part of the ISA contract for authors.

## Compact Mode

Compact mode (also called **tail/part mode**) handles edge tiles where the physical tile dimensions are larger than the valid region. When a matrix dimension is not evenly divisible by the tile size, padding is added to fill the physical tile, and compact mode determines how that padding is managed.

### Why Compact Mode Matters

In matmul, when `M % tile_M ≠ 0` or `N % tile_N ≠ 0`, the last tile in each row/column has fewer valid elements. Compact mode controls:

1. Whether padding elements are included in the matmul computation
2. Whether the fractal layout addresses only valid elements or skips over padding
3. How `TEXTRACT` and `TINSERT` handle partial tiles

### Compact Mode in TEXTRACT

`TEXTRACT` supports four compact modes that control how the extracted data is arranged:

| Mode | Description | Behavior |
|------|-------------|----------|
| `ND2NZ` | Normal → NZ fractal | Extract from normal row-major tile into NZ fractal tile. Padding rows/cols are skipped; valid data is packed contiguously in Z-order. |
| `NZ2ND` | NZ fractal → Normal | Extract from NZ fractal tile back to normal row-major tile. Valid data is unpacked from Z-order to row-major. |
| `ND` | Normal → Normal | Straight copy, no layout transformation. |
| `ND2NZ2` | Normal → NZ (row-major group) | Like `ND2NZ` but groups rows in blocks of 2 for specific CUBE access patterns. |

The A2/A3 compact test (`textract_compact`) validates all four modes with edge-case tile dimensions (where `baseM`, `baseN`, or `baseK` are non-zero).

### Compact Mode in TMATMUL_MX

For MX-format matmul (`TMATMUL_MX`), the Left tile (`TileType::Left`) with `NZ` fractal layout uses compact addressing. When the LHS matrix has fewer rows than the tile's physical height, the fractal address generator only produces addresses for valid rows. Padding rows are excluded from both address computation and CUBE processing.

### Compact Addressing in A5 TMov

The A5 `TMovmx` operation with `ZZ` layout (`NZZN` / `ZZNN` variants) uses compact addressing when tile dimensions exceed the valid region. The test `tmov_mx` with `base_m != 0` validates this behavior.

### When to Use Compact Mode

- Use `ND2NZ` / `NZ2ND` when transferring data between normal and fractal layouts across non-multiple tile boundaries
- Use `ND` for same-layout transfers (no transformation overhead)
- Use `ND2NZ2` when the CUBE requires row-grouped data alignment
- Compact mode is **automatic** in matmul when valid region < physical tile size; the fractal address generator handles it transparently

## TileType–Layout Compatibility Matrix

The combination of `TileType`, `BLayout`, `SLayout`, and `Fractal` is **jointly constrained**. Not all nine-parameter combinations are legal.

| TileType | Supported BLayout | Supported SLayout | Supported Fractal | Typical Ops |
|----------|------------------|-------------------|-------------------|-------------|
| `Vec` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TADD`, `TMUL`, `TCVT`, `TLOAD/TSTORE` |
| `Mat` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` |
| `Acc` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TMATMUL`, `TMATMUL_ACC` output |
| `Left` | `RowMajor` | `RowMajor` | `NZ` | LHS of `TMATMUL_MX` |
| `Right` | backend-specific | backend-specific | backend-specific | L0B-backed RHS matmul operand; do not assume one portable right-tile layout across A2A3 and A5 |
| `ScaleLeft` | `RowMajor` | `RowMajor` | MX scale fractal | L0A-side scale tile for MX block-scale formats |
| `ScaleRight` | `ColMajor` | `ColMajor` | MX scale fractal | L0B-side scale tile for MX block-scale formats |
| `Scalar` | `RowMajor` | `NoneBox` | `None` | Single-element scalar tiles |

Using a combination not listed in this table is an **illegal PTO program**. The verifier or backend will reject it.

For matrix roles, treat the role-to-buffer mapping as primary:

- `Left` means the L0A-backed operand
- `Right` means the L0B-backed operand
- `ScaleLeft` / `ScaleRight` are the extra L0A/L0B scale tiles required by MX block-scale formats

The exact `Right`-tile layout contract is backend-sensitive. The current manual should not describe A2A3 and A5 right tiles as one interchangeable layout rule; use the relevant per-op target-profile restrictions or the typed aliases from `include/pto/common/pto_tile.hpp`.

## Padding

Elements outside the valid region may be initialized with a padding value. The `Pad` parameter controls this:

| Pad Value | Meaning |
|-----------|---------|
| `Zero` | Out-of-valid-region elements are initialized to zero |
| `Null` | Out-of-valid-region elements are undefined; must not be read |
| `Invalid` | Elements are marked invalid; reading is undefined behavior |

Custom pad values on A5: `PadValueCustom(value)` allows compile-time-specified float patterns as pad values (e.g., `-1.0f` for softmax masking).

## Layout Conversion Patterns

### Normal → Fractal (TEXTRACT with ND2NZ)

```cpp
// Extract from normal Vec tile to Left tile with NZ fractal layout
using SrcTile = Tile<TileType::Vec, int8_t, 16, 16, RowMajor, NoneBox, None, Null>;
using DstTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
TEXTRACT(dstLeft, srcVec, ExtractMode::ND2NZ);
```

### Fractal → Normal (TINSERT with NZ2ND)

```cpp
// Insert from Left tile with NZ fractal back to normal Vec tile
using SrcTile = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
using DstTile = Tile<TileType::Vec, int8_t, 16, 16, RowMajor, NoneBox, None, Null>;
TINSERT(dstVec, srcLeft, InsertMode::NZ2ND);
```

## Constants Reference

| Constant | Value | Units | Use |
|----------|-------|-------|-----|
| `BLOCK_BYTE_SIZE` | 32 | bytes | DMA block transfer unit |
| `FIXP_BURST_UNIT_LEN` | 64 | half-words | DMA burst length |
| `FRACTAL_NZ_ROW` | 16 | elements | Fractal row dimension for NZ/ZN |
| `CUBE_BLOCK_SIZE` | 512 | bytes | CUBE fractal block |
| `MX_COL_LEN` | 2 | elements | MX matmul column block |
| `MX_ROW_LEN` | 16 | elements | MX matmul row block |
| `MX_BLOCK_SIZE` | 32 | elements | MX matmul block |

## See Also

- [Tiles and Valid Regions](../programming-model/tiles-and-valid-regions.md) — Programming model context, valid-region semantics
- [Element Types and SSA Names](./type-system.md) — Complete element type inventory
- [Tile Buffer SSA Type](./type-system.md#tile-buffer-types) — `!pto.tile<...>` vs `!pto.tile_buf<...>`
- [TEXTRACT](../tile/ops/layout-and-rearrangement/textract.md) — Layout conversion with compact mode
- [TINSERT](../tile/ops/layout-and-rearrangement/tinsert.md) — Layout conversion with compact mode
- [Tile Instruction Set](../instruction-families/tile-families.md) — How layouts interact with tile operations
