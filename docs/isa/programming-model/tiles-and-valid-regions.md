# Tiles And Valid Regions

Tiles are the primary payload objects in PTO. Most `pto.t*` semantics are defined over tiles, which is why tile shape, layout, location role, and valid-region metadata are architecture-visible.

Real kernels rarely fill an entire physical rectangle: edge tiles, partial blocks, and padding are normal. If the ISA pretends every element of the stored rectangle is meaningful, backends and authors disagree in silence. PTO instead carries **valid rows and columns** (`Rv`, `Cv`) so legality and semantics are defined on the meaningful domain first.

## Mechanism

### Tile Template Signature

A PTO tile is declared with the following template parameters:

```
Tile<TileType, DType, Rows, Cols, BLayout, SLayout, Fractal, Pad>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `TileType` | enum | Storage role: `Vec`, `Mat`, `Acc`, `Scalar`, `Left`, `Right` |
| `DType` | C++ type | Element type: `half`, `bfloat16_t`, `float`, `int8_t`, etc. |
| `Rows` | positive integer | Physical row count of the tile buffer |
| `Cols` | positive integer | Physical column count of the tile buffer |
| `BLayout` | enum | Block layout: `RowMajor` (C-contiguous) or `ColMajor` (Fortran-contiguous) |
| `SLayout` | enum | Stripe layout: `NoneBox` (uniform rectangular), `RowMajor` (fractal/strided), `ColMajor` (fractal/strided) |
| `Fractal` | enum | Fractal encoding: `None`, `NZ`, `ZN`, `FR`, `RN` (valid only when `SLayout != NoneBox`) |
| `Pad` | enum | Padding value for out-of-valid-region elements: `Zero`, `Null`, `Invalid` |

### TileType

Every tile buffer carries a `TileType` that determines which execution pipeline processes it:

| TileType | Pipeline | Typical Use |
|----------|----------|-------------|
| `Vec` | Vector Pipe (V) | General elementwise, unary, binary, reduce operations |
| `Mat` | Matrix/CUBE Pipe (M) | Matmul input operands (`TMATMUL`, `TGEMV`) |
| `Acc` | Matrix Pipe (accumulator) | Matmul output accumulator; may accumulate across iterations |
| `Scalar` | Scalar Unit | Scalar tile with `Rows = Cols = 1` |
| `Left` | Matrix Pipe | Left-hand operand of `TMATMUL_MX` (A matrix, must be `NZ` layout) |
| `Right` | Matrix Pipe | Right-hand operand of `TMATMUL_MX` (B matrix, must be `NN` layout) |

### Valid Region

The valid region is the architecture-visible statement of which elements are meaningful. It is expressed as a pair `(Rv, Cv)` — valid rows and valid columns — accessible at runtime via `tile.GetValidRow()` and `tile.GetValidCol()`.

**Semantics**: For any tile operation, `dst[i, j]` is defined if and only if `0 ≤ i < dst.Rv` and `0 ≤ j < dst.Cv`. Elements outside this domain have **no architectural meaning** unless a specific instruction page explicitly defines their behavior.

**Formula**:
```
Domain(dst) = { (i, j) | 0 ≤ i < dst.Rv  and  0 ≤ j < dst.Cv }
```

**Per-instruction iteration domain**: Unless a specific instruction states otherwise, the iteration domain is the **destination tile's valid region**:
```
for i in [0, dst.Rv):
    for j in [0, dst.Cv):
        dst[i, j] = f(src0[i, j], src1[i, j], ...)
```
For source tiles, `src[i, j]` is read regardless of whether `(i, j)` falls within the source's own valid region; the value read for out-of-region lanes is **target-specific and undefined**:
  - On A2/A3 and A5: The hardware may read whatever values happen to be in the underlying tile buffer SRAM; the content is not guaranteed to be stable across invocations or consistent with initialization
  - On CPU simulator: The underlying memory location is read directly, producing whatever bits are stored there

### Block Layout (BLayout)

`BLayout` describes the in-memory stride between adjacent elements in the row and column directions. Full reference: [Layout Reference](../state-and-types/layout.md).

| BLayout | Stride in Row Direction | Stride in Col Direction |
|---------|------------------------|------------------------|
| `RowMajor` (default) | `Cols` (elements per row) | `1` (contiguous in memory) |
| `ColMajor` | `1` (strided) | `Rows` (elements per column) |

`RowMajor` is the CPU/GPU conventional layout (row 0 is contiguous in memory). `ColMajor` is the Fortran/matrix-convention layout (column 0 is contiguous).

### Stripe Layout (SLayout)

`SLayout` describes whether the tile's sub-elements use a uniform rectangular layout or a fractal/strided layout:

| SLayout | Description | Requires |
|---------|-------------|----------|
| `NoneBox` | Uniform rectangular tile: all elements equally spaced | Default for most ops |
| `RowMajor` | Strided/fractal row layout | Fractal encoding (`NZ`, `FR`) |
| `ColMajor` | Strided/fractal column layout | Fractal encoding (`ZN`, `RN`) |

### Fractal Layout

When `SLayout != NoneBox`, the `Fractal` parameter encodes the striding pattern for matrix multiplication or other strided access patterns:

| Fractal | Layout | Typical Use |
|---------|--------|-------------|
| `None` | Not fractal (standard rectangular tile) | Elementwise ops, general compute |
| `NZ` | Row-major fractal (`NZ` = Z-order row-major) | LHS matmul operand on A5; `SLayout::RowMajor` |
| `ZN` | Column-major fractal | Symmetric variant of NZ |
| `FR` | Row-fractal | CUBE-specific strided pattern |
| `RN` | Row-N-fractal | CUBE-specific strided pattern |

### Layout Combinations by TileType

| TileType | Supported BLayout | Supported SLayout | Supported Fractal | Typical Ops |
|----------|------------------|-------------------|-------------------|-------------|
| `Vec` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TADD`, `TMUL`, `TCVT`, `TLOAD/TSTORE` |
| `Mat` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` |
| `Acc` | `RowMajor`, `ColMajor` | `NoneBox` | `None` | `TMATMUL`, `TMATMUL_ACC` output |
| `Left` | `RowMajor` | `RowMajor` | `NZ` | LHS of `TMATMUL_MX` |
| `Right` | `RowMajor` | `NoneBox` | `NN` (implicit) | RHS of `TMATMUL_MX` |
| `Scalar` | `RowMajor` | `NoneBox` | `None` | Single-element scalar tiles |

### Padding

Elements outside the valid region may be initialized with a padding value. The `Pad` parameter controls this:

| Pad Value | Meaning |
|-----------|---------|
| `Zero` | Out-of-valid-region elements are initialized to zero |
| `Null` | Out-of-valid-region elements are undefined; must not be read |
| `Invalid` | Elements are marked invalid; reading is undefined |

## Compact Mode

When a tile's physical dimensions exceed the valid region (common at matrix edges), compact mode determines how padding elements are handled. This is especially important for matmul and `TEXTRACT`/`TINSERT` operations.

### Compact Mode in TEXTRACT

`TEXTRACT` supports four compact modes for layout conversion between normal and fractal tiles:

| Mode | Description |
|------|-------------|
| `ND2NZ` | Normal row-major → NZ fractal. Valid data is packed contiguously in Z-order; padding is excluded. |
| `NZ2ND` | NZ fractal → Normal row-major. Valid data is unpacked from Z-order to row-major. |
| `ND` | Straight copy, no layout transformation. |
| `ND2NZ2` | Like `ND2NZ` but groups rows in blocks of 2 for specific CUBE access patterns. |

### Compact Mode in TMATMUL_MX

For MX-format matmul, the Left tile uses NZ fractal layout with compact addressing. When `M % tile_M ≠ 0` or `N % tile_N ≠ 0`, the fractal address generator produces addresses only for valid rows, excluding padding from CUBE processing.

## Inputs

The programming model expects the author or the frontend to supply:

- tiles with a legal type and layout combination (see layout combinations table above)
- valid-row and valid-column information when edge tiles or partial tiles exist
- instruction operands whose tile roles make sense together (e.g., `Left` + `Right` → `Acc` for matmul)

## Expected Outputs

Tile-producing operations yield a destination tile whose payload, valid region, and legality are defined by the selected instruction set and the interaction of the input tiles. The destination `TileType` and layout must be compatible with the instruction.

## Constraints

- Semantics are defined only inside the declared valid region unless an instruction page says otherwise
- Multi-input tile instructions iterate over the **destination** valid region, reading source tiles lane-by-lane at the corresponding indices regardless of the source's own valid region (hardware-specific, undefined values for out-of-region source lanes on A2/A3 and A5; raw memory reads on CPU simulator)
- A legal tile type is not enough by itself; shape, layout, location intent, and target profile also matter
- The combination of `TileType`, `BLayout`, `SLayout`, and `Fractal` MUST match one of the supported combinations in the layout table above

## Cases That Are Not Allowed

- Treating out-of-valid-region elements as architecturally meaningful data
- Assuming every backend will silently repair mismatched valid-region use
- Using tile roles or layouts that an instruction set or target profile does not permit
- Relying on any specific value from a source tile lane outside its valid region (undefined on A2/A3 and A5; raw memory on CPU simulator)

## Examples

### Example 1: Edge Tile with Partial Valid Region

An edge tile may have a physical shape of `16 x 16` while only `5 x 9` values are valid:

```cpp
using EdgeTile = Tile<TileType::Vec, half, 16, 16, RowMajor, NoneBox, None, Zero>;
EdgeTile tile;
tile.SetValidRegion(5, 9);
// Only tile[0..4][0..8] is architecturally meaningful
```

### Example 2: Matmul Tile Roles

```cpp
using A = Tile<TileType::Left, int8_t, 16, 16, RowMajor, RowMajor, NZ, Null>;
using B = Tile<TileType::Right, int8_t, 16, 16, RowMajor, NoneBox, NN, Null>;
using C = Tile<TileType::Acc, int32_t, 16, 16, RowMajor, NoneBox, None, Zero>;
A a; B b; C c;
TMATMUL(c, a, b);  // c[i,j] = sum_k a[i,k] * b[k,j]
```

### Example 3: Elementwise Operation Over Valid Region

```cpp
// TADD iterates over dst's valid region:
// for i in [0, dst.Rv), for j in [0, dst.Cv):
//     dst[i,j] = src0[i,j] + src1[i,j]
Tile<TileType::Vec, float, 16, 16> dst, src0, src1;
dst.SetValidRegion(8, 8);
// Only dst[0..7][0..7], src0[0..7][0..7], src1[0..7][0..7] participate
TADD(dst, src0, src1);
```

## See Also

- [Introduction: what PTO is](../introduction/what-is-pto-visa.md)
- [GlobalTensor And Data Movement](./globaltensor-and-data-movement.md)
- [Type System](../state-and-types/type-system.md)
- [Layout Reference](../state-and-types/layout.md)
- [Tile Instruction Set](../instruction-families/tile-families.md)
