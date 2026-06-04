# TTRANS


## Tile Operation Diagram

![TTRANS tile operation](../figures/isa/TTRANS.svg)

## Introduction

Transpose with an implementation-defined temporary tile.

## Math Interpretation

For a 2D tile, over the effective transpose domain:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

Exact shape/layout and the transpose domain depend on the target (see Constraints).

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
    - Source layout must be row-major (`TileDataSrc::isRowMajor`).
    - Element size must be `1`, `2`, or `4` bytes.
    - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
    - The transpose size is taken from `src.GetValidRow()` / `src.GetValidCol()`.
- **Implementation checks (A5)**:
    - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
    - 32-byte alignment constraints are enforced on the major dimension of both input and output (row-major checks `Cols * sizeof(T) % 32 == 0`, col-major checks `Rows * sizeof(T) % 32 == 0`).
    - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
    - The implementation operates over the static tile shape (`TileDataSrc::Rows/Cols`) and does not consult `GetValidRow/GetValidCol`.
- **Temporary tile**:
    - The C++ API requires `tmp`. The tmp space size calculation formulas are as follows:
    - **Basic parameters**:
        - RowStride: 32 for b8 types, 16 for b16/b32 types (corresponding to Y_ELEM_B8 and Y_ELEM_OTHER)
        - ElemPerBlock: 32/sizeof(T), i.e., number of elements per 32-byte block
        - b8: uint8_t/int8_t, b16: uint16_t/int16_t/half/bfloat16_t, b32: uint32_t/int32_t/float
    - **Alignment conditions**:
        - When stride meets alignment requirements (dstStride % RowStride == 0, srcStride % ElemPerBlock == 0, srcStride/ElemPerBlock <= 255), tmp is used for efficient transpose; otherwise, scalar copy is used without needing tmp.
    - **2D Tile transpose [H, W] -> [W, H]**:
        $$ \text{tmpSize} = W \times \lceil\frac{H}{\text{RowStride}}\rceil \times \text{RowStride} \times \text{sizeof(DType)} $$
        where W is the column count (validCol), H is the row count (validRow). tmpStride must be aligned to RowStride. tmp is needed only when stride meets alignment conditions.
    - **NCHW <-> NC1HWC0 bidirectional conversion**:
        - **Forward [N, C, H, W] -> [N, C1, H, W, C0]**:
        $$ \text{tmpSize} = H \times W \times \lceil\frac{C0}{\text{RowStride}}\rceil \times \text{RowStride} \times \text{sizeof(DType)} $$
        where C1 = (C + C0 - 1) / C0, transpose domain is C0 rows and H*W columns.
        - **Reverse [N, C1, H, W, C0] -> [N, C, H, W]**:
        $$ \text{tmpSize} = C0 \times \lceil\frac{H \times W}{\text{RowStride}}\rceil \times \text{RowStride} \times \text{sizeof(DType)} $$
        transpose domain is H*W rows and C0 columns.
    - **GNCHW <-> GNC1HWC0 bidirectional conversion**:
        - **Forward [G, N, C, H, W] -> [G, N, C1, H, W, C0]**:
        $$ \text{tmpSize} = H \times W \times \lceil\frac{C0}{\text{RowStride}}\rceil \times \text{RowStride} \times \text{sizeof(DType)} $$
        where C1 = (C + C0 - 1) / C0, transpose domain is C0 rows and H*W columns.
        - **Reverse [G, N, C1, H, W, C0] -> [G, N, C, H, W]**:
        $$ \text{tmpSize} = C0 \times \lceil\frac{H \times W}{\text{RowStride}}\rceil \times \text{RowStride} \times \text{sizeof(DType)} $$
        transpose domain is H*W rows and C0 columns.
    - **NC1HWC0 -> FRACTAL_Z and GNC1HWC0 -> FRACTAL_Z**:
        - These two conversions do not require tmp space, they directly execute memory reorganization operations.
    - **NCDHW to Fractal_Z_3D [N, C, D, H, W] -> [D, C1, H, W, N1, N0, C0]**:
        $$ \text{tmpSize} = (N \times C1 \times C0 \times H \times W + \max(N \times C1 \times C0 \times H \times W, H \times W \times \lceil\frac{C0}{\text{RowStride}}\rceil \times \text{RowStride})) \times \text{sizeof(DType)} $$
        where C1 = (C + C0 - 1) / C0, N1 = (N + N0 - 1) / N0. RowStride is 32 for 8-bit data and 16 for 16/32-bit data. This conversion has two stages with different execution paths: first stage extracts NCDHW d-plane to NCHW format (needs N*C1*C0*H*W space for planePtr), second stage either writes result to secondPtr (needs N*C1*C0*H*W) or uses secondPtr as transpose tmp (needs H*W*ceil(C0/RowStride)*RowStride). Since the path is chosen at runtime, secondPtr requires max of both sizes.
- **ConvTile**:
    - Transpose of ConvTile for `TileType::Vec` is supported。 Element size must be `1`、`2` or `4` bytes. Supported element types are `uint32_t`、`int32_t`、`float`、`uint16_t`、`int16_t`、`half`、`bfloat16_t`、`uint8_t`、`int8_t`.
    - Format transformation from `NCHW` to `NC1HWC0` is supported, while `C1 == (C + C0 - 1)/C0`，HW matches alignment constraint，which means `H*W*sizeof(T)==0`. C0 means `c0_size`, which `C0 * sizeof(T) == 32`。C0 can also be 4.
    - Format transformation from `NC1HWC0` to `FRACTAL_Z` is supported， while `N1 == (N + N0 - 1)/N0`。N0 should be 16.
    - Format transformation from `NCDHW` to `FRACTAL_Z_3D` is supported, with the destination shape `[D * C1 * H * W, N1, N0, C0]`, where `C1 == (C + C0 - 1)/C0` and `N1 == (N + N0 - 1)/N0`. `N0` is `16`. `C0` depends on element width: `64` for 4-bit data, `32` for 8-bit data, `16` for 16-bit data and `8` for 32-bit data. See the **Temporary tile** section above for the tmp size calculation formula.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TTRANS(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TTRANS(dst, src, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

