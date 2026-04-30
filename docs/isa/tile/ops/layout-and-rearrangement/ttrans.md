# pto.ttrans

`pto.ttrans` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Transpose with a temporary tile whose allocation and usage depend on the target. On A2/A3 the transpose is performed in-place using the scratchpad as staging; on A5 the operation requires an explicit `tmp` tile passed via the C++ API because the A5 DMA engine cannot perform a true in-place transpose and needs a scratch buffer of the same shape as the source. On the CPU simulator the `tmp` tile is not used but must still be provided.

## Mechanism

Transpose with a temporary tile whose allocation and usage depend on the target. On A2/A3 the transpose is performed in-place using the scratchpad as staging; on A5 the operation requires an explicit `tmp` tile passed via the C++ API because the A5 DMA engine cannot perform a true in-place transpose and needs a scratch buffer of the same shape as the source. On the CPU simulator the `tmp` tile is not used but must still be provided. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For a 2D tile, over the effective transpose domain:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

Exact shape/layout and the transpose domain depend on the target (see Constraints).

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

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

## Inputs

- `src` is the source tile.
- `tmp` is a temporary tile used during transpose (may not be used by all implementations).
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the transposed version of `src`: `dst[i,j]` = `src[j,i]`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Temporary tile**:
        - The C++ API requires `tmp`, but some implementations may not use it.

    - **ConvTile**:
        - Transpose of ConvTile for `TileType::Vec` is supported。 Element size must be `1`、`2` or `4` bytes. Supported element types are `uint32_t`、`int32_t`、`float`、`uint16_t`、`int16_t`、`half`、`bfloat16_t`、`uint8_t`、`int8_t`.
        - Format transformation from `NCHW` to `NC1HWC0` is supported, while `C1 == (C + C0 - 1)/C0`，HW matches alignment constraint，which means `H*W*sizeof(T)==0`. C0 means `c0_size`, which `C0 * sizeof(T) == 32`。C0 can also be 4.
        - Format transformation from `NC1HWC0` to `FRACTAL_Z` is supported， while `N1 == (N + N0 - 1)/N0`。N0 should be 16.

## Performance

### A2/A3 Cycle Count

`pto.ttrans` performs a transpose via the data-rearrangement path. On the vector pipe this lowers to a `vtranspose` sequence operating on 16×16 (FP16) or 8×8 (FP32) sub-blocks; the full tile is built by tiling these blocks.

**Cycle model**: `total ≈ startup + ⌈R/B⌉ × ⌈C/B⌉ × per_block_transpose`, where `B = 16` (FP16) or `B = 8` (FP32).

### Layout and Shape Impact

| Source/Dest layout | Effect |
|---|---|
| RowMajor → ColMajor | Native fast path |
| ColMajor → RowMajor | Native fast path |
| Same major axis | Forbidden (verifier rejects) |

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
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

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
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

## See Also

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.treshape](./treshape.md)
- Next op in instruction set: [pto.tconcat](./tconcat.md)

