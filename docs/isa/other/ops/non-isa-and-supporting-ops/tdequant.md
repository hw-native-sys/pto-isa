# pto.tdequant

`pto.tdequant` is part of the [Non-ISA And Supporting Operations](../../non-isa-and-supporting-ops.md) instruction set.

## Summary

Dequantize an integer tile into a floating-point tile using per-row scale and zero-point tiles.

## Mechanism

`pto.tdequant` converts a quantized source tile back into floating-point form. In the current repo surface, the destination tile is floating-point, the source tile is integer, and the `scale` / zero-point tiles provide one parameter value per destination row. The current public intrinsic spells the zero-point operand as `offset`.

For each lane `(r, c)` in the destination valid region:

$$ \mathrm{dst}_{r,c} = \left(\mathrm{src}_{r,c} - \mathrm{offset}_r\right) \cdot \mathrm{scale}_r $$

The operation iterates over the destination valid region. `scale` and `offset` are broadcast across columns within a row.

## Syntax

### AS Level 1 (SSA)

```text
%dst = pto.tdequant %src, %scale, %offset : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tdequant ins(%src, %scale, %offset : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `dst` | Destination tile | Floating-point output tile |
| `src` | Source tile | Quantized integer tile |
| `scale` | Parameter tile | Per-row scale values broadcast across columns |
| `offset` | Parameter tile | Per-row zero-point values broadcast across columns; the current intrinsic names this operand `offset` |

## Expected Outputs

`dst` contains the dequantized floating-point values for all lanes in its valid region.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- `dst` and `src` must have matching valid-row and valid-column extents.
- `scale` and `offset` must have row counts matching `dst.GetValidRow()`.
- In this repo surface, `dst` is `float`/`float32_t`.
- In this repo surface, `src` is `int8_t` or `int16_t`.
- In this repo surface, `scale` and `offset` use the same floating-point element type as `dst`.
- Current concrete implementations require row-major tiles.

## Exceptions

- Unsupported type pairs, incompatible layouts, or mismatched valid regions are rejected by the backend checks.
- Programs must not assume column-wise scale or offset variation unless a target profile explicitly documents it.

## Target-Profile Restrictions

| Feature | CPU Simulator | A2/A3 | A5 |
|---------|:-------------:|:-----:|:--:|
| `float <- int8_t` | Yes | Yes | Yes |
| `float <- int16_t` | Yes | Yes | Yes |
| Row-major tiles | Yes | Yes | Yes |

This checkout contains concrete implementations for CPU simulation, A2/A3, and A5.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using DstTile = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  using SrcTile = Tile<TileType::Vec, int8_t, 64, 64, BLayout::RowMajor>;
  using ParaTile = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor>;

  DstTile dst;
  SrcTile src;
  ParaTile scale;
  ParaTile offset;

  TDEQUANT(dst, src, scale, offset);
}
```

## See Also

- [Non-ISA And Supporting Operations](../../non-isa-and-supporting-ops.md)
- [pto.tquant](../../../tile/ops/irregular-and-complex/tquant.md)
