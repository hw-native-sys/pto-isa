# TADDReluConv


## Tile Operation Diagram

![TADDReluConv tile operation](../figures/isa/TADDReluConv.svg)

## Introduction

Fused elementwise add, ReLU clamp, and type conversion. Per element: `dst = convert(max(0, src0 + src1))`.

At the ISA level, this is a single fused instruction (TADDRELUCONV): add + ReLU clamp + down-cast in one semantic step. Backend realization is architecture-dependent (for example, A2A3 can use a single fused intrinsic, while A5 lowers to an equivalent VF sequence), but user-visible semantics are identical.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{convert}\!\left(\max\!\left(0,\;\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j}\right)\right) $$

where `convert` narrows the result from the source type to the destination type with saturating behavior. For floating-point down-casts, rounding follows round-to-nearest-even.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = taddreluconv %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.taddreluconv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.taddreluconv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADDRELUCONV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- **Source type pair**: `src0` and `src1` must have the same element type (`ST`).
- **Supported source -> destination type pairs**:
    - `float` -> `half`
    - `half` -> `int8_t`
    - `int16_t` -> `int8_t`
- **Layout**: All tiles must be row-major (`TileData::isRowMajor`).
- **Location**: All tiles must live in `TileType::Vec`.
- **Valid region**: `validRow > 0` and `validCol > 0`; `src0` and `src1` valid shapes must match `dst` valid shapes.
- **Implementation notes (A2A3)**: Uses the single `vaddreluconv_*` hardware intrinsic (e.g. `vaddreluconv_f322f16`, `vaddreluconv_f162s8`, `vaddreluconv_s162s8`) for the fused add + relu + convert.
- **Implementation notes (A5)**: No single `vaddreluconv_*` intrinsic; the fusion is expressed in the VF register-compute model: `vlds(src0); vlds(src1); -> add -> maxs(0) -> vcvt -> vsts`. For `int16_t -> int8_t`, A5 has no s16->s8 vcvt, so the (already non-negative) sum is clamped to `[0, 127]` and narrowed via s16->u8 vcvt; for values in `[0, 127]` the `uint8_t` and `int8_t` byte patterns are equal.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcTileT = Tile<TileType::Vec, float, 16, 16>;
  using DstTileT = Tile<TileType::Vec, half, 16, 16>;
  SrcTileT src0, src1;
  DstTileT dst;
  TADDRELUCONV(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcTileT = Tile<TileType::Vec, float, 16, 16>;
  using DstTileT = Tile<TileType::Vec, half, 16, 16>;
  SrcTileT src0, src1;
  DstTileT dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TADDRELUCONV(dst, src0, src1);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.taddreluconv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.taddreluconv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = taddreluconv %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.taddreluconv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
