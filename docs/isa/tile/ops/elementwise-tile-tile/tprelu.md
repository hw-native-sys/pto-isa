# TPRELU


## Tile Operation Diagram

![TPRELU tile operation](../../../../figures/isa/TPRELU.svg)

## Introduction

Elementwise PReLU (parametric ReLU) with a per-element slope tile.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}) $$

## Assembly Syntax

Synchronous form:

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- **Implementation checks (A2A3)**:
    - `dst`, `src0`, and `src1` element types must match. Supported types: `half`, `float`.
    - `tmp` element type must be `uint8_t` (used as comparison mask buffer).
    - All tiles must be row-major.
    - `src0` and `src1` valid shapes must match `dst`.
    - `tmp.GetValidRow() > dst.GetValidRow()` (tmp needs extra rows for mask storage).
    - In manual mode, `src0`, `src1`, `dst`, and `tmp` must not overlap in memory.
- **Implementation checks (A5)**:
    - `dst`, `src0`, and `src1` element types must match. Supported types: `half`, `float`.
    - All tiles must be row-major.
    - `src0` and `src1` valid shapes must match `dst`.

## Temporary Space

### A2A3

`tmp` **is used** as comparison mask storage. The A2A3 implementation decomposes PReLU as: `dst = src0 > 0 ? src0 : src0 * src1`, using `TCMPS` to write the comparison mask into `tmp`, then `TSEL` to select between `src0` and `dst` (= `src0 * src1`).

- `tmp` element type must be `uint8_t`.
- `tmp.GetValidRow() > dst.GetValidRow()` (the extra row region after the valid rows is used for the `TSEL` mask via `TSUBVIEW`).
- `tmp` must be row-major.
- In manual mode, `tmp` must not overlap with `dst`, `src0`, or `src1`.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses the `vprelu` vector instruction directly and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, slope, out, tmp;
  TPRELU(out, x, slope, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
