<!-- Generated from `docs/isa/tile/ops/irregular-and-complex/tpartmin.md` -->

# pto.tpartmin

Standalone reference page for `pto.tpartmin`. This page belongs to the [Irregular And Complex](../../irregular-and-complex.md) family in the PTO ISA manual.

## Summary

Partial elementwise min with implementation-defined handling of mismatched valid regions.

## Mechanism

Performs elementwise minimum selection over the destination valid region. When both `src0` and `src1` are valid at an element, the result is `min(src0, src1)`; when only one input is valid there, the result copies that input value. Handling of other mismatched-validity cases is implementation-defined. It belongs to the tile surface and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For each element `(i, j)` in the destination valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tpartmin %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile.
- `src1` is the second source tile.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the elementwise partial minimum: both valid gives min; one valid gives the valid value.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `dst`, `src0`, and `src1` must use the same element type.

- The destination valid region defines the result domain.

- For each element in the destination valid region:
  - if both inputs are valid, the instruction applies the elementwise minimum;
  - if only one input is valid, the result copies that input value.

- If `dst` has a zero valid region, the instruction returns early.

- Supported partial-validity patterns require at least one source tile to have a valid region exactly equal to `dst`, while the other source tile's valid region must not exceed `dst` in either dimension.

- Supported element types: `int32_t`, `int16_t`, `half`, `float`.

- Supported element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- Handling of any validity pattern not explicitly listed above is implementation-defined.

### A2A3 implementation checks

- `dst`, `src0`, and `src1` must all be row-major (`isRowMajor`).

### A5 implementation checks

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMIN(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TPARTMIN(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpartmin %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in family: [pto.tpartmax](./tpartmax.md)
- Next op in family: [pto.tgatherb](./tgatherb.md)
