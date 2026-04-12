<!-- Generated from `docs/isa/tile/ops/irregular-and-complex/tpartmul.md` -->

# pto.tpartmul

Standalone reference page for `pto.tpartmul`. This page belongs to the [Irregular And Complex](../../irregular-and-complex.md) family in the PTO ISA manual.

## Summary

Partial elementwise multiply with implementation-defined handling of mismatched valid regions.

## Mechanism

Performs elementwise multiplication over the destination valid region. When both `src0` and `src1` are valid at an element, the result is their product; when only one input is valid there, the result copies that input value. Handling of other mismatched-validity cases is implementation-defined. It belongs to the tile surface and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For each element `(i, j)` in the destination valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile.
- `src1` is the second source tile.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the elementwise partial product: both valid gives product; one valid gives the valid value.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `dst`, `src0`, and `src1` must use the same element type.

- The destination valid region defines the result domain.

- For each element in the destination valid region:
  - if both inputs are valid, the instruction applies its elementwise operator;
  - if only one input is valid, the result copies that input value.

- If `dst` has a zero valid region, the instruction returns early.

- Supported partial-validity patterns require at least one source tile to have a valid region exactly equal to `dst`, while the other source tile's valid region must not exceed `dst` in either dimension.

- Supported element types: `int32_t`, `int16_t`, `half`, `float`.

- Supported element types: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`, `bfloat16_t`.

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
  TPARTMUL(dst, src0, src1);
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
  TPARTMUL(dst, src0, src1);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in family: [pto.tpartadd](./tpartadd.md)
- Next op in family: [pto.tpartmax](./tpartmax.md)
