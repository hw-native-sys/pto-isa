<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b16.md` -->

# pto.plt_b16

Standalone reference page for `pto.plt_b16`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 16-bit predicate with lane index less-than comparison, and atomically decrement the scalar operand.

## Mechanism

`pto.plt_b16` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i < scalar`, then decrements the scalar by 16.

For lane index `i` (0 ≤ i < 16) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i < s \\ 0 & \text{if } i \geq s \end{cases} $$
$$ s_{\mathrm{out}} = s - 16 $$

## Syntax

### PTO Assembly Form

```text
plt_b16 %dst, %scalar_in : !pto.mask, i16 -> !pto.mask, i16
```

### AS Level 1 (SSA)

```mlir
%mask, %scalar_out = pto.plt_b16 %scalar_in : i16 -> !pto.mask, i16
```

### AS Level 2 (DPS)

```mlir
pto.plt_b16 ins(%scalar_in : i16) outs(%mask, %scalar_out : !pto.mask, i16)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PLT_B16(RegBuf<predicate_t>& dst,
                      int16_t& scalar_inout);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar_in` | `i16` | Lane-index threshold; lanes i < scalar_in are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 16-bit predicate with active lanes below threshold |
| `%scalar_out` | `i16` | Decremented scalar: `scalar_in - 16` |

## Side Effects

- The scalar operand is **modified in place**: `scalar_out = scalar_in - 16`.

## Constraints

- **Scalar range**: `scalar_in` MUST be in the range `[0, 16]`. After subtraction, `scalar_out` may be negative.
- **Chain requirement**: Programs MUST use `scalar_out` from one iteration as `scalar_in` of the next. Breaking the chain produces **implementation-defined** predicates.
- **Predicate width**: The produced predicate is 16 bits wide. For wider predicates, use `ppack`.

## Exceptions

- Illegal if `scalar_in` is outside the range `[0, 16]` for the target profile.
- Illegal if the scalar chain is broken.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| Scalar decrement | Simulated | Supported | Supported |
| 16-bit predicate width | Supported | Supported | Supported |

## Examples

### Software-pipelined remainder loop (f16/bf16)

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void process_remainder(int16_t& rem, RegBuf<predicate_t>& mask) {
    // rem = remainder count
    // predicate: lanes 0..(rem-1) active
    // rem = rem - 16
    PLT_B16(mask, rem);
}
```

### Chained remainder loop

```mlir
// Iteration 1: rem = 28
//   mask: 16 lanes active, rem_out = 12
%mask1, %rem1 = pto.plt_b16 %rem0 : i16 -> !pto.mask, i16

// Iteration 2: rem = 12
//   mask: 12 lanes active, rem_out = -4
%mask2, %rem2 = pto.plt_b16 %rem1 : i16 -> !pto.mask, i16
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.plt_b8](./plt-b8.md)
- Next op in family: [pto.plt_b32](./plt-b32.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
