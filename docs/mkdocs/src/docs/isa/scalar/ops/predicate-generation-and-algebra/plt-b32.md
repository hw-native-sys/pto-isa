<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b32.md` -->

# pto.plt_b32

Standalone reference page for `pto.plt_b32`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 32-bit predicate with lane index less-than comparison, and atomically decrement the scalar operand.

## Mechanism

`pto.plt_b32` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i < scalar`, then decrements the scalar by 32.

For lane index `i` (0 ≤ i < 32) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i < s \\ 0 & \text{if } i \geq s \end{cases} $$
$$ s_{\mathrm{out}} = s - 32 $$

The `_b32` variant is the widest directly-generable predicate segment. For f32 (N=64), two `_b32` predicates from `plt_b32` can be combined with `ppack` to form a full-width mask.

## Syntax

### PTO Assembly Form

```text
plt_b32 %dst, %scalar_in : !pto.mask, i32 -> !pto.mask, i32
```

### AS Level 1 (SSA)

```mlir
%mask, %scalar_out = pto.plt_b32 %scalar_in : i32 -> !pto.mask, i32
```

### AS Level 2 (DPS)

```mlir
pto.plt_b32 ins(%scalar_in : i32) outs(%mask, %scalar_out : !pto.mask, i32)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PLT_B32(RegBuf<predicate_t>& dst,
                      int32_t& scalar_inout);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar_in` | `i32` | Lane-index threshold; lanes i < scalar_in are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 32-bit predicate with active lanes below threshold |
| `%scalar_out` | `i32` | Decremented scalar: `scalar_in - 32` |

## Side Effects

- The scalar operand is **modified in place**: `scalar_out = scalar_in - 32`.

## Constraints

- **Scalar range**: `scalar_in` MUST be in the range `[0, 32]`. After subtraction, `scalar_out` may be negative.
- **Chain requirement**: Programs MUST use `scalar_out` from one iteration as `scalar_in` of the next. Breaking the chain produces **implementation-defined** predicates.
- **Predicate width**: The produced predicate is 32 bits wide. For f32 (N=64), two `_b32` results can be combined with `ppack`.

## Exceptions

- Illegal if `scalar_in` is outside the range `[0, 32]` for the target profile.
- Illegal if the scalar chain is broken.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| Scalar decrement | Simulated | Supported | Supported |
| 32-bit predicate width | Supported | Supported | Supported |

## Examples

### Software-pipelined remainder loop (f32)

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void process_remainder(int32_t& rem, RegBuf<predicate_t>& mask) {
    // rem = remainder count
    // predicate: lanes 0..(rem-1) active
    // rem = rem - 32
    PLT_B32(mask, rem);
}
```

### Chained remainder loop with pack for f32

```mlir
// rem = 47: two iterations needed for 64 lanes

// Iteration 1: rem = 47
//   lo_mask: 32 lanes active, rem_out = 15
%lo, %rem1 = pto.plt_b32 %rem0 : i32 -> !pto.mask, i32

// Iteration 2: rem = 15
//   hi_mask: 15 lanes active, rem_out = -17
%hi, %rem2 = pto.plt_b32 %rem1 : i32 -> !pto.mask, i32

// Combine two b32 predicates into one 64-bit predicate
%full_tail = pto.ppack %lo, %hi : !pto.mask, !pto.mask -> !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.plt_b16](./plt-b16.md)
- Next op in family: [pto.ppack](./ppack.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
