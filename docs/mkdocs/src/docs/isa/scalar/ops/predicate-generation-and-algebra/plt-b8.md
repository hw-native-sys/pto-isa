<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/plt-b8.md` -->

# pto.plt_b8

Standalone reference page for `pto.plt_b8`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 8-bit predicate with lane index less-than comparison, and atomically decrement the scalar operand.

## Mechanism

`pto.plt_b8` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i < scalar`. Unlike `pge_b8`, this operation **also** decrements the scalar operand by the predicate width, enabling chained remainder-loop generation.

For lane index `i` (0 ≤ i < 8) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i < s \\ 0 & \text{if } i \geq s \end{cases} $$
$$ s_{\mathrm{out}} = s - 8 $$

## Syntax

### PTO Assembly Form

```text
plt_b8 %dst, %scalar_in : !pto.mask, i8 -> !pto.mask, i8
```

### AS Level 1 (SSA)

```mlir
%mask, %scalar_out = pto.plt_b8 %scalar_in : i8 -> !pto.mask, i8
```

### AS Level 2 (DPS)

```mlir
pto.plt_b8 ins(%scalar_in : i8) outs(%mask, %scalar_out : !pto.mask, i8)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PLT_B8(RegBuf<predicate_t>& dst,
                     int8_t& scalar_inout);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar_in` | `i8` | Lane-index threshold; lanes i < scalar_in are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 8-bit predicate with active lanes below threshold |
| `%scalar_out` | `i8` | Decremented scalar: `scalar_in - 8` |

## Side Effects

- The scalar operand is **modified in place**: `scalar_out = scalar_in - 8`.

## Constraints

- **Scalar range**: `scalar_in` MUST be in the range `[0, 8]`. After subtraction, `scalar_out` may be negative.
- **Chain requirement**: Programs MUST use `scalar_out` from one iteration as `scalar_in` of the next. Breaking the chain without re-initializing the scalar produces **implementation-defined** predicates.
- **Predicate width**: The produced predicate is 8 bits wide. For wider predicates, use `ppack` to combine multiple `_b8` results.

## Exceptions

- Illegal if `scalar_in` is outside the range `[0, 8]` for the target profile.
- Illegal if the scalar chain is broken (use of uninitialized or stale scalar values).

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| Scalar decrement | Simulated | Supported | Supported |
| 8-bit predicate width | Supported | Supported | Supported |

## Examples

### Software-pipelined remainder loop

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void process_remainder(int8_t& rem, RegBuf<predicate_t>& mask) {
    // rem = remainder count
    // predicate: lanes 0..(rem-1) active
    // rem = rem - 8
    PLT_B8(mask, rem);
}
```

### Chained remainder loop in SSA form

```mlir
// Iteration 1: rem = 23
//   mask = [1,1,1,1,1,1,1,1] (8 lanes), rem_out = 15
%mask1, %rem1 = pto.plt_b8 %rem0 : i8 -> !pto.mask, i8

// Iteration 2: rem = 15
//   mask = [1,1,1,1,1,1,1,1] (8 lanes), rem_out = 7
%mask2, %rem2 = pto.plt_b8 %rem1 : i8 -> !pto.mask, i8

// Iteration 3: rem = 7
//   mask = [1,1,1,1,1,1,1,0] (7 lanes), rem_out = -1
%mask3, %rem3 = pto.plt_b8 %rem2 : i8 -> !pto.mask, i8
```

### Compare with pge_b8

```mlir
// pge_b8: lane i is active iff i >= scalar (tail mask)
//   input: %rem = 3
//   result: [0,0,0,0,0,1,1,1] (lanes 5,6,7 active)

// plt_b8: lane i is active iff i < scalar; decrements scalar
//   input: %rem = 3
//   result: [1,1,1,0,0,0,0,0] (lanes 0,1,2 active)
//   output: %scalar_out = -5 (3 - 8)
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pge_b32](./pge-b32.md)
- Next op in family: [pto.plt_b16](./plt-b16.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
