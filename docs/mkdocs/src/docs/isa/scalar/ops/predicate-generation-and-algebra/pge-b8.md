<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b8.md` -->

# pto.pge_b8

Standalone reference page for `pto.pge_b8`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 8-bit predicate: lanes where the lane index is greater-than-or-equal to a scalar threshold.

## Mechanism

`pto.pge_b8` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i ≥ scalar`. This is the scalar complement of `plt_b8`, commonly used for tail-mask generation in remainder loops.

For lane index `i` (0 ≤ i < 8) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i \geq s \\ 0 & \text{if } i < s \end{cases} $$

## Syntax

### PTO Assembly Form

```text
pge_b8 %dst, %scalar : !pto.mask, i8
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pge_b8 %scalar : i8 -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pge_b8 ins(%scalar : i8) outs(%mask : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PGE_B8(RegBuf<predicate_t>& dst, int8_t scalar);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar` | `i8` | Lane-index threshold; lanes i ≥ scalar are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 8-bit predicate with active lanes above threshold |

## Side Effects

None.

## Constraints

- **Scalar range**: `scalar` MUST be in the range `[0, 8]`. Values outside this range produce all-1 (if scalar ≤ 0) or all-0 (if scalar ≥ 8) predicates.具体的实现行为取决于目标 Profile。
- **Predicate width**: The produced predicate is 8 bits wide. Programs that need wider predicates MUST use `ppack` to combine multiple `_b8` results.
- **No side effect on scalar**: Unlike `plt_b8`, this operation does NOT modify the scalar operand.

## Exceptions

- Illegal if `scalar` is outside the representable range for the target profile (typically `[0, 8]`).
- Illegal if the operation is used in a context requiring a predicate width other than 8 bits.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| 8-bit predicate width | Supported | Supported | Supported |
| Scalar range enforcement | Enforced | Enforced | Enforced |

## Examples

### Tail mask for remainder loop

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void generate_tail_mask(RegBuf<predicate_t>& dst, int8_t remainder) {
    // remainder lanes active (i >= (8 - remainder))
    PGE_B8(dst, 8 - remainder);
}
```

### SSA form

```mlir
// %c0 holds the remainder count
%tail = pto.pge_b8 %c0 : i8 -> !pto.mask

// Use in predicated vector operation
%result = pto.vsel %v_true, %v_false, %tail : !pto.vreg<8xf32>, !pto.vreg<8xf32>, !pto.mask -> !pto.vreg<8xf32>
```

### Comparison with plt_b8

```mlir
// pge_b8: lane i is active iff i >= scalar
//   input: %rem = 3
//   result: [0,0,0,0,0,1,1,1] (lanes 5,6,7 active)

// plt_b8: lane i is active iff i < scalar; also decrements scalar
//   input: %rem = 3
//   result: [1,1,1,0,0,0,0,0] (lanes 0,1,2 active)
//   output: %scalar_out = 0
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pset_b32](./pset-b32.md)
- Next op in family: [pto.pge_b16](./pge-b16.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
