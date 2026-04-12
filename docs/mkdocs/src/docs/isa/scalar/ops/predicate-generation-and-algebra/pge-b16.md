<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b16.md` -->

# pto.pge_b16

Standalone reference page for `pto.pge_b16`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 16-bit predicate: lanes where the lane index is greater-than-or-equal to a scalar threshold.

## Mechanism

`pto.pge_b16` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i ≥ scalar`.

For lane index `i` (0 ≤ i < 16) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i \geq s \\ 0 & \text{if } i < s \end{cases} $$

This operation is the scalar complement of `plt_b16`. It is used for tail-mask generation when the vector width is 16 (f16/bf16 with predicate packing context).

## Syntax

### PTO Assembly Form

```text
pge_b16 %dst, %scalar : !pto.mask, i16
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pge_b16 %scalar : i16 -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pge_b16 ins(%scalar : i16) outs(%mask : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PGE_B16(RegBuf<predicate_t>& dst, int16_t scalar);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar` | `i16` | Lane-index threshold; lanes i ≥ scalar are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 16-bit predicate with active lanes above threshold |

## Side Effects

None.

## Constraints

- **Scalar range**: `scalar` MUST be in the range `[0, 16]`. Values outside this range produce all-1 (if scalar ≤ 0) or all-0 (if scalar ≥ 16) predicates.
- **Predicate width**: The produced predicate is 16 bits wide. Programs that need wider predicates MUST use `ppack` to combine multiple `_b16` results.
- **No side effect on scalar**: Unlike `plt_b16`, this operation does NOT modify the scalar operand.

## Exceptions

- Illegal if `scalar` is outside the range `[0, 16]` for the target profile.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| 16-bit predicate width | Supported | Supported | Supported |
| Scalar range enforcement | Enforced | Enforced | Enforced |

## Examples

### Tail mask for remainder loop (f16/bf16)

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void generate_tail_mask(RegBuf<predicate_t>& dst, int16_t remainder) {
    // remainder lanes active (i >= (16 - remainder))
    PGE_B16(dst, 16 - remainder);
}
```

### SSA form

```mlir
// %rem holds the remainder count
%tail = pto.pge_b16 %rem : i16 -> !pto.mask

// Use in predicated vector operation on f16 (128 lanes = 8 × 16-bit predicates)
%result = pto.vsel %v_true, %v_false, %tail : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask -> !pto.vreg<128xf16>
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pge_b8](./pge-b8.md)
- Next op in family: [pto.pge_b32](./pge-b32.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
