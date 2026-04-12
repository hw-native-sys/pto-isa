<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pge-b32.md` -->

# pto.pge_b32

Standalone reference page for `pto.pge_b32`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Generate a dynamic 32-bit predicate: lanes where the lane index is greater-than-or-equal to a scalar threshold.

## Mechanism

`pto.pge_b32` compares the lane index against a runtime scalar value and produces a predicate where active lanes satisfy `i ≥ scalar`.

For lane index `i` (0 ≤ i < 32) and scalar threshold `s`:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if } i \geq s \\ 0 & \text{if } i < s \end{cases} $$

The `_b32` variant is the widest directly-generable predicate segment. For f32 (N=64), two `_b32` predicates can be combined with `ppack` to form a full-width mask.

## Syntax

### PTO Assembly Form

```text
pge_b32 %dst, %scalar : !pto.mask, i32
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pge_b32 %scalar : i32 -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pge_b32 ins(%scalar : i32) outs(%mask : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PGE_B32(RegBuf<predicate_t>& dst, int32_t scalar);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar` | `i32` | Lane-index threshold; lanes i ≥ scalar are active |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 32-bit predicate with active lanes above threshold |

## Side Effects

None.

## Constraints

- **Scalar range**: `scalar` MUST be in the range `[0, 32]`. Values outside this range produce all-1 (if scalar ≤ 0) or all-0 (if scalar ≥ 32) predicates.
- **Predicate width**: The produced predicate is 32 bits wide. For wider predicates, use `ppack` to combine multiple `_b32` results.
- **No side effect on scalar**: Unlike `plt_b32`, this operation does NOT modify the scalar operand.

## Exceptions

- Illegal if `scalar` is outside the range `[0, 32]` for the target profile.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Dynamic predicate generation | Simulated | Supported | Supported |
| 32-bit predicate width | Supported | Supported | Supported |
| Scalar range enforcement | Enforced | Enforced | Enforced |

## Examples

### Tail mask for remainder loop (f32, 64 lanes)

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void generate_tail_mask_hi(RegBuf<predicate_t>& dst, int32_t remainder) {
    // upper half: lanes 32..63 that are active
    // remainder is already subtracted from the lower half
    PGE_B32(dst, 32 - remainder);
}
```

### SSA form

```mlir
// %rem holds the remainder count (0..63)
// Generate lower-half tail: lanes 0..31
%lo = pto.pge_b32 %rem : i32 -> !pto.mask

// Generate upper-half tail: lanes 32..63
%hi_rem = arith.subi %rem, 32 : i32
%hi = pto.pge_b32 %hi_rem : i32 -> !pto.mask

// Combine for full 64-lane predicate
%tail = pto.ppack %lo, %hi : !pto.mask, !pto.mask -> !pto.mask

// Use in predicated vector operation
%result = pto.vsel %v_true, %v_false, %tail : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pge_b16](./pge-b16.md)
- Next op in family: [pto.plt_b8](./plt-b8.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
