<!-- Generated from `docs/isa/scalar/ops/predicate-load-store/psts.md` -->

# pto.psts

Standalone reference page for `pto.psts`. This page belongs to the [Predicate Load Store](../../predicate-load-store.md) family in the PTO ISA manual.

## Summary

Store the full predicate register to a contiguous UB location.

## Mechanism

`pto.psts` writes a predicate word from `!pto.mask` to a UB address. The operation covers the full predicate width for the active element type (64 bits for f32, 128 bits for f16/bf16, 256 bits for i8/u8).

For predicate width `Pw` and UB address `base`:

$$ \mathrm{WRITE\_UB}_{64}(base, mask) $$

The predicate register is read atomically. Only bits within the current element-type predicate width are transferred; bits outside are **implementation-defined**.

## Syntax

### PTO Assembly Form

```text
psts %mask, %ub_ptr : !pto.mask, !pto.ptr<i64, ub>
```

### AS Level 1 (SSA)

```mlir
pto.psts %mask, %ub_ptr : !pto.mask, !pto.ptr<i64, ub>
```

### AS Level 2 (DPS)

```mlir
pto.psts ins(%mask, %ub_ptr : !pto.mask, !pto.ptr<i64, ub>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PSTS(RegBuf<predicate_t>& src, const Ptr<ub_space_t, ub_t>& dst);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%mask` | `!pto.mask` | Predicate register to store |
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB destination address (must be 64-bit aligned) |

## Expected Outputs

None. This form is defined by its side effect on UB memory.

## Side Effects

- Writes the predicate register value to the UB location.
- UB memory at the target address is modified.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`. Pointers to other spaces are illegal.
- **Alignment**: The effective address MUST be 64-bit aligned. Misaligned addresses are **illegal**.
- **Predicate width**: The store transfers exactly 64 bits. The caller MUST ensure this matches the active element type context.
- **Write atomicity**: The 64-bit predicate word is written atomically.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address is not 64-bit aligned.
- Illegal if predicate width does not match the active element type context.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Contiguous store | Simulated | Supported | Supported |
| 64-bit alignment requirement | Enforced | Enforced | Enforced |
| Predicate width (f32 / f16,bf16 / i8) | N=64/128/256 | N=64/128/256 | N=64/128/256 |

## Examples

### Store predicate to UB

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void save_mask(RegBuf<predicate_t>& src, Ptr<ub_space_t, ub_t> dst) {
    PSTS(src, dst);
}
```

### SSA form

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store predicate to UB for later reuse
pto.psts %mask, %ub_mask_slot0 : !pto.mask, !pto.ptr<i64, ub>
```

## Related Ops / Family Links

- Family overview: [Predicate Load Store](../../predicate-load-store.md)
- Next op in family: [pto.pst](./pst.md)
- Previous op in family: [pto.pldi](./pldi.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
