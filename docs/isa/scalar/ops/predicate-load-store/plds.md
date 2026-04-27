# pto.plds

`pto.plds` is part of the [Predicate Load Store](../../predicate-load-store.md) instruction set.

## Summary

Load the full predicate register from a contiguous UB location.

## Mechanism

`pto.plds` reads a predicate word from a UB address and materializes it as `!pto.mask`. The operation covers the full predicate width for the active element type (64 bits for f32, 128 bits for f16/bf16, 256 bits for i8/u8).

For predicate width `Pw` and UB address `base`:

$$ \mathrm{mask} = \mathrm{READ\_UB}_{64}(base) $$

The predicate register is updated atomically. All bits are meaningful only within the current element-type context; on A2/A3 and A5 unused upper bits are zeroed by the hardware; on CPU simulator they retain the value from the UB read.

## Syntax

### PTO Assembly Form

```mlir
%mask = pto.plds %ub_ptr : !pto.ptr<i64, ub> -> !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.plds %ub_ptr : !pto.ptr<i64, ub> -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.plds ins(%ub_ptr : !pto.ptr<i64, ub>) outs(%mask : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool dst;
__ubuf__ uint32_t *base;
int32_t offset = 0;
plds(dst, base, offset, __cce_simd::NORM);
plds(dst, base, offset, __cce_simd::NORM, __cce_simd::POST_UPDATE);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB base address (must be 64-bit aligned) |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Loaded predicate register |

## Side Effects

None. Does not implicitly fence or synchronize with any pipeline.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`. Pointers to other spaces are illegal.
- **Alignment**: The effective address MUST be 64-bit aligned. Misaligned addresses are **illegal**.
- **Predicate width**: The load transfers exactly 64 bits. The caller MUST ensure this matches the active element type context.
- **Single active predicate**: Loading a new predicate does not implicitly clear or save a prior predicate. Programs that need to preserve predicate state MUST save it to UB before loading.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address is not 64-bit aligned.
- Illegal if predicate width does not match the active element type context.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Contiguous load | Simulated | Supported | Supported |
| 64-bit alignment requirement | Enforced | Enforced | Enforced |
| Predicate width (f32 / f16,bf16 / i8) | N=64/128/256 | N=64/128/256 | N=64/128/256 |

## Examples

### Load predicate from UB

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void load_saved_mask(RegBuf<predicate_t>& dst, Ptr<ub_space_t, ub_t> src) {
    PLDS(dst, src);
}
```

### SSA form

```mlir
// Load predicate from UB slot 0
%mask = pto.plds %ub_mask_slot0 : !pto.ptr<i64, ub> -> !pto.mask

// Use predicate in vector select
%result = pto.vsel %v_true, %v_false, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Load Store](../../predicate-load-store.md)
- Next op in instruction set: [pto.pld](./pld.md)
- Previous op in instruction set: (none — first in instruction set)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
