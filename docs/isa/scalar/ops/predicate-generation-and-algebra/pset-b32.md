# pto.pset_b32

`pto.pset_b32` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Construct a 32-bit predicate mask from a compile-time pattern token.

## Mechanism

`pto.pset_b32` sets the predicate register to a static pattern encoded by the pattern token. No runtime data is consumed; the entire result is determined at assembly time.

For a predicate register of width 32 bits:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if lane } i \text{ matches pattern} \\ 0 & \text{otherwise} \end{cases} $$

The `_b32` variant is the widest directly-constructable predicate segment. For wider predicates, use `ppack` to combine two `_b32` predicates.

## Syntax

### PTO Assembly Form

```mlir
%mask = pto.pset_b32 "PATTERN" : !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pset_b32 "PATTERN" : !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pset_b32 "PATTERN" outs(%mask : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool mask = pset_b32(__cce_simd::PAT_VL16);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `"PATTERN"` | string attribute | Compile-time pattern token |

### Supported Pattern Tokens

| Pattern | Predicate Width | Meaning |
|---------|:--------------:|---------|
| `PAT_ALL` | 32 | All 32 bits set to 1 |
| `PAT_ALLF` | 32 | All 32 bits set to 0 |
| `PAT_VL1` … `PAT_VL32` | 32 | First N bits set to 1 |
| `PAT_H` | 32 | Bits 16–31 set to 1 (high half), bits 0–15 set to 0 |
| `PAT_Q` | 32 | Bits 24–31 set to 1 (upper quarter), bits 0–23 set to 0 |
| `PAT_M3` | 32 | Modular 3 pattern |
| `PAT_M4` | 32 | Modular 4 pattern |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Constructed 32-bit predicate |

## Side Effects

None.

## Constraints

- **Pattern token validity**: The pattern token MUST be valid for a 32-bit predicate width. Using a `PAT_VL*` token with N > 32 is **illegal**.
- **Predicate context**: The `_b32` predicate can be combined with another `_b32` using `ppack` to form a 64-bit predicate for f32 vector width (N=64).

## Exceptions

- Illegal if the pattern token is not valid for the `_b32` (32-bit) variant.
- Illegal if the pattern token is not supported by the target profile.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| All pattern tokens | Simulated | Supported | Supported |
| 32-bit predicate width | Supported | Supported | Supported |

## Examples

### Construct all-active mask

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void set_all_active(RegBuf<predicate_t>& dst) {
    PSET_B32(dst, "PAT_ALL");
}
```

### Use as active-lane mask for f32 vector operations

```mlir
// All lanes active for f32 (64-bit predicate = pack two b32)
%all32 = pto.pset_b32 "PAT_ALL" : !pto.mask
%all64_lo = pto.pset_b32 "PAT_ALL" : !pto.mask
%all64_hi = pto.pset_b32 "PAT_ALL" : !pto.mask
%all64 = pto.ppack %all64_lo, "LOWER" : !pto.mask -> !pto.mask
```

### Construct remainder mask

```mlir
// First 12 lanes active (remainder loop)
%remainder = pto.pset_b32 "PAT_VL12" : !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.pset_b16](./pset-b16.md)
- Next op in instruction set: [pto.pge_b8](./pge-b8.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
