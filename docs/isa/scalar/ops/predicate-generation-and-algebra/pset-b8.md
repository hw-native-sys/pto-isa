# pto.pset_b8

`pto.pset_b8` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Construct an 8-bit predicate mask from a compile-time pattern token.

## Mechanism

`pto.pset_b8` sets the predicate register to a static pattern encoded by the pattern token. No runtime data is consumed; the entire result is determined at assembly time.

For a predicate register of width 8 bits:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if lane } i \text{ matches pattern} \\ 0 & \text{otherwise} \end{cases} $$

The pattern token fully determines which bits are set. The operation is purely combinational — no pipeline resources are consumed beyond the scalar unit.

## Syntax

### PTO Assembly Form

```mlir
%mask = pto.pset_b8 "PATTERN" : !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pset_b8 "PATTERN" : !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pset_b8 "PATTERN" outs(%mask : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool mask = pset_b8(__cce_simd::PAT_VL4);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `"PATTERN"` | string attribute | Compile-time pattern token |

### Supported Pattern Tokens

| Pattern | Predicate Width | Meaning |
|---------|:--------------:|---------|
| `PAT_ALL` | 8 | All 8 bits set to 1 |
| `PAT_ALLF` | 8 | All 8 bits set to 0 |
| `PAT_VL1` | 8 | Bit 0 set to 1, bits 1–7 set to 0 |
| `PAT_VL2` | 8 | Bits 0–1 set to 1, bits 2–7 set to 0 |
| `PAT_H` | 8 | Bits 4–7 set to 1 (high half), bits 0–3 set to 0 |
| `PAT_Q` | 8 | Bits 6–7 set to 1 (upper quarter), bits 0–5 set to 0 |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Constructed 8-bit predicate |

## Side Effects

None. This operation does not modify architectural state other than the destination predicate register.

## Constraints

- **Pattern token validity**: The pattern token MUST be valid for an 8-bit predicate width. Using a `PAT_VL*` token with N > 8 is **illegal**.
- **Predicate context**: This operation produces a fixed-width predicate. Programs that use it in a wider predicate context MUST ensure width compatibility or use pack/unpack operations to adapt.
- **No dynamic component**: There are no runtime operands; the result is fully determined by the pattern token.

## Exceptions

- Illegal if the pattern token is not valid for the `_b8` (8-bit) variant.
- Illegal if the pattern token is not supported by the target profile.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| All pattern tokens | Simulated | Supported | Supported |
| 8-bit predicate width | Supported | Supported | Supported |

## Examples

### Construct all-active mask

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void set_all_active(RegBuf<predicate_t>& dst) {
    PSET_B8(dst, "PAT_ALL");
}
```

### Construct all-inactive mask

```mlir
%none = pto.pset_b8 "PAT_ALLF" : !pto.mask
```

### Construct first-3-lanes-active mask

```mlir
%first3 = pto.pset_b8 "PAT_VL3" : !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Next op in instruction set: [pto.pset_b16](./pset-b16.md)
- Previous op in instruction set: (none — first in pattern group)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
