<!-- Generated from `docs/isa/scalar/ops/predicate-generation-and-algebra/pset-b16.md` -->

# pto.pset_b16

Standalone reference page for `pto.pset_b16`. This page belongs to the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) family in the PTO ISA manual.

## Summary

Construct a 16-bit predicate mask from a compile-time pattern token.

## Mechanism

`pto.pset_b16` sets the predicate register to a static pattern encoded by the pattern token. No runtime data is consumed; the entire result is determined at assembly time.

For a predicate register of width 16 bits:

$$ \mathrm{mask}_i = \begin{cases} 1 & \text{if lane } i \text{ matches pattern} \\ 0 & \text{otherwise} \end{cases} $$

The pattern token fully determines which bits are set.

## Syntax

### PTO Assembly Form

```text
pset_b16 %dst, "PATTERN" : !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pset_b16 "PATTERN" : !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pset_b16 "PATTERN" outs(%mask : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PSET_B16(RegBuf<predicate_t>& dst, const char* pattern);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `"PATTERN"` | string attribute | Compile-time pattern token |

### Supported Pattern Tokens

| Pattern | Predicate Width | Meaning |
|---------|:--------------:|---------|
| `PAT_ALL` | 16 | All 16 bits set to 1 |
| `PAT_ALLF` | 16 | All 16 bits set to 0 |
| `PAT_VL1` … `PAT_VL16` | 16 | First N bits set to 1 |
| `PAT_H` | 16 | Bits 8–15 set to 1 (high half), bits 0–7 set to 0 |
| `PAT_Q` | 16 | Bits 12–15 set to 1 (upper quarter), bits 0–11 set to 0 |
| `PAT_M3` | 16 | Modular: repeat 1-1-1-0 pattern (lanes 3, 7, 11, 15 active) |
| `PAT_M4` | 16 | Modular: repeat 1-1-1-1-0-0-0-0 pattern (lanes 0–3, 8–11 active) |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Constructed 16-bit predicate |

## Side Effects

None.

## Constraints

- **Pattern token validity**: The pattern token MUST be valid for a 16-bit predicate width. Using a `PAT_VL*` token with N > 16 is **illegal**.
- **Predicate context**: This operation produces a fixed-width predicate. Programs that use it in a wider context MUST use pack/unpack to adapt.

## Exceptions

- Illegal if the pattern token is not valid for the `_b16` (16-bit) variant.
- Illegal if the pattern token is not supported by the target profile.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| All pattern tokens | Simulated | Supported | Supported |
| 16-bit predicate width | Supported | Supported | Supported |

## Examples

### Construct all-active mask

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void set_all_active(RegBuf<predicate_t>& dst) {
    PSET_B16(dst, "PAT_ALL");
}
```

### Construct modular pattern

```mlir
// Modular 3 pattern: lanes 3, 7, 11, 15 active
%mod3 = pto.pset_b16 "PAT_M3" : !pto.mask
```

### Construct first-half-active mask

```mlir
// High half: bits 8–15 active, bits 0–7 inactive
%high = pto.pset_b16 "PAT_H" : !pto.mask
```

## Related Ops / Family Links

- Family overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in family: [pto.pset_b8](./pset-b8.md)
- Next op in family: [pto.pset_b32](./pset-b32.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
