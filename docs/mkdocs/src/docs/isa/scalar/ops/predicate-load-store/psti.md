<!-- Generated from `docs/isa/scalar/ops/predicate-load-store/psti.md` -->

# pto.psti

Standalone reference page for `pto.psti`. This page belongs to the [Predicate Load Store](../../predicate-load-store.md) family in the PTO ISA manual.

## Summary

Store the full predicate register to a UB location with an immediate (compile-time constant) byte offset.

## Mechanism

`pto.psti` writes a predicate word from `!pto.mask` to a UB address computed as `base + imm * 8`. The offset is a compile-time immediate, enabling address resolution at assembly time.

For predicate `mask`, UB base `base`, and immediate offset `imm`:

$$ \mathrm{addr} = base + imm \times 8 $$
$$ \mathrm{WRITE\_UB}_{64}(\mathrm{addr}, mask) $$

The immediate offset is encoded directly in the instruction word, in units of 8 bytes (64 bits).

## Syntax

### PTO Assembly Form

```text
psti %mask, %ub_ptr[%imm], "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32
```

### AS Level 1 (SSA)

```mlir
pto.psti %mask, %ub_ptr, %imm, "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32
```

### AS Level 2 (DPS)

```mlir
pto.psti ins(%mask, %ub_ptr, %imm, "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PSTI(RegBuf<predicate_t>& src,
                    const Ptr<ub_space_t, ub_t>& base,
                    int32_t imm,
                    const char* dist = "PK");
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%mask` | `!pto.mask` | Predicate register to store |
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB base address |
| `%imm` | `i32` | Immediate byte offset in 8-byte units (compile-time constant) |
| `"DIST"` | string attribute | Distribution mode: `"NORM"` or `"PK"` |

## Expected Outputs

None. This form is defined by its side effect on UB memory.

## Side Effects

- Writes the predicate register value to the UB location.
- UB memory at the target address is modified.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`.
- **Offset alignment**: The effective address MUST be 64-bit aligned. That is, `imm * 8` MUST be a multiple of 8. Misaligned effective addresses are **illegal**.
- **Immediate range**: The offset immediate MUST fit in the instruction encoding.具体的立即数范围由目标 Profile 定义；超出范围的值为 **illegal**。
- **Distribution mode**: The `dist` attribute MUST be `"NORM"` or `"PK"`.
- **Predicate width**: The store transfers exactly 64 bits, which MUST match the active element type context.
- **Write atomicity**: The 64-bit predicate word is written atomically.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address is not 64-bit aligned.
- Illegal if the immediate offset is out of range for the target profile.
- Illegal if `dist` attribute is not `"NORM"` or `"PK"`.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Immediate-offset predicate store | Simulated | Supported | Supported |
| `"NORM"` distribution mode | Supported | Supported | Supported |
| `"PK"` (packed) distribution mode | Not supported | Supported | Supported |
| Immediate offset range | Implementation-defined | 0–255 (8-byte units) | 0–1023 (8-byte units) |

## Examples

### Store predicate with immediate offset

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void store_immediate(RegBuf<predicate_t>& src,
                     Ptr<ub_space_t, ub_t> base) {
    // Store predicate to base + 2 * 8 = base + 16 bytes
    PSTI(src, base, 2, "NORM");
}
```

### SSA form

```mlir
// Generate predicate from comparison
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store predicate to UB at base + 4 * 8 = base + 32 bytes
pto.psti %mask, %ub_base, 4, "NORM" : !pto.mask, !pto.ptr<i64, ub>, i32
```

## Related Ops / Family Links

- Family overview: [Predicate Load Store](../../predicate-load-store.md)
- Previous op in family: [pto.pst](./pst.md)
- Next op in family: [pto.pstu](./pstu.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
