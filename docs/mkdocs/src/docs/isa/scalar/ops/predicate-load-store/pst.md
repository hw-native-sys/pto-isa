<!-- Generated from `docs/isa/scalar/ops/predicate-load-store/pst.md` -->

# pto.pst

Standalone reference page for `pto.pst`. This page belongs to the [Predicate Load Store](../../predicate-load-store.md) family in the PTO ISA manual.

## Summary

Store the full predicate register to a UB location with a register-relative address offset.

## Mechanism

`pto.pst` writes a predicate word from `!pto.mask` to a UB address computed as `base + areg * 8`. The offset is sourced from a scalar register, enabling data-dependent addressing.

For predicate `mask`, UB base `base`, and offset register `areg`:

$$ \mathrm{addr} = base + areg \times 8 $$
$$ \mathrm{WRITE\_UB}_{64}(\mathrm{addr}, mask) $$

The predicate register is read atomically. Only bits within the current element-type predicate width are transferred.

## Syntax

### PTO Assembly Form

```text
pst %mask, %ub_ptr[%areg], "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32
```

### AS Level 1 (SSA)

```mlir
pto.pst %mask, %ub_ptr, %areg, "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32
```

### AS Level 2 (DPS)

```mlir
pto.pst ins(%mask, %ub_ptr, %areg, "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PST(RegBuf<predicate_t>& src,
                   const Ptr<ub_space_t, ub_t>& base,
                   int32_t areg,
                   const char* dist = "PK");
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%mask` | `!pto.mask` | Predicate register to store |
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB base address |
| `%areg` | `i32` | Scalar register holding the byte offset in 8-byte units |
| `"DIST"` | string attribute | Distribution mode: `"NORM"` or `"PK"` |

## Expected Outputs

None. This form is defined by its side effect on UB memory.

## Side Effects

- Writes the predicate register value to the UB location.
- UB memory at the target address is modified.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`.
- **Offset alignment**: The effective address MUST be 64-bit aligned. Misaligned effective addresses are **illegal**.
- **Distribution mode**: The `dist` attribute MUST be `"NORM"` or `"PK"`. The `"PK"` mode packs two 32-bit predicate segments into one 64-bit word for stores.
- **Predicate width**: The store transfers exactly 64 bits, which MUST match the active element type context.
- **Write atomicity**: The 64-bit predicate word is written atomically.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address is not 64-bit aligned.
- Illegal if `dist` attribute is not `"NORM"` or `"PK"`.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Register-offset predicate store | Simulated | Supported | Supported |
| `"NORM"` distribution mode | Supported | Supported | Supported |
| `"PK"` (packed) distribution mode | Not supported | Supported | Supported |

## Examples

### Store predicate with register offset

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void store_with_offset(RegBuf<predicate_t>& src,
                       Ptr<ub_space_t, ub_t> base,
                       int32_t slot) {
    // slot is in units of 8 bytes (one predicate word per slot)
    PST(src, base, slot, "NORM");
}
```

### SSA form

```mlir
// Generate predicate from comparison
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store predicate to UB at base + slot * 8
pto.pst %mask, %ub_base, %slot, "NORM" : !pto.mask, !pto.ptr<i64, ub>, i32
```

## Related Ops / Family Links

- Family overview: [Predicate Load Store](../../predicate-load-store.md)
- Previous op in family: [pto.psts](./psts.md)
- Next op in family: [pto.psti](./psti.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
