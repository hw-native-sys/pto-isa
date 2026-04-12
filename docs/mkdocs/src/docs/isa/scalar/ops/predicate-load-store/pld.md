<!-- Generated from `docs/isa/scalar/ops/predicate-load-store/pld.md` -->

# pto.pld

Standalone reference page for `pto.pld`. This page belongs to the [Predicate Load Store](../../predicate-load-store.md) family in the PTO ISA manual.

## Summary

Load the full predicate register from a UB location with a register-relative address offset.

## Mechanism

`pto.pld` reads a predicate word from a UB address computed as `base + areg * sizeof(predicate)`, then materializes it as `!pto.mask`. The offset is sourced from a scalar register, making the effective address data-dependent.

For predicate width `Pw`, UB base `base`, and offset register `areg`:

$$ \mathrm{addr} = base + areg \times 8 $$
$$ \mathrm{mask} = \mathrm{READ\_UB}_{64}(\mathrm{addr}) $$

The offset register value is interpreted as a byte displacement in units of 8 bytes (64 bits). The register must contain a value such that the resulting effective address is 64-bit aligned.

## Syntax

### PTO Assembly Form

```text
pld %mask, %ub_ptr[%areg], "DIST" : !pto.mask, !pto.ptr<i64, ub>, i32
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pld %ub_ptr, %areg, "DIST" : !pto.ptr<i64, ub>, i32 -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pld ins(%ub_ptr, %areg, "DIST" : !pto.ptr<i64, ub>, i32) outs(%mask : !pto.mask)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PLD(RegBuf<predicate_t>& dst,
                  const Ptr<ub_space_t, ub_t>& base,
                  int32_t areg,
                  const char* dist = "NORM");
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB base address |
| `%areg` | `i32` | Scalar register holding the byte offset in 8-byte units |
| `"DIST"` | string attribute | Distribution mode: `"NORM"`, `"US"`, or `"DS"` |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Loaded predicate register |

## Side Effects

None.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`.
- **Offset alignment**: The offset register MUST be set such that `base + areg * 8` is 64-bit aligned. Misaligned effective addresses are **illegal**.
- **Distribution mode**: The `dist` attribute MUST be one of `"NORM"`, `"US"`, or `"DS"`. Other modes are **illegal** for this form.
- **Predicate width**: The load transfers exactly 64 bits, which MUST match the active element type context.
- **Single active predicate**: Loading a new predicate does not implicitly save a prior predicate. Programs that need to preserve predicate state MUST save it first.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address (base + areg * 8) is not 64-bit aligned.
- Illegal if `dist` attribute is not a supported distribution mode.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Register-offset predicate load | Simulated | Supported | Supported |
| `"NORM"` distribution mode | Supported | Supported | Supported |
| `"US"` / `"DS"` distribution modes | Simulated | Supported | Supported |

## Examples

### Load predicate with register offset

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void load_with_offset(RegBuf<predicate_t>& dst,
                      Ptr<ub_space_t, ub_t> base,
                      int32_t slot) {
    // slot is in units of 8 bytes (one predicate word per slot)
    PLD(dst, base, slot, "NORM");
}
```

### SSA form

```mlir
// UB base at %ub_base; %c1 holds slot index (in 8-byte units)
%mask = pto.pld %ub_base, %c1, "NORM" : !pto.ptr<i64, ub>, i32 -> !pto.mask

// Use predicate in predicated vector operation
%result = pto.vsel %v_a, %v_b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Family Links

- Family overview: [Predicate Load Store](../../predicate-load-store.md)
- Previous op in family: [pto.plds](./plds.md)
- Next op in family: [pto.pldi](./pldi.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
