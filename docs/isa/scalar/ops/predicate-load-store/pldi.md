# pto.pldi

`pto.pldi` is part of the [Predicate Load Store](../../predicate-load-store.md) instruction set.

## Summary

Load the full predicate register from a UB location with an immediate (compile-time constant) byte offset.

## Mechanism

`pto.pldi` reads a predicate word from a UB address computed as `base + imm * 8`, then materializes it as `!pto.mask`. The offset is a compile-time immediate, enabling address resolution at assembly time.

For predicate width `Pw`, UB base `base`, and immediate offset `imm`:

$$ \mathrm{addr} = base + imm \times 8 $$
$$ \mathrm{mask} = \mathrm{READ\_UB}_{64}(\mathrm{addr}) $$

The immediate offset is encoded directly in the instruction word, in units of 8 bytes (64 bits).

## Syntax

### PTO Assembly Form

```mlir
%mask = pto.pldi %ub_ptr, %imm, "DIST" : !pto.ptr<i64, ub>, i32 -> !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pldi %ub_ptr, %imm, "DIST" : !pto.ptr<i64, ub>, i32 -> !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pldi ins(%ub_ptr, %imm, "DIST" : !pto.ptr<i64, ub>, i32) outs(%mask : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool dst;
__ubuf__ uint32_t *base;
int32_t offset = 0;
pldi(dst, base, offset, __cce_simd::NORM);
pldi(dst, base, offset, __cce_simd::NORM, __cce_simd::POST_UPDATE);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%ub_ptr` | `!pto.ptr<i64, ub>` | UB base address |
| `%imm` | `i32` | Immediate byte offset in 8-byte units (compile-time constant) |
| `"DIST"` | string attribute | Distribution mode: `"NORM"`, `"US"`, or `"DS"` |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | Loaded predicate register |

## Side Effects

None.

## Constraints

- **UB address space**: `%ub_ptr` MUST have address space `ub`.
- **Offset alignment**: The effective address MUST be 64-bit aligned. That is, `imm * 8` MUST be a multiple of 8. Misaligned effective addresses are **illegal**.
- **Immediate range**: The offset immediate MUST fit in the instruction encoding.具体的立即数范围由目标 Profile 定义；超出范围的值为 **illegal**。
- **Distribution mode**: The `dist` attribute MUST be one of `"NORM"`, `"US"`, or `"DS"`.
- **Predicate width**: The load transfers exactly 64 bits, which MUST match the active element type context.

## Exceptions

- Illegal if `%ub_ptr` is not a UB-space pointer.
- Illegal if the effective address is not 64-bit aligned.
- Illegal if the immediate offset is out of range for the target profile.
- Illegal if `dist` attribute is not a supported distribution mode.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Immediate-offset predicate load | Simulated | Supported | Supported |
| `"NORM"` distribution mode | Supported | Supported | Supported |
| `"US"` / `"DS"` distribution modes | Simulated | Supported | Supported |
| Immediate offset range | Implementation-defined | 0–255 (8-byte units) | 0–1023 (8-byte units) |

## Examples

### Load predicate with immediate offset

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void load_immediate(RegBuf<predicate_t>& dst,
                    Ptr<ub_space_t, ub_t> base) {
    // Load predicate from base + 3 * 8 = base + 24 bytes
    PLDI(dst, base, 3, "NORM");
}
```

### SSA form

```mlir
// Load predicate from slot 2 (2 * 8 = 16 bytes offset)
%mask = pto.pldi %ub_base, 2, "NORM" : !pto.ptr<i64, ub>, i32 -> !pto.mask

// Use in predicated vector select
%result = pto.vsel %v_true, %v_false, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Load Store](../../predicate-load-store.md)
- Previous op in instruction set: [pto.pld](./pld.md)
- Next op in instruction set: [pto.psts](./psts.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
