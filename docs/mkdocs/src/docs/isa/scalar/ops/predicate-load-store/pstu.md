<!-- Generated from `docs/isa/scalar/ops/predicate-load-store/pstu.md` -->

# pto.pstu

Standalone reference page for `pto.pstu`. This page belongs to the [Predicate Load Store](../../predicate-load-store.md) family in the PTO ISA manual.

## Summary

Stream predicate register to UB with alignment state tracking. High-throughput variant that relaxes alignment requirements at the cost of weaker write atomicity guarantees.

## Mechanism

`pto.pstu` writes a predicate word from `!pto.mask` to a UB address while tracking and updating alignment state. Unlike `psts`, this operation does not require 64-bit alignment and may batch multiple predicate writes into a single DMA transaction.

For alignment state `align_in`, predicate `mask`, and base address `base`:

$$ align\_out = align\_in \oplus mask $$
$$ base\_out = base + \mathrm{sizeof}(predicate) $$

The `%align_out` state carries forward into the next `pstu` call, enabling streaming writes without per-op synchronization.

## Syntax

### PTO Assembly Form

```text
pstu %align_in, %mask, %base_in : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

### AS Level 1 (SSA)

```mlir
%align_out, %base_out = pto.pstu %align_in, %mask, %base_in : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

### AS Level 2 (DPS)

```mlir
pto.pstu ins(%align_in, %mask, %base_in : !pto.align, !pto.mask, !pto.ptr<T, ub>)
       outs(%align_out, %base_out : !pto.align, !pto.ptr<T, ub>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST void PSTU(PredicateReg& dst,
                    Ptr<ub_space_t, ub_t> ub_ptr,
                    predicate_t align_in,
                    predicate_t align_out);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%align_in` | `!pto.align` | Alignment state from previous `pstu` or `pld`-family operation |
| `%mask` | `!pto.mask` | Predicate register to stream-store |
| `%base_in` | `!pto.ptr<T, ub>` | UB base address (no alignment requirement) |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%align_out` | `!pto.align` | Updated alignment state for next `pstu` call |
| `%base_out` | `!pto.ptr<T, ub>` | Incremented base address (base + predicate width in bytes) |

## Side Effects

- Writes the predicate register value to UB memory at the target address.
- Updates alignment state for use by subsequent `pstu` calls.
- UB memory at the target address is modified; write atomicity per 64-bit word is **not guaranteed**.

## Constraints

- **Alignment state**: `%align_in` MUST be the alignment state from the previous `pstu` call, or from a `pld`-family operation. Using an uninitialized alignment state is **illegal**.
- **Alignment state chaining**: Programs MUST pass `%align_out` from one `pstu` to the `%align_in` of the next. Breaking the chain without re-initializing the alignment state is **illegal**.
- **Write atomicity**: Unlike `psts`, the 64-bit predicate word is NOT guaranteed to be atomically written. Programs that require exact predicate state restoration MUST use `psts`, not `pstu`.
- **UB address space**: `%base_in` MUST have address space `ub`.

## Exceptions

- Illegal if `%align_in` is not initialized from a prior `pstu` or `pld` operation.
- Illegal if alignment state chain is broken.
- Illegal if `%base_in` is not a UB-space pointer.
- `pstu` MUST NOT be used when exact predicate save/restore is required.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Stream predicate store | Not supported | Supported | Supported |
| Alignment state tracking | Not applicable | Supported | Supported |
| Write atomicity guarantee | Not applicable | Not guaranteed | Not guaranteed |

CPU simulator does not implement `pstu`. Portable programs MUST use `psts` for exact predicate persistence or provide a CPU-sim fallback.

## Examples

### Streaming predicate writes

```c
#include <pto/pto-inst.hpp>
using namespace pto;

void stream_masks(Ptr<ub_space_t, ub_t> dst_base,
                  predicate_t* masks,
                  int count) {
    predicate_t align_state = 0;
    for (int i = 0; i < count; ++i) {
        PSTU(masks[i], dst_base, align_state, align_state);
        dst_base = dst_base + (predicate_width_bytes);
    }
}
```

### SSA form — chaining stream stores

```mlir
// Initialize alignment state (e.g., from a dummy load or zero)
%align0 = pto.plds %ub_dummy : !pto.ptr<i64, ub> -> !pto.mask

// Stream store first predicate; align_out carries forward
%align1, %base1 = pto.pstu %align0, %mask0, %base0 : !pto.align, !pto.mask, !pto.ptr<i64, ub> -> !pto.align, !pto.ptr<i64, ub>

// Stream store second predicate using updated alignment state
%align2, %base2 = pto.pstu %align1, %mask1, %base1 : !pto.align, !pto.mask, !pto.ptr<i64, ub> -> !pto.align, !pto.ptr<i64, ub>
```

> **Note**: For exact predicate save/restore across kernel boundaries, use `psts` instead. `pstu` is intended for high-throughput streaming scenarios where some loss of per-word atomicity is acceptable.

## Related Ops / Family Links

- Family overview: [Predicate Load Store](../../predicate-load-store.md)
- Previous op in family: [pto.psti](./psti.md)
- Next op in family: (none — last in family)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
