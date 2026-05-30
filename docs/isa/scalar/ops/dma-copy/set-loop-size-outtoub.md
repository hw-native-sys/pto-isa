# pto.set_loop_size_outtoub

`pto.set_loop_size_outtoub` is part of the [DMA Copy](../../dma-copy.md) instruction set.

!!! warning "Deprecated in PTO ISA v0.6"
    The v0.6 PTO micro-instruction surface no longer uses standalone loop / stride configuration registers for GM→UB DMA. The information previously carried in `pto.set_loop_size_outtoub` (loop counts) is now expressed inline on the grouped transfer op via `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` clauses on [`pto.mte_gm_ub`](./copy-gm-to-ubuf.md). New code should use the grouped form. This page is retained for historical reference and pre-v0.6 ports.

## Summary

Configure the inner and outer loop counts that the GM→UB DMA engine will use for subsequent transfers.

## Mechanism

This operation programs the GM→UB DMA loop-count registers. Later `pto.copy_gm_to_ubuf` instructions consume these loop counters to build a two-level nested transfer schedule: loop2 surrounds loop1, and loop1 surrounds the per-row burst engine.

## Syntax

### PTO Assembly Form

```text
set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %loop1_count | `i64` | Inner hardware-loop iteration count |
| %loop2_count | `i64` | Outer hardware-loop iteration count |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form only updates the GM→UB DMA loop-count configuration state. |

## Side Effects

Programs the loop-count state consumed by subsequent GM→UB DMA copies. The configuration remains in effect until another loop-size operation overrides it.

## Constraints

!!! warning "Constraints"
    - Both counts MUST be non-negative and within the width supported by the target profile.
    - When multi-level looping is not needed, both counts SHOULD be set to 1.
    - The configured counts apply only to later GM→UB DMA operations and do not retroactively affect already-issued copies.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible configuration contract but may not expose all hardware loop hazards.
    - A2/A3 and A5 may use different concrete register widths or reset behavior; portable code must follow the documented PTO contract plus the selected target profile.

## Examples

```mlir
pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.set_loop2_stride_outtoub](./set-loop2-stride-outtoub.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
