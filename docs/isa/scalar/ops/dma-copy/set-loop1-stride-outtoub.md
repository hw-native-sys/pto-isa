# pto.set_loop1_stride_outtoub

`pto.set_loop1_stride_outtoub` is part of the [DMA Copy](../../dma-copy.md) instruction set.

!!! warning "Deprecated in PTO ISA v0.6"
    The v0.6 PTO micro-instruction surface no longer uses standalone loop / stride configuration registers for GM→UB DMA. The stride information previously carried by `pto.set_loop1_stride_outtoub` is now expressed inline on the grouped transfer op as part of the inner `nburst(%n_burst, %src_stride, %dst_stride)` clause on [`pto.mte_gm_ub`](./copy-gm-to-ubuf.md). New code should use the grouped form. This page is retained for historical reference and pre-v0.6 ports.

## Summary

Configure the inner-loop pointer advance used by the GM→UB DMA engine.

## Mechanism

This operation programs the loop1 stride registers for GM→UB DMA. After each inner-loop iteration, the DMA engine advances the GM source pointer by `%src_stride` bytes and the UB destination pointer by `%dst_stride` bytes before issuing the next burst row set.

## Syntax

### PTO Assembly Form

```text
set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64
```

### AS Level 1 (SSA)

```mlir
pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src_stride | `i64` | GM source pointer advance per inner-loop iteration, in bytes |
| %dst_stride | `i64` | UB destination pointer advance per inner-loop iteration, in bytes |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form only updates the GM→UB DMA loop1-stride configuration state. |

## Side Effects

Programs the loop1 stride state consumed by subsequent GM→UB DMA copies. The configuration remains in effect until another loop1-stride operation overrides it.

## Constraints

!!! warning "Constraints"
    - GM stride values MUST be representable in the source-stride field width of the selected target profile.
    - UB stride values MUST obey the destination-stride field width and alignment rules of the selected target profile.
    - The configured strides apply only to later GM→UB DMA operations.

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
pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Previous op in instruction set: [pto.set_loop2_stride_outtoub](./set-loop2-stride-outtoub.md)
- Next op in instruction set: [pto.set_loop_size_ubtoout](./set-loop-size-ubtoout.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
