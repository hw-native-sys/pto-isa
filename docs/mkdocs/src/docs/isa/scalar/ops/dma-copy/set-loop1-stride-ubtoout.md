<!-- Generated from `docs/isa/scalar/ops/dma-copy/set-loop1-stride-ubtoout.md` -->

# pto.set_loop1_stride_ubtoout

Standalone reference page for `pto.set_loop1_stride_ubtoout`. This page belongs to the [DMA Copy](../../dma-copy.md) family in the PTO ISA manual.

## Summary

Configure inner loop (loop1) pointer advance for UB→GM DMA.

## Mechanism

`pto.set_loop1_stride_ubtoout` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax

```mlir
pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64
```

## Inputs

The inputs are the architecture-visible control operands shown in the syntax: pipe ids, event ids, buffer ids, loop/stride values, pointers, or configuration words used to drive later execution.

## Expected Outputs

This form is primarily defined by the side effect it has on control state, predicate state, or memory. It does not publish a new payload SSA result beyond any explicit state outputs shown in the syntax.

## Side Effects

This operation updates control, synchronization, or DMA configuration state. Depending on the form, it may stall a stage, establish a producer-consumer edge, reserve or release a buffer token, or configure later copy behavior.

## Constraints

This operation inherits the legality and operand-shape rules of its family overview. Any target-specific narrowing of element types, distributions, pipe/event spaces, or configuration tuples must be stated by the selected target profile.

## Exceptions

- It is illegal to use unsupported pipe ids, event ids, buffer ids, or configuration tuples for the selected target profile.
- Waiting on state that was never established by a matching producer or prior configuration is an illegal PTO program.

## Target-Profile Restrictions

- CPU simulation preserves the visible dependency/configuration contract, but it may not expose every low-level hazard that motivates the form on hardware targets.
- A2/A3 and A5 profiles may use different concrete pipe, DMA, predicate, or event spaces. Portable code must rely on the documented PTO contract plus the selected target profile.

## Examples

```mlir
pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64
```

## Detailed Notes

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src_stride` | 21 bits | UB source pointer advance per loop1 iteration (bytes) |
| `%dst_stride` | 40 bits | GM destination pointer advance per loop1 iteration (bytes) |

## DMA Transfer Execution

## Related Ops / Family Links

- Family overview: [DMA Copy](../../dma-copy.md)
- Previous op in family: [pto.set_loop2_stride_ubtoout](./set-loop2-stride-ubtoout.md)
- Next op in family: [pto.copy_gm_to_ubuf](./copy-gm-to-ubuf.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
