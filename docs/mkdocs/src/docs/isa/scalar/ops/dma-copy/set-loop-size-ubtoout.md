<!-- Generated from `docs/isa/scalar/ops/dma-copy/set-loop-size-ubtoout.md` -->

# pto.set_loop_size_ubtoout

Standalone reference page for `pto.set_loop_size_ubtoout`. This page belongs to the [DMA Copy](../../dma-copy.md) family in the PTO ISA manual.

## Summary

Configure HW loop iteration counts for UB→GM DMA.

## Mechanism

`pto.set_loop_size_ubtoout` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax

```mlir
pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64
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
pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64
```

## Detailed Notes

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%loop1_count` | 21 bits | Inner HW loop iteration count |
| `%loop2_count` | 21 bits | Outer HW loop iteration count |

## Related Ops / Family Links

- Family overview: [DMA Copy](../../dma-copy.md)
- Previous op in family: [pto.set_loop1_stride_outtoub](./set-loop1-stride-outtoub.md)
- Next op in family: [pto.set_loop2_stride_ubtoout](./set-loop2-stride-ubtoout.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
