<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/set-flag.md` -->

# pto.set_flag

Standalone reference page for `pto.set_flag`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Signal event from source pipe to destination pipe.

## Mechanism

`pto.set_flag` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax

```mlir
pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
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

```c
set_flag(src_pipe, dst_pipe, event_id);
```

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

## Detailed Notes

```c
set_flag(src_pipe, dst_pipe, event_id);
```

**Example:** After MTE2 completes GM→UB transfer, signal Vector pipe:
```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Next op in family: [pto.wait_flag](./wait-flag.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
