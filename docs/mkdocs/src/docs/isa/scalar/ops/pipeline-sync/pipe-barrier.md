<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/pipe-barrier.md` -->

# pto.pipe_barrier

Standalone reference page for `pto.pipe_barrier`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Drain all pending ops in the specified pipe. All previously issued operations on that pipe complete before any subsequent operation begins.

## Mechanism

`pto.pipe_barrier` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax

```mlir
pto.pipe_barrier "PIPE_*"
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
pipe_barrier(pipe);
```

```mlir
// Both stores target the same GM address — order matters!
pto.copy_ubuf_to_gm %ub_partial_0, %gm_result, ...
// Without pipe_barrier, MTE3 could execute the second copy before the first
// completes, producing a non-deterministic result at %gm_result.
pto.pipe_barrier "PIPE_MTE3"
// After barrier: first copy is guaranteed complete. Second copy overwrites deterministically.
pto.copy_ubuf_to_gm %ub_partial_1, %gm_result, ...
```

## Detailed Notes

```c
pipe_barrier(pipe);
```

**Pipe identifiers:** `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3`

**Example:** Two back-to-back `copy_ubuf_to_gm` calls writing to the same GM address. Without a barrier, MTE3 may reorder them and the final GM value is non-deterministic:

```mlir
// Both stores target the same GM address — order matters!
pto.copy_ubuf_to_gm %ub_partial_0, %gm_result, ...
// Without pipe_barrier, MTE3 could execute the second copy before the first
// completes, producing a non-deterministic result at %gm_result.
pto.pipe_barrier "PIPE_MTE3"
// After barrier: first copy is guaranteed complete. Second copy overwrites deterministically.
pto.copy_ubuf_to_gm %ub_partial_1, %gm_result, ...
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in family: [pto.wait_flag](./wait-flag.md)
- Next op in family: [pto.get_buf](./get-buf.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
