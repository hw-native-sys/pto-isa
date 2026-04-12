<!-- Generated from `docs/isa/scalar/ops/dma-copy/copy-gm-to-ubuf.md` -->

# pto.copy_gm_to_ubuf

Standalone reference page for `pto.copy_gm_to_ubuf`. This page belongs to the [DMA Copy](../../dma-copy.md) family in the PTO ISA manual.

## Summary

DMA transfer from Global Memory (`!pto.ptr<T, gm>`) to Unified Buffer (`!pto.ptr<T, ub>`).

## Mechanism

`pto.copy_gm_to_ubuf` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax


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
pto.copy_gm_to_ubuf %gm_src, %ub_dst,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride
    : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

## Detailed Notes

```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride
    : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%gm_src` | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%sid` | Stream ID (usually 0) |
| `%n_burst` | Number of burst rows (innermost loop count) |
| `%len_burst` | Contiguous bytes transferred per burst row |
| `%left_padding` | Left padding count (bytes) |
| `%right_padding` | Right padding count (bytes) |
| `%data_select_bit` | Padding / data-select control bit (`i1`) |
| `%l2_cache_ctl` | L2 cache allocate control (TBD — controls whether DMA allocates in L2 cache) |
| `%src_stride` | GM source stride: start-to-start distance between consecutive burst rows (bytes) |
| `%dst_stride` | UB destination stride: start-to-start distance between consecutive burst rows (bytes, 32B-aligned) |

## Related Ops / Family Links

- Family overview: [DMA Copy](../../dma-copy.md)
- Previous op in family: [pto.set_loop1_stride_ubtoout](./set-loop1-stride-ubtoout.md)
- Next op in family: [pto.copy_ubuf_to_gm](./copy-ubuf-to-gm.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
