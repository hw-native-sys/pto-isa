# pto.mte_gm_ub

`pto.mte_gm_ub` is part of the [DMA Copy](../../dma-copy.md) instruction set.

!!! note "PTO ISA v0.6 surface"
    The v0.6 PTO micro-instruction surface replaces the earlier `pto.copy_gm_to_ubuf` plus standalone `set_loop_size_*` / `set_loop_stride_*` configuration ops with a single grouped instruction: `pto.mte_gm_ub` with inline `nburst(...)`, optional `loop(...)`, and optional `pad(...)` clauses. The information previously carried in separate loop / stride configuration registers is now expressed directly on the transfer op.

## Summary

Execute a grouped GM→UB DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, optional `loop(...)` groups add outer repetition levels, and optional `pad(...)` controls UB row padding.

## Mechanism

The MTE2 engine reads `%n_burst` source rows from `%gm_src` and writes them to `%ub_dst`. Each row transfers `%len_burst` contiguous bytes, and the source / destination stride operands give the start-to-start byte distance from one row to the next. Optional outer `loop(...)` groups wrap `nburst(...)` to express multi-level repetition without external loop-config state. When `pad(...)` is present, UB rows are padded up to the next 32-byte aligned boundary using the supplied fill value.

## Syntax

```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
  : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
    [pad T[, i64, i64]]
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%gm_src` | ptr | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l2_cache_ctl` | 2 bits | L2 cache allocate control |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 40 bits / 21 bits | Required innermost burst group: count, GM source stride, UB destination stride |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 40 bits / 21 bits | Optional outer repetition group: count, GM source stride, UB destination stride |
| `pad(%pad_value[, %left_padding_count, %right_padding_count])` | scalar / 8 bits / 8 bits | Optional padding: fill value, optional left padding count, optional right padding count |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into Unified Buffer memory. |

## Side Effects

Reads GM-visible storage and writes UB-visible storage. The MTE2 pipe is engaged for the duration of the transfer; downstream consumers must synchronize through `pto.set_flag` / `pto.wait_flag` (`PIPE_MTE2` → `PIPE_V`).

## Constraints

!!! warning "Constraints"
    - `nburst(...)` is always required.
    - Each `loop(...)` group must be provided as a complete triple when present.
    - `nburst(...)` is the innermost group.
    - `loop(...)` groups are ordered from inner to outer; the first `loop(...)` group wraps `nburst(...)`, and each additional `loop(...)` group wraps all earlier groups.
    - `pad(...)` may contain only `%pad_value`; omitted left and right padding counts default to 0. If either left or right count is provided, both must be provided.
    - `pad(...)` is independent of the optional `loop(...)` groups. A DMA load may use `nburst(...) pad(...)` without any `loop(...)` group.
    - `%ub_dst` MUST be 32-byte aligned. When `pad(...)` is present, each UB row is padded from `%len_burst` up to the 32B-aligned boundary of the UB destination stride, ensuring every row starts at a 32B-aligned offset.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, malformed clause groups, and attribute combinations not valid for the selected target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible copy contract but may not expose all DMA overlap hazards.
    - A2/A3 and A5 may narrow supported element sizes, row widths, or cache-control semantics.

## Examples

```mlir
// Single-level transfer with padding: rows of `%len_burst` bytes,
// padded to 32B-aligned UB rows using %pad as fill.
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
    pad f16
```

```mlir
// Two-level transfer: rows × tiles, with UB row padding.
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  loop(%tiles, %gm_tile_stride, %ub_tile_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
    loop i64, i64, i64, pad f16
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Reverse direction: [pto.mte_ub_gm](./copy-ubuf-to-gm.md)
- Intra-UB copy: [pto.mte_ub_ub](./copy-ubuf-to-ubuf.md)
- Pipeline sync: [pto.set_flag](../pipeline-sync/set-flag.md), [pto.wait_flag](../pipeline-sync/wait-flag.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
