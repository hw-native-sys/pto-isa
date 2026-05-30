# pto.mte_ub_gm

`pto.mte_ub_gm` is part of the [DMA Copy](../../dma-copy.md) instruction set.

!!! note "PTO ISA v0.6 surface"
    The v0.6 PTO micro-instruction surface replaces the earlier `pto.copy_ubuf_to_gm` plus standalone `set_loop_size_*` / `set_loop_stride_*` configuration ops with a single grouped instruction: `pto.mte_ub_gm` with inline `nburst(...)` and optional `loop(...)` clauses.

## Summary

Execute a grouped UB→GM DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, and optional `loop(...)` groups add outer repetition levels.

## Mechanism

The MTE3 engine reads `%n_burst` source rows from `%ub_src` and writes them to `%gm_dst`. Each row transfers `%len_burst` contiguous bytes, and the source / destination stride operands give the start-to-start byte distance from one row to the next. Optional outer `loop(...)` groups wrap `nburst(...)` to express multi-level repetition without external loop-config state. Padding bytes added during a previous GM→UB load are stripped: MTE3 reads `%len_burst` bytes from each 32B-aligned UB row and writes only valid data to GM.

## Syntax

```mlir
pto.mte_ub_gm %ub_src, %gm_dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | ptr | GM destination pointer (`!pto.ptr<T, gm>`) |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 21 bits / 40 bits | Required innermost burst group: count, UB source stride, GM destination stride |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 21 bits / 40 bits | Optional outer repetition group: count, UB source stride, GM destination stride |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into Global Memory. |

## Side Effects

Reads UB-visible storage and writes GM-visible storage. The MTE3 pipe is engaged for the duration of the transfer; downstream consumers (and ordering against further GM writes) must synchronize through `pto.set_flag` / `pto.wait_flag` (`PIPE_V` → `PIPE_MTE3`, and `pto.mem_bar` between back-to-back stores to the same GM address).

## Constraints

!!! warning "Constraints"
    - `nburst(...)` is always required.
    - Each `loop(...)` group must be provided as a complete triple when present.
    - `nburst(...)` is the innermost group.
    - `loop(...)` groups are ordered from inner to outer; the first `loop(...)` group wraps `nburst(...)`, and each additional `loop(...)` group wraps all earlier groups.
    - `%ub_src` MUST be 32-byte aligned.

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
// Two-level outbound transfer: rows x tiles.
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64,
    loop i64, i64, i64
```

```mlir
// Three-level outbound transfer: rows x tiles x batches.
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  loop(%batches, %ub_batch_stride, %gm_batch_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64,
    loop i64, i64, i64, loop i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Reverse direction: [pto.mte_gm_ub](./copy-gm-to-ubuf.md)
- Intra-UB copy: [pto.mte_ub_ub](./copy-ubuf-to-ubuf.md)
- Pipeline sync: [pto.set_flag](../pipeline-sync/set-flag.md), [pto.wait_flag](../pipeline-sync/wait-flag.md), [pto.mem_bar](../pipeline-sync/mem-bar.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
