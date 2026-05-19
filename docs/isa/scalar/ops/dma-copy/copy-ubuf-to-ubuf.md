# pto.mte_ub_ub

`pto.mte_ub_ub` is part of the [DMA Copy](../../dma-copy.md) instruction set.

!!! note "PTO ISA v0.6 surface"
    The v0.6 PTO micro-instruction surface replaces the earlier `pto.copy_ubuf_to_ubuf` with the grouped `pto.mte_ub_ub` instruction, expressed with an inline `nburst(...)` clause. Burst length, source gap, and destination gap are all encoded in units of 32 bytes.

## Summary

Execute a grouped intra-UB copy. `nburst(...)` defines the repeated burst transfer between two UB regions.

## Mechanism

The MTE engine reads `%n_burst` blocks of `%len_burst * 32` bytes from `%ub_src` and writes them to `%ub_dst`. Between bursts, the source advances by `(len_burst + src_gap) * 32` bytes and the destination advances by `(len_burst + dst_gap) * 32` bytes. `src_gap` and `dst_gap` are the inter-burst gap fields (in 32-byte units) that follow each copied block.

## Syntax

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into the Unified Buffer destination region. |

## Side Effects

Reads UB-visible storage and writes UB-visible storage. The MTE pipe is engaged for the duration of the transfer; downstream consumers must synchronize through the appropriate pipeline-sync primitives.

## Constraints

!!! warning "Constraints"
    - UB source and destination addresses must be 32-byte aligned.
    - `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes and clause groups not valid for the selected target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible copy contract but may not expose all DMA overlap hazards.
    - A2/A3 and A5 may narrow supported element sizes, burst lengths, or gap encodings.

## Examples

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- GM↔UB transfers: [pto.mte_gm_ub](./copy-gm-to-ubuf.md), [pto.mte_ub_gm](./copy-ubuf-to-gm.md)
- UB→L1 transfer (cube path): [pto.mte_ub_l1](./mte-ub-l1.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
