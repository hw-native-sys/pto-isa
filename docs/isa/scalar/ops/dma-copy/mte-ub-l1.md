# pto.mte_ub_l1

`pto.mte_ub_l1` is part of the [DMA Copy](../../dma-copy.md) instruction set.

## Summary

Execute a grouped UB→L1 (CBUF) copy. `nburst(...)` defines the repeated burst transfer that stages a UB tile into the cube-side L1 (CBUF) buffer.

## Mechanism

The MTE engine reads `%n_burst` blocks of `%len_burst * 32` bytes from `%ub_src` and writes them to `%l1_dst`. Between bursts, the source advances by `(len_burst + src_gap) * 32` bytes and the destination advances by `(len_burst + dst_gap) * 32` bytes. `src_gap` and `dst_gap` are the inter-burst gap fields (in 32-byte units) that follow each copied block.

`pto.mte_ub_l1` is the architecturally-supported fallback for moving a Vector-produced tile back into the cube's L1 staging buffer. The hardware applies the ND→NZ layout conversion required by L1's fractal format; see [NZ Fractal Layout](../../../cube/nz-fractal-layout.md) (when authored) for details.

## Syntax

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, l1>, i64, i64, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l1_dst` | ptr | L1 destination pointer (`!pto.ptr<T, l1>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into the L1 destination region. |

## Side Effects

Reads UB-visible storage and writes L1-visible storage. The MTE pipe is engaged for the duration of the transfer. Downstream cube consumers (e.g., `pto.mte_l1_l0a` / `pto.mte_l1_l0b`) must wait on the appropriate pipeline-sync events before reading from L1.

## Constraints

!!! warning "Constraints"
    - UB source and L1 destination addresses must be 32-byte aligned.
    - `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes and clause groups not valid for the selected target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - The UB↔L1 dedicated data path is an architectural feature of cube-equipped profiles (A2/A3/A5). CPU simulation models the copy contract but does not surface the underlying NZ fractal staging.
    - For 1:2 Cube/Vector cooperation, both AIV0 and AIV1 each issue their own `pto.mte_ub_l1` against the same L1 base offset; the cube assembles the two sub-tiles into a single contiguous NZ Mat tile.

## Examples

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, l1>, i64, i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Intra-UB copy: [pto.mte_ub_ub](./copy-ubuf-to-ubuf.md)
- GM↔UB transfers: [pto.mte_gm_ub](./copy-gm-to-ubuf.md), [pto.mte_ub_gm](./copy-ubuf-to-gm.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
