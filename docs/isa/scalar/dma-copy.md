# DMA Copy

These `pto.*` forms configure and execute scalar-side DMA movement between GM, UB, and L1. They are part of the scalar and control instructions because they describe DMA configuration and copy behavior, not vector-register compute.

## What This Instruction Set Covers

- Grouped GM↔UB transfers with inline burst / loop / pad clauses
- Grouped UB↔UB and UB→L1 copies
- (Pre-v0.6) standalone loop-size and loop-stride configuration registers

## v0.6 Grouped Transfer Ops

These are the four public grouped DMA interfaces in the PTO ISA v0.6 micro-instruction surface. Each instruction expresses its repetition structure via inline `nburst(...)` / `loop(...)` clauses on the op itself; standalone loop / stride configuration registers are no longer required.

- [pto.mte_gm_ub](./ops/dma-copy/copy-gm-to-ubuf.md) — GM → UB, with optional `pad(...)` for 32B-aligned row padding
- [pto.mte_ub_gm](./ops/dma-copy/copy-ubuf-to-gm.md) — UB → GM, strips padding added during load
- [pto.mte_ub_ub](./ops/dma-copy/copy-ubuf-to-ubuf.md) — intra-UB copy in 32B-unit bursts with gap fields
- [pto.mte_ub_l1](./ops/dma-copy/mte-ub-l1.md) — UB → L1 (cube CBUF), 32B-unit bursts with gap fields

## Deprecated Pre-v0.6 Configuration Ops

These ops correspond to the older surface where loop counts and per-level strides were programmed via standalone configuration registers and then consumed by a separate copy op. In v0.6 the same information lives inline on the grouped transfer op (`nburst(...)` and outer `loop(...)` clauses). The pages below are retained for historical reference and pre-v0.6 ports.

- [pto.set_loop_size_outtoub](./ops/dma-copy/set-loop-size-outtoub.md)
- [pto.set_loop2_stride_outtoub](./ops/dma-copy/set-loop2-stride-outtoub.md)
- [pto.set_loop1_stride_outtoub](./ops/dma-copy/set-loop1-stride-outtoub.md)
- [pto.set_loop_size_ubtoout](./ops/dma-copy/set-loop-size-ubtoout.md)
- [pto.set_loop2_stride_ubtoout](./ops/dma-copy/set-loop2-stride-ubtoout.md)
- [pto.set_loop1_stride_ubtoout](./ops/dma-copy/set-loop1-stride-ubtoout.md)

The legacy execution ops `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm` / `pto.copy_ubuf_to_ubuf` have been replaced by the v0.6 grouped forms `pto.mte_gm_ub` / `pto.mte_ub_gm` / `pto.mte_ub_ub` linked above. Their per-op pages (URL slugs preserved) now document the v0.6 surface.

## Related Material

- [Control and configuration](./control-and-configuration.md)
- [Vector Instruction Set: DMA Copy](../vector/dma-copy.md)
- [Pipeline Synchronization](./ops/pipeline-sync/)
