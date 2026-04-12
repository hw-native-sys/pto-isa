<!-- Generated from `docs/isa/scalar/dma-copy.md` -->

# DMA Copy

These `pto.*` forms configure and execute scalar-side DMA movement between GM and UB or inside UB. They are part of the scalar/control surface because they define configuration and copy behavior, not vector-register compute.

## What This Family Covers

- nested-loop size and stride registers for GM↔UB transfers
- GM to UB copies
- UB to GM copies
- UB to UB copies

## Per-Op Pages

- [pto.set_loop_size_outtoub](./ops/dma-copy/set-loop-size-outtoub.md)
- [pto.set_loop2_stride_outtoub](./ops/dma-copy/set-loop2-stride-outtoub.md)
- [pto.set_loop1_stride_outtoub](./ops/dma-copy/set-loop1-stride-outtoub.md)
- [pto.set_loop_size_ubtoout](./ops/dma-copy/set-loop-size-ubtoout.md)
- [pto.set_loop2_stride_ubtoout](./ops/dma-copy/set-loop2-stride-ubtoout.md)
- [pto.set_loop1_stride_ubtoout](./ops/dma-copy/set-loop1-stride-ubtoout.md)
- [pto.copy_gm_to_ubuf](./ops/dma-copy/copy-gm-to-ubuf.md)
- [pto.copy_ubuf_to_gm](./ops/dma-copy/copy-ubuf-to-gm.md)
- [pto.copy_ubuf_to_ubuf](./ops/dma-copy/copy-ubuf-to-ubuf.md)

## Related Material

- [Control and configuration](./control-and-configuration.md)
- [Vector Families: DMA Copy](../vector/dma-copy.md)
