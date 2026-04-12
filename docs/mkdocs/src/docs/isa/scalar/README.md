<!-- Generated from `docs/isa/scalar/README.md` -->

# Scalar And Control Reference

This tree documents the `pto.*` scalar/control surface of PTO ISA: synchronization, DMA configuration, predicate-state movement, predicate construction, and the shared scalar source shell around tile and vector payload execution.

The key distinction is architectural role, not only spelling. `pto.*` pages live here when they expose control, DMA, predicate, or other non-payload state directly. When a family exists only to summarize how those forms interact with vector execution, the vector family overviews remain linked as related material rather than acting as the primary per-op reference.

## Families

- [Control and configuration](./control-and-configuration.md)
- [Pipeline sync](./pipeline-sync.md)
- [DMA copy](./dma-copy.md)
- [Predicate load store](./predicate-load-store.md)
- [Predicate generation and algebra](./predicate-generation-and-algebra.md)
- [Shared scalar arithmetic](./shared-arith.md)
- [Shared structured control flow](./shared-scf.md)
