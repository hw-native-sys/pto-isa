# Operator Fusion

This document describes operator-fusion-related considerations in PTO Tile Lib from the perspective of kernel structuring, intermediate data movement, and on-chip resource usage.

It focuses on fusion opportunities that can be expressed directly through the PTO programming model.

## 1. Operator fusion in PTO

In the PTO context, operator fusion generally means reducing unnecessary intermediate GM traffic by keeping multiple stages of work inside a single kernel or tile-level computation flow when that is legal and practical.

Typical goals are:

- reduce GM reads and writes of intermediate data
- improve data reuse in on-chip storage
- improve steady-state overlap between load, transform, compute, and store stages

This interpretation is consistent with the broader optimization guidance in `docs/coding/opt.md`.

## 2. Fusion characteristics

The following characteristics are typical of fusion in PTO kernels:

- combining multiple logical stages in one kernel can reduce intermediate memory traffic
- fusion opportunities depend on tile layout, valid-region constraints, backend support, and available on-chip storage
- a fused structure is only useful if it preserves correctness and actually improves the bottleneck stage
- some high-performance kernels in this repository already combine multiple stages rather than materializing every intermediate result in GM

These statements are much safer than claiming fixed speedups or describing undocumented compiler fusion passes as public PTO behavior.

## 3. Fusion in developer-written kernels

In this repository, fusion is best understood first as a **kernel-structuring technique** rather than as an automatically guaranteed compiler feature.

That means developers may:

- keep several dependent operations in the same tile-level kernel
- avoid storing intermediate tiles to GM when they can remain on chip
- structure the kernel so that intermediate values flow directly into the next stage

For example, a row-wise normalization or attention-related kernel may naturally combine:

- load
- reduction
- elementwise transform
- normalization
- store

within one kernel body.

## 4. Practical fusion considerations

### 4.1 Intermediate storage cost

Fusion is attractive when intermediate values would otherwise be written to GM and read back shortly afterward.

If an intermediate tile can remain on chip and feed the next stage directly, GM traffic may decrease substantially.

### 4.2 On-chip resource limits

Fusion is never free.

A fused kernel may need more:

- tile buffers
- temporary tiles
- synchronization edges
- layout conversions

If the fused version increases pressure on on-chip storage too much, the result may be worse rather than better.

### 4.3 Instruction legality and layout constraints

Even if a fused mathematical expression is conceptually simple, the PTO implementation still has to satisfy real instruction constraints:

- tile types must match the instruction requirements
- layouts must be legal for the participating operations
- valid-region handling must remain correct
- the needed instructions must exist on the selected backend

Therefore, every fusion attempt should be checked against `docs/isa/` and `include/README.md`.

### 4.4 Bottleneck awareness

Fusion should be guided by bottlenecks.

If a kernel is dominated by GM traffic, reducing intermediate GM writes may help a lot.

If a kernel is dominated by compute or transform cost, the main improvement may need to come from tiling, overlap, or layout choices instead.

## 5. Relationship with pipeline overlap

Fusion and pipeline overlap are related but different:

- **fusion** reduces unnecessary intermediate materialization and stage separation
- **pipeline overlap** improves utilization by overlapping stages that still remain

A fused kernel still needs a good pipeline structure. In many cases, the best result comes from combining:

- fewer GM round-trips for intermediate data
- better buffering and synchronization between the remaining stages

See also:

- `docs/coding/pipeline-parallel.md`
- `docs/coding/Event.md`
- `docs/coding/opt.md`

## 6. Scope and limits of the description

The following kinds of statements are not rigorous unless they are backed by repository code or formal docs:

- exact kernel-launch overhead numbers presented as universal PTO constants
- guaranteed cache-hit claims such as “100% cache hit”
- fixed speedup claims such as “3x” without shape- and backend-specific evidence
- invented APIs or pseudo-instructions that are not part of PTO public intrinsics
- undocumented automatic fusion passes presented as if they are current public compiler guarantees

Such claims may be useful in informal discussion, but they do not belong in strict repository documentation.

## 7. Fusion workflow

A practical workflow is:

1. start from a correct unfused or minimally fused kernel
2. identify whether intermediate GM traffic is actually a bottleneck
3. keep only the stages together that are legal and beneficial to combine
4. re-check tile constraints, backend support, and synchronization
5. validate correctness first, then compare performance on representative shapes

This keeps fusion decisions tied to measured benefit rather than intuition alone.

## 8. Conclusion

In PTO Tile Lib, operator fusion should be described conservatively as:

- a way to reduce unnecessary intermediate GM traffic
- a kernel-structuring and optimization technique
- something constrained by tile legality, backend support, and on-chip resources

This is more accurate than describing speculative fused APIs, guaranteed speedups, or undocumented compiler automation as part of the stable PTO interface.
