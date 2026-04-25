# Multi-core Programming

This document describes common multi-core programming patterns in PTO Tile Lib and emphasizes work partitioning styles that align with the current programming model.

It focuses on tile-based decomposition, output ownership, load balancing, and locality.

## 1. Overview

PTO kernels commonly follow an SPMD-style execution model: multiple cores run the same kernel body, and work assignment is derived from block or core identity.

This style matches the tile-oriented programming model used throughout the PTO documentation.

For introductory examples, see:

- [Quickstart Tutorial](tutorial.md)
- [Vector Add Tutorial](tutorials/vec-add.md)
- [Optimization Guide](opt.md)

## 2. The main multi-core model used in this repository

### 2.1 SPMD-style work partitioning

The most common model in this repository is:

- all cores execute the same kernel code
- each core handles a different region of the input or output
- partitioning is usually expressed in terms of rows, columns, tiles, or block ranges

This approach is a natural fit for:

- elementwise kernels
- reductions over tiled data
- GEMM-like kernels
- tiled attention-style kernels

### 2.2 Why this model is preferred

SPMD-style partitioning is easier to reason about in PTO because it aligns with:

- tile-based work decomposition
- predictable GM access patterns
- straightforward load balancing
- simpler synchronization structure

In most cases, keeping each core responsible for a regular, contiguous region is preferable to introducing irregular inter-core coordination.

## 3. Practical partitioning guidance

### 3.1 Partition by output ownership

A good default strategy is to partition work by the output region each core owns.

For example:

- for vector-style operators, split the output along a linear range
- for matrix-style operators, split along tile rows, tile columns, or a 2D block grid
- for row-wise reductions, assign one or more output rows per core

This keeps the ownership model simple:

- the core that computes an output tile also stores it
- intermediate state remains local when possible
- inter-core write conflicts are avoided

### 3.2 Balance work across cores

When assigning tiles to cores:

- prefer partitions with similar compute cost per core
- avoid leaving one small tail region to a single overloaded core if it can be redistributed cleanly
- keep GM access regular and contiguous where possible

A partition that is mathematically even but creates poor memory locality may still perform badly, so balance and locality should be considered together.

### 3.3 Prefer regular tile loops

Multi-core kernels are easier to validate and optimize when each core follows the same tile loop structure.

Typical structure:

1. determine the tile range owned by the current core
2. iterate over the assigned tile range
3. perform `TLOAD -> transform / compute -> TSTORE`
4. handle edge tiles through valid-region control when needed

This style follows the tile-oriented programming model described in [Quickstart Tutorial](tutorial.md) and [Tile Programming Model](Tile.md).

## 4. Multi-core concerns that matter in PTO

### 4.1 Load balancing

Load balancing is important because PTO kernels often combine:

- GM movement
- layout transforms
- vector or cube compute
- explicit synchronization

If one core receives substantially more tiles or more expensive tiles than the others, overall throughput may be limited by the slowest core.

In practice, check:

- whether the output space is partitioned evenly
- whether edge tiles are concentrated on too few cores
- whether some cores perform extra transform or reduction work

### 4.2 Memory locality

Good multi-core partitioning should preserve locality in GM.

Preferred patterns usually have:

- contiguous reads or writes
- repeated reuse of nearby tensor regions
- stable tile shapes and strides

Poor locality often shows up as a high data-movement cost relative to compute.

### 4.3 Cross-core communication

This repository does document communication instructions under `docs/isa/comm/`, but general multi-core kernels should not assume that arbitrary producer-consumer scheduling across cores is the default model.

For most compute kernels, it is better to:

- minimize cross-core dependencies
- partition outputs cleanly
- keep synchronization local to true producer-consumer relationships when required

If a kernel depends on communication instructions, see [Communication ISA Reference](../isa/comm/README.md) and the corresponding instruction pages.

## 5. Relationship with pipeline optimization

Multi-core parallelism and pipeline overlap solve different problems:

- **multi-core parallelism** increases throughput by distributing work across cores
- **pipeline overlap** increases per-core utilization by overlapping load / transform / compute / store stages

A high-performance kernel usually needs both:

1. a sensible per-core tile partition
2. an efficient intra-core pipeline

For overlap and buffering guidance, see [Pipeline Parallelism](pipeline-parallel.md) and [Events and Synchronization](Event.md).

## 6. Programming boundaries

Multi-core PTO kernels are normally described in terms of tile ownership, regular work partitioning, and explicit dependencies. The following items fall outside that description unless they are introduced by dedicated runtime or backend documents:

- imaginary runtime APIs such as unspecified `get_block_idx()` contracts without repository context
- placeholder instructions such as `TCOMPUTE` or `TFILL` when those are not actual PTO public intrinsics in the described form
- generic pseudo-syntax such as Python-style tensor slicing inside `TLOAD` / `TSTORE`
- unsupported claims that MPMD is a standard public programming model for ordinary PTO kernels in this repository

Such examples may be useful as intuition elsewhere, but they are not rigorous repository documentation.

## 7. Multi-core development process

A common development process is:

1. start from a correct single-tile or single-core structure
2. define output ownership per core
3. partition work into regular tile ranges
4. validate correctness on CPU simulation
5. tune tile sizes, partitioning, and overlap on the target backend

This workflow keeps the programming model aligned with the rest of the PTO documentation and avoids introducing unnecessary complexity too early.

## 8. Notes

Multi-core programming in PTO Tile Lib is generally organized around:

- SPMD-style work partitioning;
- output ownership and regular tile ranges;
- regular, contiguous, and balanced access patterns;
- coordination between multi-core partitioning and per-core pipeline optimization.
