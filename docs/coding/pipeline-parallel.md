# Pipeline Parallelism

This document describes pipeline overlap in PTO Tile Lib from the perspective of staged execution, buffering, and explicit synchronization.

It focuses on software-visible organization of load, transform, compute, and store stages.

## 1. Pipeline overlap in PTO

In PTO kernels, pipeline optimization usually means overlapping different stages of work so that data movement and compute are not serialized more than necessary.

A typical high-level view is:

```text
TLOAD -> layout / staging transform -> compute -> TSTORE
```

Depending on the kernel, the middle stages may include operations such as:

- `TEXTRACT`
- `TMOV`
- `TTRANS`
- vector compute instructions
- matrix instructions such as `TMATMUL`

This interpretation is consistent with the optimization guidance in `docs/coding/opt.md`.

## 2. Hardware-visible stages and software-visible stages

At the software level, PTO developers usually reason about stages such as:

- loading tiles from GM
- transforming or reshaping staged data
- executing vector or cube compute
- storing results back to GM

The exact hardware mapping is backend-dependent, but the programming goal is stable:

- keep useful stages overlapped when dependencies allow it
- avoid unnecessary stalls
- avoid draining the whole pipeline when only a local dependency needs to be enforced

## 3. Role of buffering

Pipeline overlap usually relies on buffering strategies such as:

- double buffering
- staged temporary tiles
- explicit producer-consumer synchronization

The core idea is simple:

- while one tile is being consumed by a later stage, another tile can be prepared for an earlier stage

Whether this helps depends on:

- the kernel structure
- the dominant bottleneck
- available on-chip resources
- correctness of the dependency structure

## 4. Role of events and synchronization

The current repository documents an explicit event model in `docs/coding/Event.md`.

That model is the most reliable public basis for describing fine-grained synchronization in PTO.

Practical guidance:

- use synchronization only for true dependencies
- prefer producer-consumer ordering over unnecessary broad barriers
- validate event usage against the documented event types and intrinsic signatures

In device builds, typed `Event<SrcOp, DstOp>` objects are used to express dependencies.
In CPU simulation, some synchronization paths are simplified.

## 5. Warm-up, steady state, and drain

A pipelined kernel is often easiest to understand in three phases:

1. **warm-up**: fill the pipeline with the first tiles
2. **steady state**: overlap stages across successive tiles
3. **drain**: finish the remaining in-flight work

This mental model is useful because many pipeline mistakes come from treating the first or last iteration the same as the steady-state loop when they actually have different dependency structure.

## 6. What makes a pipeline effective

A pipeline tends to help when:

- data movement and compute are both significant
- intermediate stages can proceed on different tiles with bounded buffering
- the synchronization graph is precise enough to preserve overlap

A pipeline may help less when:

- one stage dominates almost all execution time
- buffering pressure becomes too high
- extra transforms or synchronization eliminate the intended overlap

Therefore, pipeline parallelism should be evaluated against the real bottleneck rather than assumed to be beneficial in every kernel.

## 6. Description boundaries

The following should be avoided unless a repository source explicitly defines them:

- placeholder instructions such as `TCOMPUTE`
- event APIs that do not match `docs/coding/Event.md`
- blanket claims such as “3-4x throughput improvement”
- exact hardware-stage mappings presented as universally guaranteed for all targets
- pseudo-code that omits the real legality constraints of tiles, layouts, and instruction support

Those patterns may be useful informally, but they are not rigorous public documentation.

## 7. Pipeline tuning flow

A practical tuning process is:

1. start from a correct non-overlapped or minimally overlapped kernel
2. identify the dominant stage using profiling or stage-level timing
3. introduce buffering and fine-grained synchronization gradually
4. verify correctness after each scheduling change
5. re-check whether overlap still helps after tile-size or partitioning changes

This mirrors the tuning philosophy in `docs/coding/opt.md`.

## 9. Related references

For the current PTO Tile Lib repository, the most relevant documents are:

- `docs/coding/Event.md`
- `docs/coding/Tile.md`
- `docs/coding/tutorial.md`
- `docs/coding/opt.md`

## 9. Conclusion

In PTO Tile Lib, pipeline parallelism is best described as:

- overlapping load / transform / compute / store stages when legal
- using buffering and explicit synchronization carefully
- tuning for the real bottleneck rather than chasing overlap in the abstract

This is more accurate than relying on invented APIs, placeholder instructions, or fixed performance claims.
