# PTO Optimization guide

This document is a practical guide for optimizing PTO kernels, with an emphasis on **software-visible** levers:

- Tiling and work partitioning across blocks/cores
- Data movement (GM ↔ on-chip) and layout choices
- Overlap and synchronization (pipelines, double buffering)
- Instruction selection and fusion (vector vs cube stages)

For end-to-end, example-driven deep dives, see:

- GEMM: [`kernels/manual/a2a3/gemm_performance/README.md`](../../kernels/manual/a2a3/gemm_performance/README.md)
- Flash Attention: [`kernels/manual/common/flash_atten/README.md`](../../kernels/manual/common/flash_atten/README.md)

## 1. The performance model: think in stages

Most high-performance kernels in this repo can be reasoned about as a pipeline of stages:

1. **TLOAD**: global memory (GM) → on-chip staging (e.g., Mat/Vec tiles)
2. **Layout / staging transforms**: `TEXTRACT`, `TMOV`, `TTRANS`, `TRESHAPE` (depending on kernel)
3. **Compute**:
   - **Cube**: `TMATMUL`, `TMATMUL_ACC`, etc.
   - **Vector**: elementwise, reductions, exp/log, compare/select, etc.
4. **TSTORE**: on-chip → GM

The optimization goal is almost always the same:

- maximize steady-state overlap between “load / transform / compute / store”
- reduce bytes moved per useful FLOP
- avoid pipeline bubbles (unnecessary sync, poor tiling, bad core partitioning)

When you have profiling ratios (like the ones recorded in the kernel READMEs), treat them as a “where is the time going?” hint:

- **TLOAD near 100%** → the pipeline is feed-limited; reduce traffic or improve reuse/overlap.
- **Transform (`TEXTRACT`/`TMOV`) dominates** → reduce layout work per FLOP, or amortize it by increasing compute per transform.
- **TMATMUL is low while TLOAD is high** → the Cube is starving; overlap is broken or memory is saturated.

## 2. A repeatable tuning workflow

1. **Start from correctness**
   - Validate on CPU first: `python3 tests/run_cpu.py --verbose`
   - Add numerical checks (max diff / relative diff) early, before changing schedules.

2. **Fix the problem shape**
   - Choose a representative set of shapes (including “small” and “large”).
   - Prefer recording results in a table in the kernel folder README so changes are reviewable.

3. **Find the bottleneck stage**
   - Use profiler output and per-stage ratios (if available).
   - If you do not have a profiler, use time deltas around major phases (load/compute/store) and compare.

4. **Change one lever at a time**
   - Change tiling, or core partitioning, or overlap strategy (not all at once).
   - Re-run the same shape set.

5. **Lock in a stable steady state**
   - Make sure warm-up and drain (first/last iterations) do not serialize the main loop.

## 3. Parallelism: blocks, cores, and tiling by `block_idx`

PTO follows an SPMD-style execution model: all cores run the same kernel, and `block_idx` (and optional sub-block IDs) determine the work assignment.

Recommended reading:

- Overview and examples: [`docs/coding/tutorial.md`](tutorial.md)
- A concrete “tile-by-block-id” mapping example: [`docs/coding/tutorials/vec-add.md`](tutorials/vec-add.md)

Guidelines:

- Prefer **2D partitioning** when both dimensions are large (e.g., split `m` and `n` for GEMM-like kernels).
- Keep each block’s GM accesses **contiguous** and **regular** to maximize burst efficiency.
- Choose per-core work so that most cores do similar amounts of compute (avoid long-tail blocks).

## 4. Tiling: pick sizes that fit and reuse

Tiling is the first-order knob for performance:

- it determines on-chip footprint (whether you spill / thrash / underutilize buffers)
- it determines reuse (how many times a loaded tile contributes to compute)
- it determines how well you can overlap stages

Checklist:

- Keep tile sizes within on-chip limits (and within any kernel’s explicit buffer partition).
- Align tile shapes/layouts with the engine you want to use (Cube vs Vector).
- Increase arithmetic intensity where possible: do more compute per byte loaded.

Useful references:

- Tile definition and constraints: [`docs/coding/Tile.md`](Tile.md)
- Global tensor views and layouts: [`docs/coding/GlobalTensor.md`](GlobalTensor.md)

## 5. Data movement: reduce traffic and avoid redundant transforms

Common wins:

- **Reuse**: stage more data per DMA and reuse it (e.g., “stepK” caching in GEMM).
- **Fewer transforms**: avoid `TTRANS`/`TRESHAPE`/extra `TEXTRACT` if you can select the right input layout up front.
- **Keep outputs simple**: write back in a GM-friendly layout that matches downstream consumption.

If your kernel uses both Cube and Vector stages, try to keep intermediate data in a layout that minimizes conversion between the stages.

## 6. Overlap and synchronization: keep the pipeline full

Manual kernels often rely on explicit double buffering and event/flag synchronization to overlap:

- next `TLOAD` while current `TMATMUL` runs
- next `TEXTRACT` while current compute runs
- current `TSTORE` while next compute runs

Rules of thumb:

- Only wait on **true dependencies** (producer/consumer); avoid global “drain everything” waits in the steady-state loop.
- Treat the pipeline as having a **warm-up**, **steady state**, and **drain**; tune the steady state first.

Reference:

- Event and synchronization model: [`docs/coding/Event.md`](Event.md)

## 7. Example-driven guides

These kernel folders contain the most complete “how to tune” notes, tied to real code:

- GEMM (tiling, stepK caching, double buffering):
  - [`kernels/manual/a2a3/gemm_performance/README.md`](../../kernels/manual/a2a3/gemm_performance/README.md)
  - Kernel code: `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
- Flash Attention (staged softmax, tiled QK/PV, per-stage tuning):
  - [`kernels/manual/common/flash_atten/README.md`](../../kernels/manual/common/flash_atten/README.md)
  - Kernel code: `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`
  - [`kernels/manual/common/flash_atten/README.md`](../../kernels/manual/common/flash_atten/README.md)
  - Kernel code: `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`

## 8. Common failure modes (and what to do)

- **Great performance on one shape, terrible on others**
  - Re-tune core partitioning and tile sizes for each shape class (small/medium/large).
  - Watch for “too small” tiles (overhead dominated) and “too large” tiles (feed-limited / overlap broken).

- **High TLOAD ratio + low TMATMUL ratio**
  - Increase reuse (larger tiles or better caching), or improve overlap (double buffering correctness).
  - Reduce redundant loads (e.g., don’t reload the same panel per inner loop).

- **Transform dominates (`TEXTRACT`/`TMOV`/layout)**
  - Increase compute per transform (batch more work per extracted tile).
  - Prefer layouts that reduce the number of transforms needed.

- **Correctness breaks after pipelining changes**
  - Re-check dependency edges and ensure every consumer waits on the right producer event/flag.
  - Validate with small shapes first; add stronger correctness checks before optimizing further.
