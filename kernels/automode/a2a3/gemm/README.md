# High-Performance GEMM Operator Example

## Overview

This example demonstrates how to implement a high-performance GEMM operator using PTO and common optimization techniques (core partitioning, base-block selection, L1 caching, and double buffering).

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/manual/a2a3/gemm_performance/
├── scripts/
│   └── gen_data.py                  # Generates input and golden output
├── CMakeLists.txt                   # Build configuration
├── gemm_performance_kernel.cpp      # Kernel implementation
├── main.cpp                         # Host-side entry point
└── run.sh                           # Convenience script
```

## Operator Description

### Function

This example implements GEMM:

$$
C = A \times B
$$

Where:

- `A` is `m×k`
- `B` is `k×n`
- `C` is `m×n`

The default reference configuration in `main.cpp` uses `m=k=n=6144`.

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `GEMM` |
| Inputs      | `a`: `m×k`, `float16`, `ND`; `b`: `n×k`, `float16`, `ND` |
| Output      | `c`: `m×n`, `float`, `ND` |
| Kernel name | `GEMMPerformance` |

## Optimization Notes

This example uses a 24-core A3 platform as the performance validation platform.

- **Core partitioning**: maximize parallelism by splitting work across Cube cores. Since `m`, `n`, and `k` are equal, prefer not splitting `k` within a single core, and split `m` and `n` across 24 cores. A `4 × 6` grouping yields `singleCoreM=1536`, `singleCoreK=6144`, `singleCoreN=1024` (chosen by this example).
- **Base block selection**: choose base blocks that maximize compute-to-memory ratio. For FP16, a common choice is `[baseM, baseN, baseK] = [128, 256, 64]`, which improves arithmetic intensity versus `[128, 128, 128]` while maintaining 512-byte-aligned GM writes.
- **L1 caching**: move multiple base blocks from GM to L1 per transfer to improve bandwidth utilization. This example sets `stepKa=stepKb=4` to cache four `k` blocks at a time.
- **Double buffering**: overlap DMA and compute by enabling double buffering in L1, L0A, and L0B.

## Tiling Parameters

| Parameter     | Value |
| ------------- | ----- |
| `m`           | 6144  |
| `k`           | 6144  |
| `n`           | 6144  |
| `singleCoreM` | 1536  |
| `singleCoreK` | 6144  |
| `singleCoreN` | 1024  |
| `baseM`       | 128   |
| `baseK`       | 64    |
| `baseN`       | 256   |
| `stepM`       | 1     |
| `stepKa`      | 4     |
| `stepKb`      | 4     |
| `stepN`       | 1     |

## Measured Performance (Reference)

The following measurements were collected on Ascend A3 (24 cores) for several `m=k=n` sizes (fp16 inputs → fp32 output).

| Parameter | TMATMUL (Cube) Ratio | TEXTRACT Ratio | TLOAD Ratio | TSTORE Ratio | Execution time (ms) |
| --- | --- | --- | --- | --- | --- |
| `m=1536` `k=1536` `n=1536` | 54.5% | 42.2% | 72.2% | 7.7% | 0.0388 |
| `m=3072` `k=3072` `n=3072` | 79.0% | 62.0% | 90.9% | 5.8% | 0.2067 |
| `m=6144` `k=6144` `n=6144` | 86.7% | 68.1% | 95.2% | 3.1% | 1.5060 |
| `m=7680` `k=7680` `n=7680` | 80.6% | 63.0% | 98.4% | 2.4% | 3.1680 |

### What the numbers suggest

These metrics are most useful for answering a single question: **which engine is limiting the end-to-end pipeline**.

- **Scaling behavior**: execution time grows super-linearly with `m=k=n` (as expected for `O(n^3)` work), and throughput typically improves from small sizes to mid sizes before flattening.
- **TMATMUL utilization rises, then drops**: TMATMUL (Cube) Ratio increases from 54.5% → 86.7% as the problem grows (better amortization and steadier pipelines), then drops to 80.6% at `7680³`. This pattern usually indicates the compute pipeline is no longer the only limiter at the largest size.
- **TLOAD is near-saturated at large sizes**: TLOAD Ratio grows to 98.4% at `7680³`, suggesting the GM feed path is close to its limit and starts throttling compute (TMATMUL Ratio decreases).
- **TSTORE is small and keeps shrinking**: output writeback is a small fraction of total time for GEMM, especially at larger sizes (one write for many FMAs).
- **TEXTRACT is meaningful overhead**: the 42%→68% range suggests L1→L0 extract/layout costs are not negligible; optimizing this stage (and overlapping it cleanly) directly impacts overall performance.

If you want a single rule of thumb: when **TLOAD Ratio approaches ~100%**, you are usually **memory-feed limited** (even if TMATMUL still looks “busy”), and further speedups come from reducing bytes moved per FLOP and improving overlap.

## Performance Optimization Guide (How to Tune This Kernel)

This example is intentionally structured around a standard GEMM pipeline:

1. **TLOAD stage**: GM → L1 (`TLOAD` into `aMatTile[]` / `bMatTile[]`)
2. **TEXTRACT stage**: L1 → L0A/L0B (`TEXTRACT` into `aTile[]` / `bTile[]`)
3. **TMATMUL stage**: L0A/L0B → L0C (`TMATMUL` / `TMATMUL_ACC` into `cTile`)
4. **TSTORE stage**: L0C → GM (`TSTORE` of `cTile`)

The core kernel implementation is in `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`, with the critical control points below.

### 1) Partition work across cores first

Look at `InitGMOffsets(...)`:

- The kernel splits the global `C[m,n]` into `blockDim` independent tiles.
- For square problems (`m≈n`), splitting across **both `m` and `n`** usually gives better balance than splitting only one dimension.

Checklist:

- Ensure `m % singleCoreM == 0` and `n % singleCoreN == 0`.
- Choose a 2D grid decomposition (`m`-tiles × `n`-tiles) that matches `blockDim` so each core gets a contiguous `A` panel and `B` panel.

### 2) Choose base tiles that fit L0A/L0B cleanly

Look at `InitBuffers(...)`:

- L0A and L0B are explicitly double-buffered with a 32 KiB ping/pang split (`0x0` and `0x0 + 32768`).
- This implies an important constraint: the per-buffer tile footprint must be ≤ 32 KiB.

For fp16 inputs (2 bytes/elem):

- L0A tile bytes ≈ `baseM * baseK * 2`
- L0B tile bytes ≈ `baseK * baseN * 2`

The reference uses:

- `baseM=128, baseK=64` → `128*64*2 = 16 KiB` (fits comfortably)
- `baseK=64, baseN=256` → `64*256*2 = 32 KiB` (fills the budget)

Guidelines:

- Prefer tile sizes that **fully utilize** the 32 KiB budget (especially for `B`), but do not exceed it.
- Keep `baseK` aligned to the Cube’s preferred K granularity (often 32/64/128 depending on data type and layout).

### 3) Increase reuse with L1 “stepK” caching (without overflowing)

Look at `ProcessKIteration(...)` and the `kModstepKa` logic:

- `stepKa` / `stepKb` control how many `K`-slices are staged into L1 per DMA.
- The example uses `stepKa=stepKb=4`: one `TLOAD` transfer brings in 4 micro-panels that are later `TEXTRACT`’d.

Guidelines:

- Increase `stepK` to reduce DMA launch overhead and improve burst efficiency **until L1 capacity or overlap breaks down**.
- If TLOAD is near 100% and TMATMUL drops, try:
  - increasing `stepK` (more reuse per fetch), or
  - increasing compute intensity (e.g., larger `baseN`/`baseM` if L0 allows), or
  - improving overlap (next section).

### 4) Keep the pipeline overlapped (avoid bubbles)

The double-buffering flags (`mte2DBFlag`, `mte1DBFlag`) and event flow are the performance heart of this kernel:

- **TLOAD** loads next `aMatTile[]/bMatTile[]` while
- **TEXTRACT** extracts next `aTile[]/bTile[]` while
- **TMATMUL** computes current `TMATMUL[_ACC]`.

If you see:

- **high TLOAD but low TMATMUL** → the Cube is starving; overlap is insufficient or TLOAD is truly saturated.
- **high TEXTRACT but low TMATMUL** → extract/layout is the limiter; reduce `TEXTRACT` cost or increase compute per extract.

Practical tuning steps:

- Make sure the “first-iteration warmup” and “last-iteration drain” do not serialize the steady-state loop. This file already includes “supplement first/last sync instr”; keep them if you refactor.
- Keep compute and data movement in separate phases per buffer index (ping/pang), and only `wait_flag` at true dependency boundaries.

### 5) When scaling to new shapes, re-tune the *core tile* first

For different `m/k/n`, do not only change the constants:

- Recompute `singleCoreM/singleCoreN` so each core gets a similar amount of work.
- Recheck `mLoop`, `nLoop`, and `kLoop` (`RunGemmE2E`), because loop trip counts strongly affect overlap efficiency.

Common failure mode:

- Very large `kLoop` with insufficient `stepK` can make TLOAD dominate; very small `kLoop` can make overhead dominate.

### 6) Use the utilization ratios to decide *what* to optimize

From the measurements above:

- `7680³` has **TLOAD=98.4%** and TMATMUL down to **80.6%** → focus on reducing GM traffic (higher reuse, better cache staging) and improving overlap rather than micro-optimizing `TMATMUL`.
- Mid sizes (`3072³`, `6144³`) show strong TMATMUL and TLOAD simultaneously → pipeline is close to balanced; improvements require careful end-to-end changes.

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Generate input + golden output:

```bash
cd ${git_clone_path}/kernels/manual/a2a3/gemm_performance
python3 scripts/gen_data.py
```

3. Run the example:

```bash
bash run.sh -r npu -v Ascend910B1
```

If the run succeeds, the output prints:

```text
test success
```

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2025-12-15 | Adjusted example directory and added this README |
