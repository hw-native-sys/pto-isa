# Matmul Schedule: N-Group Swizzle + M-Direction Zigzag

A drop-in scheduler for L2-bound matmul-shaped kernels on Ascend A2/A3. Everything you need to apply, port, and tune the pattern lives in this file — no source-file lookup required. The end of the file points at the in-tree implementation if you want to see it running.

## 1. When to Use This Pattern

Reach for this schedule when **all** of these hold:

- The kernel is matmul-shaped: an output grid of `m_tiles x n_tiles` produced by accumulating along K.
- The chip is A2/A3 24-core class with on the order of tens of MiB of shared L2.
- `n_tiles = n // baseN` is materially larger than 1, so the same B panel can serve multiple M rows.
- `b_tile_bytes = baseN * k * sizeof(elem)` is small enough that several B tiles co-reside in L2 (use the budget formula in section 4).
- A manual SPMD-static partition (e.g. `mIterIdx = block_idx % mIter; nIterIdx = block_idx / mIter`) profiles as TLOAD-saturated on B.
- You want persistent block scheduling — one launched block walks many output tiles — rather than one-block-per-output-tile.

Skip this pattern (see section 9) when total tiles barely exceed `blockDim`, `n_tiles == 1`, one B tile already does not fit in L2, or K is so small the K-panel double-buffer is already cube-idle.

## 2. Why It Works

**N-group swizzle** groups `swizzleCountN` adjacent N tiles and runs the inner walk along M within the group. The same B panel stays hot in L2 across `m_tiles` rows, so B GM traffic drops by roughly a factor of `swizzleCountN` in the inner region.

**M-direction zigzag** reverses the M walk every other N-group. The boundary A tile between groups is the A tile the previous group just finished with, so A also gets boundary reuse instead of a cold restart at every group switch.

Together the two cut GM traffic enough that the K-panel double-buffered L1 pipeline can keep the cube fed at large square shapes. On Ascend910B2 the recipe reaches 0.87x–0.98x of `torch_npu` (ABt) on compute-bound shapes, peaking around 309 TFLOPS at 6144³.

## 3. Tile-Id Math (Drop-In Formulas)

The scheduler runs a single persistent block loop and computes per-output-tile `(m_idx, n_idx)` indices from `tile_id`. Copy the formulas verbatim into any new persistent-block kernel; only rename the two outer axes if needed.

```text
# bid          = block_idx
# block_num    = number of launched blocks
# m_tiles      = m // baseM
# n_tiles      = n // baseN
# output_tiles = m_tiles * n_tiles

for tile_id in range(bid, output_tiles, step=block_num):           # persistent block

    # --- N-group swizzle: split the grid into groups of swizzleCountN N-tiles ---
    tile_block_loop   = ceil_div(n_tiles, swizzleCountN)            # number of N-groups
    tile_block_span   = swizzleCountN * m_tiles                     # tiles per N-group
    tile_block_idx    = tile_id // tile_block_span                  # which N-group
    in_tile_block_idx = tile_id %  tile_block_span                  # position within group

    is_last_block     = tile_block_idx == (tile_block_loop - 1)
    n_col_tail        = n_tiles - swizzleCountN * tile_block_idx    # tail-group width
    n_col             = n_col_tail if is_last_block else swizzleCountN

    base_m_idx        = in_tile_block_idx // n_col
    base_n_idx        = tile_block_idx * swizzleCountN + (in_tile_block_idx % n_col)

    # --- M-direction zigzag: reverse M on odd N-groups ---
    odd_block         = (tile_block_idx % 2) == 1
    flipped_m_idx     = m_tiles - base_m_idx - 1
    m_idx             = flipped_m_idx if odd_block else base_m_idx
    n_idx             = base_n_idx

    # m_offset = m_idx * baseM
    # n_offset = n_idx * baseN
    # ... load A[m_offset:m_offset+baseM, :], B[:, n_offset:n_offset+baseN], accumulate K, store C tile ...
```

Notes for a porter:

- The math is shape-agnostic. Any two-axis output grid traversed by a persistent block loop can use it.
- `swizzleCountN = 1` degenerates to row-major over the base-tile grid — useful as a debug fallback.
- The tail-group case (`is_last_block`) keeps coverage correct when `n_tiles % swizzleCountN != 0`. Do not skip it.
- The K-panel pipeline inside one output tile is orthogonal to this scheduler. Pair it with a standard double-buffered L1 + ping-pong L0A/L0B layout (see section 7 for the B-MAT side).

## 4. L2 Budget Formula and the 32 MiB Cliff

`swizzleCountN` is not a free knob: it is bounded by how many B tiles plus one active A tile fit in L2 with safety margin.

```text
a_tile_bytes      = baseM * k * sizeof(elem)          # e.g. fp16 -> 2 bytes
b_tile_bytes      = baseN * k * sizeof(elem)

# L2-usage ratio: bumped UP (less safety margin) when one B tile already
# dominates L2, otherwise the conservative 0.70 budget would refuse to fit
# even one B tile and swizzleCountN would collapse to 1.
safety            = 0.90 if b_tile_bytes >= 32 MiB else 0.70

l2_budget         = l2_size * safety
available_for_b   = l2_budget - a_tile_bytes
l2_max_count_n    = available_for_b // b_tile_bytes

# Tested upper cap from the PR sweep. Not an architectural limit.
SWIZZLE_TARGET_CAP = 5

swizzleCountN     = clamp(1, min(n_tiles, SWIZZLE_TARGET_CAP, l2_max_count_n))
```

Why the **32 MiB cliff** exists: a B tile near or above 32 MiB plus an A tile already crowds L2. At the conservative 0.70 budget, `available_for_b = 0.70 * l2_size - a_tile_bytes` would go negative or near-zero and `swizzleCountN` would collapse to 1, killing the schedule. The L2-usage ratio is therefore **raised** to **0.90** — using more of L2, accepting a tighter margin against noise (concurrent traffic, conflict misses, prefetch overshoot) — so the large B tile and the active A tile still fit and grouping is still possible. Below 32 MiB the **0.70** ratio holds because the working set is small enough to leave generous headroom for absorbing noise.

The `SWIZZLE_TARGET_CAP = 5` is empirical from the PR run-table. Porters with a different shape mix may sweep higher in their own benchmarks; it is a heuristic, not a hard limit.

## 5. Runtime Wiring from CANN platform_config

The schedule's two runtime knobs both come from the SoC's CANN platform config — no recompile for a new chip.

- Resolve the CANN `platform_config` directory via `ASCEND_HOME_PATH` or `ASCEND_TOOLKIT_HOME` (the standard `aarch64-linux/data/platform_config` and `arm64-linux/data/platform_config` subpaths, plus the usual `/usr/local/Ascend/...` fallbacks).
- Read two integers from `<soc>.ini`: `cube_core_cnt` and `l2_size`.
- Then:
  ```text
  total_base_tiles = m_tiles * n_tiles
  blockDim         = min(cube_core_cnt, total_base_tiles)
  swizzleCountN    = compute_from_l2_budget(baseM, baseN, k, l2_size)   # section 4
  ```

Porter checklist when adding a new SoC: drop the `.ini` into the platform_config directory CANN already publishes, then run the kernel. No code change required.

## 6. Constraints the Porter Must Enforce

Validate on the host before building the kernel — surface a clear error rather than letting it mis-schedule:

- `m % baseM == 0`, `k % baseK == 0`, `n % baseN == 0`.
- `stepKa == stepKb == 4` for the canonical K-panel pipeline. Other values require revisiting the L1 buffer math.
- `(k // baseK) % (stepKa * 2) == 0` — required so the double-buffered K-panel loop closes cleanly.
- `1 <= blockDim <= total_base_tiles` — the persistent loop must have at least one output tile per block.
- `1 <= swizzleCountN <= n // baseN`.

## 7. DN-B Layout Note

When the host already stores B in its torch-style transposed `[n, k]` layout, declare the B GM tensor with `layout="DN"` and strides `[1, K]`. The kernel still sees a logical `[K, N]` B without an explicit transpose pass.

Pair the DN B with the matching B-MAT tile config:

```python
pto.TileBufConfig(
    blayout="RowMajor",
    slayout="ColMajor",
    s_fractal_size=512,
)
```

That config is required to feed L0B correctly from a DN-laid-out B.

## 8. Verification

The PTO-DSL GEMM kernel ships nine shape-specialized presets and a `torch_npu` baseline harness. Use it as the regression rig:

- Correctness: run a single preset and check the output against `torch.matmul` with tolerance `max(0.5, 3e-5 * k)`.
- Latency: run with `--torch-npu --benchmark` to report per-case latency, TFLOPS, and the PTO / `torch_npu` ratio.
- A full sweep covers `512×2048×1536` up to `4096×16384×16384`.

Regression watermark on Ascend910B2:

- Compute-bound shapes (3072³ and above): 0.87x–0.98x of `torch_npu`.
- Peak ~309 TFLOPS at 6144³.
- Launch-bound small shape (`512×2048×1536`): ~0.56x, expected.

If the ratio drops materially below this band on the same SoC and shape, the scheduler is the first place to look — re-derive `swizzleCountN`, then check the K-panel pipeline.

## 9. Anti-Patterns

Do not pick this schedule when:

- `total_base_tiles` barely exceeds `blockDim`. The persistent loop has no payload to amortize over.
- `n_tiles == 1`. There is no N to group; swizzle degenerates and zigzag has no boundary to exploit.
- One B tile does not fit in L2 even at safety = 1.0 (`l2_max_count_n < 1`). Pick a different blocking strategy entirely; this scheduler cannot rescue a B tile that does not fit.
- K is so small that the K-panel double-buffer is already cube-idle. The bottleneck is the inner pipeline, not GM B traffic — schedule changes will not help.

## 10. Contrast: Manual SPMD-Static Baseline

The schedule this recipe replaces (when L2 reuse matters) is the standard 2D-partition GEMM:

- Work is assigned statically per core: `mIterIdx = block_idx % mIter; nIterIdx = block_idx / mIter`.
- Each core then does a flat `for-i / for-j / for-k` over its `singleCoreM / baseM`, `singleCoreN / baseN`, `singleCoreK / baseK`.
- The K-panel double-buffer + ping-pong L0A/L0B pipeline is identical.

Use the manual baseline when L2 reuse is not the bottleneck — small shapes, single-block-per-output-tile work mixes, or debugging. Switch to this recipe when the manual baseline runs out of L2 reuse and TLOAD saturates on B.

## In-Tree Reference

The recipe was extracted from the PTO-DSL GEMM under `kernels/python/gemm/`, with the contrasting manual baseline under `kernels/manual/a2a3/gemm_performance/`. Open those directories only when you want to see the recipe wired end-to-end (buffer pyramid, K-panel pipeline, per-case build flow, `torch_npu` benchmark harness). The formulas, constraints, and verification path in this file should be enough to apply the recipe to a new kernel on their own.
