# Flash Attention 4-Stage Cube/Vec Pipeline (A3 Non-Causal Prefill)

A drop-in software-pipeline recipe for `Q @ K^T -> softmax -> @ V -> rescale-accumulate` on Ascend A3, run as a cooperating pair of cube + vec kernels staged through a GM software FIFO. Everything you need to apply, tune, and port the pattern lives in this file -- no source-file lookup required. The end of the file points at the in-tree implementation if you want to see it running.

## 1. When to Use This Pattern

Reach for this pipeline when **all** of these hold:

- You are doing Flash Attention prefill on Ascend A3 (`--pto-arch=a3 --npu-arch=dav-2201`).
- `HEAD = 128`, `S0 = 128` per Q block, `CUBE_S1 = 128`, non-causal attention. (Other variants belong in a sibling skill -- see scope.)
- The work splits cleanly into four stages that alternate cube and vector: matmul `Q K^T`, streaming softmax, matmul `P V`, rescale + accumulate into `O`.
- The single-core throughput of either pipe is limited by the other -- cube idling on QK while vec is still rescaling, or vec stalling on `pipe_barrier(PIPE_V)` while cube has the next QK ready.

Skip this pattern (see section 9) when `S1 < S1_TILE * QK_PRELOAD` (FIFO cannot prime), when the shape is small enough that the prologue dominates (e.g. `Q_ROWS == S0` and `S1 == S1_TILE * QK_PRELOAD`), or when the kernel needs causal masking -- `tile.triu` is not exposed in the current DSL.

## 2. Why It Works

**Four-stage cross-core pipeline** splits the inner attention work into two cube stages (`compute_qk`, `compute_pv`) and two vector stages (`compute_p`, `compute_gu`). Each pair runs on its own pipe, so cube matmul and vec softmax / rescale overlap by construction:

```text
   tile_id:       0       1       2       3       4   ...
   cube QK:    [QK0]   [QK1]   [QK2]   [QK3]   [QK4] ...
   vec  P :          [P0]    [P1]    [P2]    [P3]   ...   (lags by QK_PRELOAD-1)
   cube PV:                 [PV0]   [PV1]   [PV2]   ...
   vec  GU:                        [GU0]   [GU1]   ...
```

**GM-staged software FIFO** between stages decouples cube and vec timing. The slot ring is 8-deep (`dir_mask=1/2 -> slot_num=8` on A3). Three logical pipes share one GM buffer:

- `QK pipe` (cube -> vec, fp32 `[S0, S1_TILE]`)
- `PV pipe` (cube -> vec, fp32 `[S0, HEAD]`)
- `P pipe`  (vec -> cube, fp16 `[VecGuRows, S1_TILE]`)

**QK_PRELOAD** lets cube emit several QK slots before vec ever starts. With `QK_PRELOAD = 3`, the steady state always has at least 3 QK tiles in flight, so any single-iteration stall on the vec side is absorbed before cube notices.

**exp_max ring** sized to `QK_PRELOAD` (hard invariant `EXP_RING == QK_PRELOAD`) keeps the streaming-softmax rescale factor for `GU(t)` alive until `GU(t)` consumes it -- but no longer. Make the ring shorter and softmax(t + QK_PRELOAD) clobbers the factor GU(t) is still using; make it longer and vec UB runs out.

## 3. Pipeline Skeleton (Drop-In Structure)

The cube and vec kernels share a three-part envelope: **prologue** (prime the FIFO), **steady** (per-tile body), **epilogue** (drain). Keep the three sections distinct -- collapsing the prologue into the steady loop tends to break the lag-by-`QK_PRELOAD` invariant.

```text
# ----- cube kernel (compute_qk + compute_pv) -----
for qb in range(qb_start, qb_end):
    load Q[qb]
    # prologue: emit QK[0..QK_PRELOAD-1] back-to-back
    for kp in range(QK_PRELOAD):
        emit_qk(kp)

    # steady: per-tile, drain current PV then issue next QK
    for tile_id in range(0, steady_tiles):
        next_tile = tile_id + QK_PRELOAD
        emit_qk_pv_interleaved(next_tile, tile_id)
        # -> inside: tpop P; for sub in TILE_FACTOR { emit_pv_sub; emit_qk_sub }
        # The interleaving keeps both ACC accumulators warm.

    # epilogue: drain the last QK_PRELOAD PVs
    for k in range(QK_PRELOAD):
        emit_pv(steady_tiles + k)

# ----- vec kernel (compute_p + compute_gu) -----
for qb in range(qb_start, qb_end):
    # prologue: softmax for the QK_PRELOAD tiles cube already pushed
    for kp in range(QK_PRELOAD):
        emit_softmax(exp_max_ring[kp], is_init=(kp == 0))

    # steady: 140tflops order -- drain current PV/GU before producing next P
    if steady_tiles > 0:
        emit_gu(exp_max_ring[0], is_init=True)
        emit_softmax(exp_max_ring[QK_PRELOAD % EXP_RING], is_init=False)
        for tile_id in range(1, steady_tiles):
            next_tile = tile_id + QK_PRELOAD
            emit_gu_update_dispatch(tile_id)       # consume current PV, update O
            emit_softmax_dispatch(next_tile)       # produce future P

    # epilogue: drain the last QK_PRELOAD gus
    for k in range(QK_PRELOAD):
        emit_gu_any(steady_tiles + k)

    # final divide + GM store of O
```

Two non-obvious rules:

- **Cube interleaving** (`emit_qk_pv_interleaved`) walks the `TILE_FACTOR` sub-tiles of one logical S1 tile, interleaving `emit_pv_sub(current)` and `emit_qk_sub(next)`. The PV `talloc` happens on the first sub and the QK `tpush` happens on the last sub -- this matches the 140 TFLOPS reference and keeps the cube pipes back-to-back.
- **Vec drain-first** is the inverse: in the steady-state body, `emit_gu_update_dispatch(tile_id)` runs **before** `emit_softmax_dispatch(next_tile)`. Producing the future P first looks symmetric but it leaves the rescale factor in `exp_max_ring[next_tile % EXP_RING]` overwritten before `GU(tile_id)` reads it.

## 4. Knob Reference

| Knob | Default | Other values | Hard invariant |
| --- | --- | --- | --- |
| `S1_TILE` (`FA_S1_TILE`) | 256 | 512 (recommended for `S1 >= 32768` by current measurements) | Must be a multiple of `CUBE_S1` (`= 128`); the current library implements only `{256, 512}`. Extending this set requires adapting the implementation and validating the new shape, not just changing the heuristic. The 32768 threshold is an empirical recommendation, not a hard rule. |
| `QK_PRELOAD` (`FA_QK_PRELOAD`) | 3 (DSL) | 4 (manual parity) | Allow-list `{3, 4}`. |
| `EXP_RING` (`FA_EXP_RING`) | `= QK_PRELOAD` | -- | Must equal `QK_PRELOAD`; host raises `ValueError` otherwise. |
| `QK_PRELOAD * S1_TILE` floor on `S1` | -- | -- | `S1 >= QK_PRELOAD * S1_TILE` and `S1 % S1_TILE == 0`. |
| `CAUSAL_MASK` | `False` | -- | `True` requires `tile.triu` which is not exposed in the current DSL. |

The DSL builder's defaults come from the 140 TFLOPS reference, not the manual C++ kernel. The manual kernel uses `QK_PRELOAD = 4`; the DSL drops to 3 and shrinks `EXP_RING` to match, which is what frees the vec UB headroom that the row_slice working-tile shrink (see the UB budget reference) needs.

## 5. Ring and Slot Invariants the Porter Must Enforce

Validate on the host before building the kernel -- surface a clear error rather than letting the FIFO mis-pace:

- `QK_PRELOAD == EXP_RING` -- see section 2; the host today refuses any mismatch.
- `S1 >= S1_TILE * QK_PRELOAD` -- the prologue needs that many tiles to prime.
- `S1 % S1_TILE == 0` -- the steady loop is `num_tiles = S1 // S1_TILE`.
- `S1_TILE % CUBE_S1 == 0` -- `TILE_FACTOR = S1_TILE // CUBE_S1` must be integer for the sub-tile loop.
- `Q_ROWS % S0 == 0` -- one Q block per AIC core walk.
- Slot count is 8 on A3 (`dir_mask=1/2 -> slot_num=8`). Do not parameterize it; it is set by the platform.

Soft rules:

- Keep `EXP_RING` slot dispatch via `tile_id % EXP_RING`. The previous 8-slot unrolled ring (where `EXP_RING` was hard-wired to the slot count) wasted vec UB and only worked at `QK_PRELOAD = 4`.
- When porting the four-stage envelope to a different cube+vec kernel, keep the three sections (prologue / steady / epilogue) and keep the GM FIFO. Direct cube-to-vec eventing without GM staging is tempting but the slot ring is what lets the two pipes lag by `QK_PRELOAD - 1` without coupling their timing.

## 6. GM FIFO Layout Note

The GM buffer is one block per AIC, partitioned per-pipe:

```text
GM_QK_OFF_F32 = 0
GM_PV_OFF_F32 = (SLOT_SIZE_QK * SLOT_NUM) // 4
GM_P_OFF_F32  = GM_PV_OFF_F32 + (SLOT_SIZE_PV * SLOT_NUM) // 4

SLOT_SIZE_QK = S0 * S1_TILE * 4   # fp32 QK accumulator
SLOT_SIZE_PV = S0 * HEAD    * 4   # fp32 PV accumulator
SLOT_SIZE_P  = S0 * S1_TILE * 2   # fp16 softmax(QK) sent vec -> cube
SLOT_NUM     = 8                  # A3 dir_mask=1/2
```

The QK slot is described to the DSL as a `[S0, S1_TILE]` fp32 tensor; cube writes it in `TILE_FACTOR` sub-tiles of `[S0, CUBE_S1]`, vec consumes it in `TILE_FACTOR` row_slices of `[Vec_S0, S1_TILE]` (`Vec_S0 = S0 / VEC_CORES / TILE_FACTOR`). The two views into the same GM region are what mediates the cube-vec width / row mismatch without any explicit transpose.

The P slot is the symmetric story for vec -> cube: vec writes `[VecGuRows, S1_TILE]` (one row_slice per slice), cube reads `TILE_FACTOR` sub-tiles of `[S0, CUBE_S1]`. The PV slot is `[S0, HEAD]` and does not scale with `S1_TILE` because one PV is produced per logical S1 tile by accumulating sub-PV matmuls into the same accumulator.

## 7. Verification

The PTO-DSL Flash Attention kernel ships eight shape-specialized presets (`case1..case8`) and a `torch_npu` baseline harness. Use it as the regression rig:

- Correctness:
  - small shapes -- compare against a host FP32 PyTorch reference,
  - all shapes -- compare against `torch_npu.npu_fused_infer_attention_score` (the run-time picks tolerance based on `Q_ROWS * S1`).
- Latency / throughput: `run.py` reports per-case `fa_us`, `fused_us`, `speedup`, and TFLOP/s (matmul + scale + softmax op count, 140 TFLOPS convention).

Regression watermark on Ascend 910B2 with the current default knobs (`QK_PRELOAD = 3`, baseline `S1_TILE = 256`, recommended/default-suite `S1_TILE = 512` for `S1 >= 32768`, default `gu` PIPE_V patch). These are the pre-cutover numbers: the `case5`/16384 row was measured at `S1_TILE = 512`, before the default-suite cutover moved to 32768:

| Case | S0 = S1 | fa us | torch_npu us | speedup |
| ---- | ------: | ----: | -----------: | ------: |
| case1 |   1024 | 43.82 |   137.71 | 3.14x |
| case2 |   2048 | 56.92 |   154.92 | 2.72x |
| case3 |   4096 | 113.83 |  207.80 | 1.83x |
| case4 |   8192 | 283.12 |  365.81 | 1.29x |
| case5 |  16384 | 953.47 |  974.65 | 1.02x |
| case6 |  32768 | 3177.41 | 3159.95 | 0.99x |
| case7 |  65536 | 12093.10 | 12152.60 | 1.00x |
| case8 | 131072 | 48380.71 | 48061.93 | 0.99x |

Reading the band:

- `case1..case4` are launch / overhead favored -- the DSL kernel beats the fused op by 3.1x -> 1.3x as the shape grows.
- `case5..case8` are compute-bound large prefill -- the kernel sits within +-2% of `torch_npu`. Anything materially below 0.95x at these shapes is a regression in the pipeline (re-check `QK_PRELOAD`, `EXP_RING`, and the `gu` PIPE_V patch).
- `case5` (16384) is the last default-suite case below the empirical 32768 recommendation boundary: it still uses `S1_TILE = 256`, so it sees the largest scheduling overhead per S1 tile.

## 8. Anti-Patterns

Do not pick this schedule when:

- `S1 < S1_TILE * QK_PRELOAD`. The prologue cannot prime and the steady loop runs zero iterations -- you spend the whole kernel in epilogue.
- `Q_ROWS == S0` and `S1 == S1_TILE * QK_PRELOAD`. Steady-state is zero; the prologue + epilogue is all there is and the pipeline overlap never kicks in.
- The shape needs causal masking. `tile.triu` is not exposed; sub-tile masking is not a workaround that fits this pipeline. Use a different recipe.
- A new SoC or a different head dim. This kernel is locked to A3 / `HEAD = 128` / `S0 = 128` / `CUBE_S1 = 128`. Re-deriving the GM FIFO and UB budget for a new envelope is a separate Skill, not a parameter sweep.
- You are tempted to make `EXP_RING > QK_PRELOAD` "just to have headroom". It is not headroom -- it is extra vec UB that the row_slice shrink needs.

## 9. Contrast: Manual C++ Reference Kernel

The kernel this DSL recipe mirrors is `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`. The pipeline structure is **identical** (same four stages, same GM FIFO, same `QK_PRELOAD`-deep prologue + epilogue) but the manual version has three things this DSL version cannot do:

- Fused "load wide K once, sub-tile matmul into ACC subview". The DSL MAT/RIGHT subview verifier rejects partial-column subviews, so the DSL kernel uses one full DSL tile at `S1_TILE` width.
- `tile.triu`-based causal masking.
- Ping-pong tile storage. `pto.alloc_tile` is single-output; the DSL builder uses Python aliases (`[buf, buf]`) to preserve the `[buf]` indexing pattern from the C++ source without pretending it has ping-pong storage. The L1 / L0 double-buffer still works because `--enable-insert-sync` schedules around it.

The DSL kernel exists to validate that the Python DSL can express a production-grade four-stage pipeline at near-`torch_npu` performance. The manual kernel is the parity target, not the always-better baseline -- on `case5..case8` the DSL kernel is within +-2% of `torch_npu` (and so within noise of the manual C++ kernel).

## In-Tree Reference

The recipe was extracted from the PTO-DSL Flash Attention under `kernels/python/flash_atten/`:

- `kernels/fa_builder.py` -- the four-stage DSL builder, the GM FIFO layout, the steady-state `emit_qk_pv_interleaved`, the vec "drain GU first" reordering, the `exp_max` ring and slot dispatch.
- `run.py` -- per-case build flow (rebuilds `fa.so` per `FA_Q_ROWS`), `S1_TILE` auto-selection, host FP32 / `torch_npu` correctness + benchmark harness.
- `compile.sh` -- DSL -> MLIR -> ptoas C++ -> optional barrier patch -> bisheng -> `build_artifacts/fa.so`.

Open those files only when you want to see the recipe wired end-to-end. The formulas, invariants, and verification path in this file should be enough to apply the recipe to a new kernel on their own.
