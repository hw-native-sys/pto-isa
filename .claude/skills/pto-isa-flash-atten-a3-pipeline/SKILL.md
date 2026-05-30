---
name: pto-isa-flash-atten-a3-pipeline
description: >-
  PTO-DSL Flash Attention four-stage cross-core software pipeline for Ascend
  A3: compute_qk (Cube) -> compute_p (Vec) -> compute_pv (Cube) -> compute_gu
  (Vec), staged through a GM software FIFO. Captures the steady-state rhythm
  (cube-side per-tile emit_qk_pv interleaving, vec-side "drain GU then produce
  P"), the QK_PRELOAD / EXP_RING / S1_TILE knobs and their invariants, the UB
  192 KiB budget with the row_slice working-tile shrink, the empirical
  S1 >= 16384 -> S1_TILE = 512 recommendation, and the op-pattern PIPE_V barrier removal recipe. Use
  when tuning the in-tree DSL Flash Attention, porting the four-stage pipeline
  to a new persistent-block kernel that mixes cube + vec stages through a GM
  FIFO, choosing QK_PRELOAD / S1_TILE for a new shape mix, or deciding when a
  PIPE_V barrier in generated C++ is safe to drop. Scoped to A3 non-causal
  prefill with HEAD=128, S0=128, CUBE_S1=128 -- other Flash Attention flavors
  (causal mask, GQA/MQA, KV-cache decode, A5 NZ/NZ+1 layout) belong in
  sibling skills.
---

# PTO-ISA Flash Attention A3 Pipeline (4-Stage Cube/Vec + GM FIFO)

A single, focused recipe: how to schedule a Flash Attention prefill kernel on Ascend A3 so the cube and vector cores cooperate through a GM-staged software FIFO at the rhythm of the 140 TFLOPS reference. The recipe is implemented in the in-tree PTO-DSL kernel under `kernels/python/flash_atten/`; everything you need to apply, tune, or port it lives in this skill -- the source tree is only there if you want to see it wired end-to-end.

## Quick Start

- Read [references/fa-4stage-pipeline-recipe.md](references/fa-4stage-pipeline-recipe.md) -- the full pipeline (stages, prologue / steady / epilogue, ring invariants, anti-patterns).
- Read [references/ub-budget-and-tile-sizing.md](references/ub-budget-and-tile-sizing.md) -- 192 KiB UB budget, row_slice 64 KiB -> 32 KiB shrink, and the empirical `S1 >= 16384 -> S1_TILE = 512` recommendation.
- Read [references/pipe-v-barrier-patterns.md](references/pipe-v-barrier-patterns.md) -- which generated-C++ `pipe_barrier(PIPE_V)` patterns are safe to drop and why.
- Reproduce the canonical numbers on Ascend 910B2:
  ```bash
  cd ${repo}/kernels/python/flash_atten
  python3 run.py                       # full case1..case8 sweep
  python3 run.py --case case1          # one shape
  ```
  `run.py` rebuilds `build_artifacts/fa.so` per `FA_Q_ROWS`, and its default suite picks `FA_S1_TILE=512` automatically for `S1 >= 16384` (`case5..case8`) as an empirically good default.

## Core Workflow

1. **Diagnose** -- confirm the kernel is pipeline-bound (cube or vec idle while the other is busy, or `pipe_barrier(PIPE_V)` stalls dominating the vec side). For the generic profiling loop see [docs/coding/performance-best-practices.md](../../../docs/coding/performance-best-practices.md).
2. **Apply the recipe** -- four-stage cross-core pipeline with prologue + steady + epilogue, GM-staged FIFO between stages, exp_max ring sized to `QK_PRELOAD`. See [references/fa-4stage-pipeline-recipe.md](references/fa-4stage-pipeline-recipe.md).
3. **Size the tiles** -- start from the empirical defaults (`S1_TILE=256`, and recommended `512` for `S1 >= 16384`), confirm the row_slice shrink keeps the per-iteration working tile at 32 KiB, validate the UB budget on the host. See [references/ub-budget-and-tile-sizing.md](references/ub-budget-and-tile-sizing.md).
4. **Patch barriers** -- run with the default `gu` pattern; only add `softmax-exp-sum` or `softmax-sum-add` after the patch tool confirms there is no direct tile dependency. See [references/pipe-v-barrier-patterns.md](references/pipe-v-barrier-patterns.md).
5. **Verify** -- correctness vs a host FP32 reference at small shapes; latency and TFLOP/s vs `torch_npu.npu_fused_infer_attention_score` for all benchmark sizes.

## Scope

This skill is **deliberately narrow** -- it covers only the A3 non-causal prefill four-stage pipeline at the in-tree shape envelope (`HEAD=128`, `S0=128`, `CUBE_S1=128`). Anything outside that scope belongs in a different skill:

- Causal mask, GQA / MQA, KV-cache decode, other head dims -> add a sibling skill, do not extend this one. `tile.triu` is not exposed in the current DSL so causal here is a hard "not supported", not a tuning knob.
- A5 NZ / NZ+1 layout Flash Attention (`kernels/manual/a5/flash_atten/`) -> separate skill; the buffer pyramid and layout assumptions are different.
- Matmul L2-reuse scheduling -> see the GEMM L2-schedule sibling skill.
- Generic PTO optimization concepts (instruction selection, pipeline overlap methodology, profiling) -> covered by [docs/coding/opt.md](../../../docs/coding/opt.md) and [docs/coding/performance-best-practices.md](../../../docs/coding/performance-best-practices.md).
- Build / constraints / debugging / review guardrails -> covered by [../pto-isa-dev/SKILL.md](../pto-isa-dev/SKILL.md).

## Working Rules

- Treat `EXP_RING == QK_PRELOAD` as a hard invariant. Changing one without the other lets softmax(t + QK_PRELOAD) clobber the rescale factor GU(t) is still using. The host raises a ValueError today; do not relax it.
- Validate `S1 >= S1_TILE * QK_PRELOAD` and `S1 % S1_TILE == 0` on the host before launching -- surface a clear error rather than letting the FIFO underrun mid-kernel.
- Keep `S1_TILE` selected per-shape rather than hard-coded: `S1_TILE=256` is the baseline empirical value, and `S1 >= 16384` is recommended to use 512 because it has measured better on the current default shapes. This is not a hard rule or a proof of optimality; treat it as the default heuristic to beat. See the UB budget reference for why a higher S1_TILE is not free.
- The default `gu` PIPE_V removal pattern is stable and on by default. Adding `softmax-exp-sum` or `softmax-sum-add` requires the patch tool's direct-tile-dependency check to pass -- do not enable them blind.
- Prefer the op-pattern barrier removal over the legacy line-number list (`FA_REMOVE_VEC_BARRIERS`). The line-number path is kept only as an experimental fallback and breaks on every ptoas emit churn.
- This kernel is A3 only (`--pto-arch=a3 --npu-arch=dav-2201` in `compile.sh`). HEAD / S0 / CUBE_S1 are baked. Do not invent a CANN platform_config wiring for it -- the shape envelope is fixed.
- When porting the four-stage pattern to a different cube+vec persistent-block kernel, keep prologue / steady / epilogue separate (do not collapse the prologue's QK_PRELOAD preloads into the steady loop) and keep the GM-staged FIFO between cube and vec rather than direct cube-to-vec eventing.

## Where the Reference Implementation Lives

`kernels/python/flash_atten/` -- the PTO-DSL Flash Attention kernel:

- `kernels/fa_builder.py` -- the four-stage DSL builder: cube `compute_qk` + `compute_pv`, vec `compute_p` + `compute_gu`, the GM FIFO layout, the steady-state `emit_qk_pv_interleaved`, the vec "drain GU first" reordering, the `exp_max` ring and slot dispatch.
- `scripts/patch_vec_barriers.py` -- the op-pattern PIPE_V barrier remover (`gu` / `softmax-exp-sum` / `softmax-sum-add`).
- `compile.sh` -- end-to-end build: DSL -> MLIR -> ptoas C++ -> optional barrier patch -> bisheng -> `build_artifacts/fa.so`.
- `run.py` -- per-case build flow (rebuilds per `FA_Q_ROWS`), `S1_TILE` auto-selection, host FP32 / `torch_npu` correctness + benchmark harness.

`kernels/manual/common/flash_atten/` -- the manual C++ kernel this DSL version mirrors. Open it when you need to see what the DSL gave up (e.g. fused "wide K load + sub-tile matmul into ACC subview", `tile.triu` causal) and why.
