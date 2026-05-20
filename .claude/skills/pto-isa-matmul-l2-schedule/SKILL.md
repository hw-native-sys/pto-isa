---
name: pto-isa-matmul-l2-schedule
description: >-
  PTO-DSL matmul L2-reuse scheduler for Ascend A2/A3: persistent-block GEMM
  with N-group swizzle along the inner M walk and M-direction zigzag at
  N-group boundaries. Captures the tile-id math, the CANN platform_config-
  driven swizzleCountN budget (with the 32 MiB safety-ratio cliff), the DN-B
  layout note, the runtime wiring, and the verification path against
  torch_npu. Use when tuning a matmul-shaped kernel that profiles as
  L2-bound, porting the swizzle/zigzag schedule to a new persistent-block
  kernel, choosing swizzleCountN for a new SoC, or deciding between the
  manual SPMD-static baseline and this persistent + swizzle schedule. Scoped
  to one schedule recipe — add a separate skill for other PTO-ISA
  performance patterns (vector reduce, flash-attention scheduling, etc.).
---

# PTO-ISA Matmul L2-Reuse Schedule (N-Group Swizzle + M-Direction Zigzag)

A single, focused recipe: how to schedule a matmul-shaped kernel on Ascend A2/A3 so the B panel stays hot in L2 across multiple M rows, and the boundary A tile is reused at every N-group switch. The recipe is implemented in the in-tree PTO-DSL GEMM under `kernels/python/gemm/`; everything you need to apply or port it lives in this skill — the source tree is only there if you want to see it wired end-to-end.

## Quick Start

- Read [references/matmul-n-swizzle-m-zigzag.md](references/matmul-n-swizzle-m-zigzag.md) — the full recipe (when to use, tile-id math, L2 budget, constraints, anti-patterns).
- Reproduce the canonical numbers on Ascend910B2:
  ```bash
  cd ${repo}/kernels/python/gemm
  python3 run.py --case a2a3_perf_6144 --torch-npu --benchmark
  ```
  The runner builds a shape-specialized kernel per case on demand and then runs it.
- To target a new SoC: drop the CANN `<soc>.ini` into the `platform_config/` directory the runner resolves (via `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME`), then run `run.py --soc <name>`. `cube_core_cnt` and `l2_size` feed `swizzleCountN` and `blockDim` directly — no code change required.

## Core Workflow

1. **Diagnose** — confirm the kernel is L2-bound (TLOAD dominates while B traffic is large and reusable). For the generic profiling loop see [docs/coding/performance-best-practices.md](../../../docs/coding/performance-best-practices.md).
2. **Apply the recipe** — persistent block + N-group swizzle + M-direction zigzag. See the reference.
3. **Wire the runtime** — derive `blockDim` and `swizzleCountN` from CANN `platform_config/<soc>.ini`. Validate against the base-tile grid before building the kernel.
4. **Verify** — correctness vs `torch.matmul`, then benchmark ratio vs `torch_npu` ABt.

## Scope

This skill is **deliberately narrow** — it covers only the matmul L2-reuse schedule. Anything outside that scope belongs in a different skill:

- Other PTO-ISA performance patterns (vector reduce, flash-attention scheduling, K-split GEMM, etc.) → add a sibling skill, do not extend this one.
- Generic PTO optimization concepts (instruction selection, pipeline overlap, profiling methodology) → covered by [docs/coding/opt.md](../../../docs/coding/opt.md) and [docs/coding/performance-best-practices.md](../../../docs/coding/performance-best-practices.md).
- Build / constraints / debugging / review guardrails → covered by [../pto-isa-dev/SKILL.md](../pto-isa-dev/SKILL.md).

## Working Rules

- Validate the schedule's constraints on the host before building the kernel; surface a clear error rather than letting it mis-schedule.
- Keep `swizzleCountN` derived from `l2_size` rather than hard-coded — the same kernel must run on multiple Ascend SoCs.
- When porting the tile-id math to a non-GEMM persistent-block kernel, only rename the two outer axes; the formula structure is shape-agnostic.
- The recipe is meant to stand on its own — the formulas, constraints, and verification path in the reference should suffice without bouncing back to source code. Adding a helper script under this skill (e.g. a `swizzleCountN` calculator that parses a CANN `<soc>.ini`, or a small benchmark sweeper) is fine if it makes applying the recipe materially easier; do not add scripts that just duplicate what `kernels/python/gemm/run.py` already does.

## Where the Reference Implementation Lives

`kernels/python/gemm/` — the PTO-DSL GEMM kernel: the schedule, the buffer pyramid, the K-panel pipeline, the CANN `platform_config` wiring, the per-case build flow, and the `torch_npu` benchmark harness, all wired together.

`kernels/manual/a2a3/gemm_performance/` — the simpler manual SPMD-static baseline this recipe replaces when L2 reuse is the bottleneck.
