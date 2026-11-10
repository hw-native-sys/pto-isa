# TMATMUL ST Testcase — Requirements

## Purpose

Verify the `TMATMUL` instruction (`dst = src0 * src1`) on the A5 backend across the supported accumulator / operand dtype combinations, the shape matrix, the split-K variants (single-core and multi-core), the TF32 rounding modes, the GEMV shape (`M = 1`), and the bias path (`TMATMUL_BIAS`).

## Coverage points

- **Dtype combinations** (operands → accumulator):
  - `half` → `float`, `bfloat16_t` → `float`, `float` → `float`.
  - `int8_t` → `int32_t`.
  - `float8_e4m3_t` / `float8_e5m2_t` → `float` (all four A/B combinations: e4m3×e4m3, e4m3×e5m2, e5m2×e4m3, e5m2×e5m2).
  - `hifloat8_t` → `float`.
- **Plain matmul, 14 cases** (`TMATMUL`, `TMATMUL_SPLIT_K`, `TGEMV`, `TMATMUL_TF32`, `TMATMUL_MULTI_CORE_SPLIT_K`):
  - `40x50x60` half→float — baseline.
  - `6x7x8` int8→int32 — minimal correctness.
  - `127x128x61` half→float — single-core split-K (`K` split into base-K chunks with `TMATMUL_ACC`).
  - `120x110x50`, `128x96x64`, `16x32x64` float→float — full float path; `16x32x64` uses `TMATMUL_TF32` with `CAST_RINT`, `128x96x64` uses `CAST_ROUND`.
  - `144x80x48`, `1x300x60`, `50x2048x64` half→float / bf16→float — `1x300x60` runs the `TGEMV` path (`M = 1`); `50x2048x64` runs `TMATMUL_MULTI_CORE_SPLIT_K` on 16 cubes with an atomic-add `TSTORE`.
  - `32x64x96`, `128x96x64`, `145x115x85`, `120x90x160`, `30x90x60` — fp8 / hifloat8 paths, including non-aligned `M/K/N` (`145x115x85`) and a K-dominant shape (`120x90x160`).
- **Bias path, 11 cases** (`TMATMUL_BIAS`, bias vector length `N`):
  - `8x7x6` int8→int32, bias `int32_t`; `16x15x16` half→float, bias `half`; `112x127x80` half→float, bias `bfloat16_t` (mixed bias dtype); `80x112x63` bf16→float, bias bf16; `127x128x63` float→float, bias float; `1x512x85` half→float, bias float.
  - `120x90x160`, `32x64x96`, `128x96x64`, `30x90x60` fp8 combinations and `145x115x85` hifloat8, all with a `float` bias.
- **Layout**: operands are fractal NZ mat tiles (`TileType::Mat`); the accumulator lives in L0C and is drained by `TSTORE` (fixpipe) at the end of the K loop.

## Golden generation

`gen_data.py` draws random integers in `[-10, 10)` and casts them to the operand dtype, then computes `golden = np.matmul(x1, x2)` in the accumulator dtype (plus the bias vector for the bias cases, which is drawn from `[1, 10)`). HiFloat8 operands are decoded from their 8-bit encoding to `float` before the reference multiply; TF32 cases round both inputs to TF32 (`CAST_RINT` / `CAST_ROUND`) first. `main.cpp` compares the NPU output against `golden.bin` with `ResultCmp`: `eps = 0.0001` for the plain matmul cases and `eps = 0.001` for the bias cases (an element passes when `|diff| <= eps` or `|diff / expected| <= eps`).

## Out of scope

- A2/A3 backend — covered by `tests/npu/a2a3/src/st/testcase/tmatmul`.
- MX-format overloads with scale tiles — covered by the dedicated `tmatmul_mx_*` testcases; the `hifloat8` case here uses the plain `TMATMUL` path.
- Auto-mode orchestration — covered by the auto-mode runner.
