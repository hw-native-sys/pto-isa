# MHC (Multi-Head Computation) Kernels

PTO-ISA kernels for the MHC architecture from [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels).

## Overview

MHC extends the standard Transformer residual connection from a single stream to multiple parallel heads with learnable mixing:

```
x[m] = Σ_in comb[in, m] * residual[in] + post_mix[m] * up(F(down(x)))
```

This directory contains 15 kernels (7 forward + 8 backward) that implement the full MHC forward and backward pass:

| Kernel | Description | Data types |
|--------|-------------|------------|
| `expand_to_mhc_fwd/bwd` | Broadcast x to multi-head / reduce gradient | bf16 |
| `pre_apply_mix_fwd/bwd` | Weighted sum across heads | bf16 + f32 |
| `pre_norm_fn_fwd` | RMSNorm + FN weight projection | bf16 → f32 |
| `fn_normw_merge_fwd/bwd` | Fuse norm weight with FN weight | f32 |
| `head_compute_mix_fwd/bwd` | Sigmoid activation for head mix | f32 |
| `pre_split_mixes_fwd/bwd` | Split raw params into pre/post/comb | f32 |
| `sinkhorn_normalize_fwd/bwd` | Sinkhorn iteration for doubly-stochastic comb matrix | f32 |
| `post_fwd/bwd` | Final multi-head residual combination | bf16 + f32 |

## Generation

These kernels were generated from [PTO-DSL](https://github.com/huawei-csl/pto-dsl) Python source via the following pipeline:

```
PTO-DSL Python → MLIR IR (.pto) → ptoas assembler → PTO-ISA C++ (.cpp)
```

Source: [PTO-Gym PR #7](https://github.com/PTO-ISA/PTO-Gym/pull/7) (`tilekernels_ptodsl/mhc/`)

Two post-processing steps were applied for Ascend A5 (dav-3510) compatibility:
1. Tile shapes padded to 32-byte alignment (via DSL `_meta_data` modification)
2. `pipe_barrier(PIPE_V)` replaced with `pipe_barrier(PIPE_ALL)`

## Build & Run

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash run.sh
```

## Parameters

All kernels use `mhc_mult=4` and `hidden_size=1280` (matching DeepSeek-V3 configuration).
