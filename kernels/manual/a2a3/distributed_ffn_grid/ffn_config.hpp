/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time shape constants for the single-device multi-block FFN demo.

#ifndef DISTRIBUTED_FFN_GRID_CONFIG_HPP
#define DISTRIBUTED_FFN_GRID_CONFIG_HPP

#ifndef CONFIG_GRID_ROWS
#define CONFIG_GRID_ROWS 2
#endif

#ifndef CONFIG_GRID_COLS
#define CONFIG_GRID_COLS 2
#endif

#ifndef CONFIG_TOKEN_TILE
#define CONFIG_TOKEN_TILE 16
#endif

#ifndef CONFIG_MODEL_TILE
#define CONFIG_MODEL_TILE 64
#endif

#ifndef CONFIG_FFN_TILE
#define CONFIG_FFN_TILE 64
#endif

constexpr int FFN_GRID_ROWS = CONFIG_GRID_ROWS;
constexpr int FFN_GRID_COLS = CONFIG_GRID_COLS;
constexpr int FFN_TOKEN_TILE = CONFIG_TOKEN_TILE;
constexpr int FFN_MODEL_TILE = CONFIG_MODEL_TILE;
constexpr int FFN_FFN_TILE = CONFIG_FFN_TILE;
constexpr int FFN_MODEL_SHARD_TILE = FFN_MODEL_TILE / FFN_GRID_COLS;
constexpr int FFN_FFN_TOTAL_TILE = FFN_FFN_TILE * FFN_GRID_COLS;

constexpr int FFN_TILE_ELEMS = FFN_TOKEN_TILE * FFN_MODEL_TILE;
constexpr int FFN_TILE_BYTES = FFN_TILE_ELEMS * 2; // half

// ReduceSum uses GridPipe to carry fp32 [T, H] down partial tiles.
// AllGather uses GridPipe to carry fp16 [T, Fi] hidden shards before down.
#ifdef CONFIG_FFN_GRID_ALLGATHER
constexpr int FFN_SLOT_BYTES = FFN_TOKEN_TILE * FFN_FFN_TILE * 2;
#else
constexpr int FFN_SLOT_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
#endif
constexpr int FFN_SLOT_COUNT = 4;

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>().
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp:
//   unicast layout = kFlagsBytes (128) + 5 dirs * SlotCount * SlotBytes.
// The AllGather variant additionally appends the TBROADCAST (scheme-②) region:
//   + BcastSlotCount * SlotBytes   (shared payload ring)
//   + 2 * GroupMax * 4             (per-source ready lanes + free lanes)
// because each cell broadcasts its own hidden shard concurrently (真·同时 MPSC)
// and the per-receiver shared ring + per-source lanes live in every window.
constexpr int FFN_GRID_DIRECTION_COUNT = 5;
constexpr int FFN_GRID_FLAGS_BYTES = 128;
constexpr int FFN_GRID_UNICAST_WINDOW_BYTES =
    FFN_GRID_FLAGS_BYTES + FFN_GRID_DIRECTION_COUNT * FFN_SLOT_COUNT * FFN_SLOT_BYTES;
#ifdef CONFIG_FFN_GRID_ALLGATHER
// Largest group this grid forms (a row or a column) = max(rows, cols).  Each
// member owns one prefix-offset slot, so the shared ring carries one slot per
// member (no reuse ⟹ the directed free path is dormant but correct).
constexpr int FFN_GRID_GROUP_MAX = (FFN_GRID_ROWS > FFN_GRID_COLS) ? FFN_GRID_ROWS : FFN_GRID_COLS;
constexpr int FFN_BCAST_SLOT_COUNT = FFN_GRID_GROUP_MAX;
constexpr int FFN_BCAST_REGION_BYTES =
    FFN_BCAST_SLOT_COUNT * FFN_SLOT_BYTES + 2 * FFN_GRID_GROUP_MAX * static_cast<int>(sizeof(uint32_t));
constexpr int FFN_GRID_WINDOW_BYTES = FFN_GRID_UNICAST_WINDOW_BYTES + FFN_BCAST_REGION_BYTES;
#else
constexpr int FFN_GRID_WINDOW_BYTES = FFN_GRID_UNICAST_WINDOW_BYTES;
#endif

// ---------------------------------------------------------------------------
// Per-cell weight + golden byte sizes.
//   - W_gate [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_up   [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_down [Fi, H]   fp16   = FFN_FFN_TILE  * FFN_MODEL_TILE * 2 in ReduceSum mode
//   - W_down [F, Hc]   fp16   = FFN_FFN_TOTAL_TILE * FFN_MODEL_SHARD_TILE * 2 in AllGather mode
//   - golden_per_row [T, H]   fp32   = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4
//     (each row owns one [T, H] slice of yOutput; the block with col == cols-1
//      writes its row's slice)
//   - golden_total   [T_total, H]    fp32   = FFN_GRID_ROWS * FFN_GOLDEN_BYTES
//     (full file on disk; data-parallel rows split T across rows)
// Source of truth for scripts/gen_data.py and both host drivers' aclrtMalloc.
// If dtype or T_total layout changes, update gen_data.py and the host driver LoadWeights
// / VerifyOutput together.
// ---------------------------------------------------------------------------
constexpr int FFN_W_GATE_BYTES = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_W_UP_BYTES = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
#ifdef CONFIG_FFN_GRID_ALLGATHER
constexpr int FFN_W_DOWN_BYTES = FFN_FFN_TOTAL_TILE * FFN_MODEL_SHARD_TILE * 2;
#else
constexpr int FFN_W_DOWN_BYTES = FFN_FFN_TILE * FFN_MODEL_TILE * 2;
#endif
constexpr int FFN_GOLDEN_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
constexpr int FFN_GOLDEN_TOTAL_BYTES = FFN_GRID_ROWS * FFN_GOLDEN_BYTES;

// ---------------------------------------------------------------------------
// Device buffer byte sizes for the mixed Cube/Vec FFN pipeline.
//   - x          [T, H]   fp16  (per-rank shard, loaded by main from
//                                pe_<r>_x.bin into an independent device
//                                buffer; cube TMATMUL input for gate/up)
//   - gatePartial[T, Fi]  fp32  (cube output: X @ W_gate)
//   - upPartial  [T, Fi]  fp32  (cube output: X @ W_up)
//   - hidden     [T, Fi]  fp16  (Vec output: TCVT(TMUL(prelu(gate), up));
//                                Cube TMATMUL input for down projection)
//   - downPartial[T, H]   fp32  (Cube output: hidden @ W_down; Vec input for
//                                EAST reduce.  EAST reduce writes the final
//                                row output from the last column block.)
// ---------------------------------------------------------------------------
constexpr int FFN_X_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 2;
constexpr int FFN_GATE_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_FFN_TILE * 4;
constexpr int FFN_UP_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_FFN_TILE * 4;
constexpr int FFN_HIDDEN_BYTES = FFN_TOKEN_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_HIDDEN_FULL_BYTES = FFN_TOKEN_TILE * FFN_FFN_TOTAL_TILE * 2;
#ifdef CONFIG_FFN_GRID_ALLGATHER
constexpr int FFN_DOWN_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_MODEL_SHARD_TILE * 4;
#else
constexpr int FFN_DOWN_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
#endif

// Matches FFN_GOLDEN_BYTES / golden.bin for direct fp32 comparison.
constexpr int FFN_Y_OUTPUT_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;

// PReLU negative-slope coefficient.  Retained for reference only -- both FFN
// variants now use the SwiGLU activation (see FFN_SILU_CLAMP_* below), so this
// constant is no longer consumed by the kernels.  Kept to avoid disturbing any
// external references and to document the legacy activation.
constexpr float FFN_PRELU_ALPHA = 0.1f;

// ---------------------------------------------------------------------------
// SwiGLU activation (SiLU(gate) * up) with a safety clamp, matching the
// WSE-FFN tile graph ("SiLU + clamp(max=10)").  The Vec branch composes SiLU
// from existing intrinsics: silu(g) = g / (1 + exp(-g)) after clamping g into
// [FFN_SILU_CLAMP_MIN, FFN_SILU_CLAMP_MAX].  gen_data.py golden MUST apply the
// identical clamp+SiLU or ResultCmp fails.
// ---------------------------------------------------------------------------
constexpr float FFN_SILU_CLAMP_MAX = 10.0f;
constexpr float FFN_SILU_CLAMP_MIN = -10.0f;

// ---------------------------------------------------------------------------
// A3 precision mapping for the WSE-FFN tile graph.
//
// The SVG tile graph carries several low precisions that the A2/A3 cube/vector
// cores do not support (FP4 weights, FP8 activations, BF16 I/O).  Per the
// extension design, every precision referenced by the tile graph is mapped to
// ONE A3-supported dtype so the same tile structure runs unmodified:
//
//   SVG stage            SVG dtype   |  A3 mapping (this demo)
//   -------------------- ----------- + ----------------------------
//   input x / x_quant    BF16 / FP8  |  half  (fp16)              [act_quant = identity]
//   weights w1/w3/w2     FP4         |  half  (fp16)              [unpack   = identity]
//   GEMM operands        FP8 x FP8   |  half x half
//   GEMM accumulator     FP32        |  float (fp32)              [unchanged]
//   gate/up/hidden act   FP32        |  float (fp32)              [unchanged]
//   hidden_quant         FP8         |  half  (fp16)              [act_quant = identity]
//   output y             BF16        |  float (fp32) on GM for direct 1e-3 compare
//
// The aliases below name each tile-graph dtype; on A3 they all collapse to
// half (activations/weights) or float (accumulators/output).  The "act_quant"
// and "unpack" stages therefore exist as named, zero-cost identity points in
// the kernel -- they document where the SVG casts would live, without adding
// any A3-unsupported conversion.
// ---------------------------------------------------------------------------
// half (fp16) is only a known type inside the CCE device TU, so the dtype
// aliases are device-only; the host Batcher uses FFN_HALF_ELEM_BYTES instead.
constexpr int FFN_HALF_ELEM_BYTES = 2;
#ifdef __CCE_AICORE__
using FfnIoDtype = half;     // SVG BF16 input/output + FP8 x_quant/hidden_quant -> half on A3
using FfnWeightDtype = half; // SVG FP4 weights -> half on A3 (unpack is identity)
using FfnAccDtype = float;   // SVG FP32 GEMM accumulator / elementwise -> float on A3
#endif

// ---------------------------------------------------------------------------
// Batcher GM arena byte sizes.
//
// The BatcheR is the external module that owns the FULL input and the FULL
// DRAM-resident weights, splits them column-parallel, broadcasts x to every
// core, and collects the H-sharded output.  A3 has no real Batcher, so this
// demo simulates it entirely in GM: the full tensors below live in GM
// (DRAM-resident, per the SVG), the host "Batcher" slices them into contiguous
// per-column shards that each core streams (DRAM->L1), and the cores write
// their output shards straight back into the Batcher output region of GM.
//
// Full tensors (Batcher-owned, GM-resident):
//   x_full      [gridRows, T, H]   half   -- one input per data-parallel row
//   w_gate_full [H, F]             half   -- F = Fi * gridCols (full intermediate)
//   w_up_full   [H, F]             half
//   w_down_full [F, H]             half
// Per-column shard regions (distributed by the Batcher, contiguous so each
// core TLOADs its shard in one piece):
//   w_gate_shard[col] [H, Fi]  = FFN_W_GATE_BYTES  (column-parallel along Fi)
//   w_up_shard  [col] [H, Fi]  = FFN_W_UP_BYTES
//   w_down_shard[col]          = FFN_W_DOWN_BYTES  ([F,Hc] AllGather / [Fi,H] ReduceSum)
// Output collection region (cores write H-shards / reduce writes per-row):
//   y_full      [gridRows, T, H]   float
// Source of truth for batcher.hpp and both host drivers.
// ---------------------------------------------------------------------------
constexpr int FFN_BATCHER_X_BYTES = FFN_GRID_ROWS * FFN_TOKEN_TILE * FFN_MODEL_TILE * 2;
constexpr int FFN_BATCHER_W_GATE_FULL_BYTES = FFN_MODEL_TILE * FFN_FFN_TOTAL_TILE * 2;
constexpr int FFN_BATCHER_W_UP_FULL_BYTES = FFN_MODEL_TILE * FFN_FFN_TOTAL_TILE * 2;
constexpr int FFN_BATCHER_W_DOWN_FULL_BYTES = FFN_FFN_TOTAL_TILE * FFN_MODEL_TILE * 2;
constexpr int FFN_BATCHER_W_GATE_SHARD_REGION_BYTES = FFN_GRID_COLS * FFN_W_GATE_BYTES;
constexpr int FFN_BATCHER_W_UP_SHARD_REGION_BYTES = FFN_GRID_COLS * FFN_W_UP_BYTES;
constexpr int FFN_BATCHER_W_DOWN_SHARD_REGION_BYTES = FFN_GRID_COLS * FFN_W_DOWN_BYTES;
constexpr int FFN_BATCHER_Y_BYTES = FFN_GRID_ROWS * FFN_Y_OUTPUT_BYTES;

#endif // DISTRIBUTED_FFN_GRID_CONFIG_HPP
