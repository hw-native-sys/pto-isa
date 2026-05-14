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

constexpr int FFN_TILE_ELEMS = FFN_TOKEN_TILE * FFN_MODEL_TILE;
constexpr int FFN_TILE_BYTES = FFN_TILE_ELEMS * 2; // half

// GridPipe EAST reduce carries fp32 [T, H] down partial tiles.
constexpr int FFN_SLOT_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
constexpr int FFN_SLOT_COUNT = 4;

// Host-visible mirror of pto::a2a3_grid::kWindowBytes<FFN_SLOT_BYTES, FFN_SLOT_COUNT>().
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp:
//   layout = kFlagsBytes (64) + 4 dirs * SlotCount * SlotBytes.
constexpr int FFN_GRID_FLAGS_BYTES = 64;
constexpr int FFN_GRID_WINDOW_BYTES = FFN_GRID_FLAGS_BYTES + 4 * FFN_SLOT_COUNT * FFN_SLOT_BYTES;

// ---------------------------------------------------------------------------
// Per-cell weight + golden byte sizes.
//   - W_gate [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_up   [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_down [Fi, H]   fp16   = FFN_FFN_TILE  * FFN_MODEL_TILE * 2
//   - golden_per_row [T, H]   fp32   = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4
//     (each row owns one [T, H] slice of yOutput; the block with col == cols-1
//      writes its row's slice)
//   - golden_total   [T_total, H]    fp32   = FFN_GRID_ROWS * FFN_GOLDEN_BYTES
//     (full file on disk; data-parallel rows split T across rows)
// Source of truth for both scripts/gen_data.py and main.cpp aclrtMalloc.
// If dtype or T_total layout changes, update gen_data.py AND main.cpp LoadWeights
// / VerifyOutput together.
// ---------------------------------------------------------------------------
constexpr int FFN_W_GATE_BYTES = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_W_UP_BYTES = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_W_DOWN_BYTES = FFN_FFN_TILE * FFN_MODEL_TILE * 2;
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
constexpr int FFN_DOWN_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;

// Matches FFN_GOLDEN_BYTES / golden.bin for direct fp32 comparison.
constexpr int FFN_Y_OUTPUT_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;

// PReLU negative-slope coefficient.  Source of truth is gen_data.py top-level
// alpha; if you change one you MUST change the other or D-3 ResultCmp fails.
// Used by the Vec branch as a TLRELU scalar (no separate alpha tile needed).
constexpr float FFN_PRELU_ALPHA = 0.1f;

#endif // DISTRIBUTED_FFN_GRID_CONFIG_HPP
