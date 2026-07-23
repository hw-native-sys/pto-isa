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

// ===========================================================================
// Pure 1D N-cut 32-cell AllGather topology (方案①).
//
// The legacy constants above model the 2D-decomposition variant (cols shard the
// intermediate I, rows are data-parallel).  This block models the WSE FFN tile
// graph (WSE_FFN_tile级全展开图.dot) under a **real 4×8 = 32-cell wafer mesh**:
// the intermediate I is split across ALL 32 cells (pure 1D N-cut), so the single
// AllGather becomes a 32-way all-to-all, decomposed into Phase 1 row-gather
// (8-way) + Phase 2 col-gather (4-way).  32 > 24 physical AICores, so the host
// runs it as 4 launches with wave-by-communication-phase + barriers
// (2026-07-21-方案①按通信相分波原理详解.md).
//
// Shapes are the real DeepSeek-v4 Pro FFN: M=T=8, H=7168, I=3072.  Per cell:
//   I_shard = 3072/32 = 96  (gate/up output N, column-parallel)
//   H_shard = 7168/32 = 224 (down  output N, column-parallel)
// The 2-phase gather rebuilds the full hidden [8,3072] on every cell; down then
// writes an H-shard [8,224] of the output.
//
// All values are -D-overridable (CONFIG_NCUT_*) so the legacy small-shape defaults
// are untouched.  Both the AllGather variant and the ReduceSum variant (Option B,
// pure N-cut) consume these FFN_NCUT_* constants.
// ===========================================================================
#ifndef CONFIG_NCUT_GRID_ROWS
#define CONFIG_NCUT_GRID_ROWS 4
#endif
#ifndef CONFIG_NCUT_GRID_COLS
#define CONFIG_NCUT_GRID_COLS 8
#endif
#ifndef CONFIG_NCUT_T
#define CONFIG_NCUT_T 8
#endif
#ifndef CONFIG_NCUT_H
#define CONFIG_NCUT_H 7168
#endif
#ifndef CONFIG_NCUT_I
#define CONFIG_NCUT_I 3072
#endif

constexpr int FFN_NCUT_ROWS = CONFIG_NCUT_GRID_ROWS;                       // 4
constexpr int FFN_NCUT_COLS = CONFIG_NCUT_GRID_COLS;                       // 8
constexpr int FFN_NCUT_CELLS = FFN_NCUT_ROWS * FFN_NCUT_COLS;              // 32
constexpr int FFN_NCUT_T = CONFIG_NCUT_T;                                  // 8  (M)
constexpr int FFN_NCUT_H = CONFIG_NCUT_H;                                  // 7168 (input/hidden dim)
constexpr int FFN_NCUT_I = CONFIG_NCUT_I;                                  // 3072 (intermediate dim)
constexpr int FFN_NCUT_I_SHARD = FFN_NCUT_I / FFN_NCUT_CELLS;              // 96  (gate/up per-cell N)
constexpr int FFN_NCUT_H_SHARD = FFN_NCUT_H / FFN_NCUT_CELLS;              // 224 (down per-cell N)
constexpr int FFN_NCUT_ROW_BLOCK = FFN_NCUT_I_SHARD * FFN_NCUT_COLS;       // 768 (= I/rows, Phase-1 gather output)
static_assert(FFN_NCUT_I_SHARD * FFN_NCUT_CELLS == FFN_NCUT_I, "I must split evenly across all 32 cells");
static_assert(FFN_NCUT_H_SHARD * FFN_NCUT_CELLS == FFN_NCUT_H, "H must split evenly across all 32 cells");
static_assert(FFN_NCUT_ROW_BLOCK * FFN_NCUT_ROWS == FFN_NCUT_I, "row block * rows must rebuild full I");

// Cube GEMM K-reduction chunking.  The real K (7168 gate/up, 3072 down) exceeds
// the 512 KB L1 if a weight is loaded whole, so every cube GEMM accumulates over
// K in baseK slices (mirror kernels/manual/a2a3/gemm_performance).
constexpr int FFN_NCUT_K_BASE = 64;      // cube K-tile (must divide both 7168 and 3072)
constexpr int FFN_NCUT_BASE_M_ALIGN = 16; // cube M0 alignment (T=8 valid in one 16-row tile)
static_assert(FFN_NCUT_H % FFN_NCUT_K_BASE == 0, "H must be K_BASE divisible (gate/up K)");
static_assert(FFN_NCUT_I % FFN_NCUT_K_BASE == 0, "I must be K_BASE divisible (down K)");
static_assert(FFN_NCUT_I_SHARD % 16 == 0, "gate/up per-cell N must be 16-aligned");
static_assert(FFN_NCUT_H_SHARD % 16 == 0, "down per-cell N must be 16-aligned");

// Per-cell GM intermediate byte sizes (the AllGather variant's working buffers).
constexpr int FFN_NCUT_X_BYTES = FFN_NCUT_T * FFN_NCUT_H * 2;                 // [T,H] half (broadcast)
constexpr int FFN_NCUT_HIDDEN_SHARD_BYTES = FFN_NCUT_T * FFN_NCUT_I_SHARD * 2; // [8,96] half (Phase A out)
constexpr int FFN_NCUT_ROW_BLOCK_BYTES = FFN_NCUT_T * FFN_NCUT_ROW_BLOCK * 2;  // [8,768] half (Phase B out)
constexpr int FFN_NCUT_HIDDEN_FULL_BYTES = FFN_NCUT_T * FFN_NCUT_I * 2;        // [8,3072] half (Phase C out)
constexpr int FFN_NCUT_GATE_PARTIAL_BYTES = FFN_NCUT_T * FFN_NCUT_I_SHARD * 4; // [8,96] fp32 (cube->vec C2V)
constexpr int FFN_NCUT_DOWN_PARTIAL_BYTES = FFN_NCUT_T * FFN_NCUT_H_SHARD * 4; // [8,224] fp32 (down out)
constexpr int FFN_NCUT_W_GATE_SHARD_BYTES = FFN_NCUT_H * FFN_NCUT_I_SHARD * 2; // [7168,96] half weight shard
constexpr int FFN_NCUT_W_UP_SHARD_BYTES = FFN_NCUT_H * FFN_NCUT_I_SHARD * 2;
constexpr int FFN_NCUT_W_DOWN_SHARD_BYTES = FFN_NCUT_I * FFN_NCUT_H_SHARD * 2; // [3072,224] half weight shard
constexpr int FFN_NCUT_Y_SHARD_BYTES = FFN_NCUT_T * FFN_NCUT_H_SHARD * 4;      // [8,224] fp32 (final output shard)

// Two GridPipe arenas (different SlotBytes ⟹ different window layouts):
//   P1 carries the [8,96] hidden shard (Phase-1 row gather, group = a row of 8);
//   P2 carries the [8,768] row block    (Phase-2 col gather, group = a col of 4).
// Each carries one tile per source (single-shot, no slot reuse) ⟹ SlotCount = 1
// and BcastSlotCount = group size.  Window = flags(128) + 5*SlotCount*SlotBytes
// + (BcastSlotCount*SlotBytes + 2*Group*4), matching the unicast+bcast layout in
// include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int FFN_NCUT_GRID_DIRECTION_COUNT = 5;
constexpr int FFN_NCUT_GRID_FLAGS_BYTES = 128;
constexpr int FFN_NCUT_SLOT_COUNT = 1;
constexpr int FFN_NCUT_SLOT_BYTES_P1 = FFN_NCUT_HIDDEN_SHARD_BYTES;
constexpr int FFN_NCUT_SLOT_BYTES_P2 = FFN_NCUT_ROW_BLOCK_BYTES;
constexpr int FFN_NCUT_GROUP_P1 = FFN_NCUT_COLS; // 8 (ROW group)
constexpr int FFN_NCUT_GROUP_P2 = FFN_NCUT_ROWS; // 4 (COL group)
constexpr int FFN_NCUT_BCAST_SLOTS_P1 = FFN_NCUT_GROUP_P1;
constexpr int FFN_NCUT_BCAST_SLOTS_P2 = FFN_NCUT_GROUP_P2;
// Per-source ready/free lane stride (bytes) -- one full cache line per lane so
// concurrent TBROADCAST producers never share a cache line (avoids the
// write-back lost-update that dropped doorbell writes).  Must match the kernel
// layout constant kBcastLaneStride in grid_intrinsic.hpp.
constexpr int FFN_NCUT_LANE_STRIDE = 64;
constexpr int FFN_NCUT_WIN_P1 =
    FFN_NCUT_GRID_FLAGS_BYTES + FFN_NCUT_GRID_DIRECTION_COUNT * FFN_NCUT_SLOT_COUNT * FFN_NCUT_SLOT_BYTES_P1 +
    FFN_NCUT_BCAST_SLOTS_P1 * FFN_NCUT_SLOT_BYTES_P1 + 2 * FFN_NCUT_GROUP_P1 * FFN_NCUT_LANE_STRIDE;
constexpr int FFN_NCUT_WIN_P2 =
    FFN_NCUT_GRID_FLAGS_BYTES + FFN_NCUT_GRID_DIRECTION_COUNT * FFN_NCUT_SLOT_COUNT * FFN_NCUT_SLOT_BYTES_P2 +
    FFN_NCUT_BCAST_SLOTS_P2 * FFN_NCUT_SLOT_BYTES_P2 + 2 * FFN_NCUT_GROUP_P2 * FFN_NCUT_LANE_STRIDE;

// ---------------------------------------------------------------------------
// TPUSH-AllGather topology (方案①).  Same pure 1D N-cut 32-cell mesh and shapes
// as the TBROADCAST variant above, but the two gather phases are driven by the
// TPUSH/TPOP unicast primitives via a nearest-neighbor relay instead of the
// TBROADCAST MPSC collective.  The relay is fan-in-1 per direction, so it needs
// NO broadcast region (GroupMax = 0): the window is the plain unicast layout
//   flags(128) + 5 dirs * SlotCount * SlotBytes
// (mirrors FFN_RS_REDUCE_WIN).  P1 relays the [8,768] row block (row gather),
// P2 relays the [8,3072] full hidden (col gather); each relay uses two opposite
// directions (EAST forward + WEST backward for P1; SOUTH forward + NORTH backward
// for P2), so SlotCount = 2 double-buffers the two hops.  The slot must hold the
// FULL relay tile (it grows during the forward pass), so P1's slot is the row
// block and P2's slot is the full hidden.
// ---------------------------------------------------------------------------
constexpr int FFN_NCUT_TPUSH_SLOT_COUNT = 2; // double-buffer the relay's two opposite directions
constexpr int FFN_NCUT_TPUSH_SLOT_BYTES_P1 = FFN_NCUT_ROW_BLOCK_BYTES;   // [8,768]  half (row relay)
constexpr int FFN_NCUT_TPUSH_SLOT_BYTES_P2 = FFN_NCUT_HIDDEN_FULL_BYTES; // [8,3072] half (col relay)
constexpr int FFN_NCUT_TPUSH_WIN_P1 =
    FFN_NCUT_GRID_FLAGS_BYTES +
    FFN_NCUT_GRID_DIRECTION_COUNT * FFN_NCUT_TPUSH_SLOT_COUNT * FFN_NCUT_TPUSH_SLOT_BYTES_P1;
constexpr int FFN_NCUT_TPUSH_WIN_P2 =
    FFN_NCUT_GRID_FLAGS_BYTES +
    FFN_NCUT_GRID_DIRECTION_COUNT * FFN_NCUT_TPUSH_SLOT_COUNT * FFN_NCUT_TPUSH_SLOT_BYTES_P2;

// ===========================================================================
// ReduceSum variant on the same pure 1D N-cut 32-cell topology (Option B).
//
// ReduceSum also splits I across all 32 cells and broadcasts x (so it reuses the
// FFN_NCUT_* shapes above), but instead of gathering hidden it computes a full-H
// down PARTIAL per cell (W_down is cut along I = the down GEMM's K-axis, a row
// shard [I_shard, H]) and REDUCES the 32 partials: EAST 8-way (row) then SOUTH
// 4-way (col).  Because partial_c is [T, H] fp32 = 224 KB > 192 KB UB, the reduce
// is H-chunked: it carries [T, H_base] tiles (32 KB) at a time.
//
// GEMM tiling (real K/N exceed L1/L0C): gate/up K-tiled (baseK=64, as AllGather);
// down K-tiled (baseK=32) + N-tiled (baseN=1024) -- L0B [32,1024]=64 KB, L1 weight
// panel [96,1024]=192 KB.  The down B (W_down row-shard [96,7168]) has row stride
// = full H = 7168, NOT baseN, so the down GEMM uses its own GlobalB stride.
// ===========================================================================
constexpr int FFN_RS_DOWN_K_BASE = 32;   // down K-tile (I_shard=96 / 32 = 3 iters)
constexpr int FFN_RS_DOWN_N_BASE = 1024; // down N-tile == reduce H-chunk (H=7168 / 1024 = 7)
constexpr int FFN_RS_REDUCE_H_BASE = FFN_RS_DOWN_N_BASE;
static_assert(FFN_NCUT_I_SHARD % FFN_RS_DOWN_K_BASE == 0, "down K must be K_BASE divisible (I_shard)");
static_assert(FFN_NCUT_H % FFN_RS_DOWN_N_BASE == 0, "H must be N_BASE divisible (down N / reduce H-chunks)");

// Per-cell down partial [T, H] fp32 (the full-H partial summed by the reduce).
constexpr int FFN_RS_PARTIAL_BYTES = FFN_NCUT_T * FFN_NCUT_H * 4;            // 229376 B (224 KB) -- lives in GM
constexpr int FFN_RS_ROW_PARTIAL_BYTES = FFN_NCUT_T * FFN_NCUT_H * 4;        // per-row EAST-reduced partial (GM)
constexpr int FFN_RS_W_DOWN_SHARD_BYTES = FFN_NCUT_I_SHARD * FFN_NCUT_H * 2; // [96,7168] half (W_down cut along I)
constexpr int FFN_RS_Y_BYTES = FFN_NCUT_T * FFN_NCUT_H * 4;                  // final [8,7168] fp32 output

// EAST/SOUTH reduce tile = [T, H_base] fp32 (fits the 192 KB UB with the add acc).
constexpr int FFN_RS_REDUCE_TILE_BYTES = FFN_NCUT_T * FFN_RS_REDUCE_H_BASE * 4; // 32768 B (32 KB)
// One slot per H-segment so the H-chunked reduce never waits a cross-segment
// free doorbell (each segment lands in its own slot; prodIndex < SlotCount always).
constexpr int FFN_RS_REDUCE_SLOT_COUNT = FFN_NCUT_H / FFN_RS_REDUCE_H_BASE; // = kHSegs (7)
constexpr int FFN_RS_REDUCE_WIN =
    FFN_NCUT_GRID_FLAGS_BYTES + FFN_NCUT_GRID_DIRECTION_COUNT * FFN_RS_REDUCE_SLOT_COUNT * FFN_RS_REDUCE_TILE_BYTES;

#endif // DISTRIBUTED_FFN_GRID_CONFIG_HPP
