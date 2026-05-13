/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time shape constants for the 1x2 distributed FFN demo.
// Kept minimal: this is the M2 skeleton; later milestones grow to 2x2 and
// real FFN shapes (see plan declarative-launching-simon.md M3/M4).

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

constexpr int FFN_GRID_ROWS  = CONFIG_GRID_ROWS;
constexpr int FFN_GRID_COLS  = CONFIG_GRID_COLS;
constexpr int FFN_TOKEN_TILE = CONFIG_TOKEN_TILE;
constexpr int FFN_MODEL_TILE = CONFIG_MODEL_TILE;
constexpr int FFN_FFN_TILE   = CONFIG_FFN_TILE;

constexpr int FFN_TILE_ELEMS = FFN_TOKEN_TILE * FFN_MODEL_TILE;
constexpr int FFN_TILE_BYTES = FFN_TILE_ELEMS * 2;  // half

// D-2: GridPipe slot capacity bumped to fp32 [T, H] = T*H*4 so the EAST
// reduce stage can carry the fp32 down_partial tile without losing precision.
// SOURCE direction is no longer used by D-2 (cube reads X from x_dev), but the
// 4-direction window layout is unchanged -- the bump only enlarges per-slot
// padding.  ResultCmp tolerance 1e-3 in D-3 depends on this fp32 path.
constexpr int FFN_SLOT_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
constexpr int FFN_SLOT_COUNT = 4;

// Host-visible mirror of pto::a2a3_grid::kWindowBytes<FFN_SLOT_BYTES, FFN_SLOT_COUNT>().
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp:
//   layout = kFlagsBytes (64) + 4 dirs * SlotCount * SlotBytes.
constexpr int FFN_GRID_FLAGS_BYTES = 64;
constexpr int FFN_GRID_WINDOW_BYTES =
    FFN_GRID_FLAGS_BYTES + 4 * FFN_SLOT_COUNT * FFN_SLOT_BYTES;

// ---------------------------------------------------------------------------
// Host-side mirror of pto::a2a3_grid slot/flag layout, used by main.cpp to
// memcpy the SOURCE-direction X shard into the GridPipe window before launch.
// Pure integer constexpr.  Do NOT add any <pto/...> include in this header --
// main.cpp parses it in HOST mode, and an AICORE attribute leaking in would
// break the host parser.
//
// Source of truth:
//   include/pto/npu/a2a3/grid_pipe_runtime.hpp
//     - kFlagsBytes              = 64
//     - kSlotRegionOffset        = kFlagsBytes
//     - kReadyFlagOffset(d)      = (uint32_t)d * sizeof(uint32_t)
//     - kDirSlotRegionOffset(d)  = kSlotRegionOffset + (uint32_t)d * SlotCount * SlotBytes
//   include/pto/common/grid_pipe.hpp
//     - GridDirection::SOURCE = 0
//
// Any change to the constexpr functions above MUST update the mirrors here.
// ---------------------------------------------------------------------------
constexpr int FFN_GRID_DIR_SOURCE_INDEX = 0;  // GridDirection::SOURCE
constexpr int FFN_GRID_SLOT_REGION_OFFSET = FFN_GRID_FLAGS_BYTES;  // 64
constexpr int FFN_GRID_READY_FLAG_SOURCE_OFFSET =
    FFN_GRID_DIR_SOURCE_INDEX * static_cast<int>(sizeof(unsigned int));  // 0
constexpr int FFN_GRID_SOURCE_SLOT0_OFFSET =
    FFN_GRID_SLOT_REGION_OFFSET +
    FFN_GRID_DIR_SOURCE_INDEX * FFN_SLOT_COUNT * FFN_SLOT_BYTES;  // 64

// ---------------------------------------------------------------------------
// Per-rank weight + golden byte sizes.
//   - W_gate [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_up   [H, Fi]   fp16   = FFN_MODEL_TILE * FFN_FFN_TILE * 2
//   - W_down [Fi, H]   fp16   = FFN_FFN_TILE  * FFN_MODEL_TILE * 2
//   - golden_per_row [T, H]   fp32   = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4
//     (each row owns one [T, H] slice of yOutput; rank with col == cols-1
//      writes its row's slice)
//   - golden_total   [T_total, H]    fp32   = FFN_GRID_ROWS * FFN_GOLDEN_BYTES
//     (full file on disk; data-parallel rows split T across rows)
// Source of truth for both scripts/gen_data.py and main.cpp aclrtMalloc.
// If dtype or T_total layout changes, update gen_data.py AND main.cpp LoadWeights
// / VerifyOutput together.
// ---------------------------------------------------------------------------
constexpr int FFN_W_GATE_BYTES = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_W_UP_BYTES   = FFN_MODEL_TILE * FFN_FFN_TILE * 2;
constexpr int FFN_W_DOWN_BYTES = FFN_FFN_TILE  * FFN_MODEL_TILE * 2;
constexpr int FFN_GOLDEN_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;
constexpr int FFN_GOLDEN_TOTAL_BYTES = FFN_GRID_ROWS * FFN_GOLDEN_BYTES;

// ---------------------------------------------------------------------------
// M3 Session D-2: ChunkFlagMatrix layout for cube/comm coordination.
//   - num_ranks            : 1 (intra-rank cube <-> comm producer/consumer).
//   - num_tiles_per_src    : 4 (gate=0, up=1, hidden=2 (comm->cube), down=3).
//   - chunk_size           : 1 (one tile per chunk -> four flags total).
// num_chunks_per_src = ceil(num_tiles_per_src / chunk_size) = 4.
// stride = ((num_chunks + 15) / 16) * 16 = 16, so each flag is 4 bytes apart
// inside the chunk_flags region (chunk_idx * 4 from base of region).
// Cube cannot pull pto/comm/pto_comm_inst.hpp (asc-only), so it writes flag 0,
// 1, 3 via raw volatile + dcci.  Comm writes flag 2 via SetChunkFlagReady
// (TNOTIFY AtomicAdd).  Read pattern: comm uses IsChunkReady on flags 0/1/3;
// cube uses a raw volatile poll on flag 2.
// ---------------------------------------------------------------------------
constexpr int FFN_CHUNK_NUM_RANKS         = 1;
constexpr int FFN_CHUNK_NUM_TILES_PER_SRC = 4;
constexpr int FFN_CHUNK_SIZE              = 1;

// Symbolic flag indices for the 4-chunk schema.  Anywhere these are passed to
// the cube kernel they go in via a raw pointer offset; cube ignores the
// SetChunkFlagReady API.  Comm uses these as the chunk_idx argument to
// IsChunkReady / SetChunkFlagReady.
constexpr int FFN_FLAG_GATE_READY    = 0;  // cube -> comm
constexpr int FFN_FLAG_UP_READY      = 1;  // cube -> comm
constexpr int FFN_FLAG_HIDDEN_READY  = 2;  // comm -> cube
constexpr int FFN_FLAG_DOWN_READY    = 3;  // cube -> comm

// Byte offset of the int32_t chunk_flags[][] region inside the per-rank
// ChunkFlagMatrix.  Source of truth is `struct alignas(64) ChunkFlagMatrix`
// in ready_queue.hpp -- header is exactly 64 bytes (7 fields + 9-element
// padding to satisfy alignas(64)).  The cube .so cannot include
// ready_queue.hpp (pulls pto/comm/pto_comm_inst.hpp), so this mirror is the
// only path for cube to compute the doorbell pointer.
constexpr int FFN_CHUNKFLAG_OFFSET_HEADER = 64;

// ---------------------------------------------------------------------------
// D-2 device buffer byte sizes (cube + comm FFN pipeline).
//   - x          [T, H]   fp16  (per-rank shard, loaded by main from
//                                pe_<r>_x.bin into an independent device
//                                buffer; cube TMATMUL input for gate/up)
//   - gatePartial[T, Fi]  fp32  (cube output: X @ W_gate)
//   - upPartial  [T, Fi]  fp32  (cube output: X @ W_up)
//   - hidden     [T, Fi]  fp16  (comm output: TCVT(TMUL(prelu(gate), up));
//                                cube TMATMUL input for down projection)
//   - downPartial[T, H]   fp32  (cube output: hidden @ W_down; comm input for
//                                EAST reduce.  EAST reduce uses fp32 across
//                                cols and writes the rank-(M-1) result into
//                                yOutput.)
// ---------------------------------------------------------------------------
constexpr int FFN_X_BYTES            = FFN_TOKEN_TILE * FFN_MODEL_TILE * 2;
constexpr int FFN_GATE_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_FFN_TILE   * 4;
constexpr int FFN_UP_PARTIAL_BYTES   = FFN_TOKEN_TILE * FFN_FFN_TILE   * 4;
constexpr int FFN_HIDDEN_BYTES       = FFN_TOKEN_TILE * FFN_FFN_TILE   * 2;
constexpr int FFN_DOWN_PARTIAL_BYTES = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;

// yOutput dtype changed from fp16 (D-1 passthrough) to fp32 (D-2 final reduce
// output).  Matches FFN_GOLDEN_BYTES / golden.bin so host ResultCmp is
// straight memcmp(fp32, fp32) without a half->float pass.
constexpr int FFN_Y_OUTPUT_BYTES     = FFN_TOKEN_TILE * FFN_MODEL_TILE * 4;

// PReLU negative-slope coefficient.  Source of truth is gen_data.py top-level
// alpha; if you change one you MUST change the other or D-3 ResultCmp fails.
// Used by the comm kernel as a TLRELU scalar (no separate alpha tile needed).
constexpr float FFN_PRELU_ALPHA = 0.1f;

#endif  // DISTRIBUTED_FFN_GRID_CONFIG_HPP
