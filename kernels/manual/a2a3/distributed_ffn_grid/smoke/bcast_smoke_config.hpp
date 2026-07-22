/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time config for the GridPipe TBROADCAST smoke kernel.
//
// One source cell broadcasts a stamped fp32 tile to every other cell on its
// group (TBROADCAST<ROW> = the whole row, TBROADCAST<COL> = the whole column)
// over the 真·同时 MPSC channel (design doc §4 方案②·前缀偏移): batched writes
// into each receiver's shared ring + ONE publish fence + per-source ready lanes.
// Each receiver drains the source's shard with TPOP<GridGroup>(pipe, tile, src).
//
// Default: a 1 x 5 row with the source in the MIDDLE (col 2) so receivers on
// BOTH sides of the source are exercised in one run -- cols 3,4 sit east of the
// source, cols 0,1 west.  Flip to a column broadcast with
// -DCONFIG_BCAST_SPAN_COL=1 and a Rx1 grid.

#ifndef BCAST_SMOKE_CONFIG_HPP
#define BCAST_SMOKE_CONFIG_HPP

#ifndef CONFIG_BCAST_ROWS
#define CONFIG_BCAST_ROWS 1
#endif

#ifndef CONFIG_BCAST_COLS
#define CONFIG_BCAST_COLS 5
#endif

// Index of the single source along the active group axis (column index for a ROW
// broadcast, row index for a COL broadcast) = the source's rank-in-group.
#ifndef CONFIG_BCAST_SRC
#define CONFIG_BCAST_SRC 2
#endif

// 0 = ROW group (EAST+WEST arms), 1 = COL group (NORTH+SOUTH arms).
#ifndef CONFIG_BCAST_SPAN_COL
#define CONFIG_BCAST_SPAN_COL 0
#endif

#ifndef CONFIG_BCAST_T
#define CONFIG_BCAST_T 16
#endif

#ifndef CONFIG_BCAST_W
#define CONFIG_BCAST_W 64
#endif

constexpr int BCAST_ROWS = CONFIG_BCAST_ROWS;
constexpr int BCAST_COLS = CONFIG_BCAST_COLS;
constexpr int BCAST_SRC = CONFIG_BCAST_SRC;
constexpr int BCAST_SPAN_COL = CONFIG_BCAST_SPAN_COL;
constexpr int BCAST_T = CONFIG_BCAST_T;
constexpr int BCAST_W = CONFIG_BCAST_W;

constexpr int BCAST_TILE_ELEMS = BCAST_T * BCAST_W;
constexpr int BCAST_TILE_BYTES = BCAST_TILE_ELEMS * 4; // fp32 payload tile

// Unicast slot ring (unused by this pure-broadcast smoke, but the GridPipe
// template still carries it).  One [T, W] fp32 tile per slot.
constexpr int BCAST_SLOT_BYTES = BCAST_TILE_BYTES;
constexpr int BCAST_SLOT_COUNT = 2;

// TBROADCAST (scheme-②) region sizing: the group is the larger of the row/col
// extent, and the shared ring carries one slot per group member.
constexpr int BCAST_GROUP_MAX = (BCAST_ROWS > BCAST_COLS) ? BCAST_ROWS : BCAST_COLS;
constexpr int BCAST_BCAST_SLOT_COUNT = BCAST_GROUP_MAX;

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>():
//   unicast layout = kFlagsBytes (128) + 5 dirs * SlotCount * SlotBytes
//   + TBROADCAST region: BcastSlotCount * SlotBytes (shared ring)
//                       + 2 * GroupMax * 4        (per-source ready + free lanes)
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int BCAST_GRID_DIRECTION_COUNT = 5;
constexpr int BCAST_GRID_FLAGS_BYTES = 128;
constexpr int BCAST_UNICAST_WINDOW_BYTES =
    BCAST_GRID_FLAGS_BYTES + BCAST_GRID_DIRECTION_COUNT * BCAST_SLOT_COUNT * BCAST_SLOT_BYTES;
constexpr int BCAST_BCAST_REGION_BYTES =
    BCAST_BCAST_SLOT_COUNT * BCAST_SLOT_BYTES + 2 * BCAST_GROUP_MAX * static_cast<int>(sizeof(uint32_t));
constexpr int BCAST_WINDOW_BYTES = BCAST_UNICAST_WINDOW_BYTES + BCAST_BCAST_REGION_BYTES;

#endif // BCAST_SMOKE_CONFIG_HPP
