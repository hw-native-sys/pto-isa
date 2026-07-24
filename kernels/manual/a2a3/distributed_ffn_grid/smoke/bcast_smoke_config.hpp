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
// group over the 真·同时 MPSC channel (design doc §4 方案②·前缀偏移): batched
// writes into each receiver's shared ring + ONE publish fence + per-source ready
// lanes.  Each receiver drains the source's shard with TPOP<GridGroup>(pipe,
// tile, src).  Three group flavours, all single-source (one root sends, every
// other member receives):
//   * TBROADCAST<ROW> = the whole row      (CONFIG_BCAST_SPAN_COL=0, default)
//   * TBROADCAST<COL> = the whole column   (CONFIG_BCAST_SPAN_COL=1, Rx1 grid)
//   * TBROADCAST<SUBRECT> = an arbitrary sub-rectangle (CONFIG_BCAST_SUBRECT=1)
//
// Default: a 1 x 5 row with the source in the MIDDLE (col 2) so receivers on
// BOTH sides of the source are exercised in one run -- cols 3,4 sit east of the
// source, cols 0,1 west.  Flip to a column broadcast with
// -DCONFIG_BCAST_SPAN_COL=1 and a Rx1 grid.  Flip to a sub-rectangle broadcast
// with -DCONFIG_BCAST_SUBRECT=1 (plus RECT_* bounds on a 2-D grid).

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

// 0 = ROW/COL span group (existing single-axis broadcast); 1 = arbitrary
// sub-rectangle group (GridGroup::SUBRECT) -- one source broadcasts its tile to
// every cell inside [RECT_R0,RECT_R1) x [RECT_C0,RECT_C1).  Cells outside the
// rectangle are no-ops.  When 1, SUBRECT takes precedence over BCAST_SPAN_COL.
#ifndef CONFIG_BCAST_SUBRECT
#define CONFIG_BCAST_SUBRECT 0
#endif

// Sub-rectangle bounds and the source's row-major rank-in-rect (used only when
// CONFIG_BCAST_SUBRECT=1).  Defaults cover the whole default grid so a bare
// -DCONFIG_BCAST_SUBRECT=1 still produces a valid full-grid broadcast.
#ifndef CONFIG_BCAST_RECT_R0
#define CONFIG_BCAST_RECT_R0 0
#endif
#ifndef CONFIG_BCAST_RECT_R1
#define CONFIG_BCAST_RECT_R1 CONFIG_BCAST_ROWS
#endif
#ifndef CONFIG_BCAST_RECT_C0
#define CONFIG_BCAST_RECT_C0 0
#endif
#ifndef CONFIG_BCAST_RECT_C1
#define CONFIG_BCAST_RECT_C1 CONFIG_BCAST_COLS
#endif
#ifndef CONFIG_BCAST_RECT_SRC
#define CONFIG_BCAST_RECT_SRC 0
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
constexpr int BCAST_SUBRECT = CONFIG_BCAST_SUBRECT;
constexpr int BCAST_RECT_R0 = CONFIG_BCAST_RECT_R0;
constexpr int BCAST_RECT_R1 = CONFIG_BCAST_RECT_R1;
constexpr int BCAST_RECT_C0 = CONFIG_BCAST_RECT_C0;
constexpr int BCAST_RECT_C1 = CONFIG_BCAST_RECT_C1;
constexpr int BCAST_RECT_SRC = CONFIG_BCAST_RECT_SRC;
constexpr int BCAST_T = CONFIG_BCAST_T;
constexpr int BCAST_W = CONFIG_BCAST_W;

constexpr int BCAST_TILE_ELEMS = BCAST_T * BCAST_W;
constexpr int BCAST_TILE_BYTES = BCAST_TILE_ELEMS * 4; // fp32 payload tile

// Unicast slot ring (unused by this pure-broadcast smoke, but the GridPipe
// template still carries it).  One [T, W] fp32 tile per slot.
constexpr int BCAST_SLOT_BYTES = BCAST_TILE_BYTES;
constexpr int BCAST_SLOT_COUNT = 2;

// TBROADCAST (scheme-②) region sizing: the group is the sub-rectangle extent
// (SUBRECT) or the larger of the row/col extent (ROW/COL); the shared ring
// carries one slot per group member.  The default (SUBRECT=0) path keeps the
// pre-SUBRECT window byte count unchanged.
constexpr int BCAST_GROUP_MAX = (BCAST_SUBRECT != 0)
    ? ((BCAST_RECT_R1 - BCAST_RECT_R0) * (BCAST_RECT_C1 - BCAST_RECT_C0))
    : ((BCAST_ROWS > BCAST_COLS) ? BCAST_ROWS : BCAST_COLS);
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
