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
// group (design doc §4 方案②·前缀偏移): batched writes into each receiver's
// shared ring + ONE publish fence + one doorbell per receiver.  Each receiver
// drains the source's shard with TPOP<GridGroup>(pipe, tile, src).  Two group
// flavours, both single-source here (one root sends, every other member
// receives, so no turn-taking is involved):
//   * TBROADCAST<ROW> = the whole row      (CONFIG_BCAST_SPAN_COL=0, default)
//   * TBROADCAST<COL> = the whole column   (CONFIG_BCAST_SPAN_COL=1, Rx1 grid)
//
// Default: a 1 x 5 row with the source in the MIDDLE (col 2) so receivers on
// BOTH sides of the source are exercised in one run -- cols 3,4 sit east of the
// source, cols 0,1 west.  Flip to a column broadcast with
// -DCONFIG_BCAST_SPAN_COL=1 and a Rx1 grid.  Those two axes are the whole group
// vocabulary -- there is no arbitrary sub-rectangle group.

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

// 0 = ONE source (CONFIG_BCAST_SRC) broadcasts, everyone else only TPOPs -- no
//     turn to pass, so neither TBWAIT nor TBNOTIFY is called.
// 1 = EVERY member broadcasts in ascending rank-in-group order (an AllGather),
//     which is the multi-source case: the group shares one ring slot per
//     direction, so each member takes the turn with TBWAIT<Group> before its own
//     TBROADCAST and hands it to rank+1 with TBNOTIFY<Group> after.  This is the
//     mode that covers the caller-side obligation.
#ifndef CONFIG_BCAST_ALL_SRC
#define CONFIG_BCAST_ALL_SRC 0
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
constexpr int BCAST_ALL_SRC = CONFIG_BCAST_ALL_SRC;
constexpr int BCAST_T = CONFIG_BCAST_T;
constexpr int BCAST_W = CONFIG_BCAST_W;

constexpr int BCAST_TILE_ELEMS = BCAST_T * BCAST_W;
constexpr int BCAST_TILE_BYTES = BCAST_TILE_ELEMS * 4; // fp32 payload tile

// One [T, W] fp32 tile per ring slot, and ONE slot per direction: the group
// rides the ordinary directional rings, where the sender addresses
// prod_idx % SlotCount and the receiver cons_idx % SlotCount.  With several
// sources on one direction those two counters cannot agree on anything but
// SlotCount = 1 (every source's prod_idx starts at 0 while the receiver's
// cons_idx runs over all of them), so one slot it is -- the same value the FFN
// demo uses.
constexpr int BCAST_SLOT_BYTES = BCAST_TILE_BYTES;
constexpr int BCAST_SLOT_COUNT = 1;
// Largest group this grid forms (a row or a column).  Only used to size the host
// output buffer in the all-source mode (one received tile per source).
constexpr int BCAST_GROUP_MAX = (BCAST_ROWS > BCAST_COLS) ? BCAST_ROWS : BCAST_COLS;

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>(): the group rides the
// DIRECTIONAL rings, so a window is the flag header plus one ring per direction
// the group spans.  Keep in sync with grid_pipe_runtime.hpp.
constexpr int BCAST_GROUP_DIRECTION_COUNT = 2; // the two directions a ROW/COL group spans
constexpr int BCAST_GRID_FLAGS_BYTES = 1024;
// The publish turn adds nothing to this: it is one increment on the scoreboard
// of the axis the group does NOT span, already sitting unused in the flag header
// below.  No second sub-window, no second pipe, no packet.
constexpr int BCAST_WINDOW_BYTES =
    BCAST_GRID_FLAGS_BYTES + BCAST_GROUP_DIRECTION_COUNT * BCAST_SLOT_COUNT * BCAST_SLOT_BYTES;

#endif // BCAST_SMOKE_CONFIG_HPP
