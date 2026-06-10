/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time config for the GridPipe routed K-hop unicast smoke kernel.
//
// A 1 x COLS row of cells.  Each cell c pushes a stamped fp32 tile DIST hops
// EAST (TPUSH<EAST, DIST>) to cell c+DIST; cell c+DIST pops it (TPOP<EAST, DIST>)
// and stores it.  This exercises the Scheme A K-hop data path *and* the routed
// ready/free doorbell (mtspr_neighbor_counter's new dist operand) end to end.
//
// DIST is a compile-time constant (TPUSH/TPOP take it as a template parameter),
// configured via -DCONFIG_KHOP_DIST=N at cmake time.

#ifndef KHOP_SMOKE_CONFIG_HPP
#define KHOP_SMOKE_CONFIG_HPP

#ifndef CONFIG_KHOP_ROWS
#define CONFIG_KHOP_ROWS 1
#endif

#ifndef CONFIG_KHOP_COLS
#define CONFIG_KHOP_COLS 4
#endif

#ifndef CONFIG_KHOP_DIST
#define CONFIG_KHOP_DIST 2
#endif

#ifndef CONFIG_KHOP_T
#define CONFIG_KHOP_T 16
#endif

#ifndef CONFIG_KHOP_W
#define CONFIG_KHOP_W 64
#endif

constexpr int KHOP_ROWS = CONFIG_KHOP_ROWS;
constexpr int KHOP_COLS = CONFIG_KHOP_COLS;
constexpr int KHOP_DIST = CONFIG_KHOP_DIST;
constexpr int KHOP_T = CONFIG_KHOP_T;
constexpr int KHOP_W = CONFIG_KHOP_W;

constexpr int KHOP_TILE_ELEMS = KHOP_T * KHOP_W;
constexpr int KHOP_TILE_BYTES = KHOP_TILE_ELEMS * 4; // fp32 payload tile

// GridPipe slot ring carries one [T, W] fp32 tile per slot.
constexpr int KHOP_SLOT_BYTES = KHOP_TILE_BYTES;
constexpr int KHOP_SLOT_COUNT = 2;

// Host-visible mirror of pto::a2a3_grid::kWindowBytes<SlotBytes, SlotCount>():
//   layout = kFlagsBytes (128) + 5 dirs * SlotCount * SlotBytes.
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int KHOP_GRID_DIRECTION_COUNT = 5;
constexpr int KHOP_GRID_FLAGS_BYTES = 128;
constexpr int KHOP_WINDOW_BYTES = KHOP_GRID_FLAGS_BYTES + KHOP_GRID_DIRECTION_COUNT * KHOP_SLOT_COUNT * KHOP_SLOT_BYTES;

#endif // KHOP_SMOKE_CONFIG_HPP
