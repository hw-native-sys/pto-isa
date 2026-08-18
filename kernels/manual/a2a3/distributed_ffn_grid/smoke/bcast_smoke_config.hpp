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

#include <cstdint>

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

// Broadcasts the source issues back to back on ONE pipe.  The ring is as deep as
// the group is wide, so round r reuses the slot round r - SlotCount used, and
// rounds > 1 is what exercises the producer-side free credit: before round r the
// source must see its free_scb reach baseline + r*(K-1), which (each receiver
// contributing at most r) is an exact "every receiver has drained it" test.
#ifndef CONFIG_BCAST_ROUNDS
#define CONFIG_BCAST_ROUNDS 1
#endif

// 1 = EVERY member of the group broadcasts its own tile (真·同时 MPSC, the shape
// the FFN AllGather uses) instead of the single BCAST_SRC.  Combined with
// BCAST_ROUNDS > 1 this is the full stress case: K concurrent senders drawing
// basek from one dense sequence, so every ring slot changes owner every round and
// the ticket window has to hand them over in order.
#ifndef CONFIG_BCAST_ALL_SOURCES
#define CONFIG_BCAST_ALL_SOURCES 0
#endif

// Spin bound for every wait this kernel's pipe performs.  It is installed on the
// PIPE (`pipe.maxSpins`) right after init, so the kernel below calls the plain PTO
// instructions -- TBROADCAST / TPOP -- either way.
//
//   0 (default) = block forever, which is what hardware WAIT_SPR does and what the
//                 shipping path wants; this is the smoke's default coverage.
//   n > 0       = bound each wait at n, turning a stuck handshake into a fault
//                 sentinel the host decodes ("which wait, which scoreboard")
//                 instead of a hang the runtime kills with a bare rc.
//
// Use a SMALL n when debugging: one spin here is a whole service pass (a bind
// queue scan plus a scoreboard sweep), not a single load, so a healthy run needs
// only a handful of them and n = 1000 already reports in milliseconds.  A large n
// (say 200000) takes minutes to trip and looks exactly like the hang it is meant
// to diagnose.
#ifndef CONFIG_BCAST_MAX_SPINS
#define CONFIG_BCAST_MAX_SPINS 0u
#endif

// Ring depth of the broadcast channel, in slots.  0 = one slot per group member,
// which is what lets every member publish before anybody drains.  A SMALLER value
// is the interesting one: the ring is then shallower than the group is wide, so
// the publishers of one round no longer fit in it and the caller must send them in
// WAVES of SlotCount, draining between waves.  That is the whole point of taking
// the address out of the writer's identity -- the receiver's SRAM sizes the ring,
// not the number of writers -- and the wave loop in the kernel is what a real
// caller with a group wider than its ring has to do.
#ifndef CONFIG_BCAST_SLOT_COUNT
#define CONFIG_BCAST_SLOT_COUNT 0
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
constexpr int BCAST_ROUNDS = CONFIG_BCAST_ROUNDS;
constexpr int BCAST_ALL_SOURCES = CONFIG_BCAST_ALL_SOURCES;
constexpr uint32_t BCAST_MAX_SPINS = CONFIG_BCAST_MAX_SPINS;
constexpr int BCAST_T = CONFIG_BCAST_T;
constexpr int BCAST_W = CONFIG_BCAST_W;

constexpr int BCAST_TILE_ELEMS = BCAST_T * BCAST_W;
constexpr int BCAST_TILE_BYTES = BCAST_TILE_ELEMS * 4; // fp32 payload tile

// The group is the sub-rectangle extent (SUBRECT) or the larger of the row/col
// extent (ROW/COL).
constexpr int BCAST_GROUP_MAX = (BCAST_SUBRECT != 0) ?
                                    ((BCAST_RECT_R1 - BCAST_RECT_R0) * (BCAST_RECT_C1 - BCAST_RECT_C0)) :
                                    ((BCAST_ROWS > BCAST_COLS) ? BCAST_ROWS : BCAST_COLS);

// The broadcast publishes into the ring of the RESERVED broadcast channel, so
// this smoke owns exactly that one channel and its ring IS the collective's
// arena.  One [T, W] fp32 tile per slot.
//
// RING DEPTH = GROUP WIDTH here, and that is a caller decision, not a library
// requirement: with a slot per publisher every member can broadcast before
// anybody drains (the ALL_SOURCES stress below does exactly that).  A shallower
// ring is legal and would make the same run publish in waves, but then the
// kernel would have to interleave its drains -- a receiver cannot free a slot
// while its own caller is blocked inside TBROADCAST.
constexpr int BCAST_SLOT_BYTES = BCAST_TILE_BYTES;
constexpr int BCAST_SLOT_COUNT = (CONFIG_BCAST_SLOT_COUNT > 0) ? CONFIG_BCAST_SLOT_COUNT : BCAST_GROUP_MAX;
static_assert(BCAST_SLOT_COUNT > 0 && BCAST_SLOT_COUNT <= 32, "the ring-slot occupancy mask holds 32 slots");

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>():
//   kSlotRegionOffset (fixed header + control + record)
//   + group mailbox: 2 * GroupMax lines (request queue + response queue, one
//                    line per RANK-IN-GROUP -- not per core in the mesh)
//   + ChanCount * SlotCount * SlotStride (the broadcast channel's ring)
//   + SlotStride (isolated local producer L1 staging slot)
// The fixed header is kFlagsBytes = 3 * kGridChanCount * kScbLineStride
// (ready/free/close, one cache line each), the two unicast bind-mailbox queues
// plus the group-reduce epoch line, and the LOCAL pipe record.
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int BCAST_GRID_CHAN_COUNT = 1;        // == pto::kGridBcastChanCount: the broadcast channel's ring
constexpr int BCAST_GRID_CHAN_MAX = 4;          // pto::kGridChanCount
constexpr int BCAST_GRID_CONS_HIST_MAX = 8;     // pto::kGridConsHistMax
constexpr int BCAST_GRID_SCB_LINE_STRIDE = 64;  // pto::grid_mock::kScbLineStride
constexpr int BCAST_GRID_BIND_QUEUE_DEPTH = 32; // pto::kGridBindQueueDepth
constexpr int BCAST_ACTIVE_VECTOR_SUBBLOCK_ID = 0;
constexpr int BCAST_GRID_FLAGS_BYTES = 3 * BCAST_GRID_CHAN_MAX * BCAST_GRID_SCB_LINE_STRIDE; // 768
// Two bind-mailbox queues (one line per peer, each end) + the group-reduce epoch line.
constexpr int BCAST_GRID_CONTROL_BYTES = (2 * BCAST_GRID_BIND_QUEUE_DEPTH + 1) * BCAST_GRID_SCB_LINE_STRIDE; // 4160
constexpr int BCAST_GRID_RECORD_WORDS = 4 + 10 * BCAST_GRID_CHAN_MAX + 4 * BCAST_GRID_CONS_HIST_MAX + 3;     // 79
constexpr int BCAST_GRID_RECORD_BYTES =
    ((BCAST_GRID_RECORD_WORDS * 4 + BCAST_GRID_SCB_LINE_STRIDE - 1) / BCAST_GRID_SCB_LINE_STRIDE) *
    BCAST_GRID_SCB_LINE_STRIDE; // 320
constexpr int BCAST_GRID_SLOT_REGION_OFFSET =
    BCAST_GRID_FLAGS_BYTES + BCAST_GRID_CONTROL_BYTES + BCAST_GRID_RECORD_BYTES; // 5248
constexpr int BCAST_GROUP_MAILBOX_BYTES = 2 * BCAST_GROUP_MAX * BCAST_GRID_SCB_LINE_STRIDE;
constexpr int BCAST_WINDOW_BYTES = BCAST_GRID_SLOT_REGION_OFFSET + BCAST_GROUP_MAILBOX_BYTES +
                                   BCAST_GRID_CHAN_COUNT * BCAST_SLOT_COUNT * BCAST_SLOT_BYTES + BCAST_SLOT_BYTES;

#endif // BCAST_SMOKE_CONFIG_HPP
