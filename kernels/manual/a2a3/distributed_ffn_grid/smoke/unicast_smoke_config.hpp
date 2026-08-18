/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time config for the GridPipe UNICAST HANDOVER smoke kernel.
//
// WHAT IT PINS DOWN: a channel changes owner while the retiring producer's items
// are STILL UNDRAINED.  That is the one thing the time-division handover promises
// and the only thing no other test in this tree reaches -- the FFN demos all
// happen to drain a producer's stream before the next one binds, so they exercise
// the easy half of the rule.
//
// Three cells, one channel:
//
//   cell 0 (A) ---- N tiles + CLOSE ----> cell 1 (C)     [consumer channel 0]
//   cell 0 (A) ---- baton + CLOSE ------> cell 2 (B)     [same producer channel, reopened]
//   cell 2 (B) ---- N tiles + CLOSE ----> cell 1 (C)     [the SAME consumer channel 0]
//
// The baton is what makes the order deterministic instead of a race: B does not
// ask C for a channel until A has finished with it.  And C's only drain names B,
// so C cannot touch A's tiles until AFTER the handover -- which is exactly the
// state under test: at the moment C hands channel 0 to B, cons_idx is still 0 and
// all N of A's tiles are sitting in the ring.
//
// What must then hold, and what this test therefore checks:
//   * the handover happens AT ALL (a rebind gated on the drain would deadlock here
//     -- C cannot drain, because its only TPOP names a producer that has not been
//     given a channel yet.  This configuration hangs on such an implementation,
//     which is what makes it a regression test rather than a demo);
//   * B does not overwrite A's undrained tiles.  With UNICAST_SLOT_COUNT == N the
//     rings are disjoint by luck; set it to N and the ring wraps, so B's first tile
//     lands exactly on A's first slot and B must WAIT on the credit baseline it was
//     handed at bind time.  That variant is the real payload-safety proof;
//   * the stream stays continuous across the handover: C drains 2N tiles in WRITE
//     order out of one ring under one absolute count, A's first, B's after.
//
// The golden is a SUM, so it does not depend on which producer wrote which tile --
// only on all 2N tiles arriving exactly once.

#ifndef UNICAST_SMOKE_CONFIG_HPP
#define UNICAST_SMOKE_CONFIG_HPP

#include <cstdint>

// Tiles each producer pushes on its turn.  The last one carries isLastTransfer,
// i.e. publishes CLOSE, which is what makes the channel reusable.
#ifndef CONFIG_UNICAST_TILES
#define CONFIG_UNICAST_TILES 2
#endif

// Ring depth of the shared unicast channel.
//   >= 2*TILES : A's turn and B's turn occupy disjoint slots -- the handover is
//                tested, the credit is dormant.
//   == TILES   : B's first tile wraps onto A's first slot, so B must wait for the
//                relayed free baseline before it may write.  This is the variant
//                that proves relay counting protects an undrained tail.
#ifndef CONFIG_UNICAST_SLOT_COUNT
#define CONFIG_UNICAST_SLOT_COUNT 4
#endif

// Spin bound installed on the pipe (pipe.maxSpins).  0 = block forever (the
// shipping path).  A non-zero value turns the deadlock an incorrect handover would
// cause into a fault sentinel the host decodes.
#ifndef CONFIG_UNICAST_MAX_SPINS
#define CONFIG_UNICAST_MAX_SPINS 0u
#endif

#ifndef CONFIG_UNICAST_T
#define CONFIG_UNICAST_T 16
#endif

#ifndef CONFIG_UNICAST_W
#define CONFIG_UNICAST_W 64
#endif

constexpr int UNICAST_TILES = CONFIG_UNICAST_TILES;
constexpr int UNICAST_SLOT_COUNT = CONFIG_UNICAST_SLOT_COUNT;
constexpr uint32_t UNICAST_MAX_SPINS = CONFIG_UNICAST_MAX_SPINS;
constexpr int UNICAST_T = CONFIG_UNICAST_T;
constexpr int UNICAST_W = CONFIG_UNICAST_W;

// One row of three: producer A, consumer C, producer B.  Roles are cell indices,
// which on a 1 x 3 row-major grid are also the logical block ids every GridPipe
// call names its peer by.
constexpr int UNICAST_ROWS = 1;
constexpr int UNICAST_COLS = 3;
constexpr int UNICAST_CELLS = UNICAST_ROWS * UNICAST_COLS;
constexpr int UNICAST_CELL_A = 0;
constexpr int UNICAST_CELL_C = 1;
constexpr int UNICAST_CELL_B = 2;

// A producer must be able to publish its whole turn without waiting on credit,
// because the consumer does not drain anything until the handover has happened.
static_assert(
    UNICAST_TILES <= UNICAST_SLOT_COUNT,
    "a turn must fit the ring: the consumer cannot free a slot before the handover, so a producer that had to "
    "wait for credit mid-turn would never publish the CLOSE the handover needs");

constexpr int UNICAST_TILE_ELEMS = UNICAST_T * UNICAST_W;
constexpr int UNICAST_TILE_BYTES = UNICAST_TILE_ELEMS * 4; // fp32
constexpr int UNICAST_SLOT_BYTES = UNICAST_TILE_BYTES;

// ONE channel, and that is the whole point: with two, C would hand A and B a
// channel each and there would be no handover to test.
constexpr int UNICAST_CHAN_COUNT = 1;

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>() for a pipe with no
// group collective (GroupMax = 0, so no group mailbox):
//   kSlotRegionOffset + ChanCount * SlotCount * SlotStride + one producer staging slot
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int UNICAST_GRID_CHAN_MAX = 4;          // pto::kGridChanCount
constexpr int UNICAST_GRID_CONS_HIST_MAX = 8;     // pto::kGridConsHistMax
constexpr int UNICAST_GRID_SCB_LINE_STRIDE = 64;  // pto::grid_mock::kScbLineStride
constexpr int UNICAST_GRID_BIND_QUEUE_DEPTH = 32; // pto::kGridBindQueueDepth
constexpr int UNICAST_ACTIVE_VECTOR_SUBBLOCK_ID = 0;
constexpr int UNICAST_GRID_FLAGS_BYTES = 3 * UNICAST_GRID_CHAN_MAX * UNICAST_GRID_SCB_LINE_STRIDE; // 768
constexpr int UNICAST_GRID_CONTROL_BYTES =
    (2 * UNICAST_GRID_BIND_QUEUE_DEPTH + 1) * UNICAST_GRID_SCB_LINE_STRIDE;                                    // 4160
constexpr int UNICAST_GRID_RECORD_WORDS = 4 + 10 * UNICAST_GRID_CHAN_MAX + 4 * UNICAST_GRID_CONS_HIST_MAX + 3; // 79
constexpr int UNICAST_GRID_RECORD_BYTES =
    ((UNICAST_GRID_RECORD_WORDS * 4 + UNICAST_GRID_SCB_LINE_STRIDE - 1) / UNICAST_GRID_SCB_LINE_STRIDE) *
    UNICAST_GRID_SCB_LINE_STRIDE; // 320
constexpr int UNICAST_GRID_SLOT_REGION_OFFSET =
    UNICAST_GRID_FLAGS_BYTES + UNICAST_GRID_CONTROL_BYTES + UNICAST_GRID_RECORD_BYTES; // 5248
constexpr int UNICAST_WINDOW_BYTES =
    UNICAST_GRID_SLOT_REGION_OFFSET + UNICAST_CHAN_COUNT * UNICAST_SLOT_COUNT * UNICAST_SLOT_BYTES + UNICAST_SLOT_BYTES;

#endif // UNICAST_SMOKE_CONFIG_HPP
