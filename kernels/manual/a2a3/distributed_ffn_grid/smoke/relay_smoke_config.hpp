/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time config for the GridPipe REDUCE <-> UNICAST CHANNEL RELAY smoke.
//
// WHAT IT PINS DOWN: a group reduce and a unicast flow TAKE TURNS on ONE channel.
// The two used to own separate indices (a reserved GroupPullChan next to the
// unicast pool), which cost a scoreboard triplet nobody could reuse; they share the
// pool now, and sharing needs a handover rule in each direction because the two
// tenants count in DIFFERENT streams -- a unicast free_scb carries its consumer's
// cons_idx, a reduce free_scb carries the sink's fold count.
//
// Three cells in a row, ONE channel in the shared pool (ChanCount = 2, of which
// channel 0 is the reserved broadcast index this test never uses), and FOUR
// launches over ONE window per cell:
//
//   phase 0  REDUCE   cells 0,1,2 fan in to the sink (cell 2), ROUNDS rounds
//   phase 1  UNICAST  cell 0 --TILES tiles + CLOSE--> cell 2, which drains them
//   phase 2  REDUCE   the same fan-in again, different contributions
//   phase 3  ALL THREE AGAIN, INSIDE ONE LAUNCH -- reduce, unicast, reduce, with
//            every participant passing `isLastRound` on the final round of each
//            reduce.  Same channel, same rules; what changes is WHO ends the
//            collective's tenancy.  Without the flag the tenancy ends at the next
//            InitGridPipeFromWindow, so phases 0-2 need a launch boundary between
//            them; with it the release happens where the caller says, so no
//            boundary is needed -- and a stage-B TPUSH that arrives while the sink
//            is still folding stage A must WAIT rather than be granted the live
//            collective's channel, which this phase also pins down.
//
// So the one shared channel carries reduce -> unicast -> reduce (twice over), and
// each boundary exercises one rule:
//
//   phase 0 -> 1  THE ZERO RULE (归约后单播复用).  The consumer's cons_idx counts
//                 FOLDS at that point, and the channel's ring was never written.
//                 Both sides restart from zero: cons_idx = 0, the channel's own
//                 ready/close scoreboards = 0, and the grant answers prod_idx = 0
//                 and stores 0 into the producer's free_scb.  Relaying the fold
//                 count as a ready baseline instead would hang the first TPOP (it
//                 would wait for a ready count no producer is going to reach) or,
//                 worse, sail through onto a ring slot this phase never wrote.
//
//   phase 1 -> 2  DRAIN, THEN BASELINE + ROUND (单播后归约复用).  The channel is
//                 granted only after the flow CLOSED *and* was fully drained --
//                 the reduce abandons the ring, so a leftover would never be read
//                 and the old consumer would still owe a FREE store that would
//                 land in the collective's credit counter.  Nothing is cleared
//                 afterwards: cons_idx == TILES becomes the collective's baseline,
//                 and every member is handed it BOTH as its round origin and as
//                 the starting value of its free_scb.  This is why TILES is
//                 deliberately non-zero and why phase 2 runs more than one round:
//                 with a baseline the member's credit wait is live from its very
//                 first round, unlike the zero-based first collective.
//
// The goldens are SUMS, so they do not depend on arrival order -- only on every
// contribution being folded exactly once, in the right phase.

#ifndef RELAY_SMOKE_CONFIG_HPP
#define RELAY_SMOKE_CONFIG_HPP

#include <cstdint>

// Rounds each reduce phase runs.  > 1 so the fold counter actually advances and
// phase 2's members exercise the credit wait against a NON-ZERO baseline.
#ifndef CONFIG_RELAY_ROUNDS
#define CONFIG_RELAY_ROUNDS 2
#endif

// Tiles the middle unicast phase pushes.  The last one carries isLastTransfer
// (CLOSE); the consumer drains all of them, which is what makes the channel
// eligible for the collective again.
// Deliberately MORE than CONFIG_RELAY_SLOT_COUNT, so the turn WRAPS the ring and
// the producer's later pushes have to wait on credit.  That is what makes the
// zeroed free_scb of the reduce -> unicast rule load-bearing: with a turn that fits
// the ring, `prod_idx < SlotCount` all the way through and the credit counter is
// never read at all -- a stale one would pass unnoticed.  (Unlike the unicast
// handover smoke, wrapping is safe here: this consumer drains as the flow runs.)
#ifndef CONFIG_RELAY_TILES
#define CONFIG_RELAY_TILES 3
#endif

// Tiles of the SIDE FLOW the unicast phase runs at the same time, from the middle
// cell to the unicast producer.  It exists to break a coincidence: the producer's
// own free_scb ends the phase holding the sink's cons_idx, which is exactly the
// baseline phase 2 then wants, so that member alone cannot tell "the sink stated my
// free baseline" from "nobody touched it".  The middle cell's counter ends up
// holding a DIFFERENT number (its own consumer's cons_idx), so if the sink did not
// state the baseline, that member's very first credit wait would block on a
// threshold nothing is going to reach -- with the sink waiting for its epoch.
// Deliberately != CONFIG_RELAY_TILES; 0 disables the side flow.
#ifndef CONFIG_RELAY_SIDE_TILES
#define CONFIG_RELAY_SIDE_TILES 1
#endif

// Ring depth of the shared channel.  Only the unicast stages use the ring at all
// (a pull reduce moves its payload through the caller's arena), and they push more
// tiles than it holds -- see CONFIG_RELAY_TILES.
#ifndef CONFIG_RELAY_SLOT_COUNT
#define CONFIG_RELAY_SLOT_COUNT 2
#endif

// Spin bound installed on the pipe (pipe.maxSpins).  0 = block forever (the
// shipping path).  Non-zero turns a handover that never happens into a fault
// sentinel the host decodes instead of a hang.
#ifndef CONFIG_RELAY_MAX_SPINS
#define CONFIG_RELAY_MAX_SPINS 0u
#endif

#ifndef CONFIG_RELAY_T
#define CONFIG_RELAY_T 16
#endif

#ifndef CONFIG_RELAY_W
#define CONFIG_RELAY_W 64
#endif

constexpr int RELAY_ROUNDS = CONFIG_RELAY_ROUNDS;
constexpr int RELAY_TILES = CONFIG_RELAY_TILES;
constexpr int RELAY_SIDE_TILES = CONFIG_RELAY_SIDE_TILES;
constexpr int RELAY_SLOT_COUNT = CONFIG_RELAY_SLOT_COUNT;
constexpr uint32_t RELAY_MAX_SPINS = CONFIG_RELAY_MAX_SPINS;
constexpr int RELAY_T = CONFIG_RELAY_T;
constexpr int RELAY_W = CONFIG_RELAY_W;

// One row of three.  Roles are cell indices, which on a 1 x 3 row-major grid are
// also the logical block ids every GridPipe call names its peer by.  The SINK of
// both reduces is also the CONSUMER of the unicast phase, and the unicast PRODUCER
// is one of the reduce members -- so both ends of the shared channel change tenant,
// not just one.
constexpr int RELAY_ROWS = 1;
constexpr int RELAY_COLS = 3;
constexpr int RELAY_CELLS = RELAY_ROWS * RELAY_COLS;
constexpr int RELAY_CELL_SINK = RELAY_CELLS - 1; // reduce sink + unicast consumer
constexpr int RELAY_CELL_PROD = 0;               // reduce member + unicast producer
constexpr int RELAY_CELL_MID = 1;                // reduce member + side-flow producer

// FOUR launches, and the fourth is the point of the second half of this test.
//
//   phase 0 / 1 / 2   reduce, unicast, reduce -- ONE STAGE PER LAUNCH.  A caller
//                     that never marks a last round gets its collective's channel
//                     back at the next InitGridPipeFromWindow, so the handover
//                     needs a launch boundary.
//   phase 3           the SAME three stages inside ONE launch, with every
//                     participant passing `isLastRound` on the final round of each
//                     reduce.  That is what releases the collective's channel
//                     mid-launch, so the unicast flow can take it over with no
//                     boundary in between -- and take it back afterwards.
constexpr int RELAY_PHASE_INLAUNCH = 3;
constexpr int RELAY_LAUNCHES = 4;

// Result tiles.  Every reduce stage, every unicast stage and every side flow gets
// its own, so a stage that folded the wrong thing cannot hide behind another's.
constexpr int RELAY_OUT_REDUCE0 = 0;  // phase 0
constexpr int RELAY_OUT_UNICAST0 = 1; // phase 1
constexpr int RELAY_OUT_REDUCE1 = 2;  // phase 2
constexpr int RELAY_OUT_SIDE0 = 3;    // phase 1's side flow
constexpr int RELAY_OUT_REDUCE2 = 4;  // phase 3, stage A
constexpr int RELAY_OUT_UNICAST1 = 5; // phase 3, stage B
constexpr int RELAY_OUT_REDUCE3 = 6;  // phase 3, stage C
constexpr int RELAY_OUT_SIDE1 = 7;    // phase 3's side flow
constexpr int RELAY_OUT_TILES = 8;

constexpr int RELAY_TILE_ELEMS = RELAY_T * RELAY_W;
constexpr int RELAY_TILE_BYTES = RELAY_TILE_ELEMS * 4; // fp32
constexpr int RELAY_SLOT_BYTES = RELAY_TILE_BYTES;

// The buffer the kernel reads and writes, in tiles:
//
//   [RELAY_SRC_BASE ..]         host-filled contribution VALUES, [stage][round][cell]
//   [RELAY_UNICAST_IN_BASE ..]  host-filled unicast input, one per cell
//   [RELAY_ARENA_BASE ..]       the CONTRIBUTION ARENA the reduce actually reads --
//                               host-ZEROED, written by each member itself, right
//                               before the TREDUCE of that round
//
// The arena is written by the kernel rather than pre-filled on purpose: a fold that
// ran ahead of a member's store would then read a ZERO tile, so the goldens actually
// test the handshake that orders the two.  Pre-filled contributions would come out
// correct even with no synchronisation at all.
//
// It is indexed by BLOCK ID within a round, which is what makes the group
// instruction's "member b's copy of this address is + (b - sink)*stride" arithmetic
// hold, with stride = one tile.
// FOUR reduce stages: phase 0, phase 2, and the two inside phase 3.
constexpr int RELAY_REDUCE_STAGES = 4;
constexpr int RELAY_REDUCE_TILES = RELAY_REDUCE_STAGES * RELAY_ROUNDS * RELAY_CELLS;
constexpr int RELAY_SRC_BASE = 0;                                  // + (stage*ROUNDS + round)*CELLS + cell
constexpr int RELAY_UNICAST_IN_BASE = RELAY_REDUCE_TILES;          // + cell
constexpr int RELAY_ARENA_BASE = RELAY_REDUCE_TILES + RELAY_CELLS; // same indexing as SRC
constexpr int RELAY_IN_TILES = 2 * RELAY_REDUCE_TILES + RELAY_CELLS;
constexpr int RELAY_IN_BYTES = RELAY_IN_TILES * RELAY_TILE_BYTES;
constexpr int RELAY_OUT_BYTES = RELAY_OUT_TILES * RELAY_TILE_BYTES;

// The stamp reduce stage `s` adds to every contribution, so the four stages produce
// clearly different goldens: a fold that read another stage's contributions is off
// by a whole multiple of RELAY_STAGE_STAMP * RELAY_CELLS, not by a rounding error.
constexpr int RELAY_STAGE_STAMP = 100;

// TWO channels: index 0 is the reserved broadcast one (GroupMax > 0 reserves it by
// index whether or not a broadcast ever runs), index 1 is the ENTIRE shared pool.
// One is the point -- with two pool channels the reduce and the flow would each get
// their own and there would be no handover to test.
constexpr int RELAY_CHAN_COUNT = 2;
constexpr int RELAY_GROUP_MAX = RELAY_CELLS;

static_assert(
    RELAY_TILES > RELAY_SLOT_COUNT,
    "the main unicast turn must OUTGROW the ring, or the producer never reads the credit counter the reduce -> "
    "unicast rule zeroes and the rule is untested");
static_assert(
    RELAY_SIDE_TILES <= RELAY_SLOT_COUNT,
    "the side flow must fit the ring: its consumer only drains it after finishing its own turn, so a side producer "
    "that had to wait for credit mid-turn would stall behind it");
static_assert(RELAY_ROUNDS >= 1 && RELAY_TILES >= 1, "each phase must move at least one item");
static_assert(
    RELAY_SIDE_TILES == 0 || RELAY_SIDE_TILES != RELAY_TILES,
    "the side flow must leave its producer a DIFFERENT free_scb value than the sink's baseline, or it proves nothing");

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>() for a pipe WITH a
// group collective (GroupMax > 0, so the group mailbox is part of the layout):
//   kSlotRegionOffset + 2*GroupMax lines + ChanCount*SlotCount*SlotStride + one
//   producer staging slot
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int RELAY_GRID_CHAN_MAX = 4;          // pto::kGridChanCount
constexpr int RELAY_GRID_CONS_HIST_MAX = 8;     // pto::kGridConsHistMax
constexpr int RELAY_GRID_SCB_LINE_STRIDE = 64;  // pto::grid_mock::kScbLineStride
constexpr int RELAY_GRID_BIND_QUEUE_DEPTH = 32; // pto::kGridBindQueueDepth
constexpr int RELAY_ACTIVE_VECTOR_SUBBLOCK_ID = 0;
constexpr int RELAY_GRID_FLAGS_BYTES = 3 * RELAY_GRID_CHAN_MAX * RELAY_GRID_SCB_LINE_STRIDE;                 // 768
constexpr int RELAY_GRID_CONTROL_BYTES = (2 * RELAY_GRID_BIND_QUEUE_DEPTH + 1) * RELAY_GRID_SCB_LINE_STRIDE; // 4160
constexpr int RELAY_GRID_RECORD_WORDS = 4 + 10 * RELAY_GRID_CHAN_MAX + 4 * RELAY_GRID_CONS_HIST_MAX + 3;     // 79
constexpr int RELAY_GRID_RECORD_BYTES =
    ((RELAY_GRID_RECORD_WORDS * 4 + RELAY_GRID_SCB_LINE_STRIDE - 1) / RELAY_GRID_SCB_LINE_STRIDE) *
    RELAY_GRID_SCB_LINE_STRIDE; // 320
constexpr int RELAY_GRID_SLOT_REGION_OFFSET =
    RELAY_GRID_FLAGS_BYTES + RELAY_GRID_CONTROL_BYTES + RELAY_GRID_RECORD_BYTES; // 5248
constexpr int RELAY_GROUP_MAILBOX_BYTES = 2 * RELAY_GROUP_MAX * RELAY_GRID_SCB_LINE_STRIDE;
constexpr int RELAY_WINDOW_BYTES = RELAY_GRID_SLOT_REGION_OFFSET + RELAY_GROUP_MAILBOX_BYTES +
                                   RELAY_CHAN_COUNT * RELAY_SLOT_COUNT * RELAY_SLOT_BYTES + RELAY_SLOT_BYTES;

#endif // RELAY_SMOKE_CONFIG_HPP
