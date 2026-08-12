/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Grid TPUSH/TPOP model + A2/A3 mock support (A2/A3 backend).
//
// Design_spec: Grid_TPUSH_TPOP_ISA...V8.md (the IPC_SCB scoreboard route).
//
// This header holds ONLY the data model and mock support; the CCE handshake
// intrinsics themselves live in grid_cce_intrinsic.hpp as the V8 two-name facade
// layer (copy_l1_to_neighbor_l1 / sync_hscb / wait_ipc_scb -> __builtin_cce_*).
// There is deliberately NO intermediate PTO wrapper (the old sync_neighbor_scb /
// wait_local_spr / mov_local_spr / ScbOperand / neighbor_sram_addr vocabulary is
// gone, per V8 section 3.4 / section 6 point 4):
//   * Section 1: GridPipe mesh model -- the concurrency array, its binding table,
//                the consumer save/restore table, and the nearest-neighbor
//                topology resolvers the call sites derive peer ranks with.
//   * Section 2: A2/A3 GM-mock support -- fault sentinels and the scoreboard
//                cache-line layout constants.
//   * Section 3: GmSramArena -- the address-segment model that enforces the NoC
//                "TPOP reads local SRAM only" rule for the GM-window mock.
//
// The GridPipe TPUSH/TPOP overloads in pto/common/pto_instr.hpp and the A2/A3
// backends in GridTPush.hpp / GridTPop.hpp both pull this single header in (which
// in turn pulls grid_cce_intrinsic.hpp).  Compiler-visible static constraints
// are still enforced via static_assert inside the overloads in pto_instr.hpp.

#ifndef PTO_A2A3_GRID_INTRINSIC_HPP
#define PTO_A2A3_GRID_INTRINSIC_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp> // for AICORE (callable from both host and aicore contexts)

#include <pto/npu/a2a3/grid_cce_intrinsic.hpp> // V8 CCE facade layer (the ONLY handshake-intrinsic layer)

// ===========================================================================
// Section 1: GridPipe -- neighbor-core FIFO communication primitives.
//
// This is the proposal-level abstraction described in the V8 design spec (the
// IPC_SCB scoreboard handshake route).  The per-channel FIFO state below is read
// by the GridTPush.hpp / GridTPop.hpp sequence expansions, which call the CCE
// facades in grid_cce_intrinsic.hpp: cross-core notify = sync_hscb
// (SYNC_HSCB / ST_HSCB, a monotone absolute count into the peer's channel
// IPC_SCB); local steady-state wait = wait_ipc_scb (WAIT_SPR, read+block in one
// instruction); control-path snapshot = MOV_SPR2X (channel-close scan and relay
// baseline capture); payload = copy_l1_to_neighbor_l1 (COPY_L1_TO_NBR), after
// staging the tile in the pipe's isolated producer L1 slot.
// On A2/A3 there is no cross-core neighbor-IPC_SCB addressing (V8 HW-DEP-1) nor a
// local-L1->neighbor-L1 write (V8 HW-DEP-0), so those facades run their GM mock and
// Section 2 stands in for the IPC_SCB scoreboards with HCCL shared windows and
// GM words.
// ===========================================================================

// Forward declaration: provided by the target backend (cpu_stub.hpp on
// CPU sim builds, CCE intrinsic / runtime header on A2/A3 NPU builds).
// GetGridCoord below uses this; we declare it here so this header can be
// included before any backend headers without triggering an undeclared name.
uint32_t get_block_idx();

namespace pto {

// ---------------------------------------------------------------------------
// 2D mesh coordinates and shape (design doc section 2).
// ---------------------------------------------------------------------------
struct GridShape {
    int gridRows = 0; // N
    int gridCols = 0; // M
};

struct GridCoord {
    int row = 0; // 0 .. gridRows-1
    int col = 0; // 0 .. gridCols-1
};

// Half-open sub-rectangle [row0, row1) x [col0, col1) describing an arbitrary
// rectangular group member set (GridGroup::SUBRECT).  ROW / COL are the special
// cases rect = {r, r+1, 0, cols} / {0, rows, c, c+1}; a general SUBRECT lets a
// single TBROADCAST reach every cell in the rectangle (any-to-any via the mock's
// logical-rank window addressing, including diagonal / far peers).
struct GridRect {
    int row0 = 0;
    int row1 = 0;
    int col0 = 0;
    int col1 = 0;
};

// ---------------------------------------------------------------------------
// Direction enum (design doc section 3.1).  Strongly-typed to avoid clashing
// with the cluster-local pto::Direction enum used by TPipe.
// ---------------------------------------------------------------------------
enum class GridDirection : uint8_t {
    SOURCE = 0, // GM/Host/Runtime injection.  Only valid for TPOP.
    NORTH = 1,  // row -> row-1
    EAST = 2,   // col -> col+1
    WEST = 3,   // col -> col-1
    SOUTH = 4,  // row -> row+1
};

inline constexpr int kGridDirectionCount = 5;

AICORE constexpr int GridDirectionIndex(GridDirection d) { return static_cast<int>(d); }

// ---------------------------------------------------------------------------
// Broadcast GROUP -- the participant set of a TBROADCAST collective.
//
// A collective uses the same channelised payload rings and ready/free/close
// scoreboards as TPUSH/TPOP.  Source ordinal `s` is assigned to channel `s % C`,
// where C is the number of rings carried by the pipe.  At most C sources are
// therefore live at once.  Sources beyond C reuse the same channels in later
// batches, after the previous owner has closed and every receiver has returned
// its reverse FREE credit.  READY/CLOSE remain single-writer absolute stores;
// only the reverse fan-in FREE edge uses atomic accumulation.
//
//   GridGroup::ROW = every cell on the source's row    (the row is the group)
//   GridGroup::COL = every cell on the source's column (the column is the group)
//
// A group still decomposes into two opposite 1-D arms for topology description
// -- ROW = EAST+WEST, COL = NORTH+SOUTH (GroupArmA / GroupArmB) -- but the
// prefix-offset send addresses peers by their rank-in-group directly, not by
// arm, so a receiver drains member `srcRank` with TPOP<GridGroup>(pipe, tile,
// srcRank) regardless of which arm it sits on.
// ---------------------------------------------------------------------------
enum class GridGroup : uint8_t {
    ROW = 0,     // group = the source's row:    EAST arm + WEST arm
    COL = 1,     // group = the source's column: NORTH arm + SOUTH arm
    SUBRECT = 2, // group = an arbitrary sub-rectangle [row0,row1)x[col0,col1)
                 //          (runtime-described via pipe.groupRect).  It subsumes
                 //          ROW/COL and needs no arm decomposition: members are
                 //          addressed by rank-in-rect directly.
};

// The two opposite GridDirections a group decomposes into (topology only).
// constexpr so they fold into non-type template arguments where useful.
AICORE constexpr GridDirection GroupArmA(GridGroup g)
{
    return g == GridGroup::ROW ? GridDirection::EAST : GridDirection::NORTH;
}

AICORE constexpr GridDirection GroupArmB(GridGroup g)
{
    return g == GridGroup::ROW ? GridDirection::WEST : GridDirection::SOUTH;
}

// ---------------------------------------------------------------------------
// Group-rank helpers.  These map between a member's rank-in-group and its grid
// coordinate.  Payload placement is deliberately NOT rank-indexed: collectives
// map source ordinals onto GridPipe channels and then use the normal TPUSH ring
// address `(sequence % SlotCount) * SlotStride` within that channel.
// ---------------------------------------------------------------------------
// Forward declaration: BlockIdFromCoord is defined further down (coordinate
// bootstrap section); the GroupMemberBlockId helper below needs it visible here.
AICORE constexpr int BlockIdFromCoord(GridCoord coord, GridShape shape);

// Number of members in the group that `coord` belongs to.  The trailing
// `rect` is consulted only for SUBRECT (ROW/COL ignore it); defaulting it keeps
// every existing ROW/COL call site unchanged.
AICORE constexpr int GridGroupSize(GridGroup g, GridShape s, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? s.gridCols :
           (g == GridGroup::COL) ? s.gridRows :
                                   ((rect.row1 - rect.row0) * (rect.col1 - rect.col0));
}

// This cell's rank within its group = its prefix-offset base (count_k = 1).
// ROW groups vary along the column axis; COL groups along the row axis; SUBRECT
// uses a row-major rank within [row0,row1)x[col0,col1).
AICORE constexpr int RankInGroup(GridGroup g, GridCoord c, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? c.col :
           (g == GridGroup::COL) ? c.row :
                                   ((c.row - rect.row0) * (rect.col1 - rect.col0) + (c.col - rect.col0));
}

// Coordinate of the member whose rank-in-group is `rankInGroup`, given this
// cell's coordinate (the member shares this cell's fixed axis for ROW/COL).
// SUBRECT inverts the row-major rank entirely from `rect` (self-independent).
AICORE constexpr GridCoord GroupMemberCoord(GridGroup g, GridCoord self, int rankInGroup, GridRect rect = {})
{
    if (g == GridGroup::ROW) {
        return GridCoord{self.row, rankInGroup};
    }
    if (g == GridGroup::COL) {
        return GridCoord{rankInGroup, self.col};
    }
    const int colSpan = rect.col1 - rect.col0;
    return GridCoord{rect.row0 + rankInGroup / colSpan, rect.col0 + rankInGroup % colSpan};
}

AICORE constexpr int GroupMemberBlockId(GridGroup g, GridCoord self, GridShape s, int rankInGroup, GridRect rect = {})
{
    return BlockIdFromCoord(GroupMemberCoord(g, self, rankInGroup, rect), s);
}

// ---------------------------------------------------------------------------
// Mesh topology -> the MOV_UBUF_GROUP group descriptor (GridBlockRect, defined
// in grid_cce_intrinsic.hpp because the member set is a machine operand).  That
// instruction names a group by the two CORNER BLOCK IDS of a sub-rectangle, so
// these are the two conversions a Tier-2 caller needs: from the half-open
// GridRect a pipe carries, and from a GridGroup + this cell's coordinate -- ROW
// and COL being the one-row / one-column rectangles through this cell, which is
// why the group intrinsic itself no longer has to know the GridGroup enum at all.
//
// Both walk members in the same row-major order RankInGroup ranks them in, so
// source ordinals, channel assignment, and the group arena stay in step.
// ---------------------------------------------------------------------------
AICORE constexpr GridBlockRect GridBlockRectFromRect(GridRect rect, GridShape s)
{
    return GridBlockRect{
        static_cast<uint32_t>(BlockIdFromCoord(GridCoord{rect.row0, rect.col0}, s)),
        static_cast<uint32_t>(BlockIdFromCoord(GridCoord{rect.row1 - 1, rect.col1 - 1}, s)),
        static_cast<uint32_t>(s.gridCols)};
}

AICORE constexpr GridBlockRect GridBlockRectOfGroup(GridGroup g, GridCoord c, GridShape s, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? GridBlockRectFromRect(GridRect{c.row, c.row + 1, 0, s.gridCols}, s) :
           (g == GridGroup::COL) ? GridBlockRectFromRect(GridRect{0, s.gridRows, c.col, c.col + 1}, s) :
                                   GridBlockRectFromRect(rect, s);
}

// ---------------------------------------------------------------------------
// Structured collective channel schedule.
//
// Source ordinal `s` owns channel `s % C` at owner position `s / C`.  The first
// C ordinals form batch zero, the next C batch one, and so on.  A channel's
// owners execute serially while different channels remain independent.  These
// helpers are shared by TBROADCAST and group TREDUCE so both operations use the
// exact same ring/sequence schedule.
// ---------------------------------------------------------------------------
AICORE constexpr uint32_t GridCollectiveChannelCount(uint32_t sourceCount, uint32_t pipeChannelCount)
{
    return sourceCount < pipeChannelCount ? sourceCount : pipeChannelCount;
}

AICORE constexpr uint32_t GridCollectiveBatchCount(uint32_t sourceCount, uint32_t channelCount)
{
    return channelCount == 0 ? 0 : (sourceCount + channelCount - 1) / channelCount;
}

AICORE constexpr uint32_t GridCollectiveChannel(uint32_t sourceOrdinal, uint32_t channelCount)
{
    return sourceOrdinal % channelCount;
}

AICORE constexpr uint32_t GridCollectiveOwnerPosition(uint32_t sourceOrdinal, uint32_t channelCount)
{
    return sourceOrdinal / channelCount;
}

AICORE constexpr uint32_t GridCollectiveOwnerCount(uint32_t sourceCount, uint32_t channelCount, uint32_t channel)
{
    return channel >= sourceCount ? 0 : (sourceCount - channel + channelCount - 1) / channelCount;
}

AICORE constexpr uint32_t GridCollectiveNextSourceOrdinal(
    uint32_t sourceOrdinal, uint32_t sourceCount, uint32_t channelCount)
{
    const uint32_t next = sourceOrdinal + channelCount;
    return next < sourceCount ? next : GridCollectiveChannel(sourceOrdinal, channelCount);
}

// `nextProducerSequence` is the persistent prodIndex word.  Zero denotes a
// fresh producer; a non-zero word is already the absolute sequence relayed to
// this producer's next turn.  Owner position disambiguates a fresh non-first
// owner without adding another persistent state word.
AICORE constexpr uint32_t GridCollectiveProducerSequence(uint32_t nextProducerSequence, uint32_t ownerPosition)
{
    return nextProducerSequence == 0 ? ownerPosition : nextProducerSequence;
}

AICORE constexpr uint32_t GridCollectiveProducerTurn(uint32_t sequence, uint32_t ownerPosition, uint32_t ownerCount)
{
    return (sequence - ownerPosition) / ownerCount;
}

// ---------------------------------------------------------------------------
// THE CONCURRENCY ARRAY -- what replaced the per-direction state.
//
// A GridPipe used to hold one (ready_scb, free_scb, slot ring, prod_idx, cons_idx)
// set per mesh DIRECTION, which welded the pipe's resources to the geometry: five
// sets, of which no kernel ever used more than two, and a channel whose peer could
// not change without changing compass point.  Both are wrong for a time-division
// MPSC schedule, where the core on the other end of an edge changes between phases
// while the edge does not.
//
// The state is now two arrays of kGridChanCount INDEPENDENT CHANNEL POOLS.  A local
// consumer channel owns ready/close, a receive ring and cons_idx; a local producer
// channel owns free and prod_idx.  The bind handshake exchanges their indices, so
// an edge may use producer channel p at core X and consumer channel c at core Y.
// For core X:
//
//   readyScb[c]   written by X's upstream producer       (X blocks in TPOP)
//   closeScb[c]   final ready count from that producer   (consumer-side close)
//   slotBase[c]   written by that producer, read by X    (receive ring)
//   consIndex[c]  X's count of what it has drained from consumer channel c
//   freeScb[p]    written by X's downstream consumer     (X blocks in TPUSH)
//   prodIndex[p]  X's count of what it has published on producer channel p
//
// BIND/RELAY POLICY.  The producer first reserves a never-used or locally CLOSED
// producer channel.  The consumer independently takes a never-used consumer channel,
// or reuses one only after closeScb says the old producer sent its last tile and
// consIndex says that tile was drained.  It relays readyScb[c] to prodIndex[p] and
// consIndex[c] to freeScb[p].  The absolute sequence therefore continues across
// producer turns; neither the ring nor an SCB is reset.
// ---------------------------------------------------------------------------

// Channels per core.  Each channel consumes three native IPC_SCB slots
// (ready/free/close), three 64 B mock lines, and one slot ring.
inline constexpr int kGridChanCount = 4;
static_assert(3 * kGridChanCount <= 16, "GridPipe ready/free/close SCBs must fit the 16 native IPC_SCB slots");

// Peer identity id -- A LOGICAL BLOCK ID.
//
// Every core in this mesh addresses every other by its logical block id: the
// row-major index of its cell, BlockIdFromCoord(coord, shape).  There is no rank
// here in the collective-library sense -- nothing is spread over several cards, and
// the "windows" the mock resolves against are the per-core SRAM segments of ONE
// device.  Both the producer used as a channel key and the consumer used as a
// transfer target are expressed in this same logical-block-id namespace.
//
// (It is deliberately NOT get_block_idx().  Under a waved launch the hardware block
// index is an index within the wave, while the logical block id names the cell in
// the whole mesh and is stable across waves -- which is what a doorbell has to be.)
//
// kGridNoPeer means "this transfer half has no peer": a mesh-edge cell with no
// upstream or downstream simply skips that TPOP or TPUSH.
inline constexpr uint32_t kGridNoPeer = 0xFFFFFFFFu;
// Bind L1 mailboxes are host-zeroed.  Ids and channel indices are stored with a
// +1 bias, so zero remains an armed/pending word even when logical block 0 or
// channel 0 participates.  This also prevents a late-scheduled consumer's init
// from clearing a request an already-running producer deposited in its window.
inline constexpr uint32_t kGridBindPending = 0u;
inline constexpr uint32_t kGridBindHandshakeComplete = 1u;
inline constexpr uint32_t kGridBindModeDynamic = 1u;
inline constexpr uint32_t kGridBindModeFixed = 2u;
inline constexpr uint32_t kGridBindModeBroadcast = 3u;
inline constexpr int kGridInvalidChan = -1;

// Dynamic TPUSH/TPOP binding uses a source-indexed MPSC pending queue rather
// than a shared mailbox.  The runtime context used by the A2/A3 GridPipe demos
// exposes at most 64 logical windows, so one fixed slot per logical peer covers
// the complete addressable mesh.  A producer may have at most one outstanding
// request to one consumer (the public bind call is synchronous); different
// producers use different request lines and therefore never overwrite each
// other.  Responses are indexed by consumer id for the symmetric reason.
inline constexpr int kGridBindQueueDepth = 64;
inline constexpr int kGridBindQueueProdIdWord = 0;
inline constexpr int kGridBindQueueProdChanWord = 1;
inline constexpr int kGridBindQueueTokenWord = 2;
inline constexpr int kGridBindQueueCommitWord = 3;
inline constexpr int kGridBindQueueResponseTokenWord = 0;
inline constexpr int kGridBindQueueResponseReadyWord = 1;
inline constexpr int kGridBindQueueResponseConsChanWord = 2;
inline constexpr int kGridBindQueueResponseCommitWord = 3;
inline constexpr int kGridBindQueueEntryBytes = 64;
inline constexpr int kGridBindQueueEntryWords = kGridBindQueueEntryBytes / static_cast<int>(sizeof(uint32_t));
static_assert(
    kGridBindQueueCommitWord < kGridBindQueueEntryWords && kGridBindQueueResponseCommitWord < kGridBindQueueEntryWords,
    "each GridPipe bind-queue entry must fit in one cache line");

// Depth of the consumer history (see GridConsumerTable).  Deliberately larger than
// the channel count: channels are a CONCURRENCY resource, consumers come and go
// over TIME, and a schedule may rotate through more consumers than it ever has
// open at once.
inline constexpr int kGridConsHistMax = 8;
// A CLOSE-only rebind may install the next producer while payload from one or
// more retired turns is still resident in the ring.  Keep the source identity
// and end-exclusive boundary of those turns until consIndex crosses them.  The
// depth is deliberately the same as the peer history: overflowing it is a
// protocol error, never a reason to forget ownership of live payload.
inline constexpr int kGridRetiredTurnMax = kGridConsHistMax;

// ---------------------------------------------------------------------------
// THE PIPE RECORD -- why the binding table lives in the WINDOW, not in the object.
//
// The resources the allocator hands out (scoreboards, slot rings) live in GM and
// SURVIVE A KERNEL LAUNCH.  A GridPipe object does not: it is an ordinary local
// declared inside the kernel, so a schedule that spans several launches builds a
// fresh one each time.  Put the bind counters in the object and the allocator loses
// its memory at exactly the boundary where it matters -- the next launch sees
// "nothing has ever been bound", hands out element 0 again, and that element's
// ready_scb still holds the previous phase's final count.  The first TPOP then
// sails straight through onto a slot nobody wrote this phase.
//
// (That is not hypothetical: it is what the TPUSH-ReduceSum demo did.  Phase B
// reduces EAST and phase C reduces SOUTH, two different producers and two separate
// launches over one window, and both landed on channel 0.)
//
// So the bindings, per-consumer FSM, close baselines, and run-counter mirrors live
// in a small record at a fixed offset in the window.  InitGridPipeFromWindow ADOPTS
// it instead of clearing it.  A freshly allocated window -- which the host memsets
// to zero -- reads as "nothing bound" because bindCnt == 0 is exactly that.  Ids
// are stored with a +1 bias so zero remains "empty" although block id 0 is valid.
//
// The record is LOCAL: no peer ever stores into it.  Binding/FSM fields change only
// on a turn boundary; TPUSH/TPOP mirror just the one advancing prod/cons word so a
// later kernel launch can continue the same absolute count.
//
// SINGLE-WRITER REQUIREMENT.  GridPipe updates this record with ordinary local
// stores.  On A2/A3 mix mode a logical block has two AIV
// sub-blocks with the same get_block_idx() and distinct get_subblockid() values.
// A kernel using GridPipe must therefore execute its vector body on exactly one of
// them; the distributed_ffn_grid kernels select sub-block 0 at their vector entry.
// ---------------------------------------------------------------------------
inline constexpr int kGridRecCurConsChan = 0;  // current local consumer channel + 1
inline constexpr int kGridRecPrevProd = 1;     // previous upstream producer id + 1
inline constexpr int kGridRecConsCur = 2;      // current downstream consumer id + 1
inline constexpr int kGridRecConsUsed = 3;     // occupied downstream-consumer history entries
inline constexpr int kGridRecConsChanProd = 4; // [kGridChanCount] upstream producer id + 1
inline constexpr int kGridRecConsChanBindCnt = kGridRecConsChanProd + kGridChanCount; // [kGridChanCount]
inline constexpr int kGridRecConsId = kGridRecConsChanBindCnt + kGridChanCount;       // [kGridConsHistMax] id + 1
inline constexpr int kGridRecConsState = kGridRecConsId + kGridConsHistMax;           // [kGridConsHistMax] binding FSM
inline constexpr int kGridRecConsProdChan = kGridRecConsState + kGridConsHistMax;     // [history] local channel + 1
inline constexpr int kGridRecConsPeerChan = kGridRecConsProdChan + kGridConsHistMax;  // [history] remote channel + 1
inline constexpr int kGridRecConsChanCloseBase = kGridRecConsPeerChan + kGridConsHistMax;       // [channels]
inline constexpr int kGridRecProdChanProdIndex = kGridRecConsChanCloseBase + kGridChanCount;    // [channels]
inline constexpr int kGridRecConsChanConsIndex = kGridRecProdChanProdIndex + kGridChanCount;    // [channels]
inline constexpr int kGridRecConsChanPeerProdChan = kGridRecConsChanConsIndex + kGridChanCount; // [channels]
inline constexpr int kGridRecProdChanCons = kGridRecConsChanPeerProdChan + kGridChanCount;      // [channels] id + 1
inline constexpr int kGridRecProdChanState = kGridRecProdChanCons + kGridChanCount;             // [channels]
inline constexpr int kGridRecCurProdChan = kGridRecProdChanState + kGridChanCount; // current local producer channel + 1
inline constexpr int kGridRecRetiredCount = kGridRecCurProdChan + 1;               // [channels]
inline constexpr int kGridRecRetiredProd = kGridRecRetiredCount + kGridChanCount;  // [channel][turn] id + 1
inline constexpr int kGridRecRetiredEnd = kGridRecRetiredProd + kGridChanCount * kGridRetiredTurnMax; // [channel][turn]
inline constexpr int kGridRecBcastFreeThreshold =
    kGridRecRetiredEnd + kGridChanCount * kGridRetiredTurnMax;                               // [producer channels]
inline constexpr int kGridRecBindRequestToken = kGridRecBcastFreeThreshold + kGridChanCount; // next dynamic token
inline constexpr int kGridRecScbSnapshot = kGridRecBindRequestToken + 1;                     // local MOV_SPR2X scratch
inline constexpr int kGridRecordWords = kGridRecScbSnapshot + 1;
static_assert(
    kGridRecordWords == 4 + 10 * kGridChanCount + 4 * kGridConsHistMax + 2 * kGridChanCount * kGridRetiredTurnMax + 3,
    "GridPipe record layout changed; update its host-visible mirrors");

// Id <-> stored-word conversion for the +1 bias described above.
AICORE inline uint32_t GridRecPackId(uint32_t id) { return id == kGridNoPeer ? 0u : id + 1u; }
AICORE inline uint32_t GridRecUnpackId(uint32_t word) { return word == 0u ? kGridNoPeer : word - 1u; }

// ---------------------------------------------------------------------------
// GridPayloadWindow -- the sub-window of a slot that one TPUSH/TPOP actually
// moves.  This is the GridPipe equivalent of a5 TPipe's `entryOffset` plus the
// shape/stride pair its TSTORE/TLOAD descriptors carry (a5 TPush.hpp:78/274/289):
// the SLOT STRIDE (Pipe::SlotStride) addresses the ring, while the fields below
// describe the transfer.  Previously both were the single constant SlotBytes, so
// every push moved a whole slot even when only a prefix was valid.
//
//   entryOffset  byte offset of the sub-window inside the slot
//   rowBytes     bytes moved per row
//   rowCount     number of rows; 0 DISABLES the window (whole slot, 1-D,
//                Pipe::SlotStride bytes at offset 0 -- the original behaviour)
//   tileStride   byte stride between rows in the local tile (0 => rowBytes)
//   slotStride   byte stride between rows inside the slot   (0 => rowBytes)
//
// The strides are named by WHICH BUFFER they walk, not by src/dst, because the
// two swap roles between the halves: a push reads the tile and writes the slot, a
// pop reads the slot and writes the tile.  (`slotStride` is the per-row stride
// INSIDE one slot; the ring's slot-to-slot stride is Pipe::SlotStride.)
//
// rowCount > 1 expresses a 2-D sub-block (e.g. the valid column prefix of a
// row-major tile).  The COPY_L1_TO_NBR machine instruction takes a single
// `bytes` operand, so the lowering emits one burst per row and ONE ready
// doorbell for the whole window -- the doorbell count per TPUSH is unchanged.
// Hardware that grows src/dst stride operands can fold the loop into one burst.
// ---------------------------------------------------------------------------
struct GridPayloadWindow {
    uint32_t entryOffset = 0;
    uint32_t rowBytes = 0;
    uint32_t rowCount = 0; // 0 => disabled: whole slot
    uint32_t tileStride = 0;
    uint32_t slotStride = 0;
};

AICORE inline uint32_t GridPayloadTileStride(const GridPayloadWindow& w)
{
    return w.tileStride != 0 ? w.tileStride : w.rowBytes;
}

AICORE inline uint32_t GridPayloadSlotStride(const GridPayloadWindow& w)
{
    return w.slotStride != 0 ? w.slotStride : w.rowBytes;
}

// Bytes spanned inside the slot, measured from the SLOT base (entryOffset
// included).  A disabled window spans the whole slot.  This is what the range
// guard compares against SlotStride.
AICORE inline uint32_t GridPayloadSlotExtent(const GridPayloadWindow& w, uint32_t slotStride)
{
    if (w.rowCount == 0) {
        return slotStride;
    }
    return w.entryOffset + (w.rowCount - 1) * GridPayloadSlotStride(w) + w.rowBytes;
}

// ---------------------------------------------------------------------------
// GridConsumerTable -- the producer-side state machine for every downstream
// consumer this core has served.  Producer and consumer channels are independent:
// `prodChan` selects this core's free_scb/prod_idx pair, while `peerConsChan`
// selects the remote consumer's ready/close/ring resources.
//
//   UNBOUND  first meeting: TPUSH must run the identity/baseline handshake
//   ACTIVE   bound and transferring: TPUSH takes the steady-state fast path
//   CLOSED   final transfer published: a later TPUSH must handshake again
//
// For an UNBOUND/CLOSED consumer, TPUSH first reserves an unused/CLOSED local
// producer channel and sends both its producer id and that local channel to the
// consumer.  The consumer independently chooses its own receive channel, relays
// ready_scb into this producer channel's prod_idx and cons_idx into this producer
// channel's free_scb, then commits the pair through an explicit L1 completion word.
// ACTIVE entries skip that handshake.
//
// ---------------------------------------------------------------------------
enum class GridConsumerState : uint32_t {
    UNBOUND = 0,
    ACTIVE = 1,
    CLOSED = 2,
};

enum class GridProducerChannelState : uint32_t {
    UNBOUND = 0,
    ACTIVE = 1,
    CLOSED = 2,
};

struct GridConsumerTable {
    uint32_t curConsId = kGridNoPeer;   // consumer this core produces for right now
    uint32_t id[kGridConsHistMax] = {}; // historical consumer ids (kGridNoPeer = empty)
    GridConsumerState state[kGridConsHistMax] = {};
    int prodChan[kGridConsHistMax] = {};
    int peerConsChan[kGridConsHistMax] = {};
    int used = 0; // occupied entries, [0, kGridConsHistMax]

    // No Reset() on purpose: this table is loaded from the window's pipe record by
    // GridPipe::LoadRecord and never cleared.  Clearing it at init is exactly the
    // bug the record exists to prevent -- both negotiated channel indices and the
    // binding table have to outlive the pipe object that created them.

    AICORE int Find(uint32_t consId) const
    {
        if (consId == kGridNoPeer) {
            return -1; // "no consumer" is not an identity and never has saved state
        }
        for (int e = 0; e < used; ++e) {
            if (id[e] == consId) {
                return e;
            }
        }
        return -1;
    }

    // Entry for `consId`, allocating one if this is its first save.  Returns -1 only
    // when the history is full, which the caller reports as a fault rather than
    // silently dropping counters (a dropped save is a lost credit, i.e. a hang or a
    // slot overwrite much later and far from the cause).
    AICORE int FindOrAlloc(uint32_t consId)
    {
        const int e = Find(consId);
        if (e >= 0) {
            return e;
        }
        if (consId == kGridNoPeer || used >= kGridConsHistMax) {
            return -1;
        }
        id[used] = consId;
        state[used] = GridConsumerState::UNBOUND;
        prodChan[used] = kGridInvalidChan;
        peerConsChan[used] = kGridInvalidChan;
        return used++;
    }

    AICORE GridConsumerState StateOf(uint32_t consId) const
    {
        const int e = Find(consId);
        return e < 0 ? GridConsumerState::UNBOUND : state[e];
    }

    AICORE int ProducerChannelOf(uint32_t consId) const
    {
        const int e = Find(consId);
        return e < 0 ? kGridInvalidChan : prodChan[e];
    }

    AICORE int PeerConsumerChannelOf(uint32_t consId) const
    {
        const int e = Find(consId);
        return e < 0 ? kGridInvalidChan : peerConsChan[e];
    }

    AICORE bool Activate(uint32_t consId, int producerChannel, int peerConsumerChannel)
    {
        const int e = FindOrAlloc(consId);
        if (e < 0) {
            return false;
        }
        state[e] = GridConsumerState::ACTIVE;
        prodChan[e] = producerChannel;
        peerConsChan[e] = peerConsumerChannel;
        curConsId = consId;
        return true;
    }

    AICORE bool Close(uint32_t consId)
    {
        const int e = Find(consId);
        if (e < 0 || state[e] != GridConsumerState::ACTIVE) {
            return false;
        }
        state[e] = GridConsumerState::CLOSED;
        return true;
    }
};

// ---------------------------------------------------------------------------
// GridPipe<TileT, SlotStride, SlotCount, ChanCount = kGridChanCount>
//
// One instance describes two independent channel pools.  Consumer channels own
// this core's receive rings, ready/close SCBs and cons_idx; producer channels own
// this core's free SCBs and prod_idx.  A handshake explicitly exchanges the two
// indices, so an edge no longer assumes they are numerically equal.
//
// The pipe carries NO PEER IDENTITY IN ITS TYPE.  A TPUSH names the consumer it is
// writing to and a TPOP names the producer it is draining, both as ordinary mesh
// ranks the call site derived from the topology, so the same pipe object serves the
// same physical resources across phases even when the cores on the other end
// change.  Every grid transfer is still exactly one hop -- there is no hop-count
// operand anywhere in the family.
//
// ChanCount trims the array for pipes that need fewer channels.  It bounds both
// the BINDING SEARCH and the window, so a
// pipe never allocates a channel it has no ring for.  The scoreboard header is a
// fixed kGridChanCount triplets regardless, so window offsets are identical for every
// pipe in a build -- required, because the peer resolver maps a local address to the
// SAME byte offset in the peer's window.
//
// TBROADCAST and group TREDUCE do not append a collective payload area.  They use
// these same ChanCount rings, assigning at most one live source to each channel.
// A collective pipe must therefore carry at least one channel and must not be
// interleaved with dynamic TPUSH bindings on the same physical window.
// ---------------------------------------------------------------------------
template <typename TileT_, int SlotStride_, int SlotCount_, int ChanCount_ = kGridChanCount>
struct GridPipe {
    static_assert(SlotCount_ > 0, "GridPipe requires SlotCount > 0");
    static_assert(SlotStride_ > 0, "GridPipe requires SlotStride > 0");
    static_assert(
        ChanCount_ > 0 && ChanCount_ <= kGridChanCount,
        "GridPipe ChanCount must be in [1, kGridChanCount] -- the window header reserves exactly kGridChanCount "
        "ready/free/close scoreboard triplets, so a pipe cannot ask for more channels than the layout supports.");

    using TileType = TileT_;
    // Ring addressing stride.  NOT the transfer length -- that comes from the
    // per-channel GridPayloadWindow below (or defaults to the whole slot).
    static constexpr int SlotStride = SlotStride_;
    // Compatibility spelling of the same constant.  Reads as "one slot is this
    // many bytes"; kept so existing call sites and window mirrors keep working.
    static constexpr int SlotBytes = SlotStride_;
    static constexpr int SlotCount = SlotCount_;
    static constexpr int ChanCount = ChanCount_;

    // Shape + coord cached from runtime (design doc 2.1 / 2.2).
    GridShape shape{};
    GridCoord coord{};
    // Sub-rectangle of the active group (GridGroup::SUBRECT only); ignored by
    // ROW/COL.  Set by the kernel after InitGridPipeFromWindow from a host/config
    // supplied rectangle so a single TBROADCAST can target any cell range.
    GridRect groupRect{};

    // Consumer-channel resources (incoming edge): slotBase, readyScb, closeScb,
    // consIndex and consChanCloseBase are indexed by the channel this core selected
    // while acting as a consumer.
    //
    // Producer-channel resources (outgoing edge): freeScb and prodIndex are indexed
    // by the independent channel this core selected while acting as a producer.
    //
    // The arrays are sized kGridChanCount, not ChanCount.  Every fixed SPR is
    // wired; only [0, ChanCount) owns a payload ring or participates in binding
    // or a structured collective.
    __gm__ uint8_t* slotBase[kGridChanCount] = {nullptr};
    __gm__ uint32_t* readyScb[kGridChanCount] = {nullptr};
    __gm__ uint32_t* freeScb[kGridChanCount] = {nullptr};
    // A producer writes its final absolute ready count here after the transfer
    // marked `isLastTransfer`.  A channel is closed when closeScb[c] advances
    // past consChanCloseBase[c], the ready count captured when that producer bound.
    // Using an absolute count instead of a Boolean avoids any local SCB reset.
    __gm__ uint32_t* closeScb[kGridChanCount] = {nullptr};
    uint32_t prodIndex[kGridChanCount] = {0};
    uint32_t consIndex[kGridChanCount] = {0};
    uint32_t consChanCloseBase[kGridChanCount] = {0};

    // Structured-collective bind lanes.  These remain channel-indexed because
    // their owners are derived from the group schedule.  Ordinary dynamic TPUSH
    // does not use these lanes; it uses the peer-indexed queues below.
    __gm__ uint32_t* bindRequestProdIdL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindRequestProdChanL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindRequestConsChanL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindRequestModeL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindResponseReadyL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindResponseConsChanL1[kGridChanCount] = {nullptr};
    __gm__ uint32_t* bindResponseCompleteL1[kGridChanCount] = {nullptr};

    // Dynamic bind queues.  Each entry occupies a full cache line.  Request
    // entry p in a consumer window has exactly one remote writer: producer p.
    // Response entry c in a producer window likewise has exactly one remote
    // writer: consumer c.  The consumer is the sole dequeuer and advances the
    // in-object scan cursor only after a request has been fully acknowledged.
    __gm__ uint32_t* bindRequestQueueBaseL1 = nullptr;
    __gm__ uint32_t* bindResponseQueueBaseL1 = nullptr;
    uint32_t bindRequestScanStart = 0;
    uint32_t bindRequestToken = 0;

    // Dedicated outbound staging slot in this core's L1 SRAM.  It is physically
    // disjoint from slotBase[], which are receive-side payload rings.  The A2/A3
    // mock represents both sides with distinct ranges in the
    // per-core GM window; native WSE maps the same layout onto unified L1 SRAM.
    __gm__ uint8_t* producerSlotBase = nullptr; // [SlotStride]

    // Opaque runtime pointer used by the A2/A3 backend to resolve cross-rank
    // addresses (HCCL device context).  Other targets may reinterpret.
    __gm__ void* runtimeCtx = nullptr;

    // Stable logical id used for runtime telemetry / per-channel scoreboard id.
    uint32_t pipeId = 0;

    // Per-channel payload sub-window (a5 TPipe's prod/cons `entryOffset` plus a
    // transfer descriptor).  All zero = disabled = move the whole slot, which is
    // what every call site did before these existed.  Set them right before the
    // TPUSH/TPOP they apply to; they persist until reset.
    GridPayloadWindow pushWindow[kGridChanCount] = {};
    GridPayloadWindow popWindow[kGridChanCount] = {};
    // Consumer-side binding table.  Each local receive channel remembers its
    // upstream producer and that producer's independently selected producer
    // channel, so TPOP returns FREE to the correct remote SCB.
    uint32_t consChanProdId[kGridChanCount] = {};
    uint32_t consChanBindCnt[kGridChanCount] = {0};
    int consChanPeerProdChan[kGridChanCount] = {};
    uint32_t consChanRetiredCount[kGridChanCount] = {0};
    uint32_t consChanRetiredProdId[kGridChanCount][kGridRetiredTurnMax] = {};
    uint32_t consChanRetiredEnd[kGridChanCount][kGridRetiredTurnMax] = {};

    // Producer-side channel table.  CLOSED is local producer state: the final
    // TPUSH for the current consumer has been published, so this producer channel
    // may be rebound without consulting the remote consumer's channel allocator.
    uint32_t prodChanConsId[kGridChanCount] = {};
    GridProducerChannelState prodChanState[kGridChanCount] = {};
    // TBROADCAST's reverse edge is F->1, so it cannot use TPUSH's one-consumer
    // absolute free baseline.  This persistent threshold counts completed
    // receiver handoffs while the bind mailbox still relays the absolute READY
    // sequence and producer identity.
    uint32_t bcastFreeThreshold[kGridChanCount] = {0};

    // The last upstream producer accepted on any local consumer channel.
    uint32_t prevProdId = kGridNoPeer;
    int curConsChan = kGridInvalidChan;
    int curProdChan = kGridInvalidChan;

    // Base of the window's pipe record.
    __gm__ uint32_t* recordBase = nullptr;

    // Current + historical downstream consumers and both channel indices negotiated
    // for each active turn.
    GridConsumerTable consumers{};
    // Sticky consumer-history overflow flag.
    bool consHistFull = false;
    AICORE __gm__ uint32_t* BindRequestQueueEntry(uint32_t prodId) const
    {
        return bindRequestQueueBaseL1 + prodId * kGridBindQueueEntryWords;
    }
    AICORE __gm__ uint32_t* BindResponseQueueEntry(uint32_t consId) const
    {
        return bindResponseQueueBaseL1 + consId * kGridBindQueueEntryWords;
    }
    AICORE uint32_t AllocateBindRequestToken()
    {
        ++bindRequestToken;
        if (bindRequestToken == kGridBindPending) {
            ++bindRequestToken;
        }
        if (recordBase != nullptr) {
            grid_cce_detail::write_local_word(recordBase + kGridRecBindRequestToken, bindRequestToken);
        }
        return bindRequestToken;
    }
    AICORE void SetPushWindow(int chan, const GridPayloadWindow& w) { pushWindow[chan] = w; }
    AICORE void SetPopWindow(int chan, const GridPayloadWindow& w) { popWindow[chan] = w; }
    AICORE void ResetPushWindow(int chan) { pushWindow[chan] = GridPayloadWindow{}; }
    AICORE void ResetPopWindow(int chan) { popWindow[chan] = GridPayloadWindow{}; }
    AICORE void SetAllPushWindows(const GridPayloadWindow& w)
    {
        for (int c = 0; c < ChanCount; ++c) {
            pushWindow[c] = w;
        }
    }
    AICORE void SetAllPopWindows(const GridPayloadWindow& w)
    {
        for (int c = 0; c < ChanCount; ++c) {
            popWindow[c] = w;
        }
    }
    AICORE void ResetAllPushWindows() { SetAllPushWindows(GridPayloadWindow{}); }
    AICORE void ResetAllPopWindows() { SetAllPopWindows(GridPayloadWindow{}); }
    AICORE uint32_t ReadChannelScb(__gm__ uint32_t* scb, uint32_t slot)
    {
        if (recordBase == nullptr) {
            return 0;
        }
        __gm__ uint32_t* snapshot = recordBase + kGridRecScbSnapshot;
        mov_ipc_scb_to_l1(snapshot, scb, slot);
        return mov_x_to_gpr(snapshot);
    }

    AICORE uint32_t ReadConsumerReadyCount(int consChan)
    {
        return ReadChannelScb(readyScb[consChan], static_cast<uint32_t>(consChan));
    }

    AICORE uint32_t ReadConsumerCloseCount(int consChan)
    {
        return ReadChannelScb(
            closeScb[consChan], 2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(consChan));
    }

    AICORE bool ConsumerChannelHasClosedProducer(int consChan)
    {
        return consChan >= 0 && consChan < ChanCount && consChanBindCnt[consChan] != 0 &&
               ReadConsumerCloseCount(consChan) > consChanCloseBase[consChan];
    }

    // CLOSE retires the old writer and is sufficient to transfer channel
    // ownership.  The ring need not be empty: the new producer receives
    // {prodIndex=oldEnd, freeScb=consIndex}, so the ordinary TPUSH free threshold
    // prevents it from overwriting any undrained slot.  Retired source/end records
    // below preserve payload identity until those older entries are consumed.
    AICORE bool ConsumerChannelIsRebindable(int consChan)
    {
        return ConsumerChannelHasClosedProducer(consChan) &&
               consChanRetiredCount[consChan] < static_cast<uint32_t>(kGridRetiredTurnMax);
    }

    // Consumer channels prefer the lowest unused index, then the lowest CLOSED
    // index.  Producer channels deliberately allocate in the opposite direction;
    // this makes accidental same-index coupling visible in normal multi-channel
    // tests instead of hiding it behind symmetric allocation.
    AICORE int PickBindableConsumerChannel()
    {
        for (int c = 0; c < ChanCount; ++c) {
            if (consChanBindCnt[c] == 0) {
                return c;
            }
        }
        for (int c = 0; c < ChanCount; ++c) {
            if (ConsumerChannelIsRebindable(c)) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    AICORE int PickBindableProducerChannel(uint32_t consId) const
    {
        const int previous = consumers.ProducerChannelOf(consId);
        if (previous >= 0 && previous < ChanCount && prodChanConsId[previous] == consId &&
            prodChanState[previous] == GridProducerChannelState::CLOSED) {
            return previous;
        }
        for (int c = ChanCount - 1; c >= 0; --c) {
            if (prodChanState[c] == GridProducerChannelState::UNBOUND) {
                return c;
            }
        }
        for (int c = ChanCount - 1; c >= 0; --c) {
            if (prodChanState[c] == GridProducerChannelState::CLOSED) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    AICORE bool ActivateConsumer(uint32_t consId, int prodChan, int peerConsChan)
    {
        if (prodChan < 0 || prodChan >= ChanCount || peerConsChan < 0 || peerConsChan >= ChanCount ||
            !consumers.Activate(consId, prodChan, peerConsChan)) {
            consHistFull = consumers.Find(consId) < 0;
            return false;
        }
        prodChanConsId[prodChan] = consId;
        prodChanState[prodChan] = GridProducerChannelState::ACTIVE;
        curProdChan = prodChan;
        StoreRecord();
        return true;
    }

    AICORE bool CloseConsumer(uint32_t consId)
    {
        const int prodChan = consumers.ProducerChannelOf(consId);
        if (prodChan < 0 || prodChan >= ChanCount || prodChanConsId[prodChan] != consId ||
            prodChanState[prodChan] != GridProducerChannelState::ACTIVE || !consumers.Close(consId)) {
            return false;
        }
        prodChanState[prodChan] = GridProducerChannelState::CLOSED;
        StoreRecord();
        return true;
    }

    AICORE void PersistProdIndex(int prodChan)
    {
        if (recordBase != nullptr) {
            grid_cce_detail::write_local_word(recordBase + kGridRecProdChanProdIndex + prodChan, prodIndex[prodChan]);
        }
    }

    AICORE void PersistConsIndex(int consChan)
    {
        if (recordBase != nullptr) {
            grid_cce_detail::write_local_word(recordBase + kGridRecConsChanConsIndex + consChan, consIndex[consChan]);
        }
    }

    AICORE uint32_t ConsumerReadyBase(int consChan)
    {
        const uint32_t ready = ReadConsumerReadyCount(consChan);
        return ready > consIndex[consChan] ? ready : consIndex[consChan];
    }

    AICORE bool EnqueueRetiredTurn(int consChan, uint32_t prodId, uint32_t endExclusive)
    {
        if (endExclusive <= consIndex[consChan]) {
            return true;
        }
        const uint32_t count = consChanRetiredCount[consChan];
        if (prodId == kGridNoPeer || count >= static_cast<uint32_t>(kGridRetiredTurnMax)) {
            consHistFull = true;
            return false;
        }
        consChanRetiredProdId[consChan][count] = prodId;
        consChanRetiredEnd[consChan][count] = endExclusive;
        consChanRetiredCount[consChan] = count + 1;
        return true;
    }

    AICORE void RetireConsumedTurns(int consChan)
    {
        bool changed = false;
        while (consChanRetiredCount[consChan] != 0 && consIndex[consChan] >= consChanRetiredEnd[consChan][0]) {
            const uint32_t count = consChanRetiredCount[consChan];
            for (uint32_t i = 1; i < count; ++i) {
                consChanRetiredProdId[consChan][i - 1] = consChanRetiredProdId[consChan][i];
                consChanRetiredEnd[consChan][i - 1] = consChanRetiredEnd[consChan][i];
            }
            consChanRetiredProdId[consChan][count - 1] = kGridNoPeer;
            consChanRetiredEnd[consChan][count - 1] = 0;
            consChanRetiredCount[consChan] = count - 1;
            changed = true;
        }
        if (changed) {
            StoreRecord();
        }
    }

    AICORE uint32_t PayloadProducerAtHead(int consChan) const
    {
        if (consChan < 0 || consChan >= ChanCount || consChanBindCnt[consChan] == 0) {
            return kGridNoPeer;
        }
        if (consChanRetiredCount[consChan] != 0) {
            return consChanRetiredProdId[consChan][0];
        }
        return consChanProdId[consChan];
    }

    AICORE int ConsumerChannelForPayloadProducer(uint32_t prodId) const
    {
        if (prodId == kGridNoPeer) {
            return kGridInvalidChan;
        }
        for (int c = 0; c < ChanCount; ++c) {
            if (PayloadProducerAtHead(c) == prodId) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    // Install a new owner after CLOSE, but before optional payload drain.  The
    // retired boundary keeps TPOP source validation intact; FREE from all later
    // drains is deliberately routed through consChanPeerProdChan to the newly
    // installed producer, which inherited the old absolute sequence.
    AICORE bool InstallIncomingBinding(
        int consChan, uint32_t prodId, int peerProdChan, uint32_t& readyBase, bool& priorAlreadyConsumed)
    {
        if (consChan < 0 || consChan >= ChanCount || peerProdChan < 0 || peerProdChan >= ChanCount) {
            return false;
        }
        priorAlreadyConsumed = true;
        if (consChanBindCnt[consChan] != 0) {
            if (!ConsumerChannelHasClosedProducer(consChan)) {
                return false;
            }
            const uint32_t retiredEnd = ReadConsumerCloseCount(consChan);
            priorAlreadyConsumed = consIndex[consChan] >= retiredEnd;
            if (!EnqueueRetiredTurn(consChan, consChanProdId[consChan], retiredEnd)) {
                return false;
            }
            readyBase = retiredEnd;
        } else {
            // A broadcast source does not bind to itself, but it advances its
            // local logical consIndex.  max(READY,consIndex) recovers that same
            // absolute baseline when the next remote owner binds this channel.
            readyBase = ConsumerReadyBase(consChan);
            priorAlreadyConsumed = consIndex[consChan] >= readyBase;
        }
        consChanProdId[consChan] = prodId;
        consChanBindCnt[consChan] += 1;
        consChanCloseBase[consChan] = readyBase;
        consChanPeerProdChan[consChan] = peerProdChan;
        curConsChan = consChan;
        prevProdId = prodId;
        StoreRecord();
        return true;
    }

    AICORE int ConsumerChannelOfProducer(uint32_t prodId) const
    {
        if (prodId == kGridNoPeer) {
            return kGridInvalidChan;
        }
        for (int c = 0; c < ChanCount; ++c) {
            if (consChanBindCnt[c] != 0 && consChanProdId[c] == prodId) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    // Adopt the window's pipe record.  Called by InitGridPipeFromWindow instead of
    // clearing the table: a zeroed window (the host memsets one at allocation) reads
    // back as "nothing bound", because bindCnt == 0 IS that, and every id is stored
    // +1 so that 0 can mean "empty" even though block id 0 is a real core.
    AICORE void LoadRecord(__gm__ uint32_t* base)
    {
        recordBase = base;
        for (int c = 0; c < kGridChanCount; ++c) {
            consChanProdId[c] = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecConsChanProd + c));
            consChanBindCnt[c] = grid_cce_detail::read_local_word(base + kGridRecConsChanBindCnt + c);
            consChanCloseBase[c] = grid_cce_detail::read_local_word(base + kGridRecConsChanCloseBase + c);
            prodIndex[c] = grid_cce_detail::read_local_word(base + kGridRecProdChanProdIndex + c);
            consIndex[c] = grid_cce_detail::read_local_word(base + kGridRecConsChanConsIndex + c);
            const uint32_t peerProdChanWord = grid_cce_detail::read_local_word(base + kGridRecConsChanPeerProdChan + c);
            consChanPeerProdChan[c] = peerProdChanWord == 0 ? kGridInvalidChan : static_cast<int>(peerProdChanWord - 1);
            prodChanConsId[c] = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecProdChanCons + c));
            const uint32_t rawProdState = grid_cce_detail::read_local_word(base + kGridRecProdChanState + c);
            prodChanState[c] = rawProdState <= static_cast<uint32_t>(GridProducerChannelState::CLOSED) ?
                                   static_cast<GridProducerChannelState>(rawProdState) :
                                   GridProducerChannelState::UNBOUND;
            consChanRetiredCount[c] = grid_cce_detail::read_local_word(base + kGridRecRetiredCount + c);
            if (consChanRetiredCount[c] > static_cast<uint32_t>(kGridRetiredTurnMax)) {
                consChanRetiredCount[c] = kGridRetiredTurnMax;
                consHistFull = true;
            }
            for (int t = 0; t < kGridRetiredTurnMax; ++t) {
                const int index = c * kGridRetiredTurnMax + t;
                consChanRetiredProdId[c][t] =
                    GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecRetiredProd + index));
                consChanRetiredEnd[c][t] = grid_cce_detail::read_local_word(base + kGridRecRetiredEnd + index);
            }
            bcastFreeThreshold[c] = grid_cce_detail::read_local_word(base + kGridRecBcastFreeThreshold + c);
        }
        const uint32_t curConsChanWord = grid_cce_detail::read_local_word(base + kGridRecCurConsChan);
        curConsChan = curConsChanWord == 0u ? kGridInvalidChan : static_cast<int>(curConsChanWord - 1u);
        const uint32_t curProdChanWord = grid_cce_detail::read_local_word(base + kGridRecCurProdChan);
        curProdChan = curProdChanWord == 0u ? kGridInvalidChan : static_cast<int>(curProdChanWord - 1u);
        prevProdId = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecPrevProd));
        bindRequestToken = grid_cce_detail::read_local_word(base + kGridRecBindRequestToken);
        consumers.curConsId = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecConsCur));
        consumers.used = static_cast<int>(grid_cce_detail::read_local_word(base + kGridRecConsUsed));
        if (consumers.used > kGridConsHistMax) {
            consumers.used = kGridConsHistMax; // corrupt record; clamp rather than run off the array
        }
        for (int e = 0; e < kGridConsHistMax; ++e) {
            consumers.id[e] = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecConsId + e));
            const uint32_t rawState = grid_cce_detail::read_local_word(base + kGridRecConsState + e);
            consumers.state[e] = rawState <= static_cast<uint32_t>(GridConsumerState::CLOSED) ?
                                     static_cast<GridConsumerState>(rawState) :
                                     GridConsumerState::UNBOUND;
            const uint32_t prodChanWord = grid_cce_detail::read_local_word(base + kGridRecConsProdChan + e);
            consumers.prodChan[e] = prodChanWord == 0 ? kGridInvalidChan : static_cast<int>(prodChanWord - 1);
            const uint32_t peerConsChanWord = grid_cce_detail::read_local_word(base + kGridRecConsPeerChan + e);
            consumers.peerConsChan[e] =
                peerConsChanWord == 0 ? kGridInvalidChan : static_cast<int>(peerConsChanWord - 1);
        }
    }

    // Write the complete record on a binding/FSM transition.  Steady-state
    // TPUSH/TPOP update only their one advancing counter word via Persist*Index.
    AICORE void StoreRecord()
    {
        if (recordBase == nullptr) {
            return;
        }
        for (int c = 0; c < kGridChanCount; ++c) {
            grid_cce_detail::write_local_word(recordBase + kGridRecConsChanProd + c, GridRecPackId(consChanProdId[c]));
            grid_cce_detail::write_local_word(recordBase + kGridRecConsChanBindCnt + c, consChanBindCnt[c]);
            grid_cce_detail::write_local_word(recordBase + kGridRecConsChanCloseBase + c, consChanCloseBase[c]);
            grid_cce_detail::write_local_word(recordBase + kGridRecProdChanProdIndex + c, prodIndex[c]);
            grid_cce_detail::write_local_word(recordBase + kGridRecConsChanConsIndex + c, consIndex[c]);
            grid_cce_detail::write_local_word(
                recordBase + kGridRecConsChanPeerProdChan + c,
                consChanPeerProdChan[c] == kGridInvalidChan ? 0u : static_cast<uint32_t>(consChanPeerProdChan[c]) + 1u);
            grid_cce_detail::write_local_word(recordBase + kGridRecProdChanCons + c, GridRecPackId(prodChanConsId[c]));
            grid_cce_detail::write_local_word(
                recordBase + kGridRecProdChanState + c, static_cast<uint32_t>(prodChanState[c]));
            grid_cce_detail::write_local_word(recordBase + kGridRecRetiredCount + c, consChanRetiredCount[c]);
            for (int t = 0; t < kGridRetiredTurnMax; ++t) {
                const int index = c * kGridRetiredTurnMax + t;
                grid_cce_detail::write_local_word(
                    recordBase + kGridRecRetiredProd + index, GridRecPackId(consChanRetiredProdId[c][t]));
                grid_cce_detail::write_local_word(recordBase + kGridRecRetiredEnd + index, consChanRetiredEnd[c][t]);
            }
            grid_cce_detail::write_local_word(recordBase + kGridRecBcastFreeThreshold + c, bcastFreeThreshold[c]);
        }
        grid_cce_detail::write_local_word(
            recordBase + kGridRecCurConsChan,
            curConsChan == kGridInvalidChan ? 0u : static_cast<uint32_t>(curConsChan) + 1u);
        grid_cce_detail::write_local_word(
            recordBase + kGridRecCurProdChan,
            curProdChan == kGridInvalidChan ? 0u : static_cast<uint32_t>(curProdChan) + 1u);
        grid_cce_detail::write_local_word(recordBase + kGridRecPrevProd, GridRecPackId(prevProdId));
        grid_cce_detail::write_local_word(recordBase + kGridRecBindRequestToken, bindRequestToken);
        grid_cce_detail::write_local_word(recordBase + kGridRecConsCur, GridRecPackId(consumers.curConsId));
        grid_cce_detail::write_local_word(recordBase + kGridRecConsUsed, static_cast<uint32_t>(consumers.used));
        for (int e = 0; e < kGridConsHistMax; ++e) {
            grid_cce_detail::write_local_word(recordBase + kGridRecConsId + e, GridRecPackId(consumers.id[e]));
            grid_cce_detail::write_local_word(
                recordBase + kGridRecConsState + e, static_cast<uint32_t>(consumers.state[e]));
            grid_cce_detail::write_local_word(
                recordBase + kGridRecConsProdChan + e,
                consumers.prodChan[e] == kGridInvalidChan ? 0u : static_cast<uint32_t>(consumers.prodChan[e]) + 1u);
            grid_cce_detail::write_local_word(
                recordBase + kGridRecConsPeerChan + e, consumers.peerConsChan[e] == kGridInvalidChan ?
                                                           0u :
                                                           static_cast<uint32_t>(consumers.peerConsChan[e]) + 1u);
        }
    }
};

// ---------------------------------------------------------------------------
// SFINAE marker: lets pto_instr.hpp's TPUSH/TPOP/TBROADCAST grid overloads
// disambiguate against the existing TPipe overloads without ambiguity.
// ---------------------------------------------------------------------------
template <typename T>
struct is_grid_pipe : std::false_type {};

template <typename TileT, int SlotStride, int SlotCount, int ChanCount>
struct is_grid_pipe<GridPipe<TileT, SlotStride, SlotCount, ChanCount>> : std::true_type {};

template <typename T>
inline constexpr bool is_grid_pipe_v = is_grid_pipe<std::remove_reference_t<T>>::value;

// ---------------------------------------------------------------------------
// Coordinate bootstrap (design doc 2.1).  Row-major mapping from launcher's
// block_idx to (row, col).  AICORE-qualified because it calls get_block_idx(),
// which is a device intrinsic and has no host implementation.
// ---------------------------------------------------------------------------
AICORE inline GridCoord GetGridCoord(GridShape shape)
{
    int blockIdx = static_cast<int>(get_block_idx());
    return GridCoord{blockIdx / shape.gridCols, blockIdx % shape.gridCols};
}

// The mesh's addressing primitive: a cell's LOGICAL BLOCK ID, row-major.  Every
// peer id in the GridPipe surface is one of these.
AICORE constexpr int BlockIdFromCoord(GridCoord coord, GridShape shape)
{
    return coord.row * shape.gridCols + coord.col;
}

// ---------------------------------------------------------------------------
// Mesh topology helpers (design doc 2.3).  These are how a CALL SITE turns "my
// upstream / downstream on this axis" into the peer id it hands to TPUSH and TPOP.
// Direction survives here and only here: it describes the mesh,
// not the pipe.  Nothing below is stored in a GridPipe, indexes any of its arrays,
// or appears in an instruction's template arguments -- a channel is bound to a
// rank, and how the caller computed that rank is its own business.
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPush(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row > 0;
        case GridDirection::EAST:
            return c.col + 1 < s.gridCols;
        case GridDirection::WEST:
            return c.col > 0;
        case GridDirection::SOUTH:
            return c.row + 1 < s.gridRows;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr bool CanPop(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + 1 < s.gridRows;
        case GridDirection::EAST:
            return c.col > 0;
        case GridDirection::WEST:
            return c.col + 1 < s.gridCols;
        case GridDirection::SOUTH:
            return c.row > 0;
        case GridDirection::SOURCE:
            return true;
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPush(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + 1};
        case GridDirection::WEST:
            return {c.row, c.col - 1};
        case GridDirection::SOUTH:
            return {c.row + 1, c.col};
        case GridDirection::SOURCE:
            return c; // Unused; static_assert blocks TPUSH<SOURCE>.
    }
    return c;
}

AICORE constexpr GridCoord NeighborForPop(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - 1};
        case GridDirection::WEST:
            return {c.row, c.col + 1};
        case GridDirection::SOUTH:
            return {c.row - 1, c.col};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

inline constexpr int kInvalidBlockId = -1;

AICORE constexpr int NeighborBlockIdForPush(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPush(dir, c, s)) {
        return kInvalidBlockId;
    }
    GridCoord n = NeighborForPush(dir, c);
    return n.row * s.gridCols + n.col;
}

AICORE constexpr int NeighborBlockIdForPop(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPop(dir, c, s)) {
        return kInvalidBlockId;
    }
    GridCoord n = NeighborForPop(dir, c);
    return n.row * s.gridCols + n.col;
}

// Peer IDENTITY (kGridNoPeer when this cell has none) of the core it would push to
// along `dir`, and of the core that would push to it along `dir`.  These two
// expressions are what every call site actually needs, so they live here rather
// than being re-derived per kernel: a boundary cell has to produce kGridNoPeer, and
// the obvious hand-rolled version -- cast NeighborBlockIdForPush's result -- turns the
// kInvalidBlockId sentinel into a huge unsigned that then matches nothing and binds
// silently.  SOURCE has no peer rank at all (it names the runtime queue, not a
// core), so it is kGridNoPeer in both halves.
AICORE constexpr uint32_t GridPeerBlockIdForPush(GridDirection dir, GridCoord c, GridShape s)
{
    return (dir != GridDirection::SOURCE && CanPush(dir, c, s)) ?
               static_cast<uint32_t>(NeighborBlockIdForPush(dir, c, s)) :
               kGridNoPeer;
}

AICORE constexpr uint32_t GridPeerBlockIdForPop(GridDirection dir, GridCoord c, GridShape s)
{
    return (dir != GridDirection::SOURCE && CanPop(dir, c, s)) ?
               static_cast<uint32_t>(NeighborBlockIdForPop(dir, c, s)) :
               kGridNoPeer;
}

// Is `peerId` a core of this mesh at all?  The boundary guard TPUSH / TPOP apply
// before touching the fabric: kGridNoPeer and anything past the last block fail it.
AICORE constexpr bool GridBlockIdValid(uint32_t peerId, GridShape s)
{
    return peerId != kGridNoPeer && peerId < static_cast<uint32_t>(s.gridRows * s.gridCols);
}

// Compile-time pin on the resolver math.  A TPUSH reaches the ADJACENT cell along
// `dir` and nothing further: there is no hop-count operand anywhere in the grid
// instruction family, so "one hop" is a structural property of the mesh model
// rather than a defaulted argument, and push/pop must stay exact mirrors of each
// other (the free-credit doorbell of a `dir` channel routes back along -dir).
AICORE constexpr bool GridNeighborSelfCheck()
{
    GridShape s{4, 4};
    GridCoord c{2, 2};
    bool ok = true;
    // Every real direction moves exactly one cell, and pop is push reversed.
    ok = ok && (NeighborBlockIdForPush(GridDirection::NORTH, c, s) == BlockIdFromCoord(GridCoord{1, 2}, s));
    ok = ok && (NeighborBlockIdForPush(GridDirection::EAST, c, s) == BlockIdFromCoord(GridCoord{2, 3}, s));
    ok = ok && (NeighborBlockIdForPush(GridDirection::WEST, c, s) == BlockIdFromCoord(GridCoord{2, 1}, s));
    ok = ok && (NeighborBlockIdForPush(GridDirection::SOUTH, c, s) == BlockIdFromCoord(GridCoord{3, 2}, s));
    ok = ok && (NeighborBlockIdForPop(GridDirection::NORTH, c, s) == BlockIdFromCoord(GridCoord{3, 2}, s));
    ok = ok && (NeighborBlockIdForPop(GridDirection::EAST, c, s) == BlockIdFromCoord(GridCoord{2, 1}, s));
    ok = ok && (NeighborBlockIdForPop(GridDirection::WEST, c, s) == BlockIdFromCoord(GridCoord{2, 3}, s));
    ok = ok && (NeighborBlockIdForPop(GridDirection::SOUTH, c, s) == BlockIdFromCoord(GridCoord{1, 2}, s));
    // Mesh edges have no peer in the outward direction, in either half.
    ok = ok && !CanPush(GridDirection::EAST, GridCoord{0, 3}, s) && !CanPush(GridDirection::NORTH, GridCoord{0, 0}, s);
    ok = ok && !CanPop(GridDirection::EAST, GridCoord{0, 0}, s) && !CanPop(GridDirection::NORTH, GridCoord{3, 0}, s);
    // TPUSH<SOURCE> is never legal; TPOP<SOURCE> always is (runtime-bound queue).
    ok = ok && !CanPush(GridDirection::SOURCE, c, s) && CanPop(GridDirection::SOURCE, c, s);
    // The peer-id helpers agree with the resolvers, and collapse every "no peer"
    // case -- mesh edge and SOURCE alike -- onto the one sentinel a binding tests.
    ok =
        ok && (GridPeerBlockIdForPush(GridDirection::EAST, c, s) == static_cast<uint32_t>(BlockIdFromCoord({2, 3}, s)));
    ok = ok && (GridPeerBlockIdForPop(GridDirection::EAST, c, s) == static_cast<uint32_t>(BlockIdFromCoord({2, 1}, s)));
    ok = ok && (GridPeerBlockIdForPush(GridDirection::EAST, GridCoord{0, 3}, s) == kGridNoPeer);
    ok = ok && (GridPeerBlockIdForPop(GridDirection::SOURCE, c, s) == kGridNoPeer);
    ok = ok && GridBlockIdValid(0, s) && GridBlockIdValid(15, s) && !GridBlockIdValid(16, s) &&
         !GridBlockIdValid(kGridNoPeer, s);
    return ok;
}
static_assert(GridNeighborSelfCheck(), "GridPipe nearest-neighbor resolver self-test failed");

} // namespace pto

// ===========================================================================
// Section 2: A2/A3 GM-mock support -- boundary-fault sentinels.
//
// The IPC_SCB / HSCB handshake mock now lives in the CCE facades themselves
// (grid_cce_intrinsic.hpp: sync_hscb / wait_ipc_scb GM branches).
// What remains here is purely the mock's out-of-mesh fault reporting: a TPUSH /
// TPOP whose direction leaves the mesh writes a sentinel GM word that the host
// launcher polls after each kernel.  Real silicon raises a hardware fault
// instead; these have no V8 machine-instruction counterpart.
// ===========================================================================

namespace pto {
namespace grid_mock {

#ifndef PTO_GRID_MOCK_WFE_MAX_SPINS
#define PTO_GRID_MOCK_WFE_MAX_SPINS 100000000U
#endif

inline constexpr uint32_t kDefaultWfeMaxSpins = PTO_GRID_MOCK_WFE_MAX_SPINS;

// ONE CACHE LINE PER INDEPENDENTLY-WRITTEN SCOREBOARD.
//
// AICORE caches are not coherent between cores and the mock's sync_hscb store
// commits through a line-granular dcci write-back, so a core that stores into
// one word of a line writes back the WHOLE line from its own (possibly stale)
// copy.  Two DIFFERENT cores storing into two words of the SAME line therefore
// lose each other's updates: the doorbell simply never appears, and the peer
// blocks forever on a threshold that was already met.
//
// So every scoreboard gets its own cache line.  Unicast scoreboards have one
// external writer; group collectives can have many writers, but their atomic DMA
// still targets only word 0 of the dedicated line.  Keeping different scoreboard
// kinds/channels on different lines prevents unrelated cache maintenance from
// writing back a stale neighbour.  Monotone counters self-heal unless it is the
// LAST update that is lost, which is why packed 4 B spacing stayed lucky for so
// long.
inline constexpr uint32_t kScbLineStride = 64;                                   // bytes; one scoreboard per line
inline constexpr uint32_t kScbLineStrideU32 = kScbLineStride / sizeof(uint32_t); // == 16 (u32 step per scoreboard)

// Fault sentinel, in u32 words from the scoreboard it belongs to.
// Every scoreboard now owns a whole line, so word 10 of that line is free real
// estate inside the reserved flag header and clear of every live word.  (The
// sentinel write is local, so it shares a line with a remotely-written word;
// harmless in practice, because a sentinel is only ever written on a run that
// has already failed.)
inline constexpr uint32_t kFaultFlagWordOffset = 10;

// The SYNC_HSCB / WAIT_SPR mocks that used to live here are now the GM-mock
// branches of the CCE facades in grid_cce_intrinsic.hpp (sync_hscb /
// wait_ipc_scb).  Only the boundary-fault sentinels remain here, because they
// are pure mock diagnostics (the host launcher polls them) with no V8
// machine-instruction counterpart.

inline AICORE void MockSetFault(__gm__ uint32_t* faultFlag, uint32_t faultCode)
{
    if (faultFlag != nullptr) {
        volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(faultFlag);
        __asm__ __volatile__("" ::: "memory");
        *ptr = faultCode;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
}

// MOCK: V6 out-of-mesh boundary fault (TPUSH/TPOP naming a peer off the mesh).
//
// V6: a TPUSH/TPOP whose target leaves the mesh raises a fault
// (raise_fault(kFaultPushOOB/kFaultPopOOB), V6 3.5.3 P0/C0).
//
// A2/A3 mock: explicit early-exit + sentinel write so the host can detect the
// out-of-bound attempt.  Real boards will raise a fault; here we trap softly
// by writing a sentinel and aborting the kernel branch.  The host launcher
// inspects a "fault sentinel" GM word after each kernel and fails the run.
inline AICORE void MockBoundaryFault(__gm__ uint32_t* faultSentinel, uint32_t faultCode)
{
    if (faultSentinel != nullptr) {
        *reinterpret_cast<volatile __gm__ uint32_t*>(faultSentinel) = faultCode;
    }
    // Best-effort halt of the current kernel branch.  Real silicon will fault
    // here; on A2/A3 we just stop emitting further GridPipe ops in this branch.
}

// Fault codes mirror SPR_BOUNDARY_MASK fields (design doc section 5.2).  The old
// per-direction spellings (0x102..0x105 / 0x202..0x204) are gone with the direction
// concept: a push/pop now names a PEER RANK, so "off the mesh" is one condition
// rather than five, and the second code in each family is the new one -- the call
// named a peer that no channel is bound to.
inline constexpr uint32_t kFaultPushOutOfMesh = 0x101;
inline constexpr uint32_t kFaultPushUnbound = 0x106;
inline constexpr uint32_t kFaultPopOutOfMesh = 0x201;
inline constexpr uint32_t kFaultPopUnbound = 0x206;
// TPOP tried to drain a slot outside this core's own SRAM segment.  The NoC
// fabric has no remote-read path, so this can only happen via a mis-wired mock;
// the GmSramArena guard in GRID_TRY_TPOP_IMPL traps it here (design: NoC is
// write-only, TPOP is local-only).
inline constexpr uint32_t kFaultPopNonLocal = 0x205;
inline constexpr uint32_t kFaultWaitReadyTimeout = 0x301;
inline constexpr uint32_t kFaultWaitFreeTimeout = 0x302;
// A GridPayloadWindow reaches past the end of its slot.  Once the transfer
// length stopped being the compile-time SlotStride, nothing statically bounds it
// any more, and a push whose window overruns writes into the PEER's window --
// silent cross-core corruption that is far harder to trace than a local overrun.
// So the range is checked at runtime and trapped here instead.  (a5 gets this for
// free: its lengths come from the tile/GlobalTensor descriptors, so the geometry
// is self-consistent by construction.)
inline constexpr uint32_t kFaultPushPayloadRange = 0x401;
inline constexpr uint32_t kFaultPopPayloadRange = 0x402;
inline constexpr uint32_t kFaultBcastPayloadRange = 0x403;

// A producer met more distinct downstream consumers than its persistent history can
// represent (kGridConsHistMax entries).  Dropping the mapping/FSM entry would make a
// later reopen skip or corrupt the dual-channel handshake, so it is reported.  The
// fix is a deeper history, not a retry.
inline constexpr uint32_t kFaultConsHistoryFull = 0x501;
// TPUSH / TPOP ran on a pipe whose ChanCount is 0, or on one that has never been
// bound.  Either way there is no channel to carry the transfer.
inline constexpr uint32_t kFaultNoChannelBound = 0x502;
// Reserved legacy code from the pre-persistent-counter implementation.  Kept so
// existing host-side fault decoders do not reinterpret 0x503; current GridPipe
// mirrors prod_idx / cons_idx and does not emit it.
inline constexpr uint32_t kFaultRelayCountersLost = 0x503;
// Time-division MPSC bind-handshake faults.  Request/response are L1 payloads.
// Consumer-channel availability comes from CLOSE plus drain progress; producer-
// channel availability comes from its local UNBOUND/ACTIVE/CLOSED state table.
inline constexpr uint32_t kFaultBindRequestTimeout = 0x504;
inline constexpr uint32_t kFaultBindResponseTimeout = 0x505;
inline constexpr uint32_t kFaultWaitBindableChannelTimeout = 0x506;
inline constexpr uint32_t kFaultBindProtocol = 0x507;
inline constexpr uint32_t kFaultBindChannelBusy = 0x508;
inline constexpr uint32_t kFaultWaitProducerChannelTimeout = 0x509;

} // namespace grid_mock
} // namespace pto

// ===========================================================================
// Section 3: GmSramArena -- GM address-segment model of per-core SRAM (mock).
//
// The neighbor-SRAM addressing / transfer that used to live here as a
// CCE-intrinsic-style API (get_neighbor_sram_addr / copy_l1_to_neighbor_l1 /
// copy_local_slot_to_ubuf / sram_pop_is_local, with neighbor_sram_addr /
// NeighborSramOperand operands and a fabricated __builtin_pto_* stub) is gone:
// payload PUSH now stages in an isolated producer L1 slot and lowers to the
// copy_l1_to_neighbor_l1 CCE
// facade (grid_cce_intrinsic.hpp) and TPOP's local drain reuses the existing
// local copy (no Grid-specific intrinsic).  The peer-window / local-slot address
// resolution is now a plain runtime helper in the demo's gridpipe_payload_inl.hpp.
//
// What remains here is the GmSramArena model the TPOP guard still needs to
// enforce the NoC "TPOP reads local SRAM only" rule against the GM-window mock.
// ===========================================================================

namespace pto {

// ---------------------------------------------------------------------------
// GmSramArena: explicit GM address-segment model of future-hardware per-core
// SRAM.
//
// Real silicon gives every core a private on-chip SRAM that the NoC fabric can
// only *write* into from a neighbor (TPUSH = cross-hop write), never *read* out
// of remotely (TPOP only drains the local core's own SRAM).  Until that
// hardware exists we model the SRAM as a contiguous GM arena cut into equal,
// per-core address segments.  Core `c` owns segment `c`:
//
//   [base + c*segBytes, base + (c+1)*segBytes)
//
// The NoC contract this encodes:
//   * a core may WRITE across segments  (TPUSH pushes into a neighbor segment),
//   * a core may only READ its own segment (TPOP pops from local SRAM).
//
// The arena is the single source of truth for "which core owns this address",
// so the mock TPOP path can reject a cross-segment read instead of silently
// servicing it through the GM-backed fake window (which physically *can* read
// any address, unlike the fabric it stands in for).
struct GmSramArena {
    uint64_t base = 0;     // segment 0 base == contiguous arena base
    uint64_t segBytes = 0; // bytes per per-core segment (== HCCL winSize in the demo)
    uint32_t numSegs = 0;  // number of cores / segments

    AICORE constexpr uint64_t SegmentBase(int seg) const { return base + static_cast<uint64_t>(seg) * segBytes; }

    // Index of the segment that owns `addr`, or -1 if `addr` is outside the arena.
    AICORE constexpr int SegmentOf(uint64_t addr) const
    {
        if (numSegs == 0 || segBytes == 0 || addr < base) {
            return -1;
        }
        uint64_t idx = (addr - base) / segBytes;
        return idx < numSegs ? static_cast<int>(idx) : -1;
    }

    // True iff [addr, addr+bytes) lies entirely within segment `seg`.  This is
    // exactly the "may core `seg` read this slot?" test used by the TPOP guard.
    AICORE constexpr bool InSegment(int seg, uint64_t addr, uint64_t bytes) const
    {
        if (seg < 0 || static_cast<uint32_t>(seg) >= numSegs) {
            return false;
        }
        uint64_t lo = SegmentBase(seg);
        uint64_t hi = lo + segBytes;
        return addr >= lo && (addr + bytes) <= hi && (addr + bytes) >= addr; // last term traps wrap-around
    }
};

// Compile-time self-test of the segment classifier.  It is built into every
// A2/A3 kernel that pulls in this header (GridTPush.hpp -> pto_instr_impl.hpp),
// so a regression in the segment math fails the build rather than silently
// mis-routing a TPOP.  It also doubles as executable documentation of the rule.
AICORE constexpr bool GmSramArenaSelfCheck()
{
    GmSramArena arena{0x1000, 0x100, 4}; // 4 cores, 0x100-byte segments, based at 0x1000
    bool ok = true;
    ok = ok && (arena.SegmentOf(0x1000) == 0);    // first byte of core 0
    ok = ok && (arena.SegmentOf(0x11FF) == 1);    // last byte of core 1
    ok = ok && (arena.SegmentOf(0x1200) == 2);    // first byte of core 2
    ok = ok && (arena.SegmentOf(0x0FFF) == -1);   // below the arena
    ok = ok && (arena.SegmentOf(0x1400) == -1);   // past the arena ([0x1000,0x1400))
    ok = ok && arena.InSegment(1, 0x1100, 0x40);  // wholly inside core 1 -> local
    ok = ok && !arena.InSegment(1, 0x11F0, 0x40); // spills past core 1 -> not local
    ok = ok && !arena.InSegment(1, 0x1200, 0x10); // core 1 reading core 2 -> not local
    return ok;
}
static_assert(GmSramArenaSelfCheck(), "GmSramArena segment classifier self-test failed");

} // namespace pto

#endif // PTO_A2A3_GRID_INTRINSIC_HPP
