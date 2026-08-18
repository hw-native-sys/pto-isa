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
// layer (copy_ubuf_to_neighbor_ubuf / sync_hscb / wait_ipc_scb -> __builtin_cce_*).
// There is deliberately NO intermediate PTO wrapper (the old sync_neighbor_scb /
// wait_local_spr / mov_local_spr / ScbOperand / neighbor_sram_addr vocabulary is
// gone, per V8 section 3.4 / section 6 point 4):
//   * Section 1: GridPipe mesh model + nearest-neighbor resolvers.
//   * Section 2: A2/A3 GM-mock support -- boundary faults + the GmSramArena
//                address-segment model that enforces the NoC "TPOP reads local
//                SRAM only" rule for the GM-window mock.
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
// IPC_SCB scoreboard handshake route).  The per-(core, direction) FIFO state
// below is read by the GridTPush.hpp / GridTPop.hpp sequence expansions, which
// call the CCE facades in grid_cce_intrinsic.hpp: cross-core notify = sync_hscb
// (SYNC_HSCB / ST_HSCB, a monotone absolute count into the neighbor's direction
// IPC_SCB); local wait = wait_ipc_scb (WAIT_SPR, read+block in one instruction;
// no MOV_SPR2X peek -- V8); payload = copy_ubuf_to_neighbor_ubuf (COPY_UBUF_TO_NBR).
// On A2/A3 there is no cross-core neighbor-IPC_SCB addressing (V8 HW-DEP-1) nor a
// UB->neighbor-UB write (V8 HW-DEP-0), so those facades run their GM mock and
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
// MESH EDGES vs DIRECTIONS.  GridDirection has five enumerators, but only FOUR
// of them are mesh edges: SOURCE is a pseudo-direction for GM/host/runtime
// injection, not a neighbor, and it carries NO scoreboard -- the runtime gates
// that queue out-of-band (which is also why GridTPop already skips the free
// store for it).  Scoreboards are therefore indexed by EDGE, not by direction:
//
//   GridEdgeIndex(NORTH/EAST/WEST/SOUTH) = 0..3      GridEdgeIndex(SOURCE) = -1
//
// The slot rings keep the 5-wide direction indexing (DirMask is defined over
// GridDirection bits), so a body that touches both uses `dirIdx` for the ring
// and `edgeIdx` for the scoreboard.  The two index spaces are named apart on
// purpose.
// ---------------------------------------------------------------------------
inline constexpr int kGridEdgeCount = kGridDirectionCount - 1; // 4 real mesh edges (SOURCE excluded)

AICORE constexpr int GridEdgeIndex(GridDirection d) { return GridDirectionIndex(d) - 1; }

// ---------------------------------------------------------------------------
// IPC_SCB slot map (the native WAIT_SPR `local_scb_id` operand).  There are
// EIGHT scoreboards and no more: ready_scb_<edge> at edgeIdx and free_scb_<edge>
// at kGridEdgeCount+edgeIdx, matching the ISA design doc's per-direction budget
// of 8.  The group collectives add NONE of their own -- TBROADCAST /
// TPOP<Group> / TREDUCE<Group> notify through this same per-edge pair,
// attributing each (producer, consumer) edge to a direction with
// GroupFlowDirection below.  A scoreboard file is a small fixed register file,
// so "one more scoreboard per feature" does not scale; reusing the directional
// pair is what keeps the group collectives implementable on real silicon.
// ---------------------------------------------------------------------------
inline constexpr int kGridScbCount = 2 * kGridEdgeCount; // 8 scoreboards per window
static_assert(kGridScbCount <= 16, "GridPipe scoreboards must fit the 16-slot IPC_SCB file");

// ---------------------------------------------------------------------------
// GroupFlowDirection -- which direction scoreboard a group collective's
// (producer -> consumer) edge notifies through.
//
// A group edge is not a mesh edge: its endpoints are two cells of the group,
// generally several hops apart.  But the scoreboards are indexed by DIRECTION,
// so every such edge must be attributed to one.  The rule is the DOMINANT AXIS
// of the coordinate delta:
//
//   |dCol| > |dRow|            -> east-west axis:   EAST if the consumer is east
//                                                   of the producer, else WEST
//   |dCol| <= |dRow|  (TIE included) -> north-south axis: SOUTH if the consumer
//                                                   is south of the producer,
//                                                   else NORTH
//
// The tie deliberately goes to north-south, so a diagonal edge has one and only
// one answer and both endpoints compute the same one (the function is evaluated
// with the same (producer, consumer) argument order on both sides).
//
// Groups are ROW or COL only, so in practice the two endpoints are ALWAYS
// co-linear: a ROW edge has dRow == 0 and resolves EAST/WEST, a COL edge has
// dCol == 0 and resolves NORTH/SOUTH.  The off-axis and tie branches are
// therefore unreachable from the collectives today; they are kept (and
// self-tested below) because the rule, not the current group set, is the
// contract -- a future non-co-linear group inherits a defined answer.
//
// The direction it returns names the direction of FLOW, matching TPUSH: a
// producer whose consumer lies east writes the consumer's ready_scb[EAST], and
// that consumer waits on its own ready_scb[EAST] -- exactly what TPUSH<EAST> /
// TPOP<EAST> do one hop apart.
//
// CONSEQUENCE FOR CALLERS: a group collective and a unicast channel that map to
// the SAME direction on the SAME pipe share a scoreboard and will corrupt each
// other's counts.  Give them separate pipes (or separate directions).
// ---------------------------------------------------------------------------
AICORE constexpr GridDirection GroupFlowDirection(GridCoord prod, GridCoord cons)
{
    const int dRow = cons.row - prod.row; // > 0: consumer lies SOUTH of producer
    const int dCol = cons.col - prod.col; // > 0: consumer lies EAST  of producer
    const int aRow = dRow < 0 ? -dRow : dRow;
    const int aCol = dCol < 0 ? -dCol : dCol;
    if (aCol > aRow) {
        return dCol > 0 ? GridDirection::EAST : GridDirection::WEST;
    }
    return dRow > 0 ? GridDirection::SOUTH : GridDirection::NORTH;
}

// Compile-time pin on the attribution rule, in the style of GridNeighborSelfCheck
// below: the tie-break and the "flow direction, not side" convention are the two
// things a reader is most likely to get backwards, so a regression in either
// fails the build.
AICORE constexpr bool GroupFlowDirectionSelfCheck()
{
    bool ok = true;
    // Pure axes: the consumer's side picks the direction.
    ok = ok && (GroupFlowDirection({2, 2}, {2, 5}) == GridDirection::EAST);  // consumer 3 cols east
    ok = ok && (GroupFlowDirection({2, 5}, {2, 2}) == GridDirection::WEST);  // consumer 3 cols west
    ok = ok && (GroupFlowDirection({2, 2}, {5, 2}) == GridDirection::SOUTH); // consumer 3 rows south
    ok = ok && (GroupFlowDirection({5, 2}, {2, 2}) == GridDirection::NORTH); // consumer 3 rows north
    // Off-axis: the LARGER delta wins, regardless of the other axis's sign.
    ok = ok && (GroupFlowDirection({2, 2}, {3, 6}) == GridDirection::EAST);  // dCol 4 > dRow 1
    ok = ok && (GroupFlowDirection({2, 2}, {6, 3}) == GridDirection::SOUTH); // dRow 4 > dCol 1
    ok = ok && (GroupFlowDirection({6, 3}, {2, 2}) == GridDirection::NORTH); // dRow -4 dominates
    // TIE -> north-south, on both diagonals and in both senses.
    ok = ok && (GroupFlowDirection({2, 2}, {4, 4}) == GridDirection::SOUTH);
    ok = ok && (GroupFlowDirection({2, 2}, {4, 0}) == GridDirection::SOUTH);
    ok = ok && (GroupFlowDirection({4, 4}, {2, 2}) == GridDirection::NORTH);
    ok = ok && (GroupFlowDirection({4, 0}, {2, 2}) == GridDirection::NORTH);
    // Both endpoints of one edge agree, because both call it (producer, consumer).
    ok = ok && (GroupFlowDirection({0, 0}, {1, 7}) == GroupFlowDirection({0, 0}, {1, 7}));
    return ok;
}
static_assert(GroupFlowDirectionSelfCheck(), "GroupFlowDirection attribution self-test failed");

// ---------------------------------------------------------------------------
// Broadcast GROUP -- the participant set of a TBROADCAST collective.
//
// GROUP replaces the old single-source GridSpan "span": the members are named
// outright instead of being a fan-in-1 span around one source.  A source writes
// the receivers' ORDINARY directional ring at prod_idx % SlotCount -- there is
// no group-private ring and no per-source slot partition, so the ring slot is
// shared by every source on that side and the caller's SPSC schedule is what
// keeps two sources out of it at once.
//
//   GridGroup::ROW = every cell on the source's row    (the row is the group)
//   GridGroup::COL = every cell on the source's column (the column is the group)
//
// SOURCES ARE SERIALIZED, NOT CONCURRENT.  The notification semaphore is the
// per-direction ready_scb/free_scb pair below -- the same objects TPUSH uses,
// with the edge's direction picked by GroupFlowDirection.  Each doorbell is an
// INCREMENT (sync_hscb_add), so the scoreboard tolerates several writers over
// TIME without them agreeing on a value; what it does not tolerate is two of
// them in flight at ONCE, sharing one ring slot.  So a group whose members all
// broadcast (an AllGather of shards) must take turns: exactly one core inside
// TBROADCAST at any instant.  The earlier 真·同时 MPSC form bought concurrency
// with a per-source lane ARRAY (one L1/GM doorbell word per rank); that is not
// a scoreboard and is gone.  The turn itself is a BATON on the idle orthogonal
// scoreboard (GroupTurnDirection below): TBROADCAST returns only once the
// back-pressure is satisfied, TBNOTIFY hands the baton to a member named by
// block id, and TBWAIT blocks until it arrives.  See GridTBroadcast.hpp.
//
// A group still decomposes into two opposite 1-D arms for topology description
// -- ROW = EAST+WEST, COL = NORTH+SOUTH (GroupArmA / GroupArmB) -- but a source
// addresses peers by their rank-in-group directly, not by arm, so a receiver
// drains a member with TPOP<GridGroup>(pipe, tile, srcBlockId) regardless of
// which arm it sits on -- the source is named by its logical block id.
// ---------------------------------------------------------------------------
enum class GridGroup : uint8_t {
    ROW = 0, // group = the source's row:    EAST arm + WEST arm
    COL = 1, // group = the source's column: NORTH arm + SOUTH arm
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

// The mesh direction whose scoreboard carries a group's PUBLISH TURN.
//
// A group spans ONE axis, so on a group pipe the OTHER axis's scoreboards are
// idle -- a ROW group touches EAST/WEST and never NORTH/SOUTH.  The turn baton
// rides one of those idle pairs, which is why serializing the sources costs no
// new scoreboard, no ring, no window and no payload: it is one increment on a
// word the group already owns and never reads.  ROW groups take NORTH, COL
// groups take EAST.
//
// The pipe's DirMask must not name that direction -- a unicast TPUSH/TPOP on it
// would share the count.  TBWAIT / TBNOTIFY assert exactly that (DirMask gates
// the ring, and TPUSH<dir> / TPOP<dir> require their direction in it, so a
// direction absent from the mask cannot ring the scoreboard either).  TBROADCAST
// does NOT assert it: the publish never touches the turn scoreboard, so a pipe
// that broadcasts without taking turns is free to use all four directions.
AICORE constexpr GridDirection GroupTurnDirection(GridGroup g)
{
    return g == GridGroup::ROW ? GridDirection::NORTH : GridDirection::EAST;
}

// ---------------------------------------------------------------------------
// Group membership helpers: they map between a member's INDEX-IN-GROUP, its grid
// coordinate, and its logical BLOCK ID (which the runtime resolves to a peer
// window).  The two integers are deliberately named apart:
//
//   BLOCK ID    -- how one core ADDRESSES another.  It is the logical block index
//                  of a cell in the launch, `row * gridCols + col`, which is what
//                  get_block_idx() returns and what the runtime's per-block window
//                  table is keyed by.  There is no multi-device rank anywhere in
//                  this family: every peer named by a grid instruction is a block
//                  of the same launch.
//   INDEX-IN-GROUP -- a member's POSITION along the group's axis, 0..groupSize-1.
//                  It orders the members (turn-taking, the two fan-out sides, the
//                  contribution arena) and is NOT an address; it is also not a
//                  ring index, because the ring is the ordinary directional one
//                  (slot = prod_idx % SlotCount).
//
// Instruction operands that name a peer take a BLOCK ID (TPOP<Group>'s source,
// TREDUCE<Group>'s sink); the helpers below are how a caller converts between the
// two when it wants to walk a group by position.
// ---------------------------------------------------------------------------
// Forward declaration: BlockIdFromCoord is defined further down (coordinate
// bootstrap section); the GroupMemberBlockId helper below needs it visible here.
AICORE constexpr int BlockIdFromCoord(GridCoord coord, GridShape shape);

// Inverse of BlockIdFromCoord: the grid coordinate a logical block id sits at.
// This is the entry point for every peer-addressing operand -- an instruction is
// handed a block id and recovers the topology from it.
AICORE constexpr GridCoord CoordFromBlockId(int blockId, GridShape shape)
{
    return GridCoord{blockId / shape.gridCols, blockId - (blockId / shape.gridCols) * shape.gridCols};
}

// Number of members in the group that `coord` belongs to.
AICORE constexpr int GridGroupSize(GridGroup g, GridShape s) { return (g == GridGroup::ROW) ? s.gridCols : s.gridRows; }

// This cell's index within its group.
// ROW groups vary along the column axis; COL groups along the row axis.
AICORE constexpr int IndexInGroup(GridGroup g, GridCoord c) { return (g == GridGroup::ROW) ? c.col : c.row; }

// Is `peer` a member of the group `self` belongs to?  Group membership is sharing
// the FIXED axis: the same row for a ROW group, the same column for a COL group.
// The group collectives check this on the block id they are handed, so a peer
// operand that names a cell outside the group is trapped instead of resolving to
// a stranger's window.
AICORE constexpr bool GroupContains(GridGroup g, GridCoord self, GridCoord peer)
{
    return (g == GridGroup::ROW) ? (peer.row == self.row) : (peer.col == self.col);
}

// Coordinate of the member whose index-in-group is `indexInGroup`, given this
// cell's coordinate: the member shares this cell's FIXED axis (its row for a ROW
// group, its column for a COL group) and varies along the other one.
AICORE constexpr GridCoord GroupMemberCoord(GridGroup g, GridCoord self, int indexInGroup)
{
    return (g == GridGroup::ROW) ? GridCoord{self.row, indexInGroup} : GridCoord{indexInGroup, self.col};
}

AICORE constexpr int GroupMemberBlockId(GridGroup g, GridCoord self, GridShape s, int indexInGroup)
{
    return BlockIdFromCoord(GroupMemberCoord(g, self, indexInGroup), s);
}

// ---------------------------------------------------------------------------
// SHARING ONE SCOREBOARD OVER TIME.
//
// A consumer has ONE scoreboard (and one ring) per direction, so a group
// collective inevitably has SEVERAL producers writing the SAME scoreboard at
// different times, and that scoreboard holds a PERSISTENT count that nothing
// clears.  The producers do NOT negotiate a value for it: they use the
// INCREMENT form of the HSCB store (sync_hscb_add, grid_cce_intrinsic.hpp) and
// each simply adds 1.  The consumer then does exactly what TPOP does -- wait
// cons_idx + 1, drain, ++ -- and neither side needs to know who else
// participates, which is what makes a single-source broadcast and an
// all-members AllGather the same code.
//
// The same increment turns the FAN-IN direction into a plain counting
// semaphore: every consumer adds 1 to the producer's free_scb when it has
// drained, and the producer waits for the total.  That is the backpressure that
// lets one ring slot serve the whole collective -- see GridTBroadcast.hpp.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Direction mask -- which per-direction unicast slot rings a pipe allocates.
//
// Before this existed, every GridPipe paid kGridDirectionCount (5) rings even
// though no kernel uses more than two directions on one pipe (a relay uses an
// opposite pair; a pure-broadcast pipe uses none).  The mask makes the ring set
// part of the pipe's type, so the window carries popcount(mask) rings instead of
// 5 and the ring index is packed (mask EAST|WEST -> EAST is ring 0, WEST ring 1).
// Default kGridDirAll reproduces the original 5-ring layout byte for byte.
// ---------------------------------------------------------------------------
// These four are deliberately NOT AICORE-qualified: the host-side window-size
// templates in grid_pipe_runtime.hpp evaluate them, and an [aicore] function
// cannot be called from a host context.  They therefore also must not call any
// AICORE helper (hence static_cast<int> rather than GridDirectionIndex).  Device
// code sticks to plain bit arithmetic on DirMask instead of calling these.
inline constexpr int GridDirBit(GridDirection d) { return 1 << static_cast<int>(d); }

inline constexpr int kGridDirNone = 0;
inline constexpr int kGridDirAll = (1 << kGridDirectionCount) - 1; // 0x1F

inline constexpr bool GridDirInMask(int dirMask, GridDirection d)
{
    return ((dirMask >> static_cast<int>(d)) & 1) != 0;
}

// Number of rings the mask allocates.
inline constexpr int GridDirRingCount(int dirMask)
{
    int n = 0;
    for (int i = 0; i < kGridDirectionCount; ++i) {
        n += (dirMask >> i) & 1;
    }
    return n;
}

// Packed index of `d`'s ring inside the slot region (-1 when not allocated).
inline constexpr int GridDirRingIndex(int dirMask, GridDirection d)
{
    if (!GridDirInMask(dirMask, d)) {
        return -1;
    }
    int idx = 0;
    for (int i = 0; i < static_cast<int>(d); ++i) {
        idx += (dirMask >> i) & 1;
    }
    return idx;
}

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
// row-major tile).  The COPY_UBUF_TO_NBR machine instruction takes a single
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
// GridPipe<TileT, SlotStride, SlotCount, DirMask = kGridDirAll>
//
// One instance describes the FIFO state of EVERY channel the current core uses;
// each (core, direction) pair has its own ring buffer with independent prod/cons
// indices, ready/free signals, and slot region, held in the per-direction arrays
// below.  Fields are populated by the runtime during InitGridPipe and are read by
// the lower half (compiler-generated intrinsic expansions).
//
// The pipe binds NO PEER IDENTITY.  A direction names an EDGE of the mesh, not a
// core: the channel a TPUSH<dir> writes into is the ADJACENT cell along `dir` and
// the channel a TPOP<dir> drains was filled by the adjacent cell along -dir, both
// derived at the call site from (dir, coord, shape).  There is no hop-count
// operand anywhere in the family -- every grid transfer is exactly one hop -- and
// no producer/consumer template argument, so the same pipe object serves the same
// direction across phases even when the core on the other end changes.
//
// There are no broadcast-specific template parameters or storage: a group
// collective uses the very same per-direction rings and scoreboards a TPUSH
// does, which is what silicon actually provides.  DirMask must therefore name
// the directions the group spans (EAST|WEST for a ROW group, NORTH|SOUTH for a
// COL group).
// ---------------------------------------------------------------------------
template <typename TileT_, int SlotStride_, int SlotCount_, int DirMask_ = kGridDirAll>
struct GridPipe {
    static_assert(SlotCount_ > 0, "GridPipe requires SlotCount > 0");
    static_assert(SlotStride_ > 0, "GridPipe requires SlotStride > 0");
    static_assert(DirMask_ >= 0 && DirMask_ <= kGridDirAll, "GridPipe DirMask must be a kGridDirBit(...) OR-mask");

    using TileType = TileT_;
    // Ring addressing stride.  NOT the transfer length -- that comes from the
    // per-direction GridPayloadWindow below (or defaults to the whole slot).
    static constexpr int SlotStride = SlotStride_;
    // Compatibility spelling of the same constant.  Reads as "one slot is this
    // many bytes"; kept so existing call sites and window mirrors keep working.
    static constexpr int SlotBytes = SlotStride_;
    static constexpr int SlotCount = SlotCount_;
    static constexpr int DirMask = DirMask_;
    // popcount(DirMask), spelled out rather than calling GridDirRingCount: this
    // initializer is evaluated while instantiating the pipe from AICORE code, and
    // that context may not call a host constexpr function.
    static_assert(kGridDirectionCount == 5, "RingCount below enumerates exactly 5 direction bits");
    static constexpr int RingCount = ((DirMask_ >> 0) & 1) + ((DirMask_ >> 1) & 1) + ((DirMask_ >> 2) & 1) +
                                     ((DirMask_ >> 3) & 1) + ((DirMask_ >> 4) & 1);

    // Shape + coord cached from runtime (design doc 2.1 / 2.2).
    GridShape shape{};
    GridCoord coord{};
    // Per-direction state.  Index by GridDirectionIndex(dir).
    //
    // readyScb / freeScb are the V6 direction scoreboards (ready_scb_<dir> /
    // free_scb_<dir>).  On native silicon each is an IPC_SCB slot carrying a
    // monotone absolute count (written by the single upstream/downstream
    // neighbor via an HSCB store, read/blocked-on locally).  In this A2/A3 mock
    // they are GM words that stand in for those IPC_SCB slots.  prodIndex /
    // consIndex are the local GPR run-counters (slot addr, threshold, and the
    // absolute count published to the peer); they never live in an IPC_SCB.
    //
    // NOTE the two index spaces: the rings are per DIRECTION (5, DirMask is
    // defined over GridDirection bits), the scoreboards per mesh EDGE (4 --
    // SOURCE is runtime-gated and has none).  Index the first group with
    // GridDirectionIndex and the second with GridEdgeIndex.
    __gm__ uint8_t* slotBase[kGridDirectionCount] = {nullptr};
    __gm__ uint32_t* readyScb[kGridEdgeCount] = {nullptr};
    __gm__ uint32_t* freeScb[kGridEdgeCount] = {nullptr};
    uint32_t prodIndex[kGridDirectionCount] = {0};
    uint32_t consIndex[kGridDirectionCount] = {0};

    // The group collectives add NO state of their own.  They ride the SAME
    // per-direction rings and scoreboards above -- which is the only thing real
    // silicon offers: a consumer has one scoreboard set and one payload ring,
    // physically split four ways by direction, and cannot afford a private
    // window per peer.  A group edge picks its direction with
    // GroupFlowDirection(); it needs no place in a shared sequence, because the
    // doorbell is an INCREMENT (sync_hscb_add) and an add assumes nothing about
    // the other writers of that persistent count.

    // Opaque runtime pointer used by the A2/A3 backend to resolve cross-rank
    // addresses (HCCL device context).  Other targets may reinterpret.
    __gm__ void* runtimeCtx = nullptr;

    // Stable logical id used for runtime telemetry / per-direction scoreboard id.
    uint32_t pipeId = 0;

    // Per-direction payload sub-window (a5 TPipe's prod/cons `entryOffset` plus a
    // transfer descriptor).  All zero = disabled = move the whole slot, which is
    // what every call site did before these existed.  Set them right before the
    // TPUSH/TPOP they apply to; they persist until reset.
    GridPayloadWindow pushWindow[kGridDirectionCount] = {};
    GridPayloadWindow popWindow[kGridDirectionCount] = {};
    // Same for the broadcast ring.  One window covers both halves of the
    // collective: a source replicates its own shard and a receiver drains another
    // source's shard, and in a group collective those are the same geometry.
    GridPayloadWindow bcastWindow{};

    AICORE void SetPushWindow(GridDirection dir, const GridPayloadWindow& w)
    {
        pushWindow[GridDirectionIndex(dir)] = w;
    }
    AICORE void SetPopWindow(GridDirection dir, const GridPayloadWindow& w) { popWindow[GridDirectionIndex(dir)] = w; }
    AICORE void ResetPushWindow(GridDirection dir) { pushWindow[GridDirectionIndex(dir)] = GridPayloadWindow{}; }
    AICORE void ResetPopWindow(GridDirection dir) { popWindow[GridDirectionIndex(dir)] = GridPayloadWindow{}; }
    AICORE void SetBcastWindow(const GridPayloadWindow& w) { bcastWindow = w; }
    AICORE void ResetBcastWindow() { bcastWindow = GridPayloadWindow{}; }
};

// ---------------------------------------------------------------------------
// SFINAE marker: lets pto_instr.hpp's TPUSH/TPOP/TBROADCAST grid overloads
// disambiguate against the existing TPipe overloads without ambiguity.
// ---------------------------------------------------------------------------
template <typename T>
struct is_grid_pipe : std::false_type {};

template <typename TileT, int SlotStride, int SlotCount, int DirMask>
struct is_grid_pipe<GridPipe<TileT, SlotStride, SlotCount, DirMask>> : std::true_type {};

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

AICORE constexpr int BlockIdFromCoord(GridCoord coord, GridShape shape)
{
    return coord.row * shape.gridCols + coord.col;
}

// ---------------------------------------------------------------------------
// Compile-time / run-time direction validity (design doc 2.3).
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
// So every word with its own external writer gets its own cache line.  This was
// first hit on the TBROADCAST per-source lanes (a doorbell per rank packed into
// one 64 B line, "wait ready timeout"); those lanes are gone -- the group
// channel now notifies through the ready_scb/free_scb pair like TPUSH -- but the
// rule outlives them and applies verbatim to every scoreboard: ready_scb[dir] is
// written by the neighbor upstream along dir and free_scb[dir] by the downstream
// one, so five directions plus the group pair put TWELVE distinct remote writers
// on what used to be a single line.  Monotone counters self-heal unless it is
// the LAST update that is lost, which is why the packed 4 B spacing stayed lucky
// for so long.
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
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
}

// MOCK: V6 out-of-mesh boundary fault (TPUSH/TPOP off the mesh edge).
//
// V6: a TPUSH/TPOP whose (dir,dist) target leaves the mesh raises a fault
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

// Fault codes mirror SPR_BOUNDARY_MASK fields (design doc section 5.2).
inline constexpr uint32_t kFaultPushNorth = 0x101;
inline constexpr uint32_t kFaultPushEast = 0x102;
inline constexpr uint32_t kFaultPushWest = 0x103;
inline constexpr uint32_t kFaultPushSouth = 0x104;
inline constexpr uint32_t kFaultPushSource = 0x105; // Always illegal.
inline constexpr uint32_t kFaultPopNorth = 0x201;
inline constexpr uint32_t kFaultPopEast = 0x202;
inline constexpr uint32_t kFaultPopWest = 0x203;
inline constexpr uint32_t kFaultPopSouth = 0x204;
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
// A group publish arrived OUT OF TURN.  The group collectives require the caller
// to keep them SPSC -- one publisher at a time on a group, in ascending
// rank-in-group order -- because several sources share one consumer scoreboard
// over time.  When that is honoured the count a consumer finds is exactly the
// one it was waiting for; anything higher means a later source published before
// an earlier one, so the sequence has desynchronised.  The consumer traps it
// here instead of silently draining the wrong tile.
inline constexpr uint32_t kFaultGroupOutOfOrder = 0x404;
// A group collective was handed a BLOCK ID that is not a member of the group --
// TREDUCE<Group>'s sink, TPOP<Group>'s source or TBNOTIFY<Group>'s successor.
// Peers are named by block id at
// runtime, so nothing statically bounds the operand; a non-member would resolve
// to an off-group coordinate and ring a stranger's window, which is cross-core
// corruption with no local symptom.  Every member checks its own copy of the
// operand, so the trap fires on the contributors too, not only on the cell that
// would have been the collector.
inline constexpr uint32_t kFaultGroupBadPeer = 0x405;

// Direction-keyed fault code lookup.  Explicit switch avoids relying on the
// numeric layout of GridDirection so renumbering the enum cannot silently
// remap fault codes.
AICORE constexpr uint32_t PushFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPushNorth;
        case GridDirection::EAST:
            return kFaultPushEast;
        case GridDirection::WEST:
            return kFaultPushWest;
        case GridDirection::SOUTH:
            return kFaultPushSouth;
        case GridDirection::SOURCE:
            return kFaultPushSource;
    }
    return kFaultPushSource;
}

AICORE constexpr uint32_t PopFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPopNorth;
        case GridDirection::EAST:
            return kFaultPopEast;
        case GridDirection::WEST:
            return kFaultPopWest;
        case GridDirection::SOUTH:
            return kFaultPopSouth;
        case GridDirection::SOURCE:
            return 0; // SOURCE pop is legal; never raises a boundary fault.
    }
    return 0;
}

} // namespace grid_mock
} // namespace pto

// ===========================================================================
// Section 3: GmSramArena -- GM address-segment model of per-core SRAM (mock).
//
// The neighbor-SRAM addressing / transfer that used to live here as a
// CCE-intrinsic-style API (get_neighbor_sram_addr / copy_ubuf_to_neighbor_ubuf /
// copy_local_slot_to_ubuf / sram_pop_is_local, with neighbor_sram_addr /
// NeighborSramOperand operands and a fabricated __builtin_pto_* stub) is gone:
// per V8, payload PUSH lowers directly to the copy_ubuf_to_neighbor_ubuf CCE
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
