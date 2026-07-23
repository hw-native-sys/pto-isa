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
//   * Section 1: GridPipe mesh model + neighbor / K-hop resolvers.
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
#include <pto/common/type.hpp>                 // for AICORE (callable from both host and aicore contexts)

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
enum class GridDirection : uint8_t
{
    SOURCE = 0, // GM/Host/Runtime injection.  Only valid for TPOP.
    NORTH = 1,  // row -> row-1
    EAST = 2,   // col -> col+1
    WEST = 3,   // col -> col-1
    SOUTH = 4,  // row -> row+1
};

inline constexpr int kGridDirectionCount = 5;

AICORE constexpr int GridDirectionIndex(GridDirection d)
{
    return static_cast<int>(d);
}

// ---------------------------------------------------------------------------
// Broadcast GROUP -- the participant set of a TBROADCAST collective.
//
// GROUP replaces the old single-source GridSpan "span" and, with it, the
// handshake model: a TBROADCAST is no longer one source multicasting to a
// fan-in-1 span (which forbade concurrent senders).  It is a 真·同时 MPSC
// channel (see Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移): every
// member of the GROUP may broadcast its own shard into each receiver's shared
// ring *concurrently*.  The prefix-offset assignment (each source owns a
// disjoint global-index interval) plus per-source ready lanes keep every
// physical edge SPSC, so K concurrent senders never clobber a shared counter.
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
enum class GridGroup : uint8_t
{
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
// Scheme-② prefix-offset helpers.  Every member contributes a statically-known
// count (1 shard for the AllGather demo), so the prefix-offset base of member k
// is just k (computed locally under SPMD -- variant a, zero atomic).  These
// helpers map between a member's rank-in-group, its grid coordinate, and the
// global index space the shared ring is addressed by (slot = gidx % SC).
// ---------------------------------------------------------------------------
// Forward declaration: RankFromCoord is defined further down (coordinate
// bootstrap section); the GroupMemberRank helper below needs it visible here.
AICORE inline int RankFromCoord(GridCoord coord, GridShape shape);

// Number of members in the group that `coord` belongs to.  The trailing
// `rect` is consulted only for SUBRECT (ROW/COL ignore it); defaulting it keeps
// every existing ROW/COL call site unchanged.
AICORE constexpr int GridGroupSize(GridGroup g, GridShape s, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? s.gridCols
         : (g == GridGroup::COL) ? s.gridRows
         : ((rect.row1 - rect.row0) * (rect.col1 - rect.col0));
}

// This cell's rank within its group = its prefix-offset base (count_k = 1).
// ROW groups vary along the column axis; COL groups along the row axis; SUBRECT
// uses a row-major rank within [row0,row1)x[col0,col1).
AICORE constexpr int RankInGroup(GridGroup g, GridCoord c, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? c.col
         : (g == GridGroup::COL) ? c.row
         : ((c.row - rect.row0) * (rect.col1 - rect.col0) + (c.col - rect.col0));
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

AICORE constexpr int GroupMemberRank(GridGroup g, GridCoord self, GridShape s, int rankInGroup, GridRect rect = {})
{
    return RankFromCoord(GroupMemberCoord(g, self, rankInGroup, rect), s);
}

// Owner (rank-in-group) of global index `gidx`.  With count_k = 1 the prefix
// partition is the identity, so owner(gidx) = gidx; general variable-count
// partitions would replace this with a prefix-sum lookup (variant a) or an
// atomic-add reservation (variant b).  Kept as a named function so the
// directed-free path reads as the design doc states it ("owner(c + SC)").
AICORE constexpr int GroupOwnerOfIndex(int gidx)
{
    return gidx;
}

// ---------------------------------------------------------------------------
// GridPipe<TileT, SlotBytes, SlotCount, BcastSlotCount = 0, GroupMax = 0>
//
// One instance describes the FIFO state for a single logical channel that the
// current core uses; each (core, direction) pair has its own ring buffer with
// independent prod/cons indices, ready/free signals, and slot region.  Fields
// are populated by the runtime during InitGridPipe and are read by the lower
// half (compiler-generated intrinsic expansions).
//
// The trailing BcastSlotCount / GroupMax template params opt the pipe into the
// scheme-② TBROADCAST region appended after the unicast slot rings.  They
// default to 0, so a plain GridPipe<TileT, SlotBytes, SlotCount> (the unicast-
// only ReduceSum / K-hop smoke pipes) carries no broadcast state and its window
// is byte-identical to the pre-TBROADCAST layout.
// ---------------------------------------------------------------------------
template <typename TileT_, int SlotBytes_, int SlotCount_, int BcastSlotCount_ = 0, int GroupMax_ = 0>
struct GridPipe {
    static_assert(SlotCount_ > 0, "GridPipe requires SlotCount > 0");
    static_assert(SlotBytes_ > 0, "GridPipe requires SlotBytes > 0");
    static_assert(BcastSlotCount_ >= 0, "GridPipe requires BcastSlotCount >= 0");
    static_assert(GroupMax_ >= 0, "GridPipe requires GroupMax >= 0");

    using TileType = TileT_;
    static constexpr int SlotBytes = SlotBytes_;
    static constexpr int SlotCount = SlotCount_;
    static constexpr int BcastSlotCount = BcastSlotCount_;
    static constexpr int GroupMax = GroupMax_;

    // Shape + coord cached from runtime (design doc 2.1 / 2.2).
    GridShape shape{};
    GridCoord coord{};
    // Sub-rectangle of the active group (GridGroup::SUBRECT only); ignored by
    // ROW/COL.  Set by the kernel after InitGridPipeFromWindow from a host/config
    // supplied rectangle so a single TBROADCAST can target any cell range.
    GridRect groupRect{};

    // Per-direction state.  Index by GridDirectionIndex(dir).
    //
    // readyScb / freeScb are the V6 direction scoreboards (ready_scb_<dir> /
    // free_scb_<dir>).  On native silicon each is an IPC_SCB slot carrying a
    // monotone absolute count (written by the single upstream/downstream
    // neighbor via an HSCB store, read/blocked-on locally).  In this A2/A3 mock
    // they are GM words that stand in for those IPC_SCB slots.  prodIndex /
    // consIndex are the local GPR run-counters (slot addr, threshold, and the
    // absolute count published to the peer); they never live in an IPC_SCB.
    __gm__ uint8_t *slotBase[kGridDirectionCount] = {nullptr};
    __gm__ uint32_t *readyScb[kGridDirectionCount] = {nullptr};
    __gm__ uint32_t *freeScb[kGridDirectionCount] = {nullptr};
    uint32_t prodIndex[kGridDirectionCount] = {0};
    uint32_t consIndex[kGridDirectionCount] = {0};

    // TBROADCAST region (scheme-② 真·同时 MPSC), populated only when GroupMax >
    // 0.  bcastRingBase is the receiver's shared payload ring (SC slots,
    // addressed by global index gidx % BcastSlotCount, into which every group
    // member writes its own disjoint shard).  bcastReadyLanes / bcastFreeLanes
    // are per-source arrays indexed by rank-in-group: variant-B ready lanes
    // (one writer each -- the source of that rank -- so every lane is SPSC and
    // K concurrent senders never clobber a shared counter), and free lanes
    // (this core, as the single consumer of its own ring, is the sole writer of
    // each, so the free direction is SPSC too -- no min-credit tree needed,
    // design doc §7.4).  On native silicon these stand in for additional
    // IPC_SCB slots; here they are GM words in this core's window.
    __gm__ uint8_t *bcastRingBase = nullptr;    // [BcastSlotCount * SlotBytes]
    __gm__ uint32_t *bcastReadyLanes = nullptr; // [GroupMax] -- per-source ready (variant B)
    __gm__ uint32_t *bcastFreeLanes = nullptr;  // [GroupMax] -- per-source free  (X sole writer)

    // Opaque runtime pointer used by the A2/A3 backend to resolve cross-rank
    // addresses (HCCL device context).  Other targets may reinterpret.
    __gm__ void *runtimeCtx = nullptr;

    // Stable logical id used for runtime telemetry / per-direction scoreboard id.
    uint32_t pipeId = 0;
};

// ---------------------------------------------------------------------------
// SFINAE marker: lets pto_instr.hpp's TPUSH/TPOP/TBROADCAST grid overloads
// disambiguate against the existing TPipe overloads without ambiguity.
// ---------------------------------------------------------------------------
template <typename T>
struct is_grid_pipe : std::false_type {};

template <typename TileT, int SlotBytes, int SlotCount, int BcastSlotCount, int GroupMax>
struct is_grid_pipe<GridPipe<TileT, SlotBytes, SlotCount, BcastSlotCount, GroupMax>> : std::true_type {};

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

AICORE inline int RankFromCoord(GridCoord coord, GridShape shape)
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

inline constexpr int kInvalidRank = -1;

AICORE constexpr int NeighborRankForPush(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPush(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPush(dir, c);
    return n.row * s.gridCols + n.col;
}

AICORE constexpr int NeighborRankForPop(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPop(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPop(dir, c);
    return n.row * s.gridCols + n.col;
}

// ---------------------------------------------------------------------------
// Multi-hop (routed K-hop unicast) generalisation of the neighbor resolvers.
//
// Scheme A: a K-hop *unicast* push keeps the receiver's per-direction slot/flag
// state at fan-in 1, so distance enters only the *target rank* (and the
// doorbell reach), never the buffer count.  A K-hop push is therefore the
// 1-hop expansion with "+1/-1" replaced by "+K/-K"; nothing in the GridPipe
// window layout changes.  The GridKHopSelfCheck() static_assert below pins
// k == 1 to the existing CanPush/NeighborRankForPush/CanPop/NeighborRankForPop
// behaviour so the default-distance (= 1) path stays byte-identical.
//
// Precondition (caller's responsibility): within one direction and phase, at
// most one (source, distance) pair targets a given receiver, i.e. fan-in <= 1.
// Concurrent multi-source receive (gather/multicast) is out of scope here and
// needs the fan-in-indexed channel layout (Scheme B).
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row - k >= 0;
        case GridDirection::EAST:
            return c.col + k < s.gridCols;
        case GridDirection::WEST:
            return c.col - k >= 0;
        case GridDirection::SOUTH:
            return c.row + k < s.gridRows;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPushK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + k};
        case GridDirection::WEST:
            return {c.row, c.col - k};
        case GridDirection::SOUTH:
            return {c.row + k, c.col};
        case GridDirection::SOURCE:
            return c; // Unused; TPUSH<SOURCE> is blocked by static_assert.
    }
    return c;
}

AICORE constexpr int RankForPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPushK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPushK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Consumer side: the producer that fed a `dir` channel sits k hops in the
// *opposite* direction (an EAST channel is fed from the WEST, etc).  Used by
// TPOP to route the free-credit doorbell back to the K-hop producer.
AICORE constexpr bool CanPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + k < s.gridRows; // upstream to the south
        case GridDirection::EAST:
            return c.col - k >= 0;         // upstream to the west
        case GridDirection::WEST:
            return c.col + k < s.gridCols; // upstream to the east
        case GridDirection::SOUTH:
            return c.row - k >= 0;         // upstream to the north
        case GridDirection::SOURCE:
            return true;                   // SOURCE pop is bound to the runtime queue, distance-free.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPopK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - k};
        case GridDirection::WEST:
            return {c.row, c.col + k};
        case GridDirection::SOUTH:
            return {c.row - k, c.col};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

AICORE constexpr int RankForPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPopK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPopK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Compile-time pin: the K-hop resolvers must collapse to the 1-hop neighbor
// resolvers at k == 1, so existing TPUSH<DIR>/TPOP<DIR> behaviour (Dist == 1)
// is preserved bit-for-bit.  A representative 2-hop case is also checked.
AICORE constexpr bool GridKHopSelfCheck()
{
    GridShape s{4, 4};
    GridCoord c{2, 2};
    bool ok = true;
    // k == 1 reproduces the 1-hop resolvers for every real direction.
    ok = ok && (CanPushK(GridDirection::NORTH, c, s, 1) == CanPush(GridDirection::NORTH, c, s));
    ok = ok && (CanPushK(GridDirection::EAST, c, s, 1) == CanPush(GridDirection::EAST, c, s));
    ok = ok && (CanPushK(GridDirection::WEST, c, s, 1) == CanPush(GridDirection::WEST, c, s));
    ok = ok && (CanPushK(GridDirection::SOUTH, c, s, 1) == CanPush(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPushK(GridDirection::NORTH, c, s, 1) == NeighborRankForPush(GridDirection::NORTH, c, s));
    ok = ok && (RankForPushK(GridDirection::EAST, c, s, 1) == NeighborRankForPush(GridDirection::EAST, c, s));
    ok = ok && (RankForPushK(GridDirection::WEST, c, s, 1) == NeighborRankForPush(GridDirection::WEST, c, s));
    ok = ok && (RankForPushK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPush(GridDirection::SOUTH, c, s));
    ok = ok && (CanPopK(GridDirection::NORTH, c, s, 1) == CanPop(GridDirection::NORTH, c, s));
    ok = ok && (CanPopK(GridDirection::EAST, c, s, 1) == CanPop(GridDirection::EAST, c, s));
    ok = ok && (CanPopK(GridDirection::WEST, c, s, 1) == CanPop(GridDirection::WEST, c, s));
    ok = ok && (CanPopK(GridDirection::SOUTH, c, s, 1) == CanPop(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPopK(GridDirection::NORTH, c, s, 1) == NeighborRankForPop(GridDirection::NORTH, c, s));
    ok = ok && (RankForPopK(GridDirection::EAST, c, s, 1) == NeighborRankForPop(GridDirection::EAST, c, s));
    ok = ok && (RankForPopK(GridDirection::WEST, c, s, 1) == NeighborRankForPop(GridDirection::WEST, c, s));
    ok = ok && (RankForPopK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPop(GridDirection::SOUTH, c, s));
    // Representative 2-hop case on a 1x4 row: col0 --EAST,2--> col2, popped at col2.
    GridShape row{1, 4};
    ok = ok && CanPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2);
    ok = ok && (RankForPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2) == 2);
    ok = ok && !CanPushK(GridDirection::EAST, GridCoord{0, 2}, row, 2); // 2+2 == 4 out of range
    ok = ok && CanPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2);
    ok = ok && (RankForPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2) == 0);
    return ok;
}
static_assert(GridKHopSelfCheck(), "GridPipe K-hop resolver self-test failed");

} // namespace pto

// ===========================================================================
// Section 2: A2/A3 GM-mock support -- boundary-fault sentinels.
//
// The IPC_SCB / HSCB handshake mock now lives in the CCE facades themselves
// (grid_cce_intrinsic.hpp: sync_hscb / wait_ipc_scb GM branches).
// What remains here is purely the mock's out-of-mesh fault reporting: a TPUSH /
// TPOP whose (dir,dist) target leaves the mesh writes a sentinel GM word that
// the host launcher polls after each kernel.  Real silicon raises a hardware
// fault instead; these have no V8 machine-instruction counterpart.
// ===========================================================================

namespace pto {
namespace grid_mock {

#ifndef PTO_GRID_MOCK_WFE_MAX_SPINS
#define PTO_GRID_MOCK_WFE_MAX_SPINS 100000000U
#endif

inline constexpr uint32_t kDefaultWfeMaxSpins = PTO_GRID_MOCK_WFE_MAX_SPINS;
inline constexpr uint32_t kFaultFlagWordOffset = 2 * kGridDirectionCount;

// TBROADCAST per-source ready/free lane stride.  Each lane occupies a FULL
// cache line (64 B) so that concurrent producers -- each writing its own lane
// word -- never land on the same cache line.  Packing all GroupMax lanes into
// one 64 B line (the old 4 B stride) let producers' write-back (line-granular)
// stores clobber each other's words (lost-update / word-tearing), which
// silently dropped doorbell writes and caused spurious "wait ready timeout".
inline constexpr uint32_t kBcastLaneStride = 64; // bytes; one lane per cache line
inline constexpr uint32_t kBcastLaneStrideU32 = kBcastLaneStride / sizeof(uint32_t); // == 16 (u32 step per lane)

// The SYNC_HSCB / WAIT_SPR mocks that used to live here are now the GM-mock
// branches of the CCE facades in grid_cce_intrinsic.hpp (sync_hscb /
// wait_ipc_scb).  Only the boundary-fault sentinels remain here, because they
// are pure mock diagnostics (the host launcher polls them) with no V8
// machine-instruction counterpart.

inline AICORE void MockSetFault(__gm__ uint32_t *faultFlag, uint32_t faultCode)
{
    if (faultFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(faultFlag);
        __asm__ __volatile__("" ::: "memory");
        *ptr = faultCode;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
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
inline AICORE void MockBoundaryFault(__gm__ uint32_t *faultSentinel, uint32_t faultCode)
{
    if (faultSentinel != nullptr) {
        *reinterpret_cast<volatile __gm__ uint32_t *>(faultSentinel) = faultCode;
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

    AICORE constexpr uint64_t SegmentBase(int seg) const
    {
        return base + static_cast<uint64_t>(seg) * segBytes;
    }

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
    GmSramArena arena{0x1000, 0x100, 4};          // 4 cores, 0x100-byte segments, based at 0x1000
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
