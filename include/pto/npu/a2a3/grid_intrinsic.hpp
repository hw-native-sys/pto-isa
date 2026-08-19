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
// layer (copy_l1_to_peer_l1 / sync_hscb / wait_ipc_scb -> __builtin_cce_*).
// There is deliberately NO intermediate PTO wrapper (the old sync_neighbor_scb /
// wait_local_spr / mov_local_spr / ScbOperand / neighbor_sram_addr vocabulary is
// gone, per V8 section 3.4 / section 6 point 4):
//   * Section 1: GridPipe mesh model -- the concurrency array, its binding table,
//                the consumer save/restore table, and the single-hop peer
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
// Section 1: GridPipe -- peer-core FIFO communication primitives.
//
// This is the proposal-level abstraction described in the V8 design spec (the
// IPC_SCB scoreboard handshake route).  The per-channel FIFO state below is read
// by the GridTPush.hpp / GridTPop.hpp sequence expansions, which call the CCE
// facades in grid_cce_intrinsic.hpp: cross-core notify = sync_hscb
// (SYNC_HSCB / ST_HSCB, a monotone absolute count into the peer's channel
// IPC_SCB); local steady-state wait = wait_ipc_scb (WAIT_SPR, read+block in one
// instruction); control-path snapshot = MOV_SPR2X (channel-close scan and relay
// baseline capture); payload = copy_l1_to_peer_l1 (COPY_L1_TO_PEER), after
// staging the tile in the pipe's isolated producer L1 slot.
// On A2/A3 there is no cross-core peer-IPC_SCB addressing (V8 HW-DEP-1) nor a
// local-L1->peer-L1 write (V8 HW-DEP-0), so those facades run their GM mock and
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
// GROUP replaces the old single-source GridSpan "span" and, with it, the
// handshake model: a TBROADCAST is no longer one source multicasting to a
// fan-in-1 span (which forbade concurrent senders).  It is a 真·同时 MPSC
// channel (see Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移): every
// member of the GROUP may broadcast its own shard into each receiver's shared
// ring *concurrently*.  The prefix-offset assignment (each source owns a
// disjoint global-index interval) keeps the PAYLOAD disjoint, and a doorbell
// channel granted for one tile at a time keeps every COUNT single-writer, so K
// concurrent senders never clobber a shared counter.
//
//   GridGroup::ROW = every cell on the source's row    (the row is the group)
//   GridGroup::COL = every cell on the source's column (the column is the group)
//
// A group still decomposes into two opposite 1-D arms for topology description
// -- ROW = EAST+WEST, COL = NORTH+SOUTH (GroupArmA / GroupArmB) -- but the send
// addresses peers by their rank-in-group directly, not by arm, so a receiver
// drains member `srcRank` with TPOP<GridGroup>(pipe, tile, srcRank) regardless of
// which arm it sits on.
//
// THE ADDRESS COMES FROM THE CALLER, NOT FROM AN IDENTITY.
//
// The ring slot a publisher writes is `basek % SlotCount`, where `basek` is a
// GLOBAL SEQUENCE NUMBER the CALLER hands to TBROADCAST.  It used to be
// `(round*K + rank) % BcastSlotCount`, which silently required the ring to be at
// least as deep as the group is wide -- the address space was sized by the number
// of WRITERS instead of by the receiver's SRAM (2026-08-13 分析, 判据 M2/M3).  A
// caller-supplied number decouples the two: the library never derives an address
// from an identity, and a caller with more publishers than slots expresses that by
// how it allocates `basek` (waves), instead of by growing every receiver's ring.
//
// `basek` is a PRODUCER-side value, identical in every receiver's window, so the
// whole fan-out is still ONE copy_l1_to_group (判据 M4).  The caller must keep it
// unique, increasing and DENSE per collective (basek = round*K + rank, or plain
// `round` for a single source) -- density is what lets every receiver derive the
// same grant order with no communication (see GridTBroadcast.hpp).
//
// THE DOORBELL IS A TICKET ON A RESERVED CHANNEL.
//
// A group needs one doorbell per publisher, but the scoreboard file is fixed at
// kGridChanCount channels.  So publishers ASK, through a group mailbox indexed by
// RANK-IN-GROUP (depth = GroupMax, not the mesh size), and the receiver hands out
// a TICKET: the right to write slot `basek % SlotCount` and then raise the
// reserved channel's ready count by ONE ATOMIC ADD.  The receiver grants at most
// kGridBcastTicketBatch at a time and closes the ticket when ready reaches
// ticketEnd = ticketBase + grants, which is the single comparison that proves ALL
// of them landed.
//
// The grant is the WRITE PERMISSION, not just the doorbell permission: the
// receiver only grants a slot whose previous tenant its own caller has drained,
// so no publisher can ever overwrite live data -- including when the previous
// tenant belonged to a DIFFERENT publisher, which is exactly what a per-publisher
// credit counter cannot see.
//
// The publisher's own free_scb (GridPipe::GroupCreditChan) remains its credit
// counter: every receiver atomic-adds 1 when it drains a tile, and the publisher
// waits for baseline + round*(K-1) before starting a new round.  Under a summed
// counter that threshold is the ONLY sound one -- each receiver can contribute at
// most `round`, so the sum reaching round*(K-1) forces every one of them to be
// there; any looser threshold lets a fast receiver mask a slow one.

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
// Scheme-② prefix-offset helpers.  Every member contributes a statically-known
// count (1 shard for the AllGather demo), so the prefix-offset base of member k
// is just k (computed locally under SPMD -- variant a, zero atomic).  These
// helpers map between a member's rank-in-group, its grid coordinate, and the
// global index space the shared ring is addressed by (slot = gidx % SC).
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
// Both walk members in the same row-major order RankInGroup ranks them in, so a
// rank-indexed structure (the TBROADCAST ready lanes) and a block-id-indexed one
// (the group arena) stay in step.
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
//
// ONE BIND PROTOCOL, TWO LIFETIMES.  The relay above is the TIME-DIVISION bind: it
// aligns a NEW writer's baseline with the count the retiring writer left behind,
// and it presumes what a handoff presumes -- that at any instant a channel has ONE
// writer.  A group BROADCAST has K writers at the same instant, and it does put
// several of them on ONE channel -- which is exactly why its doorbell is an ATOMIC
// ADD against a ticket rather than an absolute store: a ticket's members raise one
// count together, and the receiver knows they have all landed when it reaches
// ticketEnd.  No baseline can be relayed onto a count several cores are adding to
// concurrently, so the broadcast keeps a RESERVED range, [0, BcastChanCount).
//
// The group REDUCE does not: its credit counter has exactly one writer (the sink),
// so it is handed over by the same relay the unicast flows use and draws from the
// SAME pool, [UnicastChanBase, ChanCount).  What differs between the two tenants is
// only WHICH BASELINE the handover states -- see THE TWO HANDOVER RULES below.
// ---------------------------------------------------------------------------

// Channels per core.  Each channel consumes three native IPC_SCB slots
// (ready/free/close), three 64 B mock lines, and one slot ring.
//
// THE SCOREBOARD FILE IS THE BUDGET, and it does not grow.  Silicon exposes 16
// IPC_SCB slots (V8 §3.2.3), so three per channel caps the array at 5 and the
// value below leaves one triplet of headroom.  Nothing -- including a group
// collective over a wider group -- may raise it: a group whose member count
// exceeds kGridChanCount SHARES channels instead, and the members sharing one
// channel take turns under the relay-count protocol (see the GROUP CHANNEL MAP
// note below).  That trades latency for registers, which is the correct direction
// when the register file is fixed.
inline constexpr int kGridChanCount = 4;
inline constexpr int kGridNativeIpcScbSlots = 16; // IPC_SCB_1..16 -- the whole file
static_assert(
    3 * kGridChanCount <= kGridNativeIpcScbSlots,
    "GridPipe ready/free/close SCBs must fit the 16 native IPC_SCB slots");

// HOW MANY OF THOSE CHANNELS A GROUP BROADCAST RESERVES.
//
// A group collective no longer owns a payload ring of its own: it publishes into
// the ORDINARY receive ring of a reserved channel, addressed by the caller's
// global sequence number (`basek % SlotCount`) rather than by anybody's rank.  So
// the only thing that has to be carved out of the fixed channel file is the
// channel INDEX RANGE.  For a pipe that opted into a collective (GroupMax > 0):
//
//   [0, kGridBcastChanCount)      the group broadcast (channel 0's ring is its arena)
//   [UnicastChanBase, ChanCount)  unicast flows AND the group reduce's credit
//
// The BROADCAST is the only reservation, and it is one because its ready count has
// K concurrent atomic-adding writers: there is no instant at which a single writer
// could hand it over, so it cannot join a relay pool at all.  The group REDUCE has
// one writer (the sink) on each counter and therefore SHARES the unicast pool --
// the two take turns on one channel instead of each holding an index of their own,
// which is what keeps a pipe that does both inside the fixed 16-slot scoreboard
// file.  A unicast-only pipe reserves nothing and allocates from channel 0 up.
//
// THE TWO HANDOVER RULES (接力复用).  A channel carries ONE tenant at a time, and
// the tenant kind decides what the next bind states as its baselines, because the
// two kinds count in DIFFERENT streams -- a unicast free_scb carries its consumer's
// cons_idx, a reduce free_scb carries the sink's fold count.
//
//   reduce -> unicast   ZERO.  The consumer sets cons_idx = 0, resets the
//                       channel's own ready/close scoreboards to 0 (MOVX2SPR: at
//                       this instant the channel has NO external writer -- a pull
//                       reduce never rings them and the new producer has not been
//                       answered yet, which is the exclusive ownership that
//                       instruction requires), answers prod_idx = 0 and stores 0
//                       into the producer's free_scb.  The channel restarts as if
//                       it had never been used, which is the only sound thing to
//                       do with a ring nobody wrote and counters from another
//                       stream.
// WHEN A TENANT LETS GO.  A unicast flow says so with CLOSE, on the transfer the
// caller marks `isLastTransfer`.  A collective says so with TREDUCE's
// `isLastRound`, which releases both halves of its credit channel -- the member's
// after waiting for the sink to fold that last round (so no credit can land after
// the next tenant's baseline), the sink's straight away.  Neither is required: a
// caller that marks nothing still gets its channels back at the next launch, since
// a collective's whole state is per-launch anyway.  Marking is what makes the
// handover work INSIDE one launch.
//
//   unicast -> reduce   DRAIN, THEN BASELINE + ROUND.  The channel is granted only
//                       after the retiring flow CLOSED *and* was fully drained --
//                       a stricter test than a unicast handover needs, because the
//                       reduce leaves the ring behind entirely and a late FREE
//                       store from the old consumer would land in the collective's
//                       credit counter.  Nothing is cleared afterwards: the
//                       surviving cons_idx becomes the collective's baseline, the
//                       sink hands it to every member as its round origin AND
//                       stores it into that member's free_scb, and both sides
//                       count baseline + round from there.
//
// The COLLECTIVE ITSELF uses broadcast channel 0's ring for every payload (the
// ring has to be at the same window offset at every receiver, so it cannot depend
// on which channel a grant happened to name).  Reserving more than one buys
// doorbell scoreboards that consecutive tickets ROTATE THROUGH, which spreads the
// atomic-add traffic of a batched ticket over several cache lines.
#ifndef PTO_GRID_BCAST_CHAN_COUNT
#define PTO_GRID_BCAST_CHAN_COUNT 1
#endif
inline constexpr int kGridBcastChanCount = PTO_GRID_BCAST_CHAN_COUNT;
static_assert(
    kGridBcastChanCount >= 1 && kGridBcastChanCount < kGridChanCount,
    "PTO_GRID_BCAST_CHAN_COUNT must reserve at least one channel for the group broadcast and leave at least one "
    "for unicast / reduce");

// TICKET BATCH -- how many publishers one receiver may have OUTSTANDING at once.
//
// A ticket is the right to write one ring slot and then raise the receiver's
// ready count by one.  The receiver hands out at most this many at a time and
// will not hand out more until every one of them has arrived (ready == ticketEnd),
// which is what makes "all n landed" a single scoreboard comparison.
//
// It is capped by SlotCount because the n baseks a ticket may cover are the n
// CONSECUTIVE values [grantHead, grantHead + n): consecutive values have pairwise
// distinct residues mod SlotCount exactly while n <= SlotCount, which is what
// makes "no two publishers of one ticket address the same slot" structural rather
// than checked.  It is also what makes the protocol deadlock-free -- see the
// window rule in GridTBroadcast.hpp.
#ifndef PTO_GRID_BCAST_TICKET_BATCH
#define PTO_GRID_BCAST_TICKET_BATCH 1
#endif
inline constexpr uint32_t kGridBcastTicketBatch = PTO_GRID_BCAST_TICKET_BATCH;
static_assert(
    kGridBcastTicketBatch >= 1 && kGridBcastTicketBatch <= 32,
    "PTO_GRID_BCAST_TICKET_BATCH must be in [1, 32] -- the grant window is tracked in one 32-bit mask");

namespace grid_mock {
// ONE CACHE LINE PER INDEPENDENTLY-WRITTEN WORD.
//
// AICORE caches are not coherent between cores and the mock's sync_hscb store
// commits through a line-granular dcci write-back, so a core that stores into
// one word of a line writes back the WHOLE line from its own (possibly stale)
// copy.  Two DIFFERENT cores storing into two words of the SAME line therefore
// lose each other's updates: the doorbell simply never appears, and the peer
// blocks forever on a threshold that was already met.
//
// So every word with its own external writer gets its own cache line.  It was
// first hit on the group collective's doorbells (GroupMax of them packed into one
// 64 B line, "wait ready timeout"), and it applies verbatim to every channel
// scoreboard -- readyScb[c] is written by whichever core currently holds channel
// c, freeScb[c] by the consumers on that edge -- and to every slot of the two
// bind-mailbox queues, whose whole purpose is to give K concurrent peers K
// non-interfering places to write.  Monotone counters self-heal unless it is the
// LAST update that is lost, which is why the packed 4 B spacing stayed lucky for
// so long.
//
// (Declared here rather than with the rest of grid_mock in Section 2: GridPipe
// itself needs the stride to index the mailbox queues.)
inline constexpr uint32_t kScbLineStride = 64;                                   // bytes; one word per line
inline constexpr uint32_t kScbLineStrideU32 = kScbLineStride / sizeof(uint32_t); // == 16 (u32 step per line)
} // namespace grid_mock

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
inline constexpr int kGridInvalidChan = -1;

// ---------------------------------------------------------------------------
// THE BIND MAILBOX IS A QUEUE -- one line per PEER, at both ends.
//
// The mailbox used to be a single request line per consumer and a single
// response line per producer.  That is a race the moment more than one producer
// wants the same consumer: overwrite stores are only safe on a word with ONE
// external writer, and K concurrent requesters are K writers (2026-08-11 分析
// §3.2 -- the reason the group collectives had to derive their channels instead
// of negotiating them).
//
// Both mailboxes are therefore arrays indexed by the PEER'S LOGICAL BLOCK ID:
//
//   request queue  (in the CONSUMER's window)  slot p written only by producer p
//   response queue (in the PRODUCER's window)  slot c written only by consumer c
//
// One cache line per slot, because the mock's write-back is line-granular and
// every slot has a different external writer (kScbLineStride, below).  Indexing
// by block id rather than by a rotating head is what makes "no two requesters
// share a word" STRUCTURAL: there is no ticket to hand out, because A2/A3 has no
// fetch-and-add (atom_add_hscb returns nothing) and the fabric has no remote
// read, so a producer could not observe a ticket even if one existed.
//
// SERVICE ORDER.  A consumer serves AT MOST ONE request per pass and rotates
// where it starts scanning, so the queue is starvation-free round-robin rather
// than strict arrival order -- arrival order is not observable without that
// missing ticket.  A request stays pending (untouched) until the consumer can
// actually serve it, which is what lets an accept be deferred instead of
// blocking the caller's drain progress (2026-08-11 分析 §3.3 循环等待).
//
// A slot is EMPTY when its word is zero and PENDING otherwise, so no separate
// sequence number is needed: a producer has at most one request outstanding per
// consumer, the consumer CLEARS the slot (a local store to a line whose only
// external writer is quiescent) before it answers, and the producer clears its
// own response slot before it asks again.  That ordering -- clear, then answer --
// is what keeps a reply from being mistaken for the next round's reply.
// ---------------------------------------------------------------------------
#ifndef PTO_GRID_BIND_QUEUE_DEPTH
#define PTO_GRID_BIND_QUEUE_DEPTH 32
#endif
// Upper bound on the mesh: the queues are indexed by block id, so this is how
// many cores a build may address, not how many may be in flight.  Raise it with
// -DPTO_GRID_BIND_QUEUE_DEPTH; each step costs two cache lines per pipe window.
inline constexpr int kGridBindQueueDepth = PTO_GRID_BIND_QUEUE_DEPTH;

// What a requester is asking for.  The mode picks the CHANNEL POOL and the
// binding's LIFETIME, and it travels in the request word because only the
// consumer can honour it.
enum class GridBindMode : uint32_t {
    NONE = 0,
    UNICAST = 1,    // TPUSH flow: a unicast channel, held until CLOSE + drain
    GROUP_PUSH = 2, // TBROADCAST ticket request.  It does NOT travel in this
                    // mailbox: a broadcast asks through the GROUP mailbox below,
                    // which is indexed by rank-in-group instead of by block id.
                    // The tag rides in the group request word all the same, so a
                    // corrupt line is still rejected rather than acted on.
    GROUP_PULL = 3, // TREDUCE (scheme C): no receive channel at all -- the sink only
                    // records where to push this member's credit
};

// Request word: one 32-bit store, so payload and commit are the same write.
//   [31:24] mode   [23:16] producer channel + 1   [15:0] producer block id + 1
AICORE inline uint32_t GridPackBindRequest(GridBindMode mode, int prodChan, uint32_t prodId)
{
    return (static_cast<uint32_t>(mode) << 24) | ((static_cast<uint32_t>(prodChan) + 1u) << 16) |
           ((prodId + 1u) & 0xFFFFu);
}

AICORE inline GridBindMode GridBindRequestMode(uint32_t word) { return static_cast<GridBindMode>(word >> 24); }

AICORE inline int GridBindRequestChan(uint32_t word) { return static_cast<int>((word >> 16) & 0xFFu) - 1; }

AICORE inline uint32_t GridBindRequestId(uint32_t word) { return (word & 0xFFFFu) - 1u; }

// Response commit word: [31:16] consumer channel + 1 (0 = granted with no receive
// channel, i.e. GROUP_PULL), [15:0] the granted marker.  Non-zero either way, so
// the producer polls this one word; the baseline rides in the payload word the
// consumer writes (and fences) first.
inline constexpr uint32_t kGridBindGranted = 1u;

AICORE inline uint32_t GridPackBindResponse(int consChan) // consChan < 0 => no receive channel
{
    const uint32_t chanWord = (consChan >= 0) ? (static_cast<uint32_t>(consChan) + 1u) : 0u;
    return (chanWord << 16) | kGridBindGranted;
}

AICORE inline int GridBindResponseChan(uint32_t word) { return static_cast<int>(word >> 16) - 1; }

// Depth of the consumer history (see GridConsumerTable).  Deliberately larger than
// the channel count: channels are a CONCURRENCY resource, consumers come and go
// over TIME, and a schedule may rotate through more consumers than it ever has
// open at once.
inline constexpr int kGridConsHistMax = 8;

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
inline constexpr int kGridRecScbSnapshot = kGridRecCurProdChan + 1;                // local MOV_SPR2X scratch word
// Baseline of the group publisher's credit counter: the value its free_scb held
// when it published its first tile on this collective.  Thresholds are stated
// relative to it because the counter is never reset -- it only ever accumulates
// atomic adds, and a window that has served an earlier phase carries that history.
inline constexpr int kGridRecGroupCreditBase = kGridRecScbSnapshot + 1;
// Tenant kind of every channel, one array per side (GridChannelTenant).  It is in
// the RECORD rather than in the object because the handover rule a bind applies is
// selected by the PREVIOUS tenant's kind, and a phase boundary is usually a kernel
// boundary: the pipe object that ran the retiring tenant is long gone by then.
inline constexpr int kGridRecConsChanKind = kGridRecGroupCreditBase + 1;           // [kGridChanCount]
inline constexpr int kGridRecProdChanKind = kGridRecConsChanKind + kGridChanCount; // [kGridChanCount]
inline constexpr int kGridRecordWords = kGridRecProdChanKind + kGridChanCount;
static_assert(
    kGridRecordWords == 4 + 10 * kGridChanCount + 4 * kGridConsHistMax + 3,
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
// row-major tile).  The COPY_L1_TO_PEER machine instruction takes a single
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

// WHICH KIND OF TENANT a channel last carried.  A unicast flow and a group reduce
// share one channel pool and take turns on it, and the two count in different
// streams (cons_idx vs the sink's fold count), so a bind cannot choose its
// baselines without knowing what it is taking over from -- see THE TWO HANDOVER
// RULES above.  Tracked per channel on BOTH sides (the consumer's ready/close/ring
// and the producer's free/prod_idx are independent resources) and persisted in the
// pipe record, because the tenant that left is by definition no longer running to
// be asked.
enum class GridChannelTenant : uint32_t {
    NONE = 0,    // never used on this side
    UNICAST = 1, // a TPUSH/TPOP flow
    GROUP = 2,   // a group collective's credit counter (pull reduce)
};

// Receive-side state of one group member, indexed by rank-in-group.
//
//   IDLE     nothing outstanding from this member
//   GRANTED  it holds a ticket for memberBasek[r]: it may write that ring slot
//            and then raise the ticket channel's ready count once
//   ARRIVED  the whole ticket landed (ready reached ticketEnd), so the payload in
//            slot memberBasek[r] % SlotCount is readable by the caller
//
// There is no "copied" state: TPOP returns the member to IDLE and frees its ring
// slot in the same step.
inline constexpr uint32_t kGridGroupMemberIdle = 0;
inline constexpr uint32_t kGridGroupMemberGranted = 1;
inline constexpr uint32_t kGridGroupMemberArrived = 2;

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
// GridPipe<TileT, SlotStride, SlotCount, GroupMax = 0, ChanCount = kGridChanCount>
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
// ChanCount trims the array for pipes that need fewer channels: a relay that only
// ever binds two flows can pass 2, and a pipe that only runs the collective passes
// exactly kGridBcastChanCount.  It bounds the BINDING SEARCH as well as the window,
// so a pipe never allocates a channel it has no ring for.  The scoreboard header is
// a fixed kGridChanCount triplets regardless, so window offsets are identical for
// every pipe in a build -- required, because the peer resolver maps a local address
// to the SAME byte offset in the peer's window.
//
// The GroupMax template param opts the pipe into a group collective: it sizes the
// group mailbox and the per-member tables, and it reserves channels
// [0, kGridBcastChanCount) for the broadcast.  It defaults to 0, so a plain
// GridPipe<TileT, SlotBytes, SlotCount> (the unicast-only ReduceSum pipes) carries
// no group state and still allocates unicast channels from index 0.
//
// There is NO BcastSlotCount any more: a broadcast publishes into the ordinary
// receive ring of a reserved channel, so its depth IS SlotCount and its bytes are
// the ones ChanCount already pays for.
// ---------------------------------------------------------------------------
template <typename TileT_, int SlotStride_, int SlotCount_, int GroupMax_ = 0, int ChanCount_ = kGridChanCount>
struct GridPipe {
    static_assert(SlotCount_ > 0, "GridPipe requires SlotCount > 0");
    static_assert(SlotStride_ > 0, "GridPipe requires SlotStride > 0");
    static_assert(GroupMax_ >= 0, "GridPipe requires GroupMax >= 0");
    static_assert(
        ChanCount_ >= 0 && ChanCount_ <= kGridChanCount,
        "GridPipe ChanCount must be in [0, kGridChanCount] -- the window header reserves exactly kGridChanCount "
        "ready/free/close scoreboard triplets, so a pipe cannot ask for more channels than the layout supports.");
    // A pipe that BROADCASTS must own the reserved channel's ring, since that ring
    // is the collective's payload arena.  It is asserted at the TBROADCAST /
    // TPOP<GridGroup> call sites rather than here, because a pipe may opt into
    // GroupMax purely for the group REDUCE, which moves its payload through the
    // caller's own arena and needs no ring at all.
    static_assert(
        GroupMax_ == 0 || SlotCount_ <= 32,
        "a group collective tracks ring-slot occupancy in one 32-bit mask, so SlotCount must be <= 32");

    using TileType = TileT_;
    // Ring addressing stride.  NOT the transfer length -- that comes from the
    // per-channel GridPayloadWindow below (or defaults to the whole slot).
    static constexpr int SlotStride = SlotStride_;
    // Compatibility spelling of the same constant.  Reads as "one slot is this
    // many bytes"; kept so existing call sites and window mirrors keep working.
    static constexpr int SlotBytes = SlotStride_;
    static constexpr int SlotCount = SlotCount_;
    static constexpr int GroupMax = GroupMax_;
    static constexpr int ChanCount = ChanCount_;
    // ONE RESERVATION, ONE SHARED POOL, ONE CHANNEL FILE.
    //
    // Channels [0, BcastChanCount) carry the group broadcast: channel 0's RING is
    // the collective's payload arena and every reserved channel's ready_scb can
    // carry a ticket.  They are reserved because a ticket's ready count has K
    // concurrent atomic-adding writers -- there is no instant at which one writer
    // could hand it over, so it cannot take part in a relay.
    //
    // Channels [UnicastChanBase, ChanCount) are the SHARED POOL: unicast flows
    // (negotiated through the block-id bind mailbox) and the group reduce's credit
    // counters take turns on it under the two handover rules at the top of this
    // file.  A pipe with no collective reserves nothing and allocates from channel
    // 0 up, which is why BcastChanCount is gated on GroupMax.
    static constexpr int BcastChanCount = (GroupMax_ > 0) ? kGridBcastChanCount : 0;
    static constexpr int UnicastChanBase = BcastChanCount;
    // The one producer channel a group publisher owns: its free_scb is the credit
    // counter EVERY receiver atomic-adds into (one counter per publisher is what
    // keeps the collective inside the 16-slot scoreboard file).  One collective
    // per pipe, so a fixed index is enough and every peer learns it from the
    // request word anyway.
    static constexpr int GroupCreditChan = 0;
    // How far up the channel file the group REDUCE may allocate.  Normally exactly
    // the unicast pool -- sharing it is the whole point -- but a reduce needs a
    // SCOREBOARD, not a ring, so a pipe that declares no unicast rings at all
    // (ChanCount <= UnicastChanBase, e.g. the pull-reduce demos with ChanCount = 0)
    // still gets one channel: the scoreboard header is a fixed kGridChanCount
    // triplets whatever ChanCount is.  Unicast keeps allocating inside ChanCount,
    // since it does need the ring.
    static constexpr int GroupChanLimit = (ChanCount_ > UnicastChanBase) ? ChanCount_ : (UnicastChanBase + 1);
    static_assert(
        GroupChanLimit <= kGridChanCount,
        "the group reduce's credit channel must exist in the fixed scoreboard header");
    // How many publishers one receiver may have in flight -- and, because the
    // grant window is exactly this wide, THE CONCURRENCY DEGREE OF THE COLLECTIVE.
    // n = 1 serialises the publishers strictly by basek; n = SlotCount lets a whole
    // ring's worth of them publish at once.  Clamped to SlotCount rather than
    // rejected, because the two are configured in different places (a build-wide
    // macro and a per-pipe ring depth) and the ring is the physical limit: n
    // consecutive baseks address n distinct slots only while n <= SlotCount.
    static constexpr uint32_t BcastTicketBatch = (kGridBcastTicketBatch < static_cast<uint32_t>(SlotCount_)) ?
                                                     kGridBcastTicketBatch :
                                                     static_cast<uint32_t>(SlotCount_);
    // Per-member group bookkeeping is sized by the group, and by 1 for a pipe
    // that has no collective (a zero-length array is not portable).
    static constexpr int GroupSlots = (GroupMax_ > 0) ? GroupMax_ : 1;

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
    // The arrays are sized kGridChanCount, not ChanCount, so that a ChanCount == 0
    // broadcast pipe does not declare zero-length arrays; only [0, ChanCount) is
    // ever wired to a ring or considered by a binding.
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

    // Bind mailbox QUEUES (L1 words, not SCBs).  `bindRequestQueue` is this
    // core's inbox as a CONSUMER -- slot p holds the request producer p wrote;
    // `bindResponseQueue` is its inbox as a PRODUCER -- slot c holds consumer c's
    // reply, [baseline, commit].  Both are indexed by the peer's logical block id
    // and one cache line apart, so concurrent peers never share a word.
    __gm__ uint32_t* bindRequestQueue = nullptr;
    __gm__ uint32_t* bindResponseQueue = nullptr;
    // Where the round-robin request scan starts next.  Local, and deliberately
    // not persisted: it only decides fairness, never correctness.
    int bindScanCursor = 0;

    // Scheme-C group-reduce epoch (2026-08-12 门铃归属分析 方案C).  A member
    // publishes "my round-r contribution is in place" by storing r+1 HERE, with an
    // ordinary LOCAL store -- zero cross-core action -- and the SINK PULLS it,
    // exactly as it already pulls the contributions themselves.
    __gm__ uint32_t* groupEpochWord = nullptr;

    // THE GROUP MAILBOX -- the same queue idea, indexed by RANK-IN-GROUP.
    //
    // The unicast mailbox above is indexed by the peer's LOGICAL BLOCK ID, so it
    // is as deep as the mesh may be wide (kGridBindQueueDepth lines).  A group
    // collective knows its member set exactly, so its mailbox is GroupMax lines --
    // the only O(K) structure the collective owns, and O(K) in the GROUP rather
    // than in the mesh.
    //
    //   request  slot r (in the RECEIVER's window)   [ basek | mode|prodChan|prodId ]
    //   response slot r (in the PUBLISHER's window)  [ granted | broadcast channel ]
    //
    // A request is TWO words because it carries the caller's sequence number as
    // well as the identity, so the payload word is written and fenced before the
    // commit word -- the same order the unicast response uses.  One cache line per
    // slot, for the reason every other queue has one: line-granular write-back and
    // a different external writer per slot.
    __gm__ uint32_t* groupRequestQueue = nullptr;
    __gm__ uint32_t* groupResponseQueue = nullptr;

    AICORE __gm__ uint32_t* BindRequestSlot(uint32_t peerBlockId) const
    {
        return bindRequestQueue + peerBlockId * grid_mock::kScbLineStrideU32;
    }

    // [0] = baseline the consumer relayed, [1] = commit word (polled).
    AICORE __gm__ uint32_t* BindResponseSlot(uint32_t peerBlockId) const
    {
        return bindResponseQueue + peerBlockId * grid_mock::kScbLineStrideU32;
    }

    // [0] = basek (payload), [1] = commit word (polled by the receiver).
    AICORE __gm__ uint32_t* GroupRequestSlot(int rankInGroup) const
    {
        return groupRequestQueue + static_cast<uint32_t>(rankInGroup) * grid_mock::kScbLineStrideU32;
    }

    // [0] = commit word: the granted broadcast channel, or pending.
    AICORE __gm__ uint32_t* GroupResponseSlot(int rankInGroup) const
    {
        return groupResponseQueue + static_cast<uint32_t>(rankInGroup) * grid_mock::kScbLineStrideU32;
    }

    // The group collective's payload arena is slotBase[GroupCreditChan] -- an
    // ordinary receive ring, addressed by the caller's sequence number
    // (basek % SlotCount).  There is deliberately no separate pointer and no
    // separate region: sharing the unicast ring is what made the collective's
    // address space a property of the RECEIVER's SRAM instead of of the number of
    // writers.

    // Dedicated outbound staging slot in this core's L1 SRAM.  It is physically
    // disjoint from slotBase[], which are receive-side payload rings.  The A2/A3 mock represents both sides with
    // distinct ranges in the per-core GM window; native WSE maps the same layout onto unified L1 SRAM.
    __gm__ uint8_t* producerSlotBase = nullptr; // [SlotStride]

    // Opaque runtime pointer used by the A2/A3 backend to resolve cross-rank
    // addresses (HCCL device context).  Other targets may reinterpret.
    __gm__ void* runtimeCtx = nullptr;

    // Stable logical id used for runtime telemetry / per-channel scoreboard id.
    uint32_t pipeId = 0;

    // Spin bound for every wait this pipe performs.  0 = block forever, which is
    // what hardware WAIT_SPR does and therefore the default; a non-zero value turns
    // a stuck handshake into a fault sentinel naming the wait instead of a hang the
    // runtime kills with a bare rc.
    //
    // It lives on the PIPE, not in the instruction signature, because it is a
    // property of how this core is being run (a debug/simulation bound), not of the
    // operation being expressed.  That is what lets a kernel call the plain PTO
    // instructions -- TPUSH / TPOP / TBROADCAST / TREDUCE -- and still get bounded
    // waits, instead of reaching past them into the GRID_TRY_* backend.
    uint32_t maxSpins = 0;

    // Per-channel payload sub-window (a5 TPipe's prod/cons `entryOffset` plus a
    // transfer descriptor).  All zero = disabled = move the whole slot, which is
    // what every call site did before these existed.  Set them right before the
    // TPUSH/TPOP they apply to; they persist until reset.
    GridPayloadWindow pushWindow[kGridChanCount] = {};
    GridPayloadWindow popWindow[kGridChanCount] = {};
    // Same for the broadcast ring.  One window covers both halves of the
    // collective: a source replicates its own shard and a receiver drains another
    // source's shard, and in a group collective those are the same geometry.
    GridPayloadWindow bcastWindow{};

    // Consumer-side binding table.  Each local receive channel remembers its
    // upstream producer and that producer's independently selected producer
    // channel, so TPOP returns FREE to the correct remote SCB.
    uint32_t consChanProdId[kGridChanCount] = {};
    uint32_t consChanBindCnt[kGridChanCount] = {0};
    int consChanPeerProdChan[kGridChanCount] = {};
    // What each side of a channel last carried (see GridChannelTenant).  Adopted
    // from the pipe record, so a bind in a LATER kernel launch still knows which
    // handover rule it owes the retiring tenant.
    GridChannelTenant consChanKind[kGridChanCount] = {};
    GridChannelTenant prodChanKind[kGridChanCount] = {};

    // ---- group-collective bookkeeping, indexed by RANK-IN-GROUP ----
    //
    // Deliberately LOCAL (not in the pipe record): a collective's live grant and
    // member tables never survive a kernel launch.  A group collective must
    // therefore run to completion inside one launch.  TREDUCE may release these
    // tables earlier, on an `isLastRound`, so another tenant can use the channel
    // later in that same launch; a unicast binding still spans launches through
    // the record.
    //
    // ONE ENTRY PER MEMBER IS ENOUGH, and that is a consequence of the credit
    // rule rather than an assumption: a publisher may not start round r+1 until
    // every receiver has drained all r of its earlier tiles, so a member never has
    // more than ONE undrained tile here.
    //
    //   memberBasek[r]     the sequence number granted to member r (its ring slot
    //                      is basek % SlotCount, and that is the ONLY thing that
    //                      decides where its payload sits)
    //   memberPeerChan[r]  member r's credit channel, learned from its request
    //   memberState[r]     GridGroupMemberState below
    uint32_t memberBasek[GroupSlots] = {0};
    int memberPeerChan[GroupSlots] = {};
    uint32_t memberState[GroupSlots] = {0};

    // ---- the ticket, one open at a time per pipe ----
    //
    //   grantHead     smallest basek never granted here.  With a dense caller
    //                 sequence every receiver derives the same value with no
    //                 communication, which is what makes the grant order global.
    //   grantedMask   bit i = basek (grantHead + i) has been granted -- the
    //                 window is [grantHead, grantHead + kGridBcastTicketBatch).
    //   slotBusyMask  bit s = ring slot s still holds a tile the caller has not
    //                 drained.  A grant is refused while its slot is busy, which
    //                 is what makes the grant a WRITE permission.
    //   ticketChan    the broadcast channel carrying the open ticket
    //   ticketBase    that channel's ready count when the ticket opened
    //   ticketEnd     ticketBase + grants issued; ready == ticketEnd means every
    //                 one of them has landed (ticketEnd == ticketBase = no ticket)
    uint32_t grantHead = 0;
    uint32_t grantedMask = 0;
    uint32_t slotBusyMask = 0;
    int ticketChan = 0;
    uint32_t ticketBase = 0;
    uint32_t ticketEnd = 0;
    // Baseline of this publisher's credit counter (see kGridRecGroupCreditBase),
    // and whether it has been captured yet on this collective.
    uint32_t groupCreditBase = 0;
    // Sink this core has a GROUP_PULL binding with (scheme-C reduce member half).
    // Local for the same reason as the tables above: it is re-established per
    // launch, and the sink's own view of it is per launch too.
    uint32_t groupSinkId = kGridNoPeer;
    // The two ends of the group reduce's credit counter, NEGOTIATED out of the
    // shared pool instead of being a fixed index (they used to be one reserved
    // GroupPullChan).  `groupCredChan` is the producer channel a MEMBER credits
    // through, `groupPullChan` the consumer channel a SINK counts its folds on --
    // one per collective, since every member is credited out of the same fold
    // count.  Both are per-launch.  TREDUCE's `isLastRound` releases them at an
    // explicit mid-launch completion point; if the caller omits it, the next
    // launch's ResetGroupState remains the backstop that returns them to the pool.
    int groupCredChan = kGridInvalidChan;
    int groupPullChan = kGridInvalidChan;

    // Producer-side channel table.  CLOSED is local producer state: the final
    // TPUSH for the current consumer has been published, so this producer channel
    // may be rebound without consulting the remote consumer's channel allocator.
    uint32_t prodChanConsId[kGridChanCount] = {};
    GridProducerChannelState prodChanState[kGridChanCount] = {};

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
    AICORE void SetBcastWindow(const GridPayloadWindow& w) { bcastWindow = w; }
    AICORE void ResetBcastWindow() { bcastWindow = GridPayloadWindow{}; }

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
        return consChan >= 0 && consChan < kGridChanCount && consChanBindCnt[consChan] != 0 &&
               ReadConsumerCloseCount(consChan) > consChanCloseBase[consChan];
    }

    // When a UNICAST channel may be handed to the next requester: as soon as the
    // current producer has published CLOSE (which, being ordered behind its final
    // READY, proves no more items will arrive).  THAT IS THE WHOLE CONDITION.
    //
    // No drain is required, of the retiring producer or of anybody before it, and
    // that is the point of relay counting.  The new producer is handed the channel's
    // surviving ready count as its prod_idx and this consumer's cons_idx as its
    // free_scb baseline, so its very first slot-reuse test -- free >= prod_idx + 1 -
    // SlotCount -- is stated in the SAME absolute stream as any tail still sitting in
    // the ring.  It can no more overwrite an undrained item than the retiring
    // producer could have.
    //
    // Nor does the READER need anything remembered about the retiring producer.  A
    // channel is ONE CONTINUOUS STREAM across the handover -- one ring, one absolute
    // count, writers in sequence -- so a TPOP after the handover does exactly what a
    // TPOP always does: read the local slot at cons_idx and store the new cons_idx
    // into the free_scb of whoever holds the channel NOW.  Which core wrote those
    // bytes never enters that arithmetic.
    //
    // The one thing that follows for the CALLER: once it has accepted a new producer
    // on a channel, it drains the leftovers under the NEW producer's name, because a
    // producer id only ever selects a channel and the retiring one no longer owns
    // any.  Naming the retired producer instead does not corrupt anything -- it waits
    // for a binding that will not come, and times out.
    //
    // A broadcast channel is never allocated here.  It is RESERVED by index
    // ([0, BcastChanCount)) and its grants are tickets, not bindings.
    //
    // A GROUP tenant answers a different question, because a pull reduce publishes
    // no CLOSE.  `groupPullChan` IS the answer: it names the collective this core is
    // running right now, and it is cleared both by an explicit end-of-collective
    // (TREDUCE's isLastRound -> ReleaseGroupPullChannel) and by the next launch
    // (ResetGroupState).  So a GROUP-kind channel is live exactly while it is that
    // one, and free the instant the collective says it is done -- mid-launch
    // included.
    AICORE bool ConsumerChannelIsReusable(int consChan)
    {
        if (consChan < 0 || consChan >= kGridChanCount) {
            return false;
        }
        if (consChanKind[consChan] == GridChannelTenant::GROUP) {
            return consChan != groupPullChan;
        }
        return consChanBindCnt[consChan] != 0 && ConsumerChannelHasClosedProducer(consChan);
    }

    // Has the flow that last held this receive channel been drained to the end?
    // Stricter than ConsumerChannelIsReusable (which only wants CLOSE) and required
    // by the unicast -> reduce handover: the reduce abandons the ring, so anything
    // still in it would never be read, and the old producer would still be owed a
    // FREE store that would land in the collective's credit counter instead.
    AICORE bool ConsumerChannelIsDrained(int consChan)
    {
        if (consChan < 0 || consChan >= kGridChanCount || consChanBindCnt[consChan] == 0) {
            return true; // nothing was ever pushed here
        }
        const uint32_t closeCount = ReadConsumerCloseCount(consChan);
        return closeCount > consChanCloseBase[consChan] && consIndex[consChan] >= closeCount;
    }

    // Producer-side twin: has this core's own retiring flow been drained by its
    // consumer?  free_scb reaching prod_idx is exactly that statement, and it is
    // what keeps a late FREE store out of a credit counter the collective is about
    // to start using.  A non-unicast tenant has no ring behind it, so nothing to
    // drain.
    AICORE bool ProducerChannelIsDrained(int prodChan)
    {
        if (prodChan < 0 || prodChan >= kGridChanCount || prodChanKind[prodChan] != GridChannelTenant::UNICAST) {
            return true;
        }
        return ReadChannelScb(
                   freeScb[prodChan], static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(prodChan)) >=
               prodIndex[prodChan];
    }

    // Consumer channels prefer the lowest unused index, then the lowest released
    // index.  Producer channels deliberately allocate in the opposite direction;
    // this makes accidental same-index coupling visible in normal multi-channel
    // tests instead of hiding it behind symmetric allocation.  Either way the
    // search starts past the reserved broadcast range.
    AICORE int PickBindableConsumerChannel()
    {
        const int lo = UnicastChanBase;
        const int hi = ChanCount;
        for (int c = lo; c < hi; ++c) {
            if (consChanBindCnt[c] == 0 && c != groupPullChan) {
                return c;
            }
        }
        for (int c = lo; c < hi; ++c) {
            if (ConsumerChannelIsReusable(c)) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    AICORE int PickBindableProducerChannel(uint32_t consId) const
    {
        const int previous = consumers.ProducerChannelOf(consId);
        if (previous >= UnicastChanBase && previous < ChanCount && prodChanConsId[previous] == consId &&
            prodChanState[previous] == GridProducerChannelState::CLOSED) {
            return previous;
        }
        for (int c = ChanCount - 1; c >= UnicastChanBase; --c) {
            if (prodChanState[c] == GridProducerChannelState::UNBOUND) {
                return c;
            }
        }
        for (int c = ChanCount - 1; c >= UnicastChanBase; --c) {
            if (prodChanState[c] == GridProducerChannelState::CLOSED) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    // ---- the group reduce's two ends of the SHARED pool ----
    //
    // Same pool, same preference order (consumer channels low-to-high, producer
    // channels high-to-low), and one extra condition in both: a UNICAST
    // predecessor must be CLOSED *and* DRAINED, not merely closed.  That is the
    // "先 drain 之前单播的 payload" half of the unicast -> reduce handover; a
    // unicast successor needs no such wait, because relay counting keeps the
    // leftovers drainable, while the reduce leaves the ring behind for good.
    //
    // Both are idempotent: once this launch's collective owns a channel, asking
    // again returns it.

    // Sink half (consumer role: the fold counter every member is credited from).
    AICORE int PickGroupPullChannel()
    {
        if (groupPullChan != kGridInvalidChan) {
            return groupPullChan;
        }
        for (int c = UnicastChanBase; c < GroupChanLimit; ++c) {
            if (consChanKind[c] == GridChannelTenant::NONE && consChanBindCnt[c] == 0) {
                return c;
            }
        }
        for (int c = UnicastChanBase; c < GroupChanLimit; ++c) {
            if (consChanKind[c] == GridChannelTenant::GROUP || ConsumerChannelIsDrained(c)) {
                return c;
            }
        }
        return kGridInvalidChan;
    }

    // Member half (producer role: the free_scb the sink credits this core through).
    AICORE int PickGroupCreditChannel()
    {
        if (groupCredChan != kGridInvalidChan) {
            return groupCredChan;
        }
        for (int c = GroupChanLimit - 1; c >= UnicastChanBase; --c) {
            if (prodChanState[c] == GridProducerChannelState::UNBOUND) {
                return c;
            }
        }
        for (int c = GroupChanLimit - 1; c >= UnicastChanBase; --c) {
            if (prodChanState[c] == GridProducerChannelState::CLOSED && ProducerChannelIsDrained(c)) {
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
        prodChanKind[prodChan] = GridChannelTenant::UNICAST;
        curProdChan = prodChan;
        StoreRecord();
        return true;
    }

    // A group reduce takes the same two resources a flow does, one on each side, so
    // it announces its tenancy the same way -- which is what keeps the allocator
    // from handing a live collective's channel to a flow, and what tells the NEXT
    // tenant which handover rule it owes.
    //
    // Sink half.  No producer id is recorded: this channel serves the whole group,
    // and a per-producer routing entry would let a stale unicast request be matched
    // against it (ConsumerChannelOfProducer).
    AICORE void ClaimGroupPullChannel(int consChan)
    {
        groupPullChan = consChan;
        consChanKind[consChan] = GridChannelTenant::GROUP;
        consChanProdId[consChan] = kGridNoPeer;
        consChanPeerProdChan[consChan] = kGridInvalidChan;
        consChanBindCnt[consChan] += 1;
        curConsChan = consChan;
        StoreRecord();
    }

    // Member half.  ACTIVE for as long as the collective runs, so no flow can be
    // handed this producer channel underneath it; the release below retires it.
    AICORE void ClaimGroupCreditChannel(int prodChan, uint32_t sinkId)
    {
        groupCredChan = prodChan;
        prodChanKind[prodChan] = GridChannelTenant::GROUP;
        prodChanConsId[prodChan] = sinkId;
        prodChanState[prodChan] = GridProducerChannelState::ACTIVE;
        curProdChan = prodChan;
        StoreRecord();
    }

    // ---- END OF TENANCY ----
    //
    // A collective has no CLOSE doorbell: there is no stream to close, and no core
    // could observe one if there were.  So what ends the tenancy is the CALLER,
    // saying so on its final round (TREDUCE's `isLastRound`), and these two are
    // where that lands.  A caller that never says it still gets the old behaviour
    // for free -- ResetGroupState retires whatever is left at the next launch --
    // which is why the flag defaults to false and no existing call site changes.
    //
    // THE KIND STAYS `GROUP` on purpose.  The next tenant has to know it is taking
    // over from a collective, so that it applies the zero rule instead of relaying
    // a baseline out of a fold stream that means nothing to it.
    //
    // The member half is the one with a precondition, and TREDUCE enforces it
    // before calling this: the sink must have folded this member's LAST round, so
    // that it can never store into that free_scb again.  A credit landing after the
    // next tenant's baseline would hand a producer credit it has not earned.
    AICORE void ReleaseGroupCreditChannel()
    {
        if (groupCredChan == kGridInvalidChan) {
            return;
        }
        prodChanState[groupCredChan] = GridProducerChannelState::CLOSED;
        groupCredChan = kGridInvalidChan;
        groupSinkId = kGridNoPeer;
        StoreRecord();
    }

    // Sink half.  Nothing to wait for here: the members never write this core's
    // channel resources at all (a pull reduce has no ready doorbell), and the
    // credits this sink owes were pushed inside the fold that just returned.  The
    // per-member table is cleared so a SECOND collective on this pipe re-binds
    // from scratch rather than trusting a channel index its predecessor negotiated.
    AICORE void ReleaseGroupPullChannel()
    {
        if (groupPullChan == kGridInvalidChan) {
            return;
        }
        groupPullChan = kGridInvalidChan;
        for (int r = 0; r < GroupSlots; ++r) {
            memberPeerChan[r] = kGridInvalidChan;
            memberState[r] = kGridGroupMemberIdle;
        }
        StoreRecord();
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

    // Group state is per-launch by design (see the arrays above), so it is built
    // fresh rather than adopted from the window.
    //
    // THIS IS ALSO THE BACKSTOP FOR A COLLECTIVE'S CHANNEL TENANCY.  A pull reduce
    // publishes no CLOSE, so the tenancy ends when the CALLER says the collective is
    // over (TREDUCE's isLastRound, which releases both halves mid-launch) -- and, for
    // a caller that never says it, here: the collective's whole state is per-launch,
    // so a channel a PREVIOUS launch's collective held cannot still be in use.  Either
    // way the KIND is left at GROUP, so that whoever takes the channel next applies
    // the reduce -> unicast zero rule rather than relaying a baseline out of the
    // wrong stream.
    AICORE void ResetGroupState()
    {
        for (int r = 0; r < GroupSlots; ++r) {
            memberBasek[r] = 0;
            memberPeerChan[r] = kGridInvalidChan;
            memberState[r] = kGridGroupMemberIdle;
        }
        grantHead = 0;
        grantedMask = 0;
        slotBusyMask = 0;
        ticketChan = 0;
        ticketBase = 0;
        ticketEnd = 0;
        groupSinkId = kGridNoPeer;
        groupCredChan = kGridInvalidChan;
        groupPullChan = kGridInvalidChan;
        bindScanCursor = 0;
        bool retired = false;
        for (int c = 0; c < kGridChanCount; ++c) {
            if (prodChanKind[c] == GridChannelTenant::GROUP && prodChanState[c] == GridProducerChannelState::ACTIVE) {
                prodChanState[c] = GridProducerChannelState::CLOSED;
                retired = true;
            }
        }
        if (retired) {
            StoreRecord();
        }
    }

    // Which channel prodId currently OWNS -- the only producer->channel mapping the
    // pipe keeps.  A producer that still owns a live channel here must not be handed
    // a second one, and after a handover the retiring producer owns nothing: its
    // leftovers are drained under the new owner's name (see ConsumerChannelIsReusable).
    AICORE int ConsumerChannelOfProducer(uint32_t prodId) const
    {
        if (prodId == kGridNoPeer) {
            return kGridInvalidChan;
        }
        for (int c = UnicastChanBase; c < ChanCount; ++c) {
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
            const uint32_t rawConsKind = grid_cce_detail::read_local_word(base + kGridRecConsChanKind + c);
            consChanKind[c] = rawConsKind <= static_cast<uint32_t>(GridChannelTenant::GROUP) ?
                                  static_cast<GridChannelTenant>(rawConsKind) :
                                  GridChannelTenant::NONE;
            const uint32_t rawProdKind = grid_cce_detail::read_local_word(base + kGridRecProdChanKind + c);
            prodChanKind[c] = rawProdKind <= static_cast<uint32_t>(GridChannelTenant::GROUP) ?
                                  static_cast<GridChannelTenant>(rawProdKind) :
                                  GridChannelTenant::NONE;
        }
        groupCreditBase = grid_cce_detail::read_local_word(base + kGridRecGroupCreditBase);
        const uint32_t curConsChanWord = grid_cce_detail::read_local_word(base + kGridRecCurConsChan);
        curConsChan = curConsChanWord == 0u ? kGridInvalidChan : static_cast<int>(curConsChanWord - 1u);
        const uint32_t curProdChanWord = grid_cce_detail::read_local_word(base + kGridRecCurProdChan);
        curProdChan = curProdChanWord == 0u ? kGridInvalidChan : static_cast<int>(curProdChanWord - 1u);
        prevProdId = GridRecUnpackId(grid_cce_detail::read_local_word(base + kGridRecPrevProd));
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
            grid_cce_detail::write_local_word(
                recordBase + kGridRecConsChanKind + c, static_cast<uint32_t>(consChanKind[c]));
            grid_cce_detail::write_local_word(
                recordBase + kGridRecProdChanKind + c, static_cast<uint32_t>(prodChanKind[c]));
        }
        grid_cce_detail::write_local_word(
            recordBase + kGridRecCurConsChan,
            curConsChan == kGridInvalidChan ? 0u : static_cast<uint32_t>(curConsChan) + 1u);
        grid_cce_detail::write_local_word(
            recordBase + kGridRecCurProdChan,
            curProdChan == kGridInvalidChan ? 0u : static_cast<uint32_t>(curProdChan) + 1u);
        grid_cce_detail::write_local_word(recordBase + kGridRecGroupCreditBase, groupCreditBase);
        grid_cce_detail::write_local_word(recordBase + kGridRecPrevProd, GridRecPackId(prevProdId));
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

template <typename TileT, int SlotStride, int SlotCount, int GroupMax, int ChanCount>
struct is_grid_pipe<GridPipe<TileT, SlotStride, SlotCount, GroupMax, ChanCount>> : std::true_type {};

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

AICORE constexpr GridCoord PeerCoordForPush(GridDirection dir, GridCoord c)
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

AICORE constexpr GridCoord PeerCoordForPop(GridDirection dir, GridCoord c)
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

// PeerBlockIdFor{Push,Pop} vs GridPeerBlockIdFor{Push,Pop}: same geometry, two
// sentinels.  These raw forms are SIGNED and say "off the mesh" with
// kInvalidBlockId; the Grid* forms below are what call sites use, and they say it
// with kGridNoPeer, which is what TPUSH / TPOP actually test.  Mixing them up is
// the bug the Grid* comment warns about, so they are deliberately not overloads.
AICORE constexpr int PeerBlockIdForPush(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPush(dir, c, s)) {
        return kInvalidBlockId;
    }
    GridCoord n = PeerCoordForPush(dir, c);
    return n.row * s.gridCols + n.col;
}

AICORE constexpr int PeerBlockIdForPop(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPop(dir, c, s)) {
        return kInvalidBlockId;
    }
    GridCoord n = PeerCoordForPop(dir, c);
    return n.row * s.gridCols + n.col;
}

// Peer IDENTITY (kGridNoPeer when this cell has none) of the core it would push to
// along `dir`, and of the core that would push to it along `dir`.  These two
// expressions are what every call site actually needs, so they live here rather
// than being re-derived per kernel: a boundary cell has to produce kGridNoPeer, and
// the obvious hand-rolled version -- cast PeerBlockIdForPush's result -- turns the
// kInvalidBlockId sentinel into a huge unsigned that then matches nothing and binds
// silently.  SOURCE has no peer rank at all (it names the runtime queue, not a
// core), so it is kGridNoPeer in both halves.
AICORE constexpr uint32_t GridPeerBlockIdForPush(GridDirection dir, GridCoord c, GridShape s)
{
    return (dir != GridDirection::SOURCE && CanPush(dir, c, s)) ? static_cast<uint32_t>(PeerBlockIdForPush(dir, c, s)) :
                                                                  kGridNoPeer;
}

AICORE constexpr uint32_t GridPeerBlockIdForPop(GridDirection dir, GridCoord c, GridShape s)
{
    return (dir != GridDirection::SOURCE && CanPop(dir, c, s)) ? static_cast<uint32_t>(PeerBlockIdForPop(dir, c, s)) :
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
AICORE constexpr bool GridPeerSelfCheck()
{
    GridShape s{4, 4};
    GridCoord c{2, 2};
    bool ok = true;
    // Every real direction moves exactly one cell, and pop is push reversed.
    ok = ok && (PeerBlockIdForPush(GridDirection::NORTH, c, s) == BlockIdFromCoord(GridCoord{1, 2}, s));
    ok = ok && (PeerBlockIdForPush(GridDirection::EAST, c, s) == BlockIdFromCoord(GridCoord{2, 3}, s));
    ok = ok && (PeerBlockIdForPush(GridDirection::WEST, c, s) == BlockIdFromCoord(GridCoord{2, 1}, s));
    ok = ok && (PeerBlockIdForPush(GridDirection::SOUTH, c, s) == BlockIdFromCoord(GridCoord{3, 2}, s));
    ok = ok && (PeerBlockIdForPop(GridDirection::NORTH, c, s) == BlockIdFromCoord(GridCoord{3, 2}, s));
    ok = ok && (PeerBlockIdForPop(GridDirection::EAST, c, s) == BlockIdFromCoord(GridCoord{2, 1}, s));
    ok = ok && (PeerBlockIdForPop(GridDirection::WEST, c, s) == BlockIdFromCoord(GridCoord{2, 3}, s));
    ok = ok && (PeerBlockIdForPop(GridDirection::SOUTH, c, s) == BlockIdFromCoord(GridCoord{1, 2}, s));
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
static_assert(GridPeerSelfCheck(), "GridPipe single-hop peer resolver self-test failed");

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

// Fault sentinel, in u32 words from the scoreboard (or lane) it belongs to.
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
// A group-reduce SINK gave up pulling the members' scheme-C epochs.  It is the
// pull-side twin of 0x301: same meaning ("a contribution never became ready"),
// different carrier, because a pulled flag cannot be waited on with WAIT_SPR
// (2026-08-12 门铃归属分析 §4.2) and therefore cannot time out the same way.
inline constexpr uint32_t kFaultGroupEpochTimeout = 0x303;
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
// The group geometry does not fit the pipe: the group is wider than GroupMax (so
// the group mailbox has no line for some member) or this cell's rank falls
// outside it.
inline constexpr uint32_t kFaultBcastGroupRange = 0x404;
// Reserved.  It used to mean "a group TPOP asked for a member out of turn", back
// when the members sharing a channel published into one monotone count in a
// DERIVED order and a receiver had to drain that channel in count order.  The
// queued bind mailbox removed the derived order, and harvesting a doorbell no
// longer implies copying its payload out, so any drain order is legal now and
// nothing emits this.  Kept so existing host-side fault decoders do not
// reinterpret 0x405.
inline constexpr uint32_t kFaultBcastDrainOrder = 0x405;
// A group broadcast request carried a `basek` this receiver has already granted
// (basek < grantHead).  The caller's sequence numbers must be unique and dense per
// collective: the receivers derive the grant ORDER from that sequence alone, with
// no communication, so a repeated or rewound number would hand the same ring slot
// to two publishers.  It is reported rather than served, because serving it is the
// one thing that cannot be undone.
inline constexpr uint32_t kFaultBcastBasekOrder = 0x406;

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
// Reserved.  It used to mean "a group channel changed owner while its previous
// owner had not closed and been drained" -- unavoidable while group channels were
// bound to a rank by construction with no handshake able to reject a stale one.
// They are negotiated now, and every grant carries the channel's current count as
// its baseline, so a new phase rebases instead of mis-counting.  Nothing emits it.
inline constexpr uint32_t kFaultGroupChannelOwner = 0x50A;
// A bind request named a peer whose logical block id is outside the mailbox
// queue (>= kGridBindQueueDepth), so there is no line reserved for it and a
// request would have to share one with another peer -- the exact collision the
// queue exists to make impossible.  Raise PTO_GRID_BIND_QUEUE_DEPTH.
inline constexpr uint32_t kFaultBindQueueRange = 0x50B;

} // namespace grid_mock
} // namespace pto

// ===========================================================================
// Section 3: GmSramArena -- GM address-segment model of per-core SRAM (mock).
//
// The peer-SRAM addressing / transfer that used to live here as a
// CCE-intrinsic-style API (get_neighbor_sram_addr / copy_l1_to_peer_l1 /
// copy_local_slot_to_ubuf / sram_pop_is_local, with neighbor_sram_addr /
// NeighborSramOperand operands and a fabricated __builtin_pto_* stub) is gone:
// payload PUSH now stages in an isolated producer L1 slot and lowers to the
// copy_l1_to_peer_l1 CCE
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
// only *write* into from a peer (TPUSH = cross-core write), never *read* out
// of remotely (TPOP only drains the local core's own SRAM).  Until that
// hardware exists we model the SRAM as a contiguous GM arena cut into equal,
// per-core address segments.  Core `c` owns segment `c`:
//
//   [base + c*segBytes, base + (c+1)*segBytes)
//
// The NoC contract this encodes:
//   * a core may WRITE across segments  (TPUSH pushes into a peer's segment),
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
