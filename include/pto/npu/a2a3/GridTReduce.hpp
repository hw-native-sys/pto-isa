/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TREDUCE<Direction, Op>: a fused
// "receive-combine-forward" reduce hop along a mesh direction.  Builds on
// GridTPush.hpp / GridTPop.hpp -- see the V7 design spec section 5 (worked
// ReduceSum example), which frames the row reduce as the SAME single-hop SPSC
// handshake as AllGather, differing ONLY in the per-hop middle operation:
// AllGather relays the tile, ReduceSum folds it in with a combine before
// forwarding.  TREDUCE is that fused hop.
//
// Like TPUSH / TPOP, a reduce hop reaches exactly the ADJACENT cell -- it is
// built out of them, so it inherits the one-hop rule rather than restating it.  A
// reduction across a whole row is the chain of such hops, which is what makes it
// systolic in the first place.
//
// Semantics per cell (all roles derived from (Dir, coord, shape); no explicit
// "am I root" flag needed -- the mesh boundary defines source/sink):
//   * interior/sink (has an upstream neighbor back along Dir):
//         recv  <- TPOP<Dir>          (drain the transiting partial from upstream)
//         acc   <- combine(acc, recv) (fold in this cell's local contribution)
//   * source/interior (has a downstream neighbor on along Dir):
//         TPUSH<Dir>(acc)             (forward the running reduction one hop)
//   * sink (no downstream): acc holds the COMPLETE reduction; the caller stores it.
//
// Along-the-path / on-transit compute (随路/过路计算): a fabric that can combine
// in the router collapses this whole receive-combine-forward body into ONE
// routed reduce-forward instruction -- `recv` and the in-core combine disappear
// into the transfer.  On A3 there is no such fabric and the adder is core-local,
// so TREDUCE lowers to exactly the local TPOP + combine + TPUSH sequence below.
// Nothing in the (Dir, Op, pipe, acc, recv) signature changes between the two
// lowerings; only the body does.

#ifndef PTO_A2A3_GRID_TREDUCE_HPP
#define PTO_A2A3_GRID_TREDUCE_HPP

#include <cstdint>

#include <pto/comm/comm_types.hpp>         // pto::comm::ReduceOp (Sum/Max/Min) -- shared with the collective TREDUCE
#include <pto/npu/a2a3/GridTPop.hpp>       // GRID_TPOP_IMPL (receive half)
#include <pto/npu/a2a3/GridTPush.hpp>      // GRID_TPUSH_IMPL (forward half) + payload hooks
#include <pto/npu/a2a3/grid_intrinsic.hpp> // CanPop / CanPush topology
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

// Per-hop combine: fold the transiting partial `recv` into the accumulator `acc`
// with the reduce operator.  On A3 the adder lives inside the core, so this is an
// ordinary in-UB Vec op (TADD/TMAX/TMIN, resolved via ADL on the tile type); a
// future along-the-path-compute lowering performs this in the fabric during the
// transfer and drops the call entirely.  `Op` is a compile-time constant, so the
// branch folds away and only the selected instruction is instantiated.
template <pto::comm::ReduceOp Op, typename TileAcc, typename TileRecv>
AICORE inline void GridReduceCombine(TileAcc& acc, TileRecv& recv)
{
    if constexpr (Op == pto::comm::ReduceOp::Sum) {
        TADD(acc, acc, recv);
    } else if constexpr (Op == pto::comm::ReduceOp::Max) {
        TMAX(acc, acc, recv);
    } else {
        static_assert(Op == pto::comm::ReduceOp::Min, "GridPipe TREDUCE supports ReduceOp Sum/Max/Min only");
        TMIN(acc, acc, recv);
    }
}

// GridPipe TREDUCE<Dir, Op>: one fused reduce hop (see the file header).
// `acc` is in/out -- on entry the cell's local contribution, on return the
// running reduction up to and including this cell (at the sink, the complete
// result).  `recv` is the landing tile for the transiting partial (mandatory on
// A3's in-core adder; unused by a fabric that combines on transit).  Both tiles
// must share the reduce dtype/shape.
//
// Fences use the same conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR) publish
// form as GridTPush.hpp (parse-safe on every target profile).  A full barrier
// subsumes the fine-grained MTE2->V->MTE3 crossings the hand-written kernel used:
//   * after the pop, before the combine: drain the MTE2 slot->recv copy (and the
//     caller's MTE2 producer of acc) so the Vec combine reads settled UB;
//   * after the combine / before the push: drain the V combine so the MTE3
//     payload copy reads the settled accumulator.
// The push half additionally carries its own data-before-ready publish fence
// inside GRID_TPUSH_IMPL.  The pop / push are gated on CanPop / CanPush so a
// boundary cell never enters GRID_TPUSH_IMPL's out-of-mesh fault path.
template <pto::GridDirection Dir, pto::comm::ReduceOp Op, typename Pipe, typename TileAcc, typename TileRecv>
AICORE void GRID_TREDUCE_IMPL(Pipe& pipe, TileAcc& acc, TileRecv& recv)
{
    static_assert(Dir != pto::GridDirection::SOURCE, "GridPipe TREDUCE<SOURCE> is illegal (SOURCE is TPOP-only)");

    // Receive-and-combine half.  A source cell (no upstream along Dir) has nothing
    // to drain and forwards its own contribution unchanged.
    const bool didCombine = CanPop(Dir, pipe.coord, pipe.shape);
    if (didCombine) {
        GRID_TPOP_IMPL<Dir, Pipe, TileRecv>(pipe, recv);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GridReduceCombine<Op, TileAcc, TileRecv>(acc, recv);
    }

    // Forward half.  A sink cell (no downstream along Dir) keeps the complete
    // reduction in `acc` for the caller to store.
    if (CanPush(Dir, pipe.coord, pipe.shape)) {
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GRID_TPUSH_IMPL<Dir, Pipe, TileAcc>(pipe, acc);
    }
}

// Forward declaration: TileUbPtr is provided by the demo's
// gridpipe_payload_inl.hpp (same pluggable payload-hook contract as the other
// a2a3_grid_payload helpers).  Kept out-of-line so this group-reduce facade
// stays tile-agnostic (it hands the mov_ubuf_group intrinsic raw UB ptrs).
namespace a2a3_grid_payload {
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);
} // namespace a2a3_grid_payload

// ===========================================================================
// GRID_TREDUCE_GROUP_IMPL: N->1 group fan-in reduce.  EVERY member of the group
// calls it, and the role comes from comparing its own position with the one the
// `sinkBlockId` OPERAND names: the member it points at collects, every other one
// contributes.  The collector is addressed the way every peer in this family is,
// by LOGICAL BLOCK ID -- the same integer get_block_idx() returns -- and its
// index along the group axis is recovered from the topology.  Still no "am I
// root" flag: the sink is named once, in a value every member passes identically
// under SPMD, and the collecting core is by construction the one that issues the
// gather.
//
// The sink is an operand rather than the last member (the old convention) because
// the collector is a placement decision of the CALLER: it is wherever the result
// is next needed, which is not generally the end of the row.  The cost is that
// contributors then sit on BOTH sides of the sink, so the fan-in uses the two
// directional scoreboard pairs its arms attribute to instead of one -- exactly
// the back/forward split TBROADCAST has.  Each side keeps its own count, so the
// two never interfere.
//
//   * sink (the block `sinkBlockId` names): waits until every contributor has published,
//     then reads each member's resolved contribution slot and folds them
//     element-wise with Op into `acc` via the mov_ubuf_group intrinsic with op =
//     SUM/MAX/MIN (the unified group-collective CCE instruction,
//     grid_cce_intrinsic.hpp), and finally returns the retire credit.
//   * contributor: publishes -- its contribution is already in the arena, so the
//     publish IS the doorbell (one increment on the sink's scoreboard).
//
// This is the hardware-accelerated single-instruction form of an N->1 fan-in --
// a DIFFERENT collective shape from the directional GRID_TREDUCE_IMPL relay
// above (§7.1): every member's contribution is read directly by the sink, not
// relayed hop by hop.  The directional relay stays available for chains.
//
// NOTIFICATION -- the sink's own directional scoreboards, shared over time.  A
// consumer has ONE scoreboard per direction, and each contributor's edge into
// the sink attributes to a direction by GroupFlowDirection, so all contributors
// on ONE SIDE ring the same scoreboard.  They need no agreed value: each simply
// ADDS 1 (sync_hscb_add), and the sink waits for cons_idx + (that side's peer
// count) -- one increment per contributor -- then advances cons_idx by the same.
// An increment carries no assumption about the other contributors, so their
// order does not matter and no turn ring is needed.  With the sink at the end of
// the group one side is empty and this degenerates to the single-scoreboard form.
//
// BACKPRESSURE.  After folding, the sink adds 1 to each contributor's free_scb --
// on the scoreboard of the side that contributor sits on, which is the same one
// it is waiting on.  A contributor needs that credit before publishing its next
// round, which is what stops it overwriting a contribution the sink has not yet
// read.  Again a plain counting threshold on both sides.
//
// SCOREBOARD SHARING: as with TBROADCAST, a group reduce and a unicast channel
// resolving to the same direction on the same pipe would corrupt each other's
// counts.  Give them separate pipes.
//
// `scratch` is the in-core combine scratch (one member's worth of UB; required
// by the A3 mock's in-core Vec combine).  `groupSlotBase` / `memberStride` /
// `memberCount` describe the resolved per-member contribution arena -- e.g. the
// contributors' partial buffer laid out uniform-stride (ROW/COL members occupy
// consecutive grid ranks, so uniform stride is always the right model).  The combine folds
// members in ascending index order (member 0 seeds acc), so an SPMD row/col
// fan-in reproduces the relay's left-to-right accumulation bit-for-bit (FP add
// is commutative).  Contributors ignore `acc` / `scratch` / the arena operands.
// ===========================================================================
// Wait out one side's arrivals: `peerCount` contributors lie along `dir` from
// this sink, each having added 1 to the scoreboard that direction owns.  Returns
// with cons_idx advanced past them, so the next round starts from a clean base.
template <typename Pipe>
AICORE inline bool WaitGroupSideArrivals(Pipe& pipe, GridDirection dir, int peerCount, uint32_t maxSpins)
{
    if (peerCount <= 0) {
        return true; // nothing on this side of the sink
    }
    const int dirIdx = GridDirectionIndex(dir);
    const int edgeIdx = GridEdgeIndex(dir);
    const uint32_t threshold = pipe.consIndex[dirIdx] + static_cast<uint32_t>(peerCount);
    if (!wait_ipc_scb_sim(pipe.readyScb[edgeIdx], threshold, static_cast<uint32_t>(edgeIdx), maxSpins)) {
        __gm__ uint32_t* readyFault =
            pipe.readyScb[edgeIdx] ? pipe.readyScb[edgeIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    // ORDER CHECK: a contributor may not run a round ahead of the sink -- the
    // retire credit is what lets it publish again, so when the protocol is
    // honoured the count is EXACTLY the threshold.  More means somebody
    // overwrote a contribution this fold has not read yet.
    if (peek_ipc_scb(pipe.readyScb[edgeIdx]) > threshold) {
        __gm__ uint32_t* orderFault =
            pipe.readyScb[edgeIdx] ? pipe.readyScb[edgeIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(orderFault, grid_mock::kFaultGroupOutOfOrder);
        return false;
    }
    pipe.consIndex[dirIdx] = threshold;
    return true;
}

template <
    pto::GridGroup Group, pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE bool GRID_TRY_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlotBase, uint32_t bytes, uint32_t memberCount,
    int sinkBlockId, uint32_t memberStride = 0, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    if (memberCount == 0) {
        return true; // empty group: nothing to publish, nothing to fold
    }
    const int lastIndex = static_cast<int>(memberCount) - 1;
    // The collector is named by its logical BLOCK ID, the same way every other
    // peer operand in this family is; its position along the group axis is
    // recovered from the topology, not passed in.
    const GridCoord sinkCoord = pto::CoordFromBlockId(sinkBlockId, pipe.shape);
    const int sinkIndex = pto::IndexInGroup(Group, sinkCoord);
    if (!pto::GroupContains(Group, pipe.coord, sinkCoord) || sinkIndex > lastIndex) {
        // A sink outside the group -- or outside the declared member set, whose
        // arena this fold walks -- would ring a stranger's window.  Trap on EVERY
        // member, not just the cell that would have collected.
        __gm__ uint32_t* sinkFault = pipe.readyScb[0] ? pipe.readyScb[0] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(sinkFault, grid_mock::kFaultGroupBadPeer);
        return false;
    }
    const int myIndex = pto::IndexInGroup(Group, pipe.coord);

    if (myIndex != sinkIndex) {
        // ---- contributor half -------------------------------------------------
        // This member's edge to the sink picks the scoreboard pair.
        const GridDirection dir = pto::GroupFlowDirection(pipe.coord, sinkCoord);
        const int dirIdx = GridDirectionIndex(dir);
        const int edgeIdx = GridEdgeIndex(dir);
        const uint32_t round = pipe.prodIndex[dirIdx];

        // Backpressure: wait for the sink's credit for the previous round before
        // publishing again (round 0 blocks on nothing, as TPUSH's first pushes do).
        if (round > 0 && !wait_ipc_scb_sim(
                             pipe.freeScb[edgeIdx], round, static_cast<uint32_t>(kGridEdgeCount + edgeIdx), maxSpins)) {
            __gm__ uint32_t* freeFault =
                pipe.freeScb[edgeIdx] ? pipe.freeScb[edgeIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }

        // Publish fence (data-before-ready): this member's contribution was
        // written into the arena before the call, and must be visible to the
        // sink before the doorbell below announces it.
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        sync_hscb_add(a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[edgeIdx], sinkBlockId), 1);
        pipe.prodIndex[dirIdx] = round + 1;
        return true;
    }

    // ---- sink half ------------------------------------------------------------
    // An interior sink has contributors on BOTH sides, and the two sides land on
    // different scoreboards (a contributor's edge attributes by
    // GroupFlowDirection).  Wait each side out separately: within one side the
    // increments partition its sequence densely, so that side is complete exactly
    // when its count reaches cons_idx + (peers on that side).  An empty side is
    // skipped entirely -- with the sink at either end this is the old
    // one-scoreboard form, unchanged.
    const int peersBack = sinkIndex;            // members at index [0, sinkIndex)
    const int peersFwd = lastIndex - sinkIndex; // members at index (sinkIndex, lastIndex]
    int backEdgeIdx = 0;
    int fwdEdgeIdx = 0;
    if (peersBack > 0) {
        const GridCoord firstCoord = pto::GroupMemberCoord(Group, pipe.coord, 0);
        const GridDirection dir = pto::GroupFlowDirection(firstCoord, pipe.coord);
        if (!WaitGroupSideArrivals(pipe, dir, peersBack, maxSpins)) {
            return false;
        }
        backEdgeIdx = GridEdgeIndex(dir);
    }
    if (peersFwd > 0) {
        const GridCoord lastCoord = pto::GroupMemberCoord(Group, pipe.coord, lastIndex);
        const GridDirection dir = pto::GroupFlowDirection(lastCoord, pipe.coord);
        if (!WaitGroupSideArrivals(pipe, dir, peersFwd, maxSpins)) {
            return false;
        }
        fwdEdgeIdx = GridEdgeIndex(dir);
    }

    __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileAcc>(acc));
    __ubuf__ T* scr = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileScratch>(scratch));
    // mov_ubuf_group with op = SUM/MAX/MIN (the reduce / combine-fan-in NoC mode).
    // GridCollOp is comm::ReduceOp + 1 (Sum=0->SUM=1, Max=1->MAX=2, Min=2->MIN=3);
    // eltype = sizeof(T) (2 -> half/_b16, 4 -> float/_b32).  reduce only READS the
    // contribution arena, so const-cast away to satisfy the shared writable base.
    pto::mov_ubuf_group(
        reinterpret_cast<__ubuf__ void*>(dst), reinterpret_cast<__gm__ void*>(const_cast<__gm__ T*>(groupSlotBase)),
        bytes, memberCount, memberStride, static_cast<pto::GridCollOp>(static_cast<uint32_t>(Op) + 1),
        static_cast<uint32_t>(sizeof(T)), reinterpret_cast<__ubuf__ void*>(scr));

    // consume-before-free fence, then credit every contributor: one increment
    // each, telling them their contribution has been read and the next round may
    // overwrite it.  The credit must land on the scoreboard of the side that
    // contributor sits on -- the same one it is blocked on, since both ends
    // derive it from the same GroupFlowDirection.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    for (int r = 0; r <= lastIndex; ++r) {
        if (r == sinkIndex) {
            continue; // the sink's own contribution needs no credit
        }
        const int edgeIdx = (r < sinkIndex) ? backEdgeIdx : fwdEdgeIdx;
        const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, r);
        sync_hscb_add(a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[edgeIdx], peerBlockId), 1);
    }
    return true;
}

template <
    pto::GridGroup Group, pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE void GRID_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlotBase, uint32_t bytes, uint32_t memberCount,
    int sinkBlockId, uint32_t memberStride = 0)
{
    (void)GRID_TRY_TREDUCE_GROUP_IMPL<Group, Op, T, Pipe, TileAcc, TileScratch>(
        pipe, acc, scratch, groupSlotBase, bytes, memberCount, sinkBlockId, memberStride, /*maxSpins=*/0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TREDUCE_HPP
