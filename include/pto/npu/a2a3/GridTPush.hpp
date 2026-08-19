/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPUSH.
//
// TPUSH names the CONSUMER it is writing to -- an ordinary mesh rank -- and the
// channel comes from the pipe's binding table, not from a compass point in the
// template arguments.  Every transfer is exactly one hop; there is no direction and
// no distance operand anywhere in the family.
//
// Producer-side expansion calls the V8 CCE facades directly (V8 section 3.5.3
// TPUSH), with no intermediate PTO wrapper:
//   - wait_ipc_scb                 (WAIT_SPR on the local free_scb; read+block in one
//                                   instruction, no MOV_SPR2X peek -- V8)
//   - copy_l1_to_peer_l1       (COPY_L1_TO_PEER payload write, via the hook)
//   - sync_hscb                    (SYNC_HSCB store prod_idx -> peer ready_scb)
// Peer address resolution (ResolvePeerSlotAddr / RemoteScbPtr) is a plain runtime
// helper in the demo's gridpipe_payload_inl.hpp, not an intrinsic.
//
// payload transfer is intentionally pluggable: the tile->producer-L1 staging and
// producer-L1->peer-L1 adapters live alongside the demo kernel (they need the
// Tile type), while the generic handshake sequence stays here.

#ifndef PTO_A2A3_GRID_TPUSH_HPP
#define PTO_A2A3_GRID_TPUSH_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by demo's gridpipe_runtime adaptor.
// At the demo level we inject a concrete implementation that knows how to
// move a specific tile type to/from a mock SRAM slot via TSTORE/TLOAD. Keeping
// the hook out-of-line avoids tying GridPipe to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

// Resolve a local GM slot address to the same byte offset in peerBlockId's window
// (mock: the GM window standing in for peerBlockId's SRAM; native: mesh geometry).
AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);

// Resolve a local scoreboard word to peerBlockId's scoreboard word (sync_hscb dst).
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);

// Stage a tile in the pipe's isolated local producer L1 slot.
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride);

// Copy the staged local L1 source into the resolved peer receive slot.  `tile`
// is scratch only in the A3 GM mock; it is not the architectural source address.
template <typename TileT>
__tf__ AICORE void CopyProducerSramToPeerSlot(
    __gm__ uint8_t* dstPeerSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void CopyProducerSramToPeerSlot2D(
    __gm__ uint8_t* dstPeerSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount,
    uint32_t producerStride, uint32_t dstStride, uint32_t tileStride);

// Drain this core's local GM slot into the tile (V7 TPOP local read: the existing
// local copy; deliberately no cross-core read of payload).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes);

// 2-D form of the drain (slot is the source, tile the destination).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile2D(
    TileT& tile, __gm__ uint8_t* localSlot, uint32_t rowBytes, uint32_t rowCount, uint32_t slotStride,
    uint32_t tileStride);

// Mock-only read-locality guard: true iff [localSlot, +bytes) is inside
// callerBlockId's own GmSramArena segment (native: always local by construction).
AICORE bool PopSlotIsLocal(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, uint32_t bytes, int callerBlockId);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

namespace grid_detail {

// Fault sentinel for a scoreboard, null-safe: nullptr + offset is UB and would slip
// a non-null (but invalid) pointer past MockSetFault's null guard.
AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

// A binding whose downstream-consumer history overflowed has already lost the
// state required to reopen that consumer safely.  The flag is sticky, so whichever
// op runs first reports it.
template <typename Pipe>
AICORE inline bool ReportPendingBindFault(Pipe& pipe)
{
    if (pipe.consHistFull) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return true;
    }
    return false;
}

AICORE inline void GridPublishFence()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

// L1 has no WAIT_SPR equivalent, so every mailbox wait is a bounded scalar poll
// (`maxSpins == 0` blocks forever, matching hardware).  What a waiter does
// BETWEEN attempts is the load-bearing part: it serves its own request queue, so
// a core blocked waiting for a grant is still handing out grants.  Two cores that
// ask each other at the same instant -- the normal state of a group collective --
// would otherwise deadlock, each holding what the other is waiting for.
AICORE inline bool GridBindSpin(uint32_t& spin, uint32_t maxSpins)
{
    constexpr uint32_t kFenceInterval = 64;
    if (maxSpins != 0 && spin >= maxSpins) {
        return false;
    }
    if ((++spin % kFenceInterval) == 0) {
        pipe_barrier(PIPE_ALL);
    }
    return true;
}

// ---------------------------------------------------------------------------
// THE QUEUED BIND MAILBOX.  Three primitives, used by the unicast TPUSH flow
// below and by the group reduce's credit binding (GridTReduce.hpp).  The group
// BROADCAST has its own mailbox, indexed by rank-in-group instead of by block id
// (GridTBroadcast.hpp): its member set is known exactly, so it does not need --
// and, at mesh scale, cannot afford -- a line per core in the mesh.
//
//   PostBindRequest        producer -> consumer, ONE 32-bit store
//   PollBindResponse       producer polls its own slot, non-blocking
//   ServiceOneBindRequest  consumer serves AT MOST ONE pending request
//
// "At most one" is the whole point of the queue: several requests may be waiting,
// exactly one is turned into a binding per pass, and the rest stay untouched in
// their own cache lines until a later pass can serve them.  Nothing is overwritten
// and nothing is lost, which is what the single-line mailbox could not offer K
// concurrent requesters (2026-08-11 分析 §3.2).
// ---------------------------------------------------------------------------

// Arm this producer's own response slot for `consId`.  Split out from the post
// below so a fan-out (one broadcast asking K-1 receivers) can arm every slot and
// pay ONE fence instead of one per peer -- the ordering that matters is only
// "every arm before any request", since a consumer may answer the first request
// before the last clear would have reached memory.
template <typename Pipe>
AICORE inline bool ArmBindResponseSlot(Pipe& pipe, uint32_t consId)
{
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    if (consId >= static_cast<uint32_t>(kGridBindQueueDepth) || selfBlockId >= kGridBindQueueDepth) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBindQueueRange);
        return false;
    }
    __gm__ uint32_t* response = pipe.BindResponseSlot(consId);
    grid_cce_detail::write_local_word(response, kGridBindPending);
    grid_cce_detail::write_local_word(response + 1, kGridBindPending);
    return true;
}

// Post an ARMED request.  A request is ONE word, so its payload and its commit are
// the same store and there is nothing to fence between.
template <typename Pipe>
AICORE inline void PostArmedBindRequest(Pipe& pipe, uint32_t consId, GridBindMode mode, int prodChan)
{
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    // Our slot in the CONSUMER's request queue is our slot in our own queue,
    // resolved into the consumer's window -- same byte offset, other window.
    __gm__ uint32_t* peerRequest = a2a3_grid_payload::RemoteScbPtr(
        pipe.runtimeCtx, pipe.BindRequestSlot(static_cast<uint32_t>(selfBlockId)), static_cast<int>(consId));
    sync_hscb(peerRequest, GridPackBindRequest(mode, prodChan, static_cast<uint32_t>(selfBlockId)));
}

// Arm + fence + post, for the single-peer callers (a unicast flow, a reduce
// member binding to its sink).
template <typename Pipe>
AICORE inline bool PostBindRequest(Pipe& pipe, uint32_t consId, GridBindMode mode, int prodChan)
{
    if (!ArmBindResponseSlot(pipe, consId)) {
        return false;
    }
    GridPublishFence();
    PostArmedBindRequest(pipe, consId, mode, prodChan);
    return true;
}

// Has `consId` answered yet?  Non-blocking.  On success the slot is cleared, so
// the next request to the same consumer starts from an armed mailbox.
// `consChan` comes back negative for a grant that carries no receive channel
// (GROUP_PULL: the reduce sink only records where to push credit).
template <typename Pipe>
AICORE inline bool PollBindResponse(Pipe& pipe, uint32_t consId, int& consChan, uint32_t& base)
{
    __gm__ uint32_t* response = pipe.BindResponseSlot(consId);
    const uint32_t commit = mov_x_to_gpr(response + 1);
    if (commit == kGridBindPending) {
        return false;
    }
    base = mov_x_to_gpr(response); // payload word, written and fenced before the commit
    consChan = GridBindResponseChan(commit);
    grid_cce_detail::write_local_word(response + 1, kGridBindPending);
    return true;
}

// The reduce -> unicast half of the channel relay (归约后单播复用): put a receive
// channel back into its as-new state before handing it to a flow.  cons_idx and the
// channel's OWN ready/close scoreboards go to zero together, and the bind then
// answers prod_idx = 0 and stores 0 into the producer's free_scb -- so both ends of
// the new flow start from one origin instead of relaying a baseline out of the
// collective's fold stream, which is not the same stream at all.
//
// Zeroing the scoreboards is what makes zeroing the GPR safe.  cons_idx = 0 next to
// a ready count some EARLIER flow on this channel left behind would let the very
// first TPOP sail straight through onto a ring slot this phase never wrote.  And
// MOVX2SPR is legal here for exactly the reason its facade documents -- the caller
// must prove exclusive ownership, and at this instant the channel has NO external
// writer: a pull reduce never rings ready/close, and the producer that is about to
// is still waiting for the answer this pass has not sent yet.
template <typename Pipe>
AICORE inline void ResetConsumerChannelCounters(Pipe& pipe, int consChan)
{
    mov_x_to_ipc_scb(pipe.readyScb[consChan], static_cast<uint32_t>(consChan), 0u);
    mov_x_to_ipc_scb(
        pipe.closeScb[consChan], 2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(consChan), 0u);
    GridPublishFence(); // the reset must be visible before the grant that follows it
    pipe.consIndex[consChan] = 0;
    pipe.PersistConsIndex(consChan);
}

// Serve at most one pending request; returns who was served, or kGridNoPeer when
// nothing could be served this pass (queue empty, or every servable request needs
// a channel this core cannot spare yet).  Never blocks.
//
// `scanRect` bounds the sweep: a group phase knows exactly which cores can be
// asking, and sweeping all kGridBindQueueDepth lines per call -- each an
// invalidate plus a GM read -- would dominate the collective.  An empty rect
// sweeps the whole queue, which is what the unicast bind path wants (it runs once
// per flow, not once per tile).  The scan start rotates so no requester starves.
template <typename Pipe>
AICORE inline uint32_t ServiceOneBindRequest(Pipe& pipe, const pto::GridBlockRect& scanRect)
{
    if (pipe.bindRequestQueue == nullptr) {
        return kGridNoPeer;
    }
    const uint32_t rectSize = pto::GridBlockRectSize(scanRect);
    const uint32_t scanCount = (rectSize != 0) ? rectSize : static_cast<uint32_t>(kGridBindQueueDepth);
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);

    for (uint32_t n = 0; n < scanCount; ++n) {
        const uint32_t cursor = (static_cast<uint32_t>(pipe.bindScanCursor) + n) % scanCount;
        const uint32_t peerId = (rectSize != 0) ? pto::GridBlockRectMember(scanRect, cursor) : cursor;
        if (peerId >= static_cast<uint32_t>(kGridBindQueueDepth) || peerId == static_cast<uint32_t>(selfBlockId)) {
            continue;
        }
        __gm__ uint32_t* request = pipe.BindRequestSlot(peerId);
        const uint32_t word = mov_x_to_gpr(request);
        if (word == kGridBindPending) {
            continue;
        }
        const GridBindMode mode = GridBindRequestMode(word);
        const uint32_t prodId = GridBindRequestId(word);
        const int peerProdChan = GridBindRequestChan(word);
        if (prodId != peerId || !GridBlockIdValid(prodId, pipe.shape) || peerProdChan < 0 ||
            peerProdChan >= kGridChanCount) {
            // A slot only its owner may write cannot hold this; drop it rather
            // than let one corrupt word wedge the scan forever.
            grid_cce_detail::write_local_word(request, kGridBindPending);
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
            continue;
        }

        int consChan = kGridInvalidChan;
        uint32_t base = 0;
        uint32_t freeBase = 0;
        if (mode == GridBindMode::GROUP_PULL) {
            // Scheme C: no receive RING at all, but the fold counter is an ordinary
            // channel out of the shared pool now.  All the sink takes from the
            // request is WHERE to push this member's credit; what it gives back is
            // how many rounds it has already folded, so a member joining a window
            // with history rebases instead of replaying.
            if (rectSize == 0) {
                continue; // cannot attribute a member without knowing the group -- leave it pending
            }
            const int rank = pto::GridBlockRectRankOf(scanRect, prodId);
            if (rank < 0 || rank >= Pipe::GroupSlots) {
                continue;
            }
            if (peerProdChan < Pipe::UnicastChanBase) {
                // A credit counter inside the reserved broadcast range would be an
                // absolute store into a count K receivers atomic-add.  Only a
                // corrupt line can ask for that.
                grid_cce_detail::write_local_word(request, kGridBindPending);
                grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
                continue;
            }
            // The collective's own end of the shared pool.  Allocating it lazily is
            // what lets a core service a member's request before it reaches its own
            // TREDUCE; PickGroupPullChannel is idempotent, so every later request of
            // this launch lands on the same fold counter.
            const int pullChan = pipe.PickGroupPullChannel();
            if (pullChan == kGridInvalidChan) {
                continue; // the pool's retiring flow has not drained yet -- the request waits
            }
            if (pullChan != pipe.groupPullChan) {
                pipe.ClaimGroupPullChannel(pullChan);
            }
            pipe.memberPeerChan[rank] = peerProdChan;
            // 单播后归约复用 -- BASELINE + ROUND.  Nothing is cleared: whatever the
            // retiring flow left in cons_idx IS the baseline, and it travels to the
            // member TWICE -- as the round origin it adopts as prod_idx, and (below)
            // as the starting value of the free_scb this sink credits it through.
            // Both sides then count baseline + round, so neither has to know what
            // the previous tenant left behind.
            base = pipe.consIndex[pullChan];
            freeBase = base;
        } else if (mode == GridBindMode::UNICAST) {
            // Hand out a channel from the unicast pool.  A producer that already
            // owns a live channel here is asking early (its previous flow has not
            // closed and drained yet); leaving the request pending IS the answer --
            // it will be served when that channel frees, and until then the old
            // routing the last TPOP needs stays intact.
            const int existing = pipe.ConsumerChannelOfProducer(prodId);
            if (existing != kGridInvalidChan) {
                if (!pipe.ConsumerChannelIsReusable(existing)) {
                    continue;
                }
                consChan = existing;
            } else {
                consChan = pipe.PickBindableConsumerChannel();
            }
            if (consChan == kGridInvalidChan) {
                continue; // pool exhausted right now -- the request waits, this core does not
            }
            if (pipe.consChanKind[consChan] == GridChannelTenant::GROUP) {
                // 归约后单播复用 -- THE ZERO RULE.  The retiring tenant was a group
                // reduce: its cons_idx counts folds, not tiles, and it never wrote
                // this channel's ring or rang its doorbells.  There is no baseline
                // in that stream worth relaying, so the channel restarts from zero
                // on both sides (see ResetConsumerChannelCounters).
                ResetConsumerChannelCounters(pipe, consChan);
                base = 0;
                freeBase = 0;
            } else {
                base = pipe.ReadConsumerReadyCount(consChan);
                freeBase = pipe.consIndex[consChan];
            }
            // The outgoing producer is simply forgotten.  Its leftovers, if any, stay
            // in the ring as part of the channel's one continuous stream and are
            // drained by the ordinary TPOP arithmetic under the new owner's name;
            // nothing about the retiring core is needed to read them.
            pipe.consChanProdId[consChan] = prodId;
            pipe.consChanKind[consChan] = GridChannelTenant::UNICAST;
            pipe.consChanBindCnt[consChan] += 1;
            pipe.consChanPeerProdChan[consChan] = peerProdChan;
            pipe.consChanCloseBase[consChan] = base;
            pipe.curConsChan = consChan;
            pipe.prevProdId = prodId;
            pipe.StoreRecord();
        } else {
            // GROUP_PUSH does not travel in this mailbox any more: a broadcast asks
            // through the group mailbox, which is indexed by rank-in-group rather
            // than by block id.  Anything else here is a corrupt line.
            grid_cce_detail::write_local_word(request, kGridBindPending);
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
            continue;
        }

        // Clear the request BEFORE answering.  A producer only asks again once it
        // has the answer, so this ordering is exactly what keeps the clear from
        // wiping the NEXT round's request.
        grid_cce_detail::write_local_word(request, kGridBindPending);

        // STATE THE PRODUCER'S FREE BASELINE, whichever tenant this is.  A unicast
        // producer's free threshold is an absolute cons_idx and a reduce member's is
        // the sink's fold count, and in both cases the counter that has to carry it
        // may still hold whatever the channel's PREVIOUS tenant left there -- so the
        // grantor states it rather than assuming a zero.  (The group BROADCAST's
        // credit counter is the one that must never be overwritten: it sums atomic
        // adds from K receivers.  It cannot appear here -- it lives in the reserved
        // range, which the GROUP_PULL guard above rejects and unicast never
        // allocates from.)
        {
            __gm__ uint32_t* peerFree =
                a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerProdChan], static_cast<int>(prodId));
            sync_hscb(peerFree, freeBase);
        }
        __gm__ uint32_t* peerResponse = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.BindResponseSlot(static_cast<uint32_t>(selfBlockId)), static_cast<int>(prodId));
        sync_hscb(peerResponse, base);
        GridPublishFence(); // baseline before commit -- the producer polls the commit
        sync_hscb(peerResponse + 1, GridPackBindResponse(consChan));

        pipe.bindScanCursor = static_cast<int>((cursor + 1u) % scanCount);
        return prodId;
    }
    return kGridNoPeer;
}

// Producer half of a UNICAST flow: pick a local producer channel, ask, and wait
// for the consumer's independently chosen receive channel + baseline.
template <typename Pipe>
AICORE inline int EnsureOutgoingConsumerBinding(Pipe& pipe, uint32_t consId, uint32_t maxSpins)
{
    if (pipe.consumers.StateOf(consId) == GridConsumerState::ACTIVE) {
        const int prodChan = pipe.consumers.ProducerChannelOf(consId);
        const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
        if (prodChan >= 0 && prodChan < Pipe::ChanCount && peerConsChan >= 0 && peerConsChan < Pipe::ChanCount &&
            pipe.prodChanConsId[prodChan] == consId &&
            pipe.prodChanState[prodChan] == GridProducerChannelState::ACTIVE) {
            pipe.curProdChan = prodChan;
            pipe.consumers.curConsId = consId;
            return prodChan;
        }
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }

    if (pipe.consumers.FindOrAlloc(consId) < 0) {
        pipe.consHistFull = true;
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }

    uint32_t spin = 0;
    int prodChan = pipe.PickBindableProducerChannel(consId);
    while (prodChan == kGridInvalidChan) {
        (void)ServiceOneBindRequest(pipe, pto::GridBlockRect{});
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.freeScb[0]), grid_mock::kFaultWaitProducerChannelTimeout);
            return kGridInvalidChan;
        }
        prodChan = pipe.PickBindableProducerChannel(consId);
    }

    if (!PostBindRequest(pipe, consId, GridBindMode::UNICAST, prodChan)) {
        return kGridInvalidChan;
    }

    int peerConsChan = kGridInvalidChan;
    uint32_t readyBase = 0;
    spin = 0;
    while (!PollBindResponse(pipe, consId, peerConsChan, readyBase)) {
        (void)ServiceOneBindRequest(pipe, pto::GridBlockRect{});
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
            return kGridInvalidChan;
        }
    }
    if (peerConsChan < 0 || peerConsChan >= Pipe::ChanCount) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }

    pipe.prodIndex[prodChan] = readyBase;
    pipe.PersistProdIndex(prodChan);
    if (!pipe.ActivateConsumer(consId, prodChan, peerConsChan)) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    return prodChan;
}

// TPOP calls this before draining `prodId`: it services the queue until that
// producer holds a channel here.
template <typename Pipe>
AICORE inline int EnsureIncomingProducerBinding(Pipe& pipe, uint32_t prodId, uint32_t maxSpins)
{
    uint32_t spin = 0;
    while (true) {
        // Finish the expected producer's current turn before looking for a new
        // request.  CLOSE is sent after the final READY, so while consIndex is below
        // that final count this TPOP must keep the current mapping to drain the last
        // item -- INCLUDING anything an earlier producer left in the same stream,
        // which is why the test is against the count and not against an identity.
        // Once the turn is closed AND drained, fall through and service instead: a
        // later phase of the same producer has to re-handshake, and only servicing
        // can accept that request.
        const int existing = pipe.ConsumerChannelOfProducer(prodId);
        if (existing != kGridInvalidChan) {
            const uint32_t closeCount = pipe.ReadConsumerCloseCount(existing);
            if (closeCount <= pipe.consChanCloseBase[existing] || pipe.consIndex[existing] < closeCount) {
                return existing;
            }
        }
        if (ServiceOneBindRequest(pipe, pto::GridBlockRect{}) == prodId) {
            continue; // re-check at the top so validation stays in one place
        }
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindRequestTimeout);
            return kGridInvalidChan;
        }
    }
}

} // namespace grid_detail

// Push `tile` to the core whose LOGICAL BLOCK ID is `consId`.  The producer and
// consumer sides use independently negotiated channel indices.
//
// The call site derives `consId` from topology (GridPeerBlockIdForPush) or its
// schedule.  The first transfer to an UNBOUND/CLOSED consumer negotiates a channel;
// later ACTIVE transfers use that state-machine entry directly.
template <typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(
    Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false,
    uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "GridPipe TPUSH needs a pipe with at least one channel (ChanCount > 0)");

    if (grid_detail::ReportPendingBindFault(pipe)) {
        return false;
    }

    // Boundary check.  A boundary cell has no downstream, and the call site says so
    // by passing kGridNoPeer; anything else outside the mesh is a mis-derived rank.
    if (!GridBlockIdValid(consId, pipe.shape)) {
        grid_mock::MockBoundaryFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultPushOutOfMesh);
        return false;
    }

    // ACTIVE is the steady-state fast path.  UNBOUND/CLOSED negotiates independent
    // local producer and remote consumer channels.
    const int prodChan = grid_detail::EnsureOutgoingConsumerBinding(pipe, consId, maxSpins);
    if (prodChan == kGridInvalidChan) {
        return false;
    }
    const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
    if (peerConsChan < 0 || peerConsChan >= Pipe::ChanCount) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return false;
    }

    // Step 1 (V8 P1): wait for a free slot.  free threshold = prod_idx-SlotCount+1;
    //   WAIT_SPR alone reads the local free_scb and blocks (read+block in one
    //   instruction; no MOV_SPR2X peek -- V8).  The `prodIndex >= SlotCount` guard is
    //   exactly threshold > 0, so the first SlotCount pushes skip the wait (startup
    //   zero-block, V8 R6).  free_scb of channel c occupies IPC_SCB slot
    //   kGridChanCount+c.
    const uint32_t idx = pipe.prodIndex[prodChan];
    const uint32_t freeSlot = static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(prodChan);
    if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
        const uint32_t freeThreshold = idx + 1 - Pipe::SlotCount;
        if (!wait_ipc_scb_sim(pipe.freeScb[prodChan], freeThreshold, freeSlot, maxSpins)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2 (V7 P2): compute the local slot address from the producer GPR
    //   (slot_off = (prod_idx % SlotCount) * SlotStride); pure local scalar math.
    //   SlotStride addresses the ring; the payload window says what part of the
    //   slot this push actually moves (a5 TPipe: entryBase + entryOffset, with the
    //   length coming from the transfer descriptor rather than the slot size).
    const GridPayloadWindow win = pipe.pushWindow[prodChan];

    // Range guard (differs from a5: there the length is implied by the tile /
    // GlobalTensor descriptors and cannot exceed the slot; here it is a runtime
    // number, and an overrun writes into the PEER's window).
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultPushPayloadRange);
        return false;
    }

    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotStride + win.entryOffset;
    // `slotBase[peerConsChan]` is used only as an address template; resolving it
    // maps to that channel's receive ring in the peer's window.
    __gm__ uint8_t* localSlot = pipe.slotBase[peerConsChan] + slotOff;
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;

    // Step 2.5: materialise the outbound payload in the dedicated producer L1
    // range.  Source formula (one synchronous producer slot per pipe):
    //   producer = producerSlotBase + entryOffset + row*slotStride
    // Destination formula remains the selected receive-ring slot above.  The two
    // bases are disjoint by the window layout in grid_pipe_runtime.hpp.
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
    grid_detail::GridPublishFence();

    // Step 3 (V7 P3): payload transfer into the CONSUMER's SRAM/L1 slot region.
    //   The runtime helper resolves that rank's slot -- same byte offset, other
    //   window -- while the payload hook reads the isolated producer L1 slot and
    //   calls the copy_l1_to_peer_l1 CCE facade (COPY_L1_TO_PEER).
    const int peerBlockId = static_cast<int>(consId);
    __gm__ uint8_t* peerSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, peerBlockId);
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyProducerSramToPeerSlot<TileProd>(peerSlot, localProducerSlot, tile, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyProducerSramToPeerSlot2D<TileProd>(
            peerSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
            GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // Publish fence (V7 P4, data-before-ready / R5). Orders the payload write
    // (MTE3 into the peer window) before the ready sync_hscb store below.  V7's
    // preferred form issues SYNC_HSCB(READY) from the payload's async pipe so it
    // *naturally* orders after the payload DMA (no explicit fence); this A2/A3
    // mock instead uses the conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
    // fallback (V7 3.4.1 grid_publish_fence).  Without it the scalar-pipe store
    // can become visible on the peer before the MTE3 slot bytes commit to DDR,
    // causing the consumer's read to pick up pre-publish (zero) data.
    grid_detail::GridPublishFence();

    // Step 4 (V7 P5): announce readiness -- sync_hscb (SYNC_HSCB) store of
    //   prod_idx (= idx+1) into the consumer's ready_scb for THIS CHANNEL INDEX
    //   (overwrite store of a monotone absolute count; single external writer per
    //   SPSC).  ready_scb of channel c occupies IPC_SCB slot c.
    //
    __gm__ uint32_t* peerReady =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[peerConsChan], peerBlockId);
    sync_hscb(peerReady, idx + 1);

    // Step 5 (V7 P5): bump the local producer GPR (drives slot addr / free
    //   threshold / the absolute count published to the consumer).
    pipe.prodIndex[prodChan] = idx + 1;
    pipe.PersistProdIndex(prodChan);

    if (isLastTransfer) {
        // CLOSE is ordered after payload + READY and carries the same final
        // absolute count.  The consumer compares it with the ready baseline it
        // captured at bind time, so no local SCB clear/reset is required.
        grid_detail::GridPublishFence();
        __gm__ uint32_t* peerClose =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.closeScb[peerConsChan], peerBlockId);
        sync_hscb(peerClose, idx + 1);
        if (!pipe.CloseConsumer(consId)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[peerConsChan]), grid_mock::kFaultBindProtocol);
            return false;
        }
    }
    return true;
}

template <typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false)
{
    (void)GRID_TRY_TPUSH_IMPL<Pipe, TileProd>(pipe, tile, consId, isLastTransfer, pipe.maxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP
