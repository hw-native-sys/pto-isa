/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 GridPipe runtime helpers: shmem window layout and the init helper that
// wires a pipe's concurrency array to it.  See the V6 IPC_SCB scoreboard design
// and its A2/A3 mock in include/pto/npu/a2a3/grid_intrinsic.hpp.

#ifndef PTO_A2A3_GRID_PIPE_RUNTIME_HPP
#define PTO_A2A3_GRID_PIPE_RUNTIME_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

namespace pto {
namespace a2a3_grid {

// shmem window layout (per rank), in bytes.  The ready/free/close scoreboard words stand
// in for the V6 IPC_SCB slots of each CHANNEL (each carries a monotone absolute
// count written by the peer bound to that channel via an HSCB store):
//
//   offset                                          contents
//   -----------------------------------------------------------------------
//   0                                               ready scoreboards, one per
//     + c * kScbLineStride                            channel, ONE CACHE LINE EACH
//   kGridChanCount * kScbLineStride                 free scoreboards, likewise
//     + c * kScbLineStride
//   2 * kGridChanCount * kScbLineStride             close scoreboards, likewise
//     + c * kScbLineStride
//   kBindRequestOffset                              bind-request QUEUE, one L1 line
//     + p * kScbLineStride                            per PRODUCER block id p
//   kBindResponseOffset                             bind-response QUEUE, one L1 line
//     + c * kScbLineStride                            per CONSUMER block id c
//   kGroupEpochOffset                               scheme-C group-reduce epoch line
//   kRecordOffset                                   pipe record: bindings, consumer
//                                                     FSM/history, both channel maps,
//                                                     close bases and run counters
//   kGroupMailboxOffset                             group-collective mailbox (GroupMax > 0):
//     + r * kScbLineStride                            request  queue, one line per RANK-IN-GROUP
//     + (GroupMax + r) * kScbLineStride               response queue, likewise
//   kSlotRegionOffset + group mailbox bytes         slot region, ChanCount rings
//     + c * SlotCount * SlotStride                    ring of channel c.  Channels
//                                                     [0, BcastChanCount) are the group
//                                                     broadcast's; channel 0's ring IS the
//                                                     collective's payload arena, addressed by
//                                                     the caller's basek % SlotCount.  There is
//                                                     no separate broadcast region any more.
//   end of receive rings                             producer staging [SlotStride]
//                                                     local L1 source for every outbound transfer
//
// TWO PROPERTIES THE REST OF THE SYSTEM LEANS ON.
//
// (1) The scoreboard header is a FIXED kGridChanCount triplets, whatever a pipe's
//     ChanCount is, and every scoreboard owns a whole cache line.  Fixed, because
//     the peer resolver maps a local address to the SAME byte offset in the peer's
//     window, so two pipes in one build must agree on where channel c's doorbell
//     lives.  A line each, because each has a DIFFERENT external writer and the
//     mock's write-back is line-granular -- see kScbLineStride in
//     grid_intrinsic.hpp for the lost-update this prevents.  Only the RINGS are
//     trimmed by ChanCount, so a pure-broadcast pipe (ChanCount = 0) pays the
//     header and no unicast payload bytes at all.
//
// (2) The pipe record is LOCAL-ONLY -- this core is its sole reader and writer, no
//     peer ever stores into it -- so its words may share cache lines freely, and it
//     sits OUTSIDE kFlagsBytes.  That boundary matters: the host launchers scan the
//     flag header for fault sentinels, and the record holds ordinary counters and
//     block ids that would read as sentinels if they were inside.
//
// (3) The final SlotStride bytes are a LOCAL PRODUCER STAGING SLOT, not another
//     receive-ring entry.  Real WSE hardware has one unified L1 SRAM address space
//     (there is no physically separate Vec UB).  An outbound tile is therefore
//     staged here first and the NoC maps this local L1 address to the peer's receive
//     payload ring.  Keeping the producer slot after every receive-side region makes
//     source and destination storage disjoint even when a cell relays a tile while
//     its own receive ring is live.
//
// Fault sentinels live at word kFaultFlagWordOffset of the scoreboard (or lane)
// they belong to, which is inside that scoreboard's own line and clear of every
// live word.
inline constexpr uint32_t kScbHeaderBytes = 3U * static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;

// The two remotely-written control regions a bind handshake runs over.  Each is a
// QUEUE of kGridBindQueueDepth cache lines indexed by the PEER'S BLOCK ID, so a
// core can be asked by every other core at once without two requesters ever
// sharing a word (grid_intrinsic.hpp, "THE BIND MAILBOX IS A QUEUE").
//
//   request  slot p = [ mode | producer channel | producer id ]   -- one word, so
//                      payload and commit are the same store
//   response slot c = [ baseline, commit(consumer channel | granted) ]
//
// A line each, for the same reason the scoreboards get one: the mock's write-back
// is line-granular, and every slot has a DIFFERENT external writer.
inline constexpr uint32_t kBindQueueBytes = static_cast<uint32_t>(kGridBindQueueDepth) * grid_mock::kScbLineStride;
inline constexpr uint32_t kBindRequestOffset = kScbHeaderBytes;
inline constexpr uint32_t kBindResponseOffset = kBindRequestOffset + kBindQueueBytes;
// One line, written LOCALLY and read REMOTELY: the scheme-C group-reduce epoch
// (2026-08-12 门铃归属分析 方案C).  It is the only word in the layout with that
// direction, which is exactly the point -- a pull collective pulls its flag the
// same way it pulls its data.
inline constexpr uint32_t kGroupEpochOffset = kBindResponseOffset + kBindQueueBytes;
inline constexpr uint32_t kControlBytes = 2U * kBindQueueBytes + grid_mock::kScbLineStride;

// Pipe record: bindings, consumer FSM/history, producer/consumer channel maps and
// states, close bases, and persistent prod/cons counter mirrors.  It lets a schedule span several kernel
// launches; rounded up to a cache line so the slot region stays aligned.
inline constexpr uint32_t kRecordOffset = kScbHeaderBytes + kControlBytes;
inline constexpr uint32_t kRecordBytes =
    ((static_cast<uint32_t>(kGridRecordWords) * static_cast<uint32_t>(sizeof(uint32_t)) + grid_mock::kScbLineStride -
      1U) /
     grid_mock::kScbLineStride) *
    grid_mock::kScbLineStride;

// Bytes the host scans for fault sentinels: the scoreboard header only.
inline constexpr uint32_t kFlagsBytes = kScbHeaderBytes;
// The group mailbox sits between the record and the rings, because it is the one
// region whose size depends on the pipe's GroupMax.  Both ends of a collective are
// the same pipe type, so the peer resolver still maps a request line to the same
// byte offset in the peer's window; pipes of DIFFERENT types never exchange group
// requests (they cannot be members of one collective).
inline constexpr uint32_t kGroupMailboxOffset = kRecordOffset + kRecordBytes;

template <int GroupMax>
inline constexpr uint32_t kGroupMailboxBytes()
{
    return 2U * static_cast<uint32_t>(GroupMax) * grid_mock::kScbLineStride; // request queue + response queue
}

// Ring region base.  Kept as the GroupMax == 0 value so unicast-only layouts read
// exactly as before; a collective pipe adds its mailbox bytes (kGroupMailboxBytes).
inline constexpr uint32_t kSlotRegionOffset = kGroupMailboxOffset;

template <int GroupMax>
inline constexpr uint32_t kSlotRegionOffsetOf()
{
    return kSlotRegionOffset + kGroupMailboxBytes<GroupMax>();
}

inline constexpr uint32_t kReadyScbOffset(int chan) { return static_cast<uint32_t>(chan) * grid_mock::kScbLineStride; }

inline constexpr uint32_t kFreeScbOffset(int chan)
{
    return (static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(chan)) * grid_mock::kScbLineStride;
}

inline constexpr uint32_t kCloseScbOffset(int chan)
{
    return (2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(chan)) * grid_mock::kScbLineStride;
}

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kSlotRegionBytes()
{
    return static_cast<uint32_t>(ChanCount) * SlotCount * SlotStride;
}

template <int SlotStride, int SlotCount, int GroupMax = 0, int ChanCount = kGridChanCount>
inline constexpr uint32_t kProducerRegionOffset()
{
    return kSlotRegionOffsetOf<GroupMax>() + kSlotRegionBytes<SlotStride, SlotCount, ChanCount>();
}

template <int SlotStride, int SlotCount, int GroupMax = 0, int ChanCount = kGridChanCount>
inline constexpr uint32_t kWindowBytes()
{
    return kProducerRegionOffset<SlotStride, SlotCount, GroupMax, ChanCount>() +
           static_cast<uint32_t>(SlotStride); // isolated local producer staging slot
}

template <int SlotStride, int SlotCount, int GroupMax = 0>
inline constexpr uint32_t kChanSlotRegionOffset(int chan)
{
    return kSlotRegionOffsetOf<GroupMax>() + static_cast<uint32_t>(chan) * SlotCount * SlotStride;
}

// Wire up a GridPipe instance from a flat GM window owned by this rank.
// The host launcher allocates WindowBytes<Pipe>() bytes per rank, then calls
// this in the kernel prologue.  `runtimeCtx` is the HCCL device context handle
// used later by GridTPush/GridTPop/GridTBroadcast to resolve cross-rank
// addresses.
//
// It wires resources and ADOPTS the window's pipe record.  No channel is bound
// here: which producer each element serves is a runtime decision the kernel makes
// dynamically by the first TPUSH/TPOP identity handshake.
//
// Adopting rather than clearing is what makes a multi-launch schedule work: the
// scoreboards and rings in this window outlive the kernel, so the allocator's
// memory of which of them are already dirty has to as well.  A window the host has
// just memset reads back as "nothing bound" on its own.
//
// The offsets use the constexpr VARIABLES above plus plain arithmetic (CCE forbids
// calling a host constexpr *function* from an AICORE context, so the kXxxOffset()
// helpers are not called here even though they are constexpr).
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(
    Pipe& pipe, GridShape shape, GridCoord coord, __gm__ uint8_t* window, __gm__ void* runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    // Scoreboards exist for all kGridChanCount channels (the header is fixed);
    // only the RINGS are trimmed to Pipe::ChanCount.  All three scoreboards of a
    // channel are a whole cache line apart, so step by kScbLineStrideU32 in u32 units.
    // Group mailbox bytes shift the ring region: it is the only pipe-dependent
    // region ahead of it.  Computed inline from the constexpr VARIABLES plus the
    // pipe's static members (CCE forbids calling a host constexpr function from an
    // AICORE context, so kSlotRegionOffsetOf<>() is not called here).
    const uint32_t groupMailboxBytes = 2U * static_cast<uint32_t>(Pipe::GroupMax) * grid_mock::kScbLineStride;
    const uint32_t slotRegionOff = kSlotRegionOffset + groupMailboxBytes;

    __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
    for (int c = 0; c < kGridChanCount; ++c) {
        const uint32_t readyWord = static_cast<uint32_t>(c) * grid_mock::kScbLineStrideU32;
        const uint32_t freeWord =
            (static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(c)) * grid_mock::kScbLineStrideU32;
        const uint32_t closeWord =
            (2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(c)) * grid_mock::kScbLineStrideU32;
        pipe.readyScb[c] = scbs + readyWord;
        pipe.freeScb[c] = scbs + freeWord;
        pipe.closeScb[c] = scbs + closeWord;
        if (c < Pipe::ChanCount) {
            pipe.slotBase[c] = window + slotRegionOff + c * Pipe::SlotCount * Pipe::SlotStride;
        } else {
            pipe.slotBase[c] = nullptr; // no ring allocated for this channel
        }
        pipe.pushWindow[c] = GridPayloadWindow{};
        pipe.popWindow[c] = GridPayloadWindow{};
    }
    pipe.consHistFull = false;
    // Binding table + consumer history come from the window, not from zero.
    pipe.LoadRecord(scbs + kRecordOffset / sizeof(uint32_t));
    pipe.bindRequestQueue = reinterpret_cast<__gm__ uint32_t*>(window + kBindRequestOffset);
    pipe.bindResponseQueue = reinterpret_cast<__gm__ uint32_t*>(window + kBindResponseOffset);
    pipe.groupEpochWord = reinterpret_cast<__gm__ uint32_t*>(window + kGroupEpochOffset);
    // Do not clear the request queue here.  A producer in an earlier hardware wave
    // may already have deposited a request in this not-yet-scheduled consumer's
    // window.  A zero word is "empty" and the +1 encoding keeps block id 0 and
    // channel 0 distinguishable from it; the consumer clears each slot itself,
    // just before it answers.
    //
    // The group tables ARE cleared: a collective's grants are one-shot and its
    // members re-request every round, so nothing in them is meant to outlive a
    // launch (see GridPipe's group bookkeeping).
    pipe.ResetGroupState();
    pipe.bcastWindow = GridPayloadWindow{};

    // Group mailbox.  Like the unicast one it is NOT cleared here: a publisher in
    // an earlier hardware wave may already have deposited a request in this
    // not-yet-scheduled receiver's window, and the +1 id/channel encoding keeps a
    // zero commit word meaning "empty".  Null for a pipe with no collective, which
    // then pays no window bytes for it either.
    if constexpr (Pipe::GroupMax > 0) {
        pipe.groupRequestQueue = reinterpret_cast<__gm__ uint32_t*>(window + kGroupMailboxOffset);
        pipe.groupResponseQueue = reinterpret_cast<__gm__ uint32_t*>(
            window + kGroupMailboxOffset + static_cast<uint32_t>(Pipe::GroupMax) * grid_mock::kScbLineStride);
    }

    const uint32_t slotRegionBytes = static_cast<uint32_t>(Pipe::ChanCount) * static_cast<uint32_t>(Pipe::SlotCount) *
                                     static_cast<uint32_t>(Pipe::SlotStride);
    const uint32_t producerOff = slotRegionOff + slotRegionBytes;

    // One synchronous outbound transfer uses this slot at a time.  It is appended
    // after all receive-side rings/lanes so a producer can never alias a payload
    // that this same cell is concurrently waiting to consume.
    pipe.producerSlotBase = window + producerOff;
}

// ===========================================================================
// GROUP-COLLECTIVE CHANNEL MODEL (shared by TBROADCAST and TREDUCE).
//
// The scoreboard file is fixed (kGridChanCount channels), so a group of K members
// is always wider than the pool of doorbell channels once K grows.  The collective
// therefore RESERVES channels [0, kGridBcastChanCount) and multiplexes every
// publisher through them with a TICKET:
//
//   1. THE ADDRESS COMES FROM THE CALLER.  A publisher writes ring slot
//      `basek % SlotCount` of channel 0, where basek is the global sequence number
//      the caller passed to TBROADCAST.  No identity enters the address, so the
//      receiver's ring is sized by ITS OWN SRAM rather than by the number of
//      writers (2026-08-13 分析, 判据 M2/M3), and the offset is still the same in
//      every receiver's window, so the fan-out stays one copy_l1_to_group (M4).
//   2. THE GRANT IS THE WRITE PERMISSION.  A receiver only grants a slot whose
//      previous tenant its own caller has drained, so nothing can be overwritten
//      while it is live -- including when that tenant belonged to a DIFFERENT
//      publisher, the case a per-publisher credit counter is blind to.
//   3. GRANT ORDER IS GLOBAL AND UNCOMMUNICATED.  Every receiver grants only the
//      window [grantHead, grantHead + n) of the dense basek sequence, so all of
//      them serve the same publishers in the same order without exchanging a
//      word.  That is what makes "hold some grants while waiting for the rest"
//      deadlock-free (proof in GridTBroadcast.hpp).
//   4. CREDIT IS STILL AN ATOMIC ADD: every receiver adds 1 to the publisher's
//      free_scb when it drains a tile, and the publisher needs baseline +
//      round*(K-1) before starting a new round.  Because a single receiver can
//      contribute AT MOST `round`, the sum reaching the threshold means EVERY
//      receiver has -- an exact "all of them" test rather than a
//      fast-receiver-masks-a-slow-one sum (选型文档 §9), and the only sound
//      threshold this counter admits.
//
// CALLER CONTRACT.  `basek` must be unique, increasing and DENSE per collective,
// counted from 0 within EACH KERNEL LAUNCH (basek = round*K + rank, or plain
// `round` for a single source).  Density is what point 3 derives the order from,
// and the per-launch origin is because the grant state resets with every other
// part of the group state.  When the group is wider than SlotCount the caller must
// also WAVE its publishers -- at most SlotCount of them in flight, draining
// between waves -- because a receiver cannot free a ring slot while its own caller
// is blocked inside TBROADCAST.  Violating any of this blocks (and times out on
// the spin bound); it does not corrupt.
//
// The grant state (grantHead, the slot mask, the per-member table) belongs to the
// PIPE, not to a group, so one pipe runs ONE collective per launch -- which is why
// the demos give each phase its own window even where the geometry would allow
// sharing.  A second collective on the same pipe would inherit the first one's
// grantHead and never see its restarted sequence.
// ===========================================================================

// Native IPC_SCB slot ids of channel c's scoreboards (V8 §3.3 G3: ready of
// channel c -> slot c, free -> kGridChanCount + c, close -> 2*kGridChanCount + c).
AICORE inline uint32_t GridReadyScbSlot(int chan) { return static_cast<uint32_t>(chan); }

AICORE inline uint32_t GridFreeScbSlot(int chan)
{
    return static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(chan);
}

// Resolved geometry of one group collective on this cell.  `ok` folds every
// runtime guard the static_asserts cannot cover -- the group size and this cell's
// rank are runtime values -- so a caller faults once and then reads straight.
// There is no ring geometry here any more: a publisher's slot comes from the
// caller's basek, and the doorbell channel from the ticket it is granted.
struct GridGroupPlan {
    int myRank = -1;
    int groupSize = 0;         // K
    pto::GridBlockRect rect{}; // the member set, as the group instructions name it
    bool ok = false;
};

template <pto::GridGroup Group, typename Pipe>
AICORE inline GridGroupPlan PlanGroup(Pipe& pipe)
{
    GridGroupPlan plan;
    plan.myRank = pto::RankInGroup(Group, pipe.coord, pipe.groupRect);
    plan.groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    if (plan.groupSize <= 0 || plan.groupSize > Pipe::GroupMax || plan.myRank < 0 || plan.myRank >= plan.groupSize) {
        return plan; // the group mailbox has no line for some member, or a rank outside the group
    }
    plan.rect = pto::GridBlockRectOfGroup(Group, pipe.coord, pipe.shape, pipe.groupRect);
    plan.ok = true;
    return plan;
}

// Host-side helper: total bytes per rank for a single GridPipe (the group mailbox
// included when the pipe opted into a collective).
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    return kWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::GroupMax, Pipe::ChanCount>();
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP
