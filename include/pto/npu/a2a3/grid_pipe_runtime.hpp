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
//   kScbHeaderBytes                                 C bind-request L1 lines
//   kBindResponseOffset                             C bind-response-data lines
//   kBindResponseCompleteOffset                     C structured-bind response-commit lines
//   kBindRequestQueueOffset                         64 dynamic-bind request entries
//   kBindResponseQueueOffset                        64 dynamic-bind response entries
//   kRecordOffset                                   pipe record: bindings, consumer
//                                                     FSM/history, both channel maps,
//                                                     close bases and run counters
//   kSlotRegionOffset                               payload region, ChanCount rings
//     + c * SlotCount * SlotStride                    ring of channel c
//   end of channel rings                            producer staging [SlotStride]
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
//     trimmed by ChanCount.  TPUSH, TBROADCAST, and group TREDUCE all address
//     the same rings; there is no appended per-source collective region.
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
// Fault sentinels live at word kFaultFlagWordOffset of the scoreboard they
// belong to, which is inside that scoreboard's own line and clear of every live
// word.
inline constexpr uint32_t kScbHeaderBytes = 3U * static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;

// Three arrays of C remotely-written structured-collective control lines.  Request = [producer-id
// commit, producer channel, requested consumer channel, mode]; response data =
// [ready baseline, consumer channel]; response completion is isolated because a
// broadcast bind has multiple external completion writers.
inline constexpr uint32_t kBindRequestOffset = kScbHeaderBytes;
inline constexpr uint32_t kBindRequestBytes = static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;
inline constexpr uint32_t kBindResponseOffset = kBindRequestOffset + kBindRequestBytes;
inline constexpr uint32_t kBindResponseBytes = static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;
inline constexpr uint32_t kBindResponseCompleteOffset = kBindResponseOffset + kBindResponseBytes;
inline constexpr uint32_t kBindResponseCompleteBytes =
    static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;

// Dynamic TPUSH/TPOP bind queues.  Request entry p is written only by logical
// producer p; response entry c is written only by logical consumer c.  One full
// line per entry is required because different entries have different external
// writers and the A3 mock commits a whole cache line on every remote store.
inline constexpr uint32_t kBindRequestQueueOffset = kBindResponseCompleteOffset + kBindResponseCompleteBytes;
inline constexpr uint32_t kBindRequestQueueBytes =
    static_cast<uint32_t>(kGridBindQueueDepth * kGridBindQueueEntryBytes);
inline constexpr uint32_t kBindResponseQueueOffset = kBindRequestQueueOffset + kBindRequestQueueBytes;
inline constexpr uint32_t kBindResponseQueueBytes =
    static_cast<uint32_t>(kGridBindQueueDepth * kGridBindQueueEntryBytes);
inline constexpr uint32_t kControlBytes = kBindRequestBytes + kBindResponseBytes + kBindResponseCompleteBytes +
                                          kBindRequestQueueBytes + kBindResponseQueueBytes;
static_assert(
    kGridBindQueueEntryBytes == static_cast<int>(grid_mock::kScbLineStride),
    "GridPipe bind-queue entries must remain isolated cache lines");

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
inline constexpr uint32_t kSlotRegionOffset = kRecordOffset + kRecordBytes;

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

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kProducerRegionOffset()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount, ChanCount>();
}

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kWindowBytes()
{
    return kProducerRegionOffset<SlotStride, SlotCount, ChanCount>() +
           static_cast<uint32_t>(SlotStride); // isolated local producer staging slot
}

template <int SlotStride, int SlotCount>
inline constexpr uint32_t kChanSlotRegionOffset(int chan)
{
    return kSlotRegionOffset + static_cast<uint32_t>(chan) * SlotCount * SlotStride;
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
            pipe.slotBase[c] = window + kSlotRegionOffset + c * Pipe::SlotCount * Pipe::SlotStride;
        } else {
            pipe.slotBase[c] = nullptr; // no ring allocated for this channel
        }
        pipe.pushWindow[c] = GridPayloadWindow{};
        pipe.popWindow[c] = GridPayloadWindow{};
    }
    pipe.consHistFull = false;
    // Binding table + consumer history come from the window, not from zero.
    pipe.LoadRecord(scbs + kRecordOffset / sizeof(uint32_t));
    for (int c = 0; c < kGridChanCount; ++c) {
        __gm__ uint32_t* request = reinterpret_cast<__gm__ uint32_t*>(
            window + kBindRequestOffset + static_cast<uint32_t>(c) * grid_mock::kScbLineStride);
        __gm__ uint32_t* response = reinterpret_cast<__gm__ uint32_t*>(
            window + kBindResponseOffset + static_cast<uint32_t>(c) * grid_mock::kScbLineStride);
        __gm__ uint32_t* complete = reinterpret_cast<__gm__ uint32_t*>(
            window + kBindResponseCompleteOffset + static_cast<uint32_t>(c) * grid_mock::kScbLineStride);
        pipe.bindRequestProdIdL1[c] = request;
        pipe.bindRequestProdChanL1[c] = request + 1;
        pipe.bindRequestConsChanL1[c] = request + 2;
        pipe.bindRequestModeL1[c] = request + 3;
        pipe.bindResponseReadyL1[c] = response;
        pipe.bindResponseConsChanL1[c] = response + 1;
        pipe.bindResponseCompleteL1[c] = complete;
    }
    pipe.bindRequestQueueBaseL1 = reinterpret_cast<__gm__ uint32_t*>(window + kBindRequestQueueOffset);
    pipe.bindResponseQueueBaseL1 = reinterpret_cast<__gm__ uint32_t*>(window + kBindResponseQueueOffset);
    pipe.bindRequestScanStart = 0;
    // Do not clear either bind protocol here.  A producer in an earlier hardware
    // wave may already have deposited a structured request or a dynamic queue
    // entry.  Host-zeroed commit words arm both protocols, and the consumer clears
    // a request before publishing its response.
    const uint32_t slotRegionBytes = static_cast<uint32_t>(Pipe::ChanCount) * static_cast<uint32_t>(Pipe::SlotCount) *
                                     static_cast<uint32_t>(Pipe::SlotStride);
    const uint32_t producerOff = kSlotRegionOffset + slotRegionBytes;

    // One synchronous outbound transfer uses this slot at a time.  It is appended
    // after all receive-side rings so a producer can never alias a payload
    // that this same cell is concurrently waiting to consume.
    pipe.producerSlotBase = window + producerOff;
}

// Host-side helper: total bytes per rank for a single GridPipe.  All operations
// use the same channel rings, so no operation-specific payload region is added.
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    return kWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::ChanCount>();
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP
