/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 GridPipe runtime helpers: shmem window layout, init helpers, neighbor
// rank resolution.  See the V6 IPC_SCB scoreboard design and its A2/A3 mock in
// include/pto/npu/a2a3/grid_intrinsic.hpp.

#ifndef PTO_A2A3_GRID_PIPE_RUNTIME_HPP
#define PTO_A2A3_GRID_PIPE_RUNTIME_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

namespace pto {
namespace a2a3_grid {

// shmem window layout (per rank), in bytes.  The ready/free scoreboard words
// stand in for the V6 ready_scb_<dir> / free_scb_<dir> IPC_SCB slots (each
// carries a monotone absolute count written by the peer's HSCB store):
//
//   offset                                         contents            written by
//   ----------------------------------------------------------------------------
//   edge * 64                                      ready scb <edge> u32 upstream nbr
//   (4 + edge) * 64                                free  scb <edge> u32 downstream nbr
//   512 .. kFlagsBytes-1                           reserved (alignment, telemetry)
//   kSlotRegionOffset (1024)                       slot region for the ALLOCATED directions
//     + ring(dir) * SlotCount * SlotStride         slot ring for that direction
//
// TEN scoreboards, full stop.  The group collectives add none: they notify
// through these same per-direction ones, attributing each (producer, consumer)
// edge to a direction with GroupFlowDirection (grid_intrinsic.hpp).  So the
// header is fixed-size regardless of group size -- which is the whole point,
// since the per-source ready/free LANE ARRAYS this replaced were sized by the
// group and appended after the ring.  The broadcast region is now exactly the
// shared ring.
//
// EVERY SCOREBOARD OWNS A FULL CACHE LINE (grid_mock::kScbLineStride).  Each of
// the eight has a DIFFERENT external writer -- ready_scb[edge] comes from the
// neighbor upstream along that edge, free_scb[edge] from the one downstream --
// and the mock's write-back is line-granular, so packing them at 4 B let one
// peer's store put back a stale copy of another peer's doorbell.  See the
// rationale on kScbLineStride in grid_intrinsic.hpp.  Eight lines = 512 B live.
//
// R = GridDirRingCount(DirMask) and ring(dir) = GridDirRingIndex(DirMask, dir):
// only the directions a pipe actually pushes/pops get a ring, and their rings are
// packed from offset 0.  The default DirMask (kGridDirAll) gives R = 5 and
// ring(dir) == GridDirectionIndex(dir).  A pure-broadcast pipe (DirMask =
// kGridDirNone) has R = 0 and pays no unicast bytes at all.
//
// There is no broadcast-specific region: a group collective rides the SAME
// per-direction rings and scoreboards a TPUSH does, so a group pipe just names
// the directions it spans in DirMask.  Each scoreboard's
// fault sentinel sits kFaultFlagWordOffset u32 words INTO ITS OWN line, so the
// sentinels stay inside the reserved header and clear of every live word:
//   readyScb[dir] + kFaultFlagWordOffset
//   freeScb[dir]  + kFaultFlagWordOffset
inline constexpr uint32_t kHeaderLineCount = kGridScbCount; // 4 ready + 4 free (one pair per mesh edge)

// Flag header: kHeaderLineCount cache lines plus reserved headroom, rounded up so
// the slot ring stays generously aligned.  Host launchers mirror this constant
// (the *_GRID_FLAGS_BYTES in the demo configs).
inline constexpr uint32_t kFlagsBytes = 1024;
static_assert(
    kFlagsBytes >= kHeaderLineCount * grid_mock::kScbLineStride, "flag header must hold one line per scoreboard");
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

// Scoreboard positions, as u32 word indices / byte offsets into the window.  The
// stride is one cache line, NOT one word -- that is the correctness requirement,
// not padding.  Indexed by mesh EDGE, so SOURCE has no entry.
//
// These are host-callable constexpr functions, so the edge index is spelled out
// as `static_cast<int>(d) - 1` rather than calling GridEdgeIndex(): that helper
// is [aicore]-qualified and a host context cannot call it (same reason
// GridDirBit and friends in grid_intrinsic.hpp avoid GridDirectionIndex).
inline constexpr uint32_t kReadyScbWord(GridDirection d)
{
    return static_cast<uint32_t>(static_cast<int>(d) - 1) * grid_mock::kScbLineStrideU32;
}

inline constexpr uint32_t kFreeScbWord(GridDirection d)
{
    return (kGridEdgeCount + static_cast<uint32_t>(static_cast<int>(d) - 1)) * grid_mock::kScbLineStrideU32;
}

inline constexpr uint32_t kReadyScbOffset(GridDirection d) { return kReadyScbWord(d) * sizeof(uint32_t); }

inline constexpr uint32_t kFreeScbOffset(GridDirection d) { return kFreeScbWord(d) * sizeof(uint32_t); }

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kSlotRegionBytes()
{
    return static_cast<uint32_t>(GridDirRingCount(DirMask)) * SlotCount * SlotStride;
}

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount, DirMask>();
}

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kDirSlotRegionOffset(GridDirection d)
{
    return kSlotRegionOffset + static_cast<uint32_t>(GridDirRingIndex(DirMask, d)) * SlotCount * SlotStride;
}

// Wire up a GridPipe instance from a flat GM window owned by this rank.
// The host launcher allocates WindowBytes<Pipe>() bytes per rank, then calls
// this in the kernel prologue.  `runtimeCtx` is the HCCL device context handle
// used later by GridTPush/GridTPop/GridTBroadcast to resolve cross-rank
// addresses.
//
// The unicast offsets use the constexpr variable kSlotRegionOffset + plain
// arithmetic (CCE forbids calling a host constexpr *function* from an AICORE
// context, so we do not call the kXxxOffset() helpers here even though they are
// constexpr -- only the variable + the pipe's static members are needed).
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(
    Pipe& pipe, GridShape shape, GridCoord coord, __gm__ uint8_t* window, __gm__ void* runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    // Scoreboards stay indexed by direction (all 5 always exist, each on its own
    // cache line -- see the layout comment); only the slot RINGS are packed by
    // DirMask.  `ring` walks the allocated directions in order, so ring(dir)
    // matches GridDirRingIndex(DirMask, dir) without calling it from this AICORE
    // context.  The line stride is likewise spelled out rather than calling
    // kReadyScbWord() / kFreeScbWord(), which are host constexpr functions.
    __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
    const int lineU32 = static_cast<int>(grid_mock::kScbLineStrideU32);
    // Scoreboards: one pair per mesh EDGE (4), not per direction -- SOURCE is a
    // runtime-gated queue and has none.
    for (int e = 0; e < kGridEdgeCount; ++e) {
        pipe.readyScb[e] = scbs + e * lineU32;
        pipe.freeScb[e] = scbs + (kGridEdgeCount + e) * lineU32;
    }
    // Rings: still per DIRECTION, because DirMask is defined over GridDirection.
    int ring = 0;
    for (int i = 0; i < kGridDirectionCount; ++i) {
        if (((Pipe::DirMask >> i) & 1) != 0) {
            pipe.slotBase[i] = window + kSlotRegionOffset + ring * Pipe::SlotCount * Pipe::SlotStride;
            ++ring;
        } else {
            pipe.slotBase[i] = nullptr; // no ring allocated for this direction
        }
        pipe.prodIndex[i] = 0;
        pipe.consIndex[i] = 0;
        pipe.pushWindow[i] = GridPayloadWindow{};
        pipe.popWindow[i] = GridPayloadWindow{};
    }
    pipe.bcastWindow = GridPayloadWindow{};

    // Nothing else to wire: the group collectives ride the rings and scoreboards
    // filled above.
}

// Host-side helper: total bytes per rank for a single GridPipe.
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    return kWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::DirMask>();
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP
