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
//   offset                                         contents
//   ----------------------------------------------------------------------
//   0                                              ready scoreboards [kGridDirectionCount] u32
//   4 * kGridDirectionCount                        free scoreboards [kGridDirectionCount] u32
//   8 * kGridDirectionCount                        reserved (fault sentinels, alignment, telemetry)
//   kSlotRegionOffset                              slot region for all directions
//     + dir * SlotCount * SlotBytes                slot ring for that direction
//   kSlotRegionOffset + 5*SlotCount*SlotBytes      TBROADCAST region (only if GroupMax > 0):
//     + 0                                            shared payload ring [BcastSlotCount * SlotBytes]
//     + BcastSlotCount*SlotBytes                     per-source ready lanes [GroupMax] u32 (variant B)
//     + GroupMax*4                                    per-source free  lanes [GroupMax] u32 (X sole writer)
//
// The unicast slot region is sized to (kGridDirectionCount * SlotCount * SlotBytes).
// The TBROADCAST region is appended only when GroupMax > 0; a unicast-only pipe
// (BcastSlotCount = GroupMax = 0) has no broadcast region and a byte-identical
// window to the pre-TBROADCAST layout.  Keep enough reserved words for
// GridTPush/GridTPop fault sentinels:
//   readyScb[dir] + kFaultFlagWordOffset
//   freeScb[dir]  + kFaultFlagWordOffset
inline constexpr uint32_t kFlagsBytes = 128;
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

inline constexpr uint32_t kReadyScbOffset(GridDirection d)
{
    return static_cast<uint32_t>(d) * sizeof(uint32_t);
}

inline constexpr uint32_t kFreeScbOffset(GridDirection d)
{
    return kGridDirectionCount * sizeof(uint32_t) + static_cast<uint32_t>(d) * sizeof(uint32_t);
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kSlotRegionBytes()
{
    return kGridDirectionCount * SlotCount * SlotBytes;
}

// TBROADCAST (scheme-②) region offsets/sizes.  No-ops (zero) when GroupMax == 0.
template <int SlotBytes, int BcastSlotCount>
inline constexpr uint32_t kBcastRingBytes()
{
    return static_cast<uint32_t>(BcastSlotCount) * static_cast<uint32_t>(SlotBytes);
}

template <int GroupMax>
inline constexpr uint32_t kBcastLaneBytes()
{
    return static_cast<uint32_t>(GroupMax) * sizeof(uint32_t);
}

template <int SlotBytes, int SlotCount, int BcastSlotCount, int GroupMax>
inline constexpr uint32_t kBcastRegionBytes()
{
    return kBcastRingBytes<SlotBytes, BcastSlotCount>() + // shared payload ring
           kBcastLaneBytes<GroupMax>() +                  // per-source ready lanes (variant B)
           kBcastLaneBytes<GroupMax>();                   // per-source free  lanes
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotBytes, SlotCount>();
}

template <int SlotBytes, int SlotCount, int BcastSlotCount, int GroupMax>
inline constexpr uint32_t kWindowBytesWithBcast()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotBytes, SlotCount>() +
           kBcastRegionBytes<SlotBytes, SlotCount, BcastSlotCount, GroupMax>();
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kDirSlotRegionOffset(GridDirection d)
{
    return kSlotRegionOffset + static_cast<uint32_t>(d) * SlotCount * SlotBytes;
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
AICORE inline void InitGridPipeFromWindow(Pipe &pipe, GridShape shape, GridCoord coord, __gm__ uint8_t *window,
                                          __gm__ void *runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    __gm__ uint32_t *scbs = reinterpret_cast<__gm__ uint32_t *>(window);
    for (int i = 0; i < kGridDirectionCount; ++i) {
        pipe.readyScb[i] = scbs + i;
        pipe.freeScb[i] = scbs + kGridDirectionCount + i;
        pipe.slotBase[i] = window + kSlotRegionOffset + i * Pipe::SlotCount * Pipe::SlotBytes;
        pipe.prodIndex[i] = 0;
        pipe.consIndex[i] = 0;
    }

    // TBROADCAST region (scheme-② 真·同时 MPSC).  Only wired when the pipe
    // opted in (GroupMax > 0); a unicast-only pipe leaves these null and pays
    // zero window bytes for broadcast.  Offsets are computed inline from the
    // constexpr variable + the pipe's static members (see the note above).
    if constexpr (Pipe::GroupMax > 0) {
        const uint32_t slotRegionBytes = static_cast<uint32_t>(kGridDirectionCount) *
                                         static_cast<uint32_t>(Pipe::SlotCount) *
                                         static_cast<uint32_t>(Pipe::SlotBytes);
        const uint32_t ringOff = kSlotRegionOffset + slotRegionBytes;
        const uint32_t readyOff =
            ringOff + static_cast<uint32_t>(Pipe::BcastSlotCount) * static_cast<uint32_t>(Pipe::SlotBytes);
        const uint32_t freeOff =
            readyOff + static_cast<uint32_t>(Pipe::GroupMax) * static_cast<uint32_t>(sizeof(uint32_t));
        pipe.bcastRingBase = window + ringOff;
        pipe.bcastReadyLanes = reinterpret_cast<__gm__ uint32_t *>(window + readyOff);
        pipe.bcastFreeLanes = reinterpret_cast<__gm__ uint32_t *>(window + freeOff);
    }
}

// Host-side helper: total bytes per rank for a single GridPipe (broadcast
// region included when the pipe opted in).
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    if constexpr (Pipe::GroupMax > 0) {
        return kWindowBytesWithBcast<Pipe::SlotBytes, Pipe::SlotCount, Pipe::BcastSlotCount, Pipe::GroupMax>();
    } else {
        return kWindowBytes<Pipe::SlotBytes, Pipe::SlotCount>();
    }
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP
