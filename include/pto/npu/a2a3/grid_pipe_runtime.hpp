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
// rank resolution.  See design doc section 5 and the mock SPR lowering in
// include/pto/npu/a2a3/grid_intrinsic.hpp.

#ifndef PTO_A2A3_GRID_PIPE_RUNTIME_HPP
#define PTO_A2A3_GRID_PIPE_RUNTIME_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

namespace pto {
namespace a2a3_grid {

// shmem window layout (per rank), in bytes:
//
//   offset                                         contents
//   ----------------------------------------------------------------------
//   0                                              ready flags [kGridDirectionCount] u32
//   4 * kGridDirectionCount                        free flags [kGridDirectionCount] u32
//   8 * kGridDirectionCount                        reserved (fault sentinels, alignment, telemetry)
//   kSlotRegionOffset                              slot region for all directions
//     + dir * SlotCount * SlotBytes                slot ring for that direction
//
// The slot region is sized to (kGridDirectionCount * SlotCount * SlotBytes).
// Keep enough reserved words for GridTPush/GridTPop fault sentinels:
//   readyFlags[dir] + kFaultFlagWordOffset
//   freeFlags[dir]  + kFaultFlagWordOffset
inline constexpr uint32_t kFlagsBytes = 128;
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

inline constexpr uint32_t kReadyFlagOffset(GridDirection d)
{
    return static_cast<uint32_t>(d) * sizeof(uint32_t);
}

inline constexpr uint32_t kFreeFlagOffset(GridDirection d)
{
    return kGridDirectionCount * sizeof(uint32_t) + static_cast<uint32_t>(d) * sizeof(uint32_t);
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kSlotRegionBytes()
{
    return kGridDirectionCount * SlotCount * SlotBytes;
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotBytes, SlotCount>();
}

template <int SlotBytes, int SlotCount>
inline constexpr uint32_t kDirSlotRegionOffset(GridDirection d)
{
    return kSlotRegionOffset + static_cast<uint32_t>(d) * SlotCount * SlotBytes;
}

// Wire up a GridPipe instance from a flat GM window owned by this rank.
// The host launcher allocates kWindowBytes() bytes per rank, then calls this
// in the kernel prologue.  `runtimeCtx` is the HCCL device context handle used
// later by GridTPush/GridTPop to resolve cross-rank addresses.
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(Pipe &pipe, GridShape shape, GridCoord coord, __gm__ uint8_t *window,
                                          __gm__ void *runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    __gm__ uint32_t *flags = reinterpret_cast<__gm__ uint32_t *>(window);
    for (int i = 0; i < kGridDirectionCount; ++i) {
        pipe.readyFlags[i] = flags + i;
        pipe.freeFlags[i] = flags + kGridDirectionCount + i;
        pipe.slotBase[i] = window + kSlotRegionOffset + i * Pipe::SlotCount * Pipe::SlotBytes;
        pipe.prodIndex[i] = 0;
        pipe.consIndex[i] = 0;
    }
}

// Host-side helper: total bytes per rank for a single GridPipe.
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    return kWindowBytes<Pipe::SlotBytes, Pipe::SlotCount>();
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP
