/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
#define DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "const_args.hpp"

AICORE inline volatile __gm__ int32_t* FixedSyncSlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t slot)
{
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    return reinterpret_cast<volatile __gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(slot) * kMegaMoeFixedSyncSlotBytes);
}

// The doorbell is a scalar GM store. The caller drains payload pipelines before publishing.
AICORE inline void PublishScalarEpoch(volatile __gm__ int32_t* slot, int32_t epoch)
{
    *slot = epoch;
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

AICORE inline void ResetFixedSyncWorkspace(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    for (uint32_t slotId = 0U; slotId < kMegaMoeFixedSyncSlotCount; ++slotId) {
        volatile __gm__ int32_t* slot = FixedSyncSlot(workspaceGM, tilingData, slotId);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        *slot = 0;
    }
    for (uint32_t slotId = 0U; slotId < kMegaMoeFixedSyncSlotCount; ++slotId) {
        dcci((__gm__ void*)FixedSyncSlot(workspaceGM, tilingData, slotId), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

AICORE inline int32_t ReadScalarEpoch(volatile __gm__ int32_t* slot)
{
    dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *slot;
}

AICORE inline void EpochPollBackoff()
{
    constexpr uint32_t kDelayTicks = 3U;
    const uint64_t deadline = get_sys_cnt() + kDelayTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline int32_t WaitEpochRaw(volatile __gm__ int32_t* slot, int32_t epoch)
{
    while (true) {
        const int32_t observed = ReadScalarEpoch(slot);
        if (observed >= epoch) {
            return observed;
        }
        EpochPollBackoff();
    }
}

AICORE inline int32_t WaitEpochAcquire(volatile __gm__ int32_t* slot, int32_t epoch)
{
    const int32_t observed = WaitEpochRaw(slot, epoch);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return observed;
}

AICORE inline void PublishGmm2EntryReady(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    PublishScalarEpoch(
        FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot), kMegaMoeFixedGmm2EntryReadyMarker);
}

AICORE inline void WaitGmm2EntryReady(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    WaitEpochAcquire(
        FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot), kMegaMoeFixedGmm2EntryReadyMarker);
}

AICORE inline bool Gmm2EntryReady(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    return ReadScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, kMegaMoeFixedSyncGmm2EntryReadySlot)) >=
           kMegaMoeFixedGmm2EntryReadyMarker;
}

#if defined(__DAV_VEC__)
// These snapshot helpers require AIV UB and MTE2/MTE3; AIC callers publish scalar arrivals only.
constexpr uint32_t kMegaMoeSyncSnapshotValuesPerSlot = kMegaMoeFixedSyncSlotBytes / sizeof(int32_t);
constexpr uint32_t kMegaMoeSyncSnapshotMaxValues = kMegaMoeFixedPhysicalAicNum * kMegaMoeSyncSnapshotValuesPerSlot;
constexpr uint64_t kMegaMoeSyncSnapshotUbBytes = static_cast<uint64_t>(kMegaMoeSyncSnapshotMaxValues) * sizeof(int32_t);
constexpr uint64_t kMegaMoeSyncSnapshotUbOffset = AtlasA5::UB_SIZE - kMegaMoeSyncSnapshotUbBytes;
constexpr event_t kMegaMoeSyncSnapshotEvent = EVENT_ID0;

static_assert(kMegaMoeFixedSyncSlotBytes % sizeof(int32_t) == 0U);
static_assert(kMegaMoeSyncSnapshotUbBytes <= A5_UB_SYNC_RESERVE_BYTES);
static_assert(kMegaMoeSyncSnapshotUbOffset >= A5_MAIN_UB_SIZE);
static_assert(kMegaMoeSyncSnapshotUbOffset % UB_ALIGN == 0U);

AICORE inline int32_t ReadArrivalMinMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t producerCount)
{
    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t* arrivalBase = reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(arrivalBaseSlot) * kMegaMoeFixedSyncSlotBytes);
    const uint32_t snapshotValues = producerCount * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(arrivalBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    pto::TLOAD(snapshotTile, snapshotGlobal);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);

    int32_t observedMin = snapshotTile.GetValue(0U);
    for (uint32_t peer = 1U; peer < producerCount; ++peer) {
        const int32_t observed = snapshotTile.GetValue(peer * kMegaMoeSyncSnapshotValuesPerSlot);
        observedMin = observed < observedMin ? observed : observedMin;
    }
    return observedMin;
}

AICORE inline int32_t WaitArrivalMinMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t producerCount,
    int32_t epoch)
{
    while (true) {
        const int32_t observedMin = ReadArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount);
        if (observedMin >= epoch) {
            return observedMin;
        }
        EpochPollBackoff();
    }
}

AICORE inline void PublishEpochRangeMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t baseSlot, uint32_t count, int32_t epoch)
{
    using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
    using SnapshotTile =
        pto::Tile<pto::TileType::Vec, int32_t, 1, kMegaMoeSyncSnapshotMaxValues, pto::BLayout::RowMajor, -1, -1>;

    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData->fixedGroupTiling;
    __gm__ int32_t* readyBase = reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + fixed.syncOffset + static_cast<uint64_t>(baseSlot) * kMegaMoeFixedSyncSlotBytes);
    const uint32_t snapshotValues = count * kMegaMoeSyncSnapshotValuesPerSlot;
    SnapshotShape snapshotShape(snapshotValues);
    SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
    SnapshotGlobal snapshotGlobal(readyBase, snapshotShape, snapshotStride);
    SnapshotTile snapshotTile(1U, snapshotValues);
    pto::TASSIGN(snapshotTile, kMegaMoeSyncSnapshotUbOffset);
    for (uint32_t localId = 0U; localId < count; ++localId) {
        snapshotTile.SetValue(localId * kMegaMoeSyncSnapshotValuesPerSlot, epoch);
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
    pto::TSTORE(snapshotGlobal, snapshotTile);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(kMegaMoeSyncSnapshotEvent, kMegaMoeSyncSnapshotEvent);
}
#endif

#if defined(__DAV_VEC__)
AICORE inline void NotifyGroupConsumersMte(
    GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t arrivalBaseSlot, uint32_t readyBaseSlot,
    uint32_t producerCount, uint32_t consumerCount, uint32_t localId, uint32_t coordinatorLocalId, uint32_t notifyCall)
{
    const int32_t epoch = static_cast<int32_t>(notifyCall + 1U);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    PublishScalarEpoch(FixedSyncSlot(workspaceGM, tilingData, arrivalBaseSlot + localId), epoch);

    if (localId == coordinatorLocalId) {
        WaitArrivalMinMte(workspaceGM, tilingData, arrivalBaseSlot, producerCount, epoch);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        PublishEpochRangeMte(workspaceGM, tilingData, readyBaseSlot, consumerCount, epoch);
    }
}
#endif

#endif // DISPATCH_MEGA_COMBINE_MEGA_EXPERT_SYNC_HPP
