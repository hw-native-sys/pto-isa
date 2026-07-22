/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP
#define PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP

// PTO-native CKE trigger.
//
// The PTO CCU dispatch chain (TREDUCE<CollEngine::CCU>, TSCATTER<...>, ...)
// terminates here.  The implementation is a single scalar store into the
// CCU gate slot, which is mapped as device memory on this architecture so
// the write is globally visible without explicit cache maintenance.
//
// Self-contained: this header pulls in only <cstdint>.  No AscendC, no
// kernel_operator.h, no TPipe / DataCopy.  CCU test kernel translation
// units therefore do not need to drag the AscendC runtime into their
// compile graph, and HcclCcuKernelRegister still succeeds against the
// resulting .so (verified end-to-end on all four collective ST tests).

#include <cstdint>

namespace pto {
namespace comm {

// CKE gate slot encoding: the most-significant bit marks the entry as
// valid; the low bits carry the rank mask.  Producer (this) writes the
// bit set; CCU hardware clears it after the trigger fires.
static constexpr uint64_t kCkeValidBit = 1ULL << 63;

namespace detail {
PTO_INTERNAL void DcciCke(__gm__ void* ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
}
} // namespace detail

PTO_INTERNAL void CkeTrigger(uint64_t ckeSlotVA, uint32_t mask, __ubuf__ uint8_t* /* ubScratch */)
{
    volatile __gm__ uint64_t* p = reinterpret_cast<volatile __gm__ uint64_t*>(ckeSlotVA);
    uint64_t payload = static_cast<uint64_t>(mask) | kCkeValidBit;

    detail::DcciCke(reinterpret_cast<__gm__ void*>(ckeSlotVA));
    *p = payload;
    detail::DcciCke(reinterpret_cast<__gm__ void*>(ckeSlotVA));
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

// Convenience wrapper used by the T{REDUCE,SCATTER,BROADCAST,GATHER}_CCU_IMPL
// fan-outs in a5/T*.hpp.  Resolves the UB scratch pointer from a tile then
// forwards to CkeTrigger; collapses two lines (reinterpret_cast + trigger)
// at every call site into one.
template <typename TileData>
PTO_INTERNAL void CkeTriggerFromTile(uint64_t ckeSlotVA, uint32_t mask, TileData& tile)
{
    __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(tile.data());
    CkeTrigger(ckeSlotVA, mask, ub);
}

// Store tileData to parallelGroup[selfIdx] (if AivStored) then trigger CKE.
// Used by TREDUCE_CCU_IMPL and TGATHER_CCU_IMPL.
template <typename ParallelGroupType, typename TileData, typename... WaitEvents>
PTO_INTERNAL void CcuStoreTriggerSelf(
    ParallelGroupType& parallelGroup, TileData& tileData, const CcuTriggerContext& ctx, WaitEvents&... events)
{
    WaitAllEvents(events...);

    if (ctx.inputSource == CcuInputSource::AivStored) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(parallelGroup[static_cast<int>(ctx.selfIdx)], tileData);
        pipe_barrier(PIPE_MTE3);
    }

    CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, tileData);
}

// Root stores tileData to srcGlobalData (if AivStored) then trigger CKE.
// Used by TSCATTER_CCU_IMPL and TBROADCAST_CCU_IMPL.
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INTERNAL void CcuStoreTriggerRoot(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& tileData, const CcuTriggerContext& ctx,
    WaitEvents&... events)
{
    WaitAllEvents(events...);

    if (ctx.inputSource == CcuInputSource::AivStored) {
        if (static_cast<int>(ctx.selfIdx) == parallelGroup.GetRootIdx()) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(srcGlobalData, tileData);
            pipe_barrier(PIPE_MTE3);
        }
    }

    CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, tileData);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP
