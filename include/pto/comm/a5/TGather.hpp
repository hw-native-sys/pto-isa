/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TGATHER_HPP
#define PTO_COMM_A5_TGATHER_HPP

// Suppress the A2/A3 deferred-fail CCU stub: A5 provides the real
// implementation below.  Without this guard the stub would join the real
// overload set in `pto::comm` and the generic `Args&&...` pack would
// out-rank the real `T&` parameters under partial ordering on some
// compilers.
#define PTO_COMM_A5_TGATHER_PROVIDED 1

// AIV path - shared with a2a3, forwarded to avoid code duplication.
#include "pto/comm/a2a3/TGather.hpp"

// CCU path - A5-only implementation used by TGATHER<CollEngine::CCU>.
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

// Leading `CollEngine` placeholder mirrors the TPUT_ASYNC_IMPL<engine> pattern:
// callers write `TGATHER_CCU_IMPL<engine>(...)`, which turns the qualified
// template-id into a dependent name so the discarded `if constexpr (engine ==
// CCU)` branch in pto_comm_inst.hpp performs no lookup on non-A5 builds.
template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TGATHER_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &stagingTileData, const CcuTriggerContext &ctx, WaitEvents &...events)
{
    CcuStoreTriggerSelf(parallelGroup, stagingTileData, ctx, events...);
    (void)dstGlobalData;
}

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TGATHER_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &pingTile,
                                   TileData &pongTile, const CcuTriggerContext &ctx, WaitEvents &...events)
{
    CcuStoreTriggerSelf(parallelGroup, pingTile, ctx, events...);
    (void)dstGlobalData;
    (void)pongTile;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TGATHER_HPP
