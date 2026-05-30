/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TREDUCE_HPP
#define PTO_COMM_A5_TREDUCE_HPP

// Suppress the a2a3-side deferred-fail TREDUCE_CCU_IMPL stub when we are
// building for A5 — the real overload lives below. Without this guard the
// `Args&&...` stub competes with our typed signature in the overload set and
// can win under partial ordering on some compilers (observed on bisheng).
// Mirrors a5/TBroadCast.hpp / a5/TScatter.hpp / a5/TGather.hpp.
#define PTO_COMM_A5_TREDUCE_PROVIDED 1

// AIV path - shared with a2a3, forwarded to avoid code duplication.
#include "pto/comm/a2a3/TReduce.hpp"

// CCU path - A5-only implementation used by TREDUCE<CollEngine::CCU>.
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TREDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &recvTileData, ReduceOp op,
                                   const CcuTriggerContext &ctx, WaitEvents &...events)
{
    CcuStoreTriggerSelf(parallelGroup, accTileData, ctx, events...);
    (void)dstGlobalData;
    (void)recvTileData;
    (void)op;
}

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TREDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &pingTileData, TileData &pongTileData, ReduceOp op,
                                   const CcuTriggerContext &ctx, WaitEvents &...events)
{
    CcuStoreTriggerSelf(parallelGroup, accTileData, ctx, events...);
    (void)dstGlobalData;
    (void)pingTileData;
    (void)pongTileData;
    (void)op;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TREDUCE_HPP
