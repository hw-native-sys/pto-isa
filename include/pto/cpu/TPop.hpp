/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOP_HPP
#define TPOP_HPP

#include <pto/cpu/TPush.hpp>

namespace pto {

template <typename Pipe, typename TileCons, TileSplitAxis Split>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, TileCons &tile)
{
    if (cpu_pipe::IsInactiveNoSplitVecLane<TileCons, Split>()) {
        cpu_pipe::EnsureTileStorage(tile);
        cpu_pipe::FillTile(tile, typename TileCons::DType{});
        return;
    }
    // 1. Wait for data
    if (pipe.cons.getWaitStatus()) {
        pipe.cons.template wait<TileCons, Split>();
    }

    // 2. Address calculation + TASSIGN + data transfer
    pipe.cons.template pop<TileCons, Split>(pipe.fifo, tile);
}

template <typename TileCons, typename Pipe>
PTO_INTERNAL void TPOP_IMPL(TileCons &tile, Pipe &pipe)
{
    TPOP_IMPL<Pipe, TileCons, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    if (cpu_pipe::IsInactiveNoSplitVecConsumerLane<Pipe, Split>()) {
        return;
    }
    if (pipe.cons.getFreeStatus()) {
        pipe.cons.template free<Split>();
    }
}

template <typename Pipe>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    TFREE_IMPL<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}

} // namespace pto

#endif
