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

template <typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<!is_global_data_v<TileCons>, int> = 0>
PTO_INTERNAL void TPOP_IMPL(Pipe& pipe, TileCons& tile)
{
    constexpr bool participateNoSplitC2V = cpu_pipe::ShouldNoSplitC2VConsumerLaneParticipate<Pipe, TileCons, Split>();
    // The second vector lane pops only when the dual-lane C2V protocol is active (>=2 vector
    // subblocks). With a single subblock it takes the inactive-lane path (zero-filled tile, no
    // pop), so the pipe runs as a single consumer.
    if ((!participateNoSplitC2V || !cpu_pipe::IsDualLaneC2VActive()) &&
        cpu_pipe::IsInactiveNoSplitVecLane<TileCons, Split>()) {
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
    if constexpr (participateNoSplitC2V) {
        if (get_subblockid() != 0) {
            cpu_pipe::FillTile(tile, typename TileCons::DType{});
        }
    }
}

template <
    typename TileCons, typename Pipe,
    std::enable_if_t<(is_tile_data_v<TileCons> || is_conv_tile_v<TileCons>) && !is_global_data_v<TileCons>, int> = 0>
PTO_INTERNAL void TPOP_REVERSED_IMPL(TileCons& tile, Pipe& pipe)
{
    TPOP_IMPL<Pipe, TileCons, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INTERNAL void TPOP_GLOBAL_IMPL(Pipe& pipe, GlobalData& gmTensor)
{
    if (pipe.cons.getWaitStatus()) {
        pipe.cons.template wait<GlobalData, Split>();
    }
    const std::size_t slotIndex = static_cast<std::size_t>(pipe.cons.getTileId() % Pipe::RingFiFo::SLOT_NUM);
    const std::size_t entryBase =
        slotIndex * Pipe::RingFiFo::SLOT_SIZE + static_cast<std::size_t>(pipe.cons.entryOffset);
    auto* addr = reinterpret_cast<typename GlobalData::DType*>(
        reinterpret_cast<std::uintptr_t>(pipe.fifo.GM_SLOT_BUFFER) + entryBase);
    TASSIGN_IMPL(gmTensor, addr);
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe& pipe)
{
    if ((!cpu_pipe::ShouldNoSplitC2VConsumerLaneFree<Pipe, Split>() || !cpu_pipe::IsDualLaneC2VActive()) &&
        cpu_pipe::IsInactiveNoSplitVecConsumerLane<Pipe, Split>()) {
        return;
    }
    if (pipe.cons.getFreeStatus()) {
        pipe.cons.template free<Split>();
    }
}

template <typename Pipe>
PTO_INTERNAL void TFREE_IMPL(Pipe& pipe)
{
    TFREE_IMPL<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INTERNAL void TFREE_GLOBAL_IMPL(Pipe& pipe, GlobalData&)
{
    if ((!cpu_pipe::ShouldNoSplitC2VConsumerLaneFree<Pipe, Split>() || !cpu_pipe::IsDualLaneC2VActive()) &&
        cpu_pipe::IsInactiveNoSplitVecConsumerLane<Pipe, Split>()) {
        return;
    }
    if (pipe.cons.getFreeStatus()) {
        pipe.cons.template free<Split>();
    }
}

} // namespace pto

#endif
