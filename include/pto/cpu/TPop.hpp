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
    if (pipe.cons.getWaitStatus()) {
        pipe.cons.template wait<Split>();
    }

    const std::size_t slotIndex = static_cast<std::size_t>(pipe.cons.getTileId() % Pipe::RingFiFo::SLOT_NUM);
    const std::size_t entryBase =
        slotIndex * Pipe::RingFiFo::SLOT_SIZE + static_cast<std::size_t>(pipe.cons.entryOffset);

    if (pipe.fifo.GM_SLOT_BUFFER != nullptr) {
        using T = typename TileCons::DType;
        constexpr int rows = TileCons::Rows;
        constexpr int cols = TileCons::Cols;
        std::size_t subOffset = 0;
        if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
            subOffset = static_cast<std::size_t>(get_subblockid()) * rows * cols * sizeof(T);
        }
        using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
        auto *addr = reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(pipe.fifo.GM_SLOT_BUFFER) +
                                                  entryBase + subOffset);
        GlobalData globalData(addr);
        TLOAD_IMPL(tile, globalData);
    } else if constexpr (Pipe::is_c2v) {
        using T = typename TileCons::DType;
        constexpr int rows = TileCons::Rows;
        constexpr int cols = TileCons::Cols;
        using SlotTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, rows, cols>;

        SlotTile slotTile;
        TASSIGN(slotTile, static_cast<uint64_t>(pipe.fifo.C2V_CONSUMER_BUF + entryBase));
        TMOV_IMPL(tile, slotTile);
    } else if constexpr (Pipe::is_v2c) {
        using T = typename TileCons::DType;
        constexpr int rows = TileCons::Rows;
        constexpr int cols = TileCons::Cols;
        using SlotTile = Tile<TileType::Mat, T, rows, cols, BLayout::RowMajor, rows, cols>;

        SlotTile slotTile;
        TASSIGN(slotTile, static_cast<uint64_t>(pipe.fifo.V2C_CONSUMER_BUF + entryBase));
        TMOV_IMPL(tile, slotTile);
    }
}

template <typename TileCons, typename Pipe>
PTO_INTERNAL void TPOP_IMPL(TileCons &tile, Pipe &pipe)
{
    TPOP_IMPL<Pipe, TileCons, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
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