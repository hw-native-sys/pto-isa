/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOP_HPP
#define TPOP_HPP

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/TPush.hpp>

namespace pto {

// pop tile from fifo for tile data
template <typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileCons>, int> = 0>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, TileCons &tile)
{
    // 1. Cross-Core: Wait for Data
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.wait();
    }

    // 2. Address Calculation & Load
    pipe.cons.template pop<TileCons, Split>(pipe.fifo, tile);
    pipe.cons.tileIndex++;

    // 3. Cross-Core: Free Space
    bool isFree = pipe.cons.getFreeStatus();
    if (isFree) {
        pipe.cons.free();
    }
}

// get sub-block offset for global data
template <typename GlobalData, TileSplitAxis Split>
PTO_INTERNAL uint64_t getPopSubAIVOffset()
{
    if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
        return 0;
    }

    constexpr int consM = GlobalData::staticShape[pto::GlobalTensorDim::DIM_3];
    constexpr int consN = GlobalData::staticShape[pto::GlobalTensorDim::DIM_4];
    if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
        return get_subblockid() * consM * consN * sizeof(typename GlobalData::RawDType);
    } else { // TILE_LEFT_RIGHT
        return get_subblockid() * consN * sizeof(typename GlobalData::RawDType);
    }
}

// pop tensor slot entry from global memory
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, GlobalData &gmTensor)
{
    static_assert(is_global_data_v<GlobalData>, "Fix: GlobalTensor must satisfy is_global_data_v<GlobalData>.");
    static_assert(Pipe::is_c2v || Pipe::is_v2c || Pipe::is_both,
                  "Fix: TPOP with GlobalTensor is only supported by C2V or V2C or Both communication on A2A3.");
    // 1. Cross-Core: Wait for Data
    pipe.cons.wait();

    // 2. Address Calculation
    uint64_t entryBase = (uint64_t)pipe.fifo.GM_SLOT_BUFFER;
    const uint64_t slotOffset = (pipe.cons.tileIndex % Pipe::RingFiFo::SLOT_NUM) * Pipe::RingFiFo::SLOT_SIZE;
    if constexpr (Pipe::is_c2v) {
        entryBase += slotOffset + getPopSubAIVOffset<GlobalData, Split>();
    } else if constexpr (Pipe::is_v2c) {
        entryBase += slotOffset;
    } else if constexpr (Pipe::is_both) {
#ifdef __DAV_VEC__
        entryBase += slotOffset + getPopSubAIVOffset<GlobalData, Split>();
#endif
#ifdef __DAV_CUBE__
        entryBase += slotOffset;
#endif
    }

    // 3. Increment tile index
    pipe.cons.tileIndex++;
    TASSIGN_IMPL(gmTensor, reinterpret_cast<typename GlobalData::DType *>(entryBase));
}

//--------------------------------------------
template <typename TileData, typename Pipe>
PTO_INTERNAL void TPOP_IMPL(TileData &tile, Pipe &pipe)
{
    // 1. Cross-Core: Wait for Data
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.wait();
    }

    // 2. Address Calculation & Load
    pipe.cons.pop(pipe.fifo, tile);
    pipe.cons.tile_id++;

    // 3. Cross-Core: Free Space
    bool isFree = pipe.cons.getFreeStatus();
    if (isFree) {
        pipe.cons.free();
    }
}

} // namespace pto

#endif
