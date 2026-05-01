/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TALLOC_HPP
#define TALLOC_HPP

#include <type_traits>
#include <pto/common/fifo.hpp>
#include <pto/npu/a5/TPush.hpp>

namespace pto {

// get sub-block offset for global data
template <typename GlobalData, TileSplitAxis Split>
PTO_INTERNAL uint64_t getSubAIVOffset()
{
    constexpr int prodN = GlobalData::staticShape[pto::GlobalTensorDim::DIM_4];
    constexpr int prodM = GlobalData::staticShape[pto::GlobalTensorDim::DIM_3];
    if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
        return get_subblockid() * prodM * prodN * sizeof(typename GlobalData::RawDType);
    } else { // TILE_LEFT_RIGHT
        return get_subblockid() * prodN * sizeof(typename GlobalData::RawDType);
    }

    if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
        return 0;
    }
}

// allocate fifo slot entry for global data
template <typename Pipe, typename GlobalData, TileSplitAxis Split>
PTO_INTERNAL void TALLOC_IMPL(Pipe &pipe, GlobalData &gmTensor)
{
    static_assert(Pipe::is_c2v_gm || Pipe::is_v2c_gm || Pipe::is_both_gm,
                  "Fix: TALLOC with GlobalTensor is only supported by GM FIFO directions on A5.");
    static_assert(is_global_data_v<GlobalData>, "Fix: GlobalTensor must satisfy is_global_data_v<GlobalData>.");

    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex);
    if (isAllocate) {
        pipe.prod.template allocate<Split>(); // wait for space
    }

    // 2. Slot entry address allocation
    uint64_t entryBase = (uint64_t)pipe.fifo.GM_SLOT_BUFFER;
    const uint64_t slotOffset = (pipe.prod.tileIndex % Pipe::RingFiFo::SLOT_NUM) * Pipe::RingFiFo::SLOT_SIZE;
    if constexpr (Pipe::is_c2v_gm) {
        entryBase += slotOffset;
    } else if constexpr (Pipe::is_v2c_gm) {
        entryBase += slotOffset + getSubAIVOffset<GlobalData, Split>();
    } else if constexpr (Pipe::is_both_gm) {
#ifdef __DAV_CUBE__
        entryBase += slotOffset;
#endif
#ifdef __DAV_VEC__
        entryBase += slotOffset + getSubAIVOffset<GlobalData, Split>();
#endif
    }

    // 3. Increment tile index
    pipe.prod.tileIndex++;
    TASSIGN_IMPL(gmTensor, reinterpret_cast<typename GlobalData::DType *>(entryBase));
}

} // namespace pto
#endif