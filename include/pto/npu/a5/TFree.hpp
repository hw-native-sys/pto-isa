/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFREE_HPP
#define TFREE_HPP

#include <type_traits>
#include <pto/common/fifo.hpp>
#include <pto/npu/a5/TPush.hpp>

namespace pto {

template <
    uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum, bool IsNoSplit,
    bool EN_UNIT_FLAG, typename CtrlConfig>
PTO_INTERNAL void
TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum, IsNoSplit, EN_UNIT_FLAG, CtrlConfig>::Consumer::freeCtrl() const
{
    if constexpr (CtrlConfig::EnableBackward) {
        if constexpr (is_c2v_ctrl) {
#ifdef PTO_COMPILE_VEC
            if (activeVector()) {
                set_intra_block(PIPE_S, FlagID + 1);
            }
#endif
        } else {
            cubeFlags<true, FlagID + 1>();
        }
    }
}

// free space for tile data
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe& pipe)
{
    // Auto-free pops and repeated frees must not return a credit twice.
    if (pipe.cons.freeIndex == pipe.cons.tileIndex) {
        return;
    }
    bool isFree = pipe.cons.getFreeStatus() && Pipe::shouldNotifyFree(pipe.cons.freeIndex);
    pipe.cons.freeIndex++;
    if (isFree) {
        pipe.cons.template free<Split>();
    }
}

// free global tensor slot entry from global memory
template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INTERNAL void TFREE_IMPL(Pipe& pipe, GlobalData& gmTensor)
{
    (void)gmTensor;
    static_assert(is_global_data_v<GlobalData>, "Fix: GlobalTensor must satisfy is_global_data_v<GlobalData>.");
    static_assert(
        Pipe::is_c2v_gm || Pipe::is_v2c_gm || Pipe::is_both_gm,
        "Fix: TFREE with GlobalTensor is only supported by GM FIFO directions on A5.");
    TFREE_IMPL<Pipe, Split>(pipe);
}

//------------------------------------------------
template <typename Pipe>
PTO_INTERNAL void TFREE_IMPL(Pipe& pipe)
{
    bool isFree = pipe.cons.getFreeStatus();
    if (isFree) {
        pipe.cons.free();
    }
}

} // namespace pto

#endif
