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

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/TPush.hpp>

namespace pto {

// free space for tile data
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}

// free space for global data
template <typename Pipe, typename GlobalData, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe, GlobalData &gmTensor)
{
    (void)gmTensor;
    static_assert(is_global_data_v<GlobalData>, "Fix: GlobalTensor must satisfy is_global_data_v<GlobalData>.");
    static_assert(Pipe::is_c2v || Pipe::is_v2c || Pipe::is_both,
                  "Fix: TFREE with GlobalTensor is only supported by C2V or V2C or Both communication on A2A3.");

    bool isFree = pipe.cons.getFreeStatus() && Pipe::shouldNotifyFree(static_cast<uint32_t>(pipe.cons.tileIndex - 1));
    if (isFree) {
        pipe.cons.free();
    }
    return;
}

//--------------------------------------------
template <typename Pipe>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}

} // namespace pto

#endif