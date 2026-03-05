/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUT_HPP
#define TPUT_HPP

#include <pto/cpu/comm/TGet.hpp>

namespace pto {
namespace comm {
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &ping, TileData &pong)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL void TGET_ASYNC_IMPL(GlobalDstData &dst, GlobalSrcData &src)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL void TPUT_ASYNC_IMPL(GlobalDstData &dst, GlobalSrcData &src)
{
    Copy_Data(src, dst);
}

} // namespace comm
} // namespace pto
#endif
