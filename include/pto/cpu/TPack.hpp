/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TPACK_CPU_HPP
#define TPACK_CPU_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TPACK_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    for (unsigned r = 0; r < src.GetValidRow(); ++r) {
        for (unsigned c = 0; c < src.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = static_cast<typename TileDataDst::DType>(
                src.data()[GetTileElementOffset<TileDataSrc>(r, c)]);
        }
    }
}
} // namespace pto

#endif
