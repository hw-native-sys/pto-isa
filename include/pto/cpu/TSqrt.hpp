/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSQRT_HPP
#define TSQRT_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include <cmath>

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
void TSqrt_Impl(TileDataDst dst, TileDataSrc src, int validRow, int validCol)
{
    for (size_t c = 0; c < (size_t)validCol; c++) {
        for (size_t r = 0; r < (size_t)validRow; r++) {
            const auto x = static_cast<double>(src.data()[GetTileElementOffset<TileDataSrc>(r, c)]);
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] =
                static_cast<typename TileDataDst::DType>(std::sqrt(x));
        }
    }
}

template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSQRT_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert((std::is_same<typename TileDataDst::DType, bfloat16_t>::value &&
                   std::is_same<typename TileDataSrc::DType, bfloat16_t>::value) ||
                      (std::is_same<typename TileDataDst::DType, half>::value &&
                       std::is_same<typename TileDataSrc::DType, half>::value) ||
                      (std::is_same<typename TileDataSrc::DType, float>::value &&
                       std::is_same<typename TileDataDst::DType, float>::value),
                  "TSQRT: Invalid data type");
    static_assert(TileDataSrc::ValidRow == TileDataDst::ValidRow && TileDataSrc::ValidCol == TileDataDst::ValidCol,
                  "TSQRT: Src valid row/col != Dst valid row/col");
    TSqrt_Impl<TileDataDst, TileDataSrc>(dst, src, dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif // TSQRT_HPP
