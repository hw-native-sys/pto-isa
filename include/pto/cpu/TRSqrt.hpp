/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TRSQRT_HPP
#define PTO_CPU_TRSQRT_HPP

#include <cmath>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TRSQRT_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert((std::is_same<typename TileDataDst::DType, half>::value &&
                   std::is_same<typename TileDataSrc::DType, half>::value) ||
                      (std::is_same<typename TileDataSrc::DType, float>::value &&
                       std::is_same<typename TileDataDst::DType, float>::value),
                  "TRSQRT: Invalid data type");
    static_assert(TileDataSrc::ValidRow == TileDataDst::ValidRow && TileDataSrc::ValidCol == TileDataDst::ValidCol,
                  "TRSQRT: Src valid row/col != Dst valid row/col");
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto x = static_cast<double>(src.data()[GetTileElementOffset<TileDataSrc>(r, c)]);
            const double y = 1.0 / std::sqrt(x);
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = static_cast<typename TileDataDst::DType>(y);
        }
    });
}

template <typename TileDataDst, typename TileDataSrc, typename TmpTileData>
PTO_INTERNAL void TSQRT_IMPL(TileDataDst &dst, TileDataSrc &src, TmpTileData &tmp)
{
    TRSQRT_IMPL(dst, src);
}
} // namespace pto

#endif
