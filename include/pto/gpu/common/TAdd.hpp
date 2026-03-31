/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TADD_HPP
#define PTO_GPU_COMMON_TADD_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/gpu/common/tile_offsets.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Fix: TADD input/output dtypes must match for the GPU backend.");

    const unsigned validRows = dst.GetValidRow();
    const unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TADD src0 valid shape mismatch with dst in GPU backend.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TADD src1 valid shape mismatch with dst in GPU backend.");

    for (unsigned r = 0; r < validRows; ++r) {
        for (unsigned c = 0; c < validCols; ++c) {
            const std::size_t idx = gpu::GetTileElementOffset<TileDataDst>(r, c);
            const std::size_t src0Idx = gpu::GetTileElementOffset<TileDataSrc0>(r, c);
            const std::size_t src1Idx = gpu::GetTileElementOffset<TileDataSrc1>(r, c);
            dst.data()[idx] = src0.data()[src0Idx] + src1.data()[src1Idx];
        }
    }
}

} // namespace pto

#endif
