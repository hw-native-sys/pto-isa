/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include <cassert>

namespace pto {
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow = 0, uint32_t idxCol = 0)
{
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());
    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
            dst.data()[dstTileIdx] = src.data()[srcTileIdx];
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, std::enable_if_t<is_tile<FpTileData>::value, int> = 0>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow = 0,
                                uint32_t idxCol = 0)
{
    (void)fp;
    (void)reluMode;
    TEXTRACT_IMPL(dst, src, idxRow, idxCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
