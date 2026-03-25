/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPRELU_HPP
#define TPRELU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TPreluCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1,
                              const TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TPRELU the data type of dst must be consistent with of src0 and src1.");
    static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value, "Fix: TPRELU has invalid data type.");
    static_assert(std::is_same<typename TileDataTmp::DType, uint8_t>::value, "Fix: TPRELU has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor && TileDataTmp::isRowMajor,
        "Fix: TPRELU only support row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TPRELU input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TPRELU input tile src1 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(tmp.GetValidRow() == validRows,
               "Fix: TPRELU input tile tmp valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TPRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    TPreluCheck(dst, src0, src1, tmp);
    TMUL_IMPL(dst, src0, src1);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    TCMPS_IMPL(tmp, src0, (T)0, CmpMode::GT);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    using SelTmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    SelTmpTile selTmp;
    // TSEL使用的selTmp取tmp后的32B
    TSUBVIEW(selTmp, tmp, TileDataTmp::Rows, 0);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    TSEL_IMPL(dst, tmp, src0, dst, selTmp);
}
} // namespace pto

#endif
