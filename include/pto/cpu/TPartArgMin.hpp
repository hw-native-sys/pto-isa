/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGMIN_HPP
#define TPARTARGMIN_HPP

#include "TPartArgCommon.hpp"

namespace pto {
struct MinCompareOp {
    template <typename ValType>
    static inline bool apply(ValType src0Val, ValType src1Val)
    {
        return src0Val <= src1Val;
    }
};

template <typename ValType, typename IdxType>
using MinWithIndexOp = PartArgSelectOp<MinCompareOp, ValType, IdxType>;

template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
struct PartArgMinOp {
    PTO_INTERNAL static void PartInstr(typename TileDataDstVal::TileDType dstVal,
                                       typename TileDataDstIdx::TileDType dstIdx,
                                       typename TileDataSrc0Val::TileDType src0Val,
                                       typename TileDataSrc0Idx::TileDType src0Idx,
                                       typename TileDataSrc1Val::TileDType src1Val,
                                       typename TileDataSrc1Idx::TileDType src1Idx, int DstOffset, int Src0Offset,
                                       int Src1Offset)
    {
        MinWithIndexOp<typename TileDataSrc0Val::DType, typename TileDataSrc0Idx::DType>::apply(
            dstVal[DstOffset], dstIdx[DstOffset], src0Val[Src0Offset], src0Idx[Src0Offset], src1Val[Src1Offset],
            src1Idx[Src1Offset]);
    }
};

template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
PTO_INTERNAL void TPARTARGMIN_IMPL(TileDataDstVal &dstVal, TileDataDstIdx &dstIdx, TileDataSrc0Val &src0Val,
                                   TileDataSrc0Idx &src0Idx, TileDataSrc1Val &src1Val, TileDataSrc1Idx &src1Idx)
{
    int dstRow, dstCol, src0Row, src0Col, src1Row, src1Col;
    CheckAndGetValidRegions(dstVal, src0Val, src1Val, dstRow, dstCol, src0Row, src0Col, src1Row, src1Col);

    TPartInstr2<PartArgMinOp<TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val,
                             TileDataSrc1Idx>,
                TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>(
        dstVal.data(), dstIdx.data(), src0Val.data(), src0Idx.data(), src1Val.data(), src1Idx.data(), dstRow, dstCol,
        src0Row, src0Col, src1Row, src1Col);
}
} // namespace pto
#endif // TPARTARGMIN_HPP