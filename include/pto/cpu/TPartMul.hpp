/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TPARTMUL_HPP
#define TPARTMUL_HPP

#include "TPartOp.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
struct PartMulOp {
    PTO_INTERNAL static void PartInstr(typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0,
                                       typename TileDataSrc1::TileDType src1, int DstOffset, int Src0Offset,
                                       int Src1Offset)
    {
        dst[DstOffset] = src0[Src0Offset] * src1[Src1Offset];
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataSrc0::DType;
    TPartCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst.GetValidRow(), dst.GetValidCol());
    TPartInstr<PartMulOp<TileDataDst, TileDataSrc0, TileDataSrc1>, TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol(), src0.GetValidRow(),
        src0.GetValidCol(), src1.GetValidRow(), src1.GetValidCol());
}
} // namespace pto

#endif