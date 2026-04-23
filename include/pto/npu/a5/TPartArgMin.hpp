/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGMIN_HPP
#define TPARTARGMIN_HPP

#include "TPartBinOps.hpp"
#include "TPartArgBinOps.hpp"

namespace pto {

template <typename T, typename U>
struct TPartArgMinOp {
    static constexpr typename Padding<T>::Type PadVal = Padding<T>::Max;
    static constexpr typename Padding<U>::Type PadIdx = Padding<U>::Max;
    PTO_INTERNAL static void BinInstr(MaskReg &maskReg, RegTensor<T> &src, RegTensor<T> &dst, MaskReg preg)
    {
        vcmp_lt(maskReg, src, dst, preg);
    }
};

template <typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData, typename DstIdxTileData,
          typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TPARTARGMIN_IMPL(DstValTileData &dstVal, Src0ValTileData &src0Val, Src1ValTileData &src1Val,
                                   DstIdxTileData &dstIdx, Src0IdxTileData &src0Idx, Src1IdxTileData &src1Idx,
                                   VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    TPartArgCheck(dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx);
    TPartArgImpl<TPartArgMinOp<typename DstValTileData::DType, typename DstIdxTileData::DType>, DstValTileData,
                 Src0ValTileData, Src1ValTileData, DstIdxTileData, Src0IdxTileData, Src1IdxTileData>(
        dstVal.data(), src0Val.data(), src1Val.data(), dstIdx.data(), src0Idx.data(), src1Idx.data(),
        dstVal.GetValidRow(), dstVal.GetValidCol(), src0Val.GetValidRow(), src0Val.GetValidCol(), src1Val.GetValidRow(),
        src1Val.GetValidCol(), version);
}
} // namespace pto
#endif