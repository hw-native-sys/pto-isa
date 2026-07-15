/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGCOMMON_HPP
#define TPARTARGCOMMON_HPP

#include "TPartOp.hpp"
#include <pto/pto-inst.hpp>

namespace pto {
template <typename DstVal, typename Src0Val, typename Src1Val>
static inline void CheckAndGetValidRegions(
    const DstVal& dstVal, const Src0Val& src0Val, const Src1Val& src1Val, int& dstRow, int& dstCol, int& src0Row,
    int& src0Col, int& src1Row, int& src1Col)
{
    using T = typename Src0Val::DType;
    TPartCheck<T, DstVal, Src0Val, Src1Val>(dstVal.GetValidRow(), dstVal.GetValidCol());
    dstRow = dstVal.GetValidRow();
    dstCol = dstVal.GetValidCol();
    src0Row = src0Val.GetValidRow();
    src0Col = src0Val.GetValidCol();
    src1Row = src1Val.GetValidRow();
    src1Col = src1Val.GetValidCol();
}

template <typename CompareOp, typename ValType, typename IdxType>
struct PartArgSelectOp {
    static inline void apply(
        ValType& dstVal, ValType src0Val, ValType src1Val, IdxType& dstIdx, IdxType src0Idx, IdxType src1Idx)
    {
        if (CompareOp::apply(src0Val, src1Val)) {
            dstVal = src0Val;
            dstIdx = src0Idx;
        } else {
            dstVal = src1Val;
            dstIdx = src1Idx;
        }
    }
};

} // namespace pto
#endif // TPARTARGCOMMON_HPP
