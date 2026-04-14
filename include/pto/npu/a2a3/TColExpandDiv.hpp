/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDDIV_HPP
#define TCOLEXPANDDIV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TColExpandBinOp.hpp>

namespace pto {

template <typename T>
struct ColExpandDivOp {
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vdiv(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                               uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                                               uint8_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, 0);
    }
};

template <typename T>
struct ColExpandDivOp2 {
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vdiv(dst, src1, src0, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                               uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                                               uint8_t src1RepeatStride)
    {
        vdiv(dst, src1, src0, repeats, 1, 1, 1, dstRepeatStride, 0, src0RepeatStride);
    }
};
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float16_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, float32_t>,
                  "Fix: TCOLEXPANDDIV Invalid data type.");
    TCOLEXPANDOP_IMPL<ColExpandDivOp<T>, ColExpandDivOp2<T>, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif
