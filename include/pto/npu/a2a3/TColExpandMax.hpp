/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDMAX_HPP
#define TCOLEXPANDMAX_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TColExpandBinOp.hpp>

namespace pto {

template <typename T>
struct ColExpandMaxOp {
    PTO_INTERNAL static void ColExpandBinInstr(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeats)
    {
        vmax(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void ColExpandBinInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeats, uint8_t dstRepeatStride,
        uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, 0);
    }
};
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDMAX_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    TCOLEXPANDOP_IMPL<ColExpandMaxOp<T>, ColExpandMaxOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif