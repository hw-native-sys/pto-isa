/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPATIALADD_HPP
#define TPATIALADD_HPP

#include "TPartBinOps.hpp"

namespace pto {
template <typename T>
struct PartAddOp {
    PTO_INTERNAL static void BinInstr(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg preg)
    {
        vadd(dst, src0, src1, preg, MODE_ZEROING);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTADD_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TPARTADD src and dst data type is different!");
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, half>::value ||
            std::is_same<T, bfloat16_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value,
        "Fix: TPARTADD Invalid data type.");
    TPARTOP_IMPL<PartAddOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif
