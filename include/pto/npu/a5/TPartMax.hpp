/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTMAX_HPP
#define TPARTMAX_HPP

#include "TPartBinOps.hpp"

namespace pto {

template <typename T>
struct TPartMaxOp {
    PTO_INTERNAL static void BinInstr(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg preg)
    {
        vmax(dst, src0, src1, preg, MODE_ZEROING);
    }
};

template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TPARTMAX_IMPL(DstTileData& dst, Src0TileData& src0, Src1TileData& src1)
{
    using T = typename DstTileData::DType;
    static_assert(
        std::is_same_v<T, typename Src0TileData::DType> && std::is_same_v<T, typename Src1TileData::DType>,
        "Fix: TPARTMAX Input and output types should match");
    static_assert(
        std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint16_t> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
            std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,
        "Fix: TPARTMAX Invalid data type.");
    TPARTOP_IMPL<TPartMaxOp<T>, DstTileData, Src0TileData, Src1TileData>(dst, src0, src1);
}
} // namespace pto
#endif