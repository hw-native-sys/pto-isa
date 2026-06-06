/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HPP_KIRINX90
#define COMMON_HPP_KIRINX90
#include "pto/npu/kirin9030/common.hpp"
#include <pto/common/arch_capability.hpp>

namespace pto {
template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isCastQuant>
PTO_INTERNAL void CheckTMovAccToMat()
{
    static_assert((SrcTileData::Loc == TileType::Acc), "Source TileType only support Acc.");
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((DstTileData::SFractalSize == TileConfig::fractalABSize),
                  "Destination SFractalSize only support 512.");
    static_assert(((DstTileData::Cols * sizeof(DstType) % C0_SIZE_BYTE == 0) && ((DstTileData::Cols) > 0)),
                  "Dst Tile Cols * sizeof(DstType) must be multiples of 32 and not 0.");
    static_assert((!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor),
                  "Src fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert(std::is_same_v<SrcType, half> || std::is_same_v<SrcType, int32_t>,
                  "Src data type only support half or int32_t.");
    if constexpr (isCastQuant) {
        if constexpr (std::is_same_v<SrcType, half>) {
            static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int8_t> ||
                              std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int16_t>,
                          "The output data type must be int8/uint8/half/int16 when input is data type half.");
        } else if constexpr (std::is_same_v<SrcType, int32_t>) {
            static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int8_t> ||
                              std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int16_t>,
                          "The output data type must be int8/uint8/half/int16/int32 when input is data type int32.");
        }
    } else {
        static_assert(std::is_same_v<DstType, SrcType>,
                      "The input data type must be consistent with the output data type when preQuantScalar is not "
                      "configured");
        static_assert(std::is_same_v<DstType, half> || std::is_same_v<DstType, int32_t>,
                      "The data type must be half or int32 when preQuantScalar is not configured");
    }
    static_assert((DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Only support nz2nz, nz2nd or nz2dn.");
}
} // namespace pto
#endif
