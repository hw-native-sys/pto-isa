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

namespace pto {
template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isCastQuant>
PTO_INTERNAL void CheckTMovAccValid()
{
    static_assert((SrcTileData::Loc == TileType::Acc), "Source TileType only support Acc.");
    static_assert(
        (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec),
        "Destination TileType only support Mat or Vec.");
    static_assert(
        ((DstTileData::Cols * sizeof(DstType) % C0_SIZE_BYTE == 0) && ((DstTileData::Cols) > 0)),
        "Dst Tile Cols * sizeof(DstType) must be multiples of 32 and not 0.");
    static_assert(
        (!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor), "Src fractal format should be nz.");
    static_assert(
        std::is_same_v<SrcType, half> || std::is_same_v<SrcType, int32_t>,
        "Src data type only support half or int32_t.");
    if constexpr (isCastQuant) {
        if constexpr (std::is_same_v<SrcType, half>) {
            static_assert(
                std::is_same_v<DstType, half> || std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int8_t> ||
                    std::is_same_v<DstType, int16_t>,
                "The output data type must be int8/uint8/half/int16 when input is data type half.");
        } else if constexpr (std::is_same_v<SrcType, int32_t>) {
            static_assert(
                std::is_same_v<DstType, half> || std::is_same_v<DstType, uint8_t> || std::is_same_v<DstType, int8_t> ||
                    std::is_same_v<DstType, int16_t>,
                "The output data type must be int8/uint8/half/int16/int32 when input is data type int32.");
        }
    } else {
        static_assert(
            std::is_same_v<DstType, SrcType>,
            "The input data type must be consistent with the output data type when preQuantScalar is not "
            "configured");
        static_assert(
            std::is_same_v<DstType, int32_t> || std::is_same_v<DstType, half>,
            "The data type must be half or int32 when preQuantScalar is not configured");
    }
    static_assert(
        (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
            (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
        "Only support nz2nz or nz2nd");
}

PTO_INTERNAL void SetLoop3Para()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
}

template <typename DstTile>
struct TMovConfig {
    using DType = typename DstTile::DType;

    static constexpr bool isNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    static constexpr bool isNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    static constexpr bool isFloat = std::is_same_v<DType, float>;

    static constexpr bool channelSplitEnable = isNz2Nz && isFloat && (DstTile::SFractalSize == 512);
    static constexpr bool is1024Nz = isNz2Nz && (DstTile::SFractalSize == 1024);

    static constexpr uint32_t baseC0Size = C0_SIZE_BYTE / sizeof(DType);
    static constexpr uint32_t specialC0Size = (!channelSplitEnable && is1024Nz) ? (2 * baseC0Size) : baseC0Size;

    static constexpr uint32_t accDstStride = isNz2Nd ? DstTile::Cols : (DstTile::Rows * specialC0Size);

    static constexpr int32_t nzAlign =
        (isFloat && channelSplitEnable) ? baseC0Size : (isFloat ? FRACTAL_NZ_ROW : baseC0Size);
};
} // namespace pto
#endif