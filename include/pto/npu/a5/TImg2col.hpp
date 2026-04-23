/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TIMG2COL_HPP
#define TIMG2COL_HPP

namespace pto {

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode>
__tf__ PTO_INTERNAL void TImg2col(typename TileData::TileDType __out__ dst, typename ConvTileData::TileDType __in__ src,
                                  uint16_t stepM, uint16_t stepK, uint16_t posM, uint16_t posK, uint8_t strideW,
                                  uint8_t strideH, uint16_t filterW, uint16_t filterH, uint8_t dilationW,
                                  uint8_t dilationH, bool transpose, uint16_t channelSize)
{
    using DstType = std::conditional_t<(sizeof(typename TileData::DType) == 2), half, typename TileData::DType>;
    using SrcType = std::conditional_t<(sizeof(typename ConvTileData::DType) == 2), half, typename ConvTileData::DType>;
    __ca__ DstType *dstAddr = (__ca__ DstType *)__cce_get_tile_ptr(dst);
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    bool fmatrixCtrl =
        (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) || (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL);
    bool highFilterH = (filterH > 255);
    bool highFilterW = (filterW > 255);
    uint8_t lowFilterH = filterH & 0xFF;
    uint8_t lowFilterW = filterW & 0xFF;
    img2colv2_cbuf_to_ca(dstAddr, srcAddr, stepK, stepM, posK, posM, strideW, strideH, lowFilterW, lowFilterH,
                         dilationW, dilationH, highFilterW, highFilterH, transpose, fmatrixCtrl, channelSize);
}

template <SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_AUTO>
PTO_INTERNAL void SetFmatrix(uint16_t fmapH, uint16_t fmapW, const uint8_t *padList)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        uint64_t fmatrixReg = 0;
        constexpr uint32_t l1ShiftBit = 16;
        fmatrixReg |= uint64_t(fmapW & 0xFFFF);
        fmatrixReg |= uint64_t(fmapH & 0xFFFF) << l1ShiftBit;

        constexpr uint32_t padListShiftBit = 8;
        constexpr uint32_t padListShiftBase = 32;
        constexpr uint32_t padNumber = 4;

        for (uint32_t i = 0; i < padNumber; i++) {
            fmatrixReg |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO) {
            set_fmatrix(fmatrixReg);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
            set_fmatrix_b(fmatrixReg);
        }
    }
}

template <SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_AUTO>
PTO_INTERNAL void SetRepeat(uint16_t repeatStride, uint8_t repeatTime, uint8_t repeatMode, uint16_t dstStride,
                            uint16_t dstMposition)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        uint64_t rptConfig = 0;
        constexpr uint32_t repeatTimeShiftBit = 16;
        constexpr uint32_t repeatModeShiftBit = 24;
        constexpr uint32_t dstStrideShiftBit = 32;
        constexpr uint32_t dstMpositionShiftBit = 48;
        rptConfig |= uint64_t(repeatStride);
        rptConfig |= uint64_t(repeatTime) << repeatTimeShiftBit;
        rptConfig |= uint64_t(repeatMode) << repeatModeShiftBit;
        rptConfig |= uint64_t(dstStride) << dstStrideShiftBit;
        rptConfig |= uint64_t(dstMposition) << dstMpositionShiftBit;
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO) {
            set_l3d_rpt(rptConfig);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
            set_l3d_rpt_b(rptConfig);
        }
    }
}

template <SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_AUTO, typename T>
PTO_INTERNAL void SetPadding(T padValue)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        uint32_t paddingValue = 0;
        constexpr uint16_t padValueShiftBit = 8;
        if constexpr (sizeof(T) == 1) {
            uint8_t u8Value = *reinterpret_cast<const uint8_t *>(&padValue);
            paddingValue = (static_cast<uint16_t>(u8Value) << padValueShiftBit) | u8Value;
        } else if constexpr (sizeof(T) == 2) {
            paddingValue = *reinterpret_cast<const uint16_t *>(&padValue);
        } else if constexpr (sizeof(T) == 4) {
            paddingValue = *reinterpret_cast<const uint32_t *>(&padValue);
        }
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO) {
            set_padding(paddingValue);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
            set_padding_b(paddingValue);
        }
    }
}

template <typename TileData, typename ConvTileData>
PTO_INTERNAL void Timg2colConvTileCheck(TileData &dst, ConvTileData &src)
{
    static_assert((ConvTileData::Loc == TileType::Mat), "TImg2col: Source TileType only support Mat.");
    static_assert((TileData::Loc == TileType::Left), "TImg2col: Destination TileType only support Left.");
    static_assert((ConvTileData::layout == Layout::NC1HWC0) || (ConvTileData::layout == Layout::NDC1HWC0),
                  "TImg2col: Source layout only support NC1HWC0.");
    static_assert(TileData::SFractal == SLayout::RowMajor && !TileData::isRowMajor,
                  "TImg2col: Destination layout only support SLayout is RowMajor ang BLayout is ColMajor.");
    static_assert(std::is_same_v<typename ConvTileData::DType, typename TileData::DType>,
                  "TImg2col: Destination and Source tile data types must be the same.");
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TIMG2COL_IMPL(TileData &dst, ConvTileData &src, uint16_t posM, uint16_t posK)
{
    Timg2colConvTileCheck<TileData, ConvTileData>(dst, src);
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_AUTO || FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
        SetFmatrix<FmatrixMode>(src.GetFmapH(), src.GetFmapW(), src.GetPadListArray());
        SetRepeat<FmatrixMode>(src.GetRepeatStride(), src.GetRepeatTime(), src.GetRepeatMode(), src.GetDstStride(),
                               src.GetDstMposition());
        SetPadding<FmatrixMode>(src.GetPadValue());
    }
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t stepK = CeilAlignment(dst.GetValidCol(), c0Size);
    uint16_t stepM = dst.GetValidRow();
    TImg2col<TileData, ConvTileData, FmatrixMode>(
        dst.data(), src.data(), stepM, stepK, posM, posK, src.GetStrideW(), src.GetStrideH(), src.GetFilterW(),
        src.GetFilterH(), src.GetDilationW(), src.GetDilationH(), src.GetTranspose(), src.GetChannelSize());
}
} // namespace pto
#endif // TIMG2COL_HPP
