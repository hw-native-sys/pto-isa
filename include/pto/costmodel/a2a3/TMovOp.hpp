/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOVOP_HPP
#define TMOVOP_HPP
#include "common.hpp"
#include "TCopy.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
AICORE void TMovToBt(std::vector<CostModelStats> &stats)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr const int BURST_LEN_UNIT = 64;

    if constexpr (std::is_same<SrcType, int32_t>::value || std::is_same<SrcType, float>::value) {
        static_assert(std::is_same<DstType, SrcType>::value,
                      "TMov: Destination and Source tile data types must be the same.");
    } else if constexpr (std::is_same<SrcType, half>::value) {
        static_assert(std::is_same<DstType, float>::value,
                      "TMov: When Source tile data types is half, dst tile data types must be float");
    }
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Bias, row must be 1");
    static_assert(SrcTileData::Cols * sizeof(SrcType) % BURST_LEN_UNIT == 0,
                  "TMov: When TileType is Bias, col * sizeof(srcDType) must be aligned to 64");

    uint16_t convControl = 0;
    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;

    if constexpr (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) {
        convControl = 1;
    }

    CostModelStats costModelStats("copy_cbuf_to_bt", 1, burstLen, 0, 0);
    costModelStats.setConvControl(convControl);
    stats.emplace_back(costModelStats);
}

template <typename DstTileData, typename SrcTileData>
AICORE void TMovToFb(std::vector<CostModelStats> &stats)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr const int BURST_LEN_UNIT = 128;
    constexpr const int RELU_BIT = 16;

    static_assert(std::is_same<DstType, SrcType>::value,
                  "TMov: Destination and Source tile data types must be the same.");
    static_assert(std::is_same<DstType, uint64_t>::value, "TMov: Invalid data type.");
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Scaling, row must be 1");
    static_assert(SrcTileData::Cols * sizeof(SrcType) % BURST_LEN_UNIT == 0,
                  "TMov: When TileType is Scaling, col * sizeof(srcType) must be aligned to 128");

    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;
    stats.emplace_back("copy_cbuf_to_fbuf", 1, burstLen, 0, 0);
}

template <typename DstTileData, typename SrcTileData>
AICORE void TMovToVec(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    constexpr unsigned srcStride = SrcTileData::RowStride;
    constexpr unsigned dstStride = DstTileData::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TCopy<DstTileData, SrcTileData, blockSizeElem, srcStride, dstStride>(stats, validRow, validCol);
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
AICORE void TMovCcToCb(std::vector<CostModelStats> &stats, uint16_t validRow, uint16_t validCol)
{
    stats.emplace_back("copy_matrix_cc_to_cbuf");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToLeft(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(stats, 0, 0, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, false>(stats, 0, 0, dst.GetValidRow(), dst.GetValidCol(),
                                                                dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(stats, 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToACompact<DstTileData, SrcTileData, true>(stats, 0, 0, dst.GetValidRow(), dst.GetValidCol(),
                                                               dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(stats, src.data(), 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToRight(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, false>(stats, 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(stats, 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBCompact<DstTileData, SrcTileData, true>(stats, 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(stats, 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_CONVTILE_IMPL(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TExtractToBConv<DstTileData, SrcTileData>(stats, src.GetShape(3), dst.GetValidRow(), dst.GetValidCol(), 0, 0);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_TILE_IMPL(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
                  "TMov: The shape of src needs to be the same as that of dst.");
    static_assert((SrcTileData::Loc == TileType::Mat &&
                   (DstTileData::Loc == TileType::Left || DstTileData::Loc == TileType::Right ||
                    DstTileData::Loc == TileType::Bias || DstTileData::Loc == TileType::Scaling)) ||
                      (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) ||
                      (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Acc),
                  "TMov: Invalid TileType.");
    if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Left) {
        TMovToLeft<DstTileData, SrcTileData>(stats, dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Right) {
        TMovToRight<DstTileData, SrcTileData>(stats, dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Bias) {
        TMovToBt<DstTileData, SrcTileData>(stats);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Scaling) {
        TMovToFb<DstTileData, SrcTileData>(stats);
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec) {
        TMovToVec<DstTileData, SrcTileData>(stats, dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TMovCcToCb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(stats, m, n);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void runTMovOp(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src)
{
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TMOV_CONVTILE_IMPL(stats, dst, src);
    } else {
        TMOV_TILE_IMPL(stats, dst, src);
    }
}
} // namespace pto
#endif
