/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP
#include "common.hpp"
#include "TExtract.hpp"
#include "pto/npu/a5/TPartAdd.hpp"

namespace pto {
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToBt(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using DstType = typename DstTile::DType;
    using SrcType = typename SrcTile::DType;
    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                      (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>),
                  "Fix: TMOV: Bias data type only supports int32_t or half.");

    constexpr const int BIAS_TABLE_UNIT = 64;
    static_assert(SrcTile::Rows == 1, "TMov: When TileType is Bias, row must be 1.");
    static_assert(DstTile::Cols * sizeof(DstType) % BIAS_TABLE_UNIT == 0,
                  "TMov: When TileType is Bias, col * sizeof(Dtype) must be aligned to 64.");
    static_assert(DstTile::Cols * sizeof(DstType) <= PTO_BIAS_SIZE_BYTES,
                  "TMov: The memory occupation of BiasTile exceeds 1.0KB bias table size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    uint64_t dstAddrP = (uint64_t)dst;

    constexpr bool convControl = false;
    constexpr uint16_t burstNum = 1;
    constexpr const int BURST_LEN_UNIT_SHIFT = 5; // BURST_LEN_UNIT = 32;
    constexpr uint16_t burstLen = SrcTile::Numel * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToFb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using SrcType = typename SrcTile::DType;
    using DstType = typename DstTile::DType;
    constexpr const int FIXPIPE_BUFFER_UNIT = 128;
    static_assert(SrcTile::Rows == 1, "TMov: When TileType is Scaling, row must be 1.");
    static_assert(DstTile::Cols * sizeof(DstType) % FIXPIPE_BUFFER_UNIT == 0,
                  "TMov: When TileType is Scaling, col * sizeof(Dtype) must be aligned to 128.");
    static_assert(DstTile::Cols * sizeof(DstType) <= PTO_FBUF_SIZE_BYTES,
                  "TMov: The memory occupation of FbTile exceeds 7.0KB fixpipe buffer size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)__cce_get_tile_ptr(dst);

    constexpr uint16_t burstNum = 1;
    constexpr uint16_t burstLen = SrcTile::Numel * sizeof(SrcType) / FIXP_BURST_UNIT_LEN;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, srcGap, dstGap);
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

template <typename DstTile, typename SrcTile>
PTO_INTERNAL constexpr uint32_t GetTmovAccDstStride()
{
    if constexpr (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox) {
        return DstTile::Cols;
    } else if constexpr (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox) {
        return DstTile::Rows;
    }
    constexpr bool channelSplitEnable = (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTile::DType, float>) &&
                                        (DstTile::SFractalSize == 512);
    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (DstTile::SFractalSize == 1024) ?
                                    2 * C0_SIZE_BYTE / sizeof(typename DstTile::DType) :
                                    C0_SIZE_BYTE / sizeof(typename DstTile::DType);
    return DstTile::Rows * c0Size;
}

template <typename DstTile, typename SrcTile, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TMovCcToCb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool enableNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable = (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTile::DType, float>) &&
                                        (DstTile::SFractalSize == 512);
    if constexpr (enableNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr (std::is_same_v<typename DstTile::DType, float>) {
            constexpr int32_t align = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
            validCol = CeilAlignment(validCol, align);
        } else {
            validCol = CeilAlignment(validCol, c0Size);
        }
    }

    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    if constexpr (enableNz2Nd || enableNz2Dn) {
        SetLoop3Para();
    }
    if constexpr (enableNz2Dn) {
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }
    auto srcStride = CeilAlignment(validRow, BLOCK_LEN);
    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);

    pto_copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, 0, 0, QuantPre,
                               static_cast<uint8_t>(reluMode), channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
                               false, false, false, false, false, enableNz2Dn);
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, QuantMode_t quantPre, ReluPreMode reluMode,
          STPhase Phase = STPhase::Unspecified>
__tf__ PTO_INTERNAL void TMovCcToUb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable = (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTile::DType, float>) &&
                                        (DstTile::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    if constexpr (enableNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr ((mode == AccToVecMode::SingleModeVec0 || mode == AccToVecMode::SingleModeVec1)) {
            if constexpr (std::is_same_v<typename DstTile::DType, float>) {
                constexpr int32_t align = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
                validCol = CeilDivision(validCol, align) * align;
            } else {
                validCol = CeilDivision(validCol, c0Size) * c0Size;
            }
        } else if constexpr (mode == AccToVecMode::DualModeSplitM) {
            validCol = CeilDivision(validCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        } else {
            validCol = CeilDivision(validCol, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
        }
    }
    if constexpr (enableNz2Nd) {
        SetLoop3Para();
    } else if constexpr (enableNz2Dn) {
        SetLoop3Para();
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }
    auto srcStride = (validRow + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);
    pto_copy_matrix_cc_to_ub(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, false, 0, unitFlagCtrl,
                             quantPre, static_cast<uint8_t>(reluMode), channelSplitEnable, enableNz2Nd, 0, 0, false,
                             false, 0, false, false, false, false, false, enableNz2Dn);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL constexpr void CommonCheck()
{
    using T = typename DstTile::DType;
    using U = typename SrcTile::DType;
    static_assert(std::is_same_v<T, U>, "Fix: TMov Destination and Source tile data types must be the same.");

    if constexpr (DstTile::Loc == TileType::Left) {
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
                      "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(DstTile::SFractal == SLayout::RowMajor && !DstTile::isRowMajor,
                      "Fix: TMov: Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    }
    if constexpr (DstTile::Loc == TileType::Right) {
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
                      "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(DstTile::SFractal == SLayout::ColMajor && DstTile::isRowMajor,
                      "Fix: TMov: Dst fractal format should be (BFractal: RowMajor, SFractal: ColMajor).");
    }

    static_assert((SrcTile::SFractal == SLayout::ColMajor && SrcTile::isRowMajor) ||
                      (SrcTile::SFractal == SLayout::RowMajor && !SrcTile::isRowMajor) || (SrcTile::isRowMajor),
                  "TMov: SrcTile Invalid Fractal.");
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToVecNd2Nz(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                        uint32_t validRow, uint32_t validCol, uint32_t srcValidRow,
                                        unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, float>::value) ||
                      (std::is_same<T, int32_t>::value) || (std::is_same<T, int8_t>::value),
                  "Dst and src must be float/int32_t/half/int8_t/.");

    using U = std::conditional_t<sizeof(T) == 1, uint8_t, T>;
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ U *srcPtr = (__ubuf__ U *)__cce_get_tile_ptr(src);
    constexpr int32_t srcRow = SrcTile::Rows;
    constexpr int32_t srcCol = SrcTile::Cols;
    constexpr int32_t srcByteSize = srcRow * srcCol * sizeof(U);
    constexpr int32_t dstByteSize = DstTile::Rows * DstTile::Cols * sizeof(U);

    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(U);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr bool isOptForConflict = DstTile::Compact == CompactMode::RowPlusOne;
    uint32_t alignRow = (srcRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = isOptForConflict ? ((alignRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
                                              (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t virtualRow = isOptForConflict ? alignRow + 1 : alignRow;
    uint32_t repeatStride = 1;
    uint16_t innerLoopNum = validRow - 1;
    uint32_t cfgVsstb = (blockStride << 16u) | (1 & 0xFFFFU);
    uint32_t repeatStrideLast = (REPEAT_BYTE * virtualRow - innerLoopNum * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE;
    uint32_t cfgVsstbLast = (blockStride << 16u) | (repeatStrideLast & 0xFFFFU);
    uint32_t srcOffset = innerLoopNum * SrcTile::RowStride;
    __VEC_SCOPE__
    {
        RegTensor<U> vreg;
        MaskReg preg;
        uint32_t cols = validCol;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            preg = CreatePredicate<U>(cols);
            for (uint16_t i = 0; i < innerLoopNum; ++i) {
                vlds(vreg, srcPtr, SrcTile::RowStride, NORM, POST_UPDATE);
                vsstb(vreg, dstPtr, cfgVsstb, preg, POST_UPDATE);
            }
            vlds(vreg, srcPtr, elementsPerRepeat, NORM, POST_UPDATE);
            vsstb(vreg, dstPtr, cfgVsstbLast, preg, POST_UPDATE);
            srcPtr -= srcOffset;
        }
    } // end of VF
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TMOV)
    OP_TYPE(element_wise) void TMovVecToVec(typename DstTile::TileDType __out__ dstData,
                                            typename SrcTile::TileDType __in__ srcData, unsigned validRow,
                                            unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    constexpr unsigned nRepeatElem = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg pReg;
        uint32_t sreg;
        uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sreg = (uint32_t)validCol;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                pReg = CreatePredicate<T>(sreg);
                vlds(vreg0, src, i * SrcTile::RowStride + j * nRepeatElem, NORM);
                vsts(vreg0, dst, i * DstTile::RowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovToVec(DstTile &dst, SrcTile &src)
{
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TMovVecToVec<DstTile, SrcTile>(dst.data(), src.data(), validRow, validCol);
}

template <typename DstTile, typename SrcTile>
AICORE void TMovToLeft(DstTile &dst, SrcTile &src)
{
    CommonCheck<DstTile, SrcTile>();
    if constexpr (SrcTile::Rows == 1 && SrcTile::isRowMajor) {
        TExtractToAVector<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidCol());
    } else if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToATransCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                      dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTile, typename SrcTile>
AICORE void TMovToRight(DstTile &dst, SrcTile &src)
{
    CommonCheck<DstTile, SrcTile>();
    if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToBTransCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                      dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    if constexpr (SrcTile::Loc == TileType::Mat) {
        static_assert((SrcTile::Rows == DstTile::Rows) && ((SrcTile::Cols == DstTile::Cols)),
                      "TMov: The shape of destination and source tile must be the same.");
        if constexpr (DstTile::Loc == TileType::Bias) {
            TMovToBt<DstTile, SrcTile>(dst.data(), src.data());
        } else if constexpr (DstTile::Loc == TileType::Scaling) {
            TMovToFb<DstTile, SrcTile>(dst.data(), src.data());
        } else if constexpr (DstTile::Loc == TileType::Left) {
            TMovToLeft(dst, src);
        } else if constexpr (DstTile::Loc == TileType::Right) {
            TMovToRight(dst, src);
        } else if constexpr (DstTile::Loc == TileType::ScaleLeft) {
            static_assert(sizeof(DstTile::DType) == 0, "TMov: ScaleLeft tile type is not supported.");
        } else if constexpr (DstTile::Loc == TileType::ScaleRight) {
            static_assert(sizeof(DstTile::DType) == 0, "TMov: ScaleRight tile type is not supported.");
        }
    } else if constexpr (SrcTile::Loc == TileType::Acc) {
        CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
        if constexpr (DstTile::Loc == TileType::Vec) {
            TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(dst.data(),
                                                                                                      src.data(), m, n);
        } else if constexpr (DstTile::Loc == TileType::Mat) {
            TMovCcToCb<DstTile, SrcTile, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
        }
    } else if constexpr (SrcTile::Loc == TileType::Vec) {
        if constexpr (DstTile::Loc == TileType::Vec) {
            if constexpr ((SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) &&
                          (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
                TMovToVecNd2Nz<typename DstTile::DType, DstTile, SrcTile>(dst.data(), src.data(), dst.GetValidRow(),
                                                                          dst.GetValidCol(), src.GetValidRow());
            } else {
                TMovToVec<DstTile, SrcTile>(dst, src);
            }
        } else if constexpr (DstTile::Loc == TileType::Mat) {
            CommonCheck<DstTile, SrcTile>();
            TExtractVecToMat<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, src.GetValidRow(), src.GetValidCol(),
                                               dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename DstTile, typename SrcTile, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename DstTile, typename SrcTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
                  "Quant is not support in dual Dst Mode.");
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename FpTile>
__tf__ PTO_INTERNAL void SetFPC(typename FpTile::TileDType __in__ fp)
{
    __fbuf__ typename FpTile::DType *dstAddrFp = (__fbuf__ typename FpTile::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

template <typename DstTile, typename SrcTile, typename FpTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, FpTile &fp)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    SetFPC<FpTile>(fp.data());
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, typename FpTile, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, FpTile &fp)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
                  "Quant is not support in dual Dst Mode.");
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTile>(fp.data());
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif
