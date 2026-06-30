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

#include "pto/npu/kirinX90/TExtract.hpp"
#include "pto/common/arch/memory/tmov_common.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TMovToBt(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using DstType = typename DstTileData::DType;
    using SrcType = typename SrcTileData::DType;
    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                      (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>),
                  "Fix: TMOV: Bias data type only supports int32_t or half.");

    constexpr const int BIAS_TABLE_UNIT = 64;
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Bias, row must be 1.");
    static_assert(DstTileData::Cols * sizeof(DstType) % BIAS_TABLE_UNIT == 0,
                  "TMov: When TileType is Bias, col * sizeof(Dtype) must be aligned to 64.");
    static_assert(DstTileData::Cols * sizeof(DstType) <= PTO_BIAS_SIZE_BYTES,
                  "TMov: The memory occupation of BiasTile exceeds 1.0KB bias table size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    uint64_t dstAddrP = (uint64_t)dst;

    constexpr bool convControl = false;
    constexpr uint16_t burstNum = 1;
    constexpr const int BURST_LEN_UNIT_SHIFT = 6; // BURST_LEN_UNIT = 64;
    constexpr uint16_t burstLen = SrcTileData::Numel * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;

    copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, burstNum, burstLen, 0, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr const int FIXPIPE_BUFFER_UNIT = 128;
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Scaling, row must be 1.");
    static_assert(DstTileData::Cols * sizeof(DstType) % FIXPIPE_BUFFER_UNIT == 0,
                  "TMov: When TileType is Scaling, col * sizeof(Dtype) must be aligned to 128.");
    static_assert(DstTileData::Cols * sizeof(DstType) <= PTO_FBUF_SIZE_BYTES,
                  "TMov: The memory occupation of FbTile exceeds 6.0KB fixpipe buffer size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)__cce_get_tile_ptr(dst);

    constexpr uint16_t burstNum = 1;
    constexpr const int BURST_LEN_UNIT_SHIFT = 7; // BURST_LEN_UNIT = 128;
    constexpr uint16_t burstLen = SrcTileData::Numel * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;

    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, 0, 0);
}

template <typename DstTileData, typename SrcTileData, unsigned blockSizeElem, unsigned srcStride, unsigned dstStride>
__tf__ PTO_INTERNAL void TMovToVecImpl(typename DstTileData::TileDType __out__ dst,
                                       typename SrcTileData::TileDType __in__ src, uint64_t validRow, uint64_t validCol)
{
    using T = typename SrcTileData::DType;
    using U = typename DstTileData::DType;
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);

    static_assert(sizeof(T) == sizeof(U), "TMOV: src and dst data type is different!");
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
                vlds(vreg0, srcPtr, i * SrcTileData::RowStride + j * nRepeatElem, NORM);
                vsts(vreg0, dstPtr, i * DstTileData::RowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TMovToVec(DstTileData &dst, SrcTileData &src)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TMovToVecImpl<DstTileData, SrcTileData, blockSizeElem, SrcTileData::RowStride, DstTileData::RowStride>(
        dst.data(), src.data(), validRow, validCol);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TMovToVecNd2Nz(typename DstTileData::TileDType __out__ dst,
                                        typename SrcTileData::TileDType __in__ src, uint32_t validRow,
                                        uint32_t validCol, uint32_t srcValidRow,
                                        unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, float>::value) ||
                      (std::is_same<T, int32_t>::value) || (std::is_same<T, int8_t>::value),
                  "Dst and src must be float/int32_t/half/int8_t/.");

    using U = std::conditional_t<sizeof(T) == 1, uint8_t, T>;
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ U *srcPtr = (__ubuf__ U *)__cce_get_tile_ptr(src);
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t srcByteSize = srcRow * srcCol * sizeof(U);
    constexpr int32_t dstByteSize = DstTileData::Rows * DstTileData::Cols * sizeof(U);

    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(U);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr bool isOptForConflict = DstTileData::Compact == CompactMode::RowPlusOne;
    uint32_t alignRow = (srcRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = isOptForConflict ? ((alignRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
                                              (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t virtualRow = isOptForConflict ? alignRow + 1 : alignRow;
    uint32_t repeatStride = 1;
    uint16_t innerLoopNum = validRow - 1;
    uint32_t cfgVsstb = (blockStride << 16u) | (1 & 0xFFFFU);
    uint32_t repeatStrideLast = (REPEAT_BYTE * virtualRow - innerLoopNum * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE;
    uint32_t cfgVsstbLast = (blockStride << 16u) | (repeatStrideLast & 0xFFFFU);
    uint32_t srcOffset = innerLoopNum * SrcTileData::RowStride;
    __VEC_SCOPE__
    {
        RegTensor<U> vreg;
        MaskReg preg;
        uint32_t cols = validCol;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            preg = CreatePredicate<U>(cols);
            for (uint16_t i = 0; i < innerLoopNum; ++i) {
                vlds(vreg, srcPtr, SrcTileData::RowStride, NORM, POST_UPDATE);
                vsstb(vreg, dstPtr, cfgVsstb, preg, POST_UPDATE);
            }
            vlds(vreg, srcPtr, elementsPerRepeat, NORM, POST_UPDATE);
            vsstb(vreg, dstPtr, cfgVsstbLast, preg, POST_UPDATE);
            srcPtr -= srcOffset;
        }
    }
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToVecNd2Zz(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
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
    uint16_t repeatTimesCols = CeilDivision(validCol, elementsPerRepeat);
    uint32_t alignRow = (srcRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t repeatStride = 1;
    uint32_t cfgVsstb = (blockStride << 16u) | (1 & 0xFFFFU);
    uint16_t repeatTimesRows = alignRow / FRACTAL_NZ_ROW;

    __VEC_SCOPE__
    {
        RegTensor<U> vreg;
        MaskReg preg;
        uint32_t rows;
        uint16_t innerLoopNum;
        for (uint16_t i = 0; i < repeatTimesRows; ++i) {
            uint32_t cols = validCol;
            preg = CreatePredicate<U>(cols);
            rows = validRow;
            for (uint16_t j = 0; j < repeatTimesCols; ++j) {
                innerLoopNum = rows % FRACTAL_ZZ_ROW;
                rows -= FRACTAL_ZZ_ROW;
                for (uint16_t k = 0; k < innerLoopNum; ++k) {
                    vlds(vreg, srcPtr, (i * FRACTAL_ZZ_ROW) + k * SrcTile::RowStride + j * elementsPerRepeat, NORM);
                    vsstb(vreg, dstPtr, cfgVsstb, preg, POST_UPDATE);
                }
            }
        }
    }
}

template <typename DstTile>
struct TmovConfig {
    using DType = typename DstTile::DType;

    static constexpr bool isNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    static constexpr bool isNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    static constexpr bool isFloat = std::is_same_v<DType, float>;

    static constexpr bool channelSplitEnable = isNz2Nz && isFloat && (DstTile::SFractalSize == 512);
    static constexpr bool is1024Nz = isNz2Nz && (DstTile::SFractalSize == 1024);

    static constexpr uint32_t baseC0Size = C0_SIZE_BYTE / sizeof(DType);
    static constexpr uint32_t specialC0Size = (!channelSplitEnable && is1024Nz) ? (2 * baseC0Size) : baseC0Size;

    // accDstStride 计算
    static constexpr uint32_t accDstStride = isNz2Nd ? DstTile::Cols : (DstTile::Rows * specialC0Size);

    // validCol 对齐大小计算
    static constexpr int32_t nzAlign =
        (isFloat && channelSplitEnable) ? baseC0Size : (isFloat ? FRACTAL_NZ_ROW : baseC0Size);
};

template <typename DstTile, typename SrcTile, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TMovCcToCb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    using Cfg = TmovConfig<DstTile>; // 引入配置

    constexpr uint32_t dstStride = Cfg::accDstStride;
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    if constexpr (Cfg::isNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr (Cfg::isFloat) {
            validCol = CeilAlignment(validCol, Cfg::nzAlign);
        } else {
            validCol = CeilAlignment(validCol, Cfg::baseC0Size);
        }
    }

    if constexpr (Cfg::isNz2Nd) {
        SetLoop3Para();
    }
    auto srcStride = CeilAlignment(validRow, BLOCK_LEN);
    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);

    copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, 0, QuantPre, reluMode,
                           Cfg::channelSplitEnable, Cfg::isNz2Nd, false);
}

template <typename DstTile, typename SrcTile, QuantMode_t QuantPre, ReluPreMode reluMode,
          STPhase Phase = STPhase::Unspecified>
__tf__ PTO_INTERNAL void TMovCcToUb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    using Cfg = TmovConfig<DstTile>; // 引入配置
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr uint32_t dstStride = Cfg::accDstStride;
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    if constexpr (Cfg::isNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr (Cfg::isFloat) {
            validCol = CeilAlignment(validCol, Cfg::nzAlign);
        } else {
            validCol = CeilAlignment(validCol, Cfg::baseC0Size);
        }
    }

    if constexpr (Cfg::isNz2Nd) {
        SetLoop3Para();
    }
    auto srcStride = CeilAlignment(validRow, BLOCK_LEN);
    __ubuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);

    copy_matrix_cc_to_ubuf(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, unitFlagCtrl, QuantPre,
                           reluMode, Cfg::channelSplitEnable, Cfg::isNz2Nd, false);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_TILE_IMPL(DstTileData &dst, SrcTileData &src)
{
    static_assert((SrcTileData::Loc == TileType::Mat &&
                   (DstTileData::Loc == TileType::Left || DstTileData::Loc == TileType::Right ||
                    DstTileData::Loc == TileType::Bias || DstTileData::Loc == TileType::Scaling)) ||
                      (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) ||
                      (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Vec) ||
                      (SrcTileData::Loc == TileType::Acc &&
                       (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec)),
                  "TMov: Invalid TileType.");
    if constexpr (SrcTileData::Loc == TileType::Mat) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
                      "TMov: The shape of src needs to be the same as that of dst.");
        if constexpr (DstTileData::Loc == TileType::Bias) {
            TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == TileType::Scaling) {
            TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == TileType::Left) {
            TMovToLeft<DstTileData, SrcTileData>(dst, src);
        } else if constexpr (DstTileData::Loc == TileType::Right) {
            TMovToRight<DstTileData, SrcTileData>(dst, src);
        }
    } else if constexpr (SrcTileData::Loc == TileType::Acc) {
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        if constexpr (DstTileData::Loc == TileType::Vec) {
            TMovCcToUb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
        } else if constexpr (DstTileData::Loc == TileType::Mat) {
            TMovCcToCb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
        }
    } else if constexpr (SrcTileData::Loc == TileType::Vec) {
        if constexpr (DstTileData::Loc == TileType::Vec) {
            if constexpr ((SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) &&
                          (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor))) {
                TMovToVecNd2Nz<typename DstTileData::DType, DstTileData, SrcTileData>(
                    dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow());
            } else if constexpr ((SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox) &&
                                 (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)) {
                TMovToVecNd2Zz<typename DstTileData::DType, DstTileData, SrcTileData>(
                    dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow());
            } else {
                TMovToVec<DstTileData, SrcTileData>(dst, src);
            }
        } else if constexpr (DstTileData::Loc == TileType::Mat) {
            if constexpr ((SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox) &&
                          (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox)) {
                TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, src.GetValidRow(),
                                                           src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
            } else if constexpr ((!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor) &&
                                 (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)) {
                TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, src.GetValidRow(),
                                                           src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
            } else {
                static_assert(sizeof(typename DstTileData::DType) == 0,
                              "TMov Vec->Mat: Only support ND->ND or ZZ->ZZ on kirinX90.");
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TMOV_CONVTILE_IMPL(dst, src);
    } else {
        TMOV_TILE_IMPL(dst, src);
    }
}
// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, quantPre, reluMode, Phase>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, quantPre, reluMode, Phase>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

// vector quant
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTileData>(fp.data());
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, quantPre, reluMode, Phase>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

} // namespace pto
#endif // TMOV_HPP
