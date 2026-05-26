/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

namespace pto {
template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TMovToBt(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr const int BURST_LEN_UNIT = 64;

    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                      (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>),
                  "Fix: TMOV: Bias data type only supports int32_t or half.");
    static_assert(SrcTileData::Rows == 1, "TMov: When TileType is Bias, row must be 1");
    static_assert(SrcTileData::Cols * sizeof(SrcType) % BURST_LEN_UNIT == 0,
                  "TMov: When TileType is Bias, col * sizeof(srcDType) must be aligned to 64");
    static_assert(DstTileData::Cols * sizeof(DstType) <= PTO_BIAS_SIZE_BYTES,
                  "Fix: TMov: The memory occupation of BiasTile exceeds 1.0KB bias table size.");
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(__cce_get_tile_ptr(src));
    uint64_t dstAddr = (uint64_t)(__cce_get_tile_ptr(dst));

    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;

    copy_cbuf_to_bt(dstAddr, srcAddr, false, 1, burstLen, 0, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
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

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(__cce_get_tile_ptr(src));
    __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)(__cce_get_tile_ptr(dst));

    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;
    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, (uint16_t)1, burstLen, (uint16_t)0, (uint16_t)0);
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
    if constexpr (DstTileData::Cols == SrcTileData::Cols || DstTileData::Rows == 1) {
        unsigned blockLen = (DstTileData::Cols * validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        if constexpr (DstTileData::Cols == DstTileData::ValidCol) {
            pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, blockLen, 0, 0);
        } else {
            if (DstTileData::Cols == validCol) {
                pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, blockLen, 0, 0);
            } else {
                blockLen = (validCol * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
                for (int i = 0; i < validRow; i++) {
                    pto_copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr + i * srcStride, 1, blockLen, 0, 0);
                }
            }
        }
    } else {
        unsigned blockLen = CeilDivision(validCol * sizeof(T), BLOCK_BYTE_SIZE);
        unsigned srcGap = SrcTileData::Cols * sizeof(T) / BLOCK_BYTE_SIZE - blockLen;
        unsigned dstGap = DstTileData::Cols * sizeof(T) / BLOCK_BYTE_SIZE - blockLen;
        for (int i = 0; i < validRow; i++) {
            pto_copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr + i * srcStride, 1, blockLen, srcGap, dstGap);
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
    } // end of VF
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToCb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                              uint16_t validRow, uint16_t validCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
    __cc__ SrcType *srcAddr = (__cc__ SrcType *)__cce_get_tile_ptr(src);
    __cbuf__ DstType *dstAddr = (__cbuf__ DstType *)__cce_get_tile_ptr(dst);

    constexpr uint32_t dstStride_dst_D = DstTileData::Rows;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    validCol = CeilDivision(validCol, c0Size) * c0Size;
    copy_matrix_cc_to_cbuf(dstAddr, srcAddr, 0, validCol, SrcTileData::Rows, dstStride_dst_D, srcStride, 0, QuantPre,
                           reluMode, false, false);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToLeft(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                                dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToACompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                               dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToRight(DstTileData &dst, SrcTileData &src)
{
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                                dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBCompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                               dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_CONVTILE_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(),
                                                  dst.GetValidCol(), 0, 0);
    }
}
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_TILE_IMPL(DstTileData &dst, SrcTileData &src)
{
    static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
                  "TMov: The shape of src needs to be the same as that of dst.");
    static_assert((SrcTileData::Loc == TileType::Mat &&
                   (DstTileData::Loc == TileType::Left || DstTileData::Loc == TileType::Right ||
                    DstTileData::Loc == TileType::Bias || DstTileData::Loc == TileType::Scaling)) ||
                      (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) ||
                      (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Vec) ||
                      (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Acc),
                  "TMov: Invalid TileType.");
    if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Left) {
        TMovToLeft<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Right) {
        TMovToRight<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Bias) {
        TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Scaling) {
        TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec) {
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
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        if constexpr ((SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox) &&
                      (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox)) {
            TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, src.GetValidRow(),
                                                       src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr ((SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor) &&
                             (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)) {
            TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, src.GetValidRow(),
                                                       src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
        } else {
            static_assert(sizeof(typename DstTileData::DType) == 0,
                          "TMov Vec->Mat: Only support ND->ND or ZZ->ZZ on kirinX90.");
        }
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TMovCcToCb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
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
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

// vector quant
template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7))
                             << 8; // fpc[15:8] means Quant_PRE_ADDR, uint of 128(2^7)bytes
    set_fpc(deqTensorAddr);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTileData>(fp.data());
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif // TMOV_HPP
