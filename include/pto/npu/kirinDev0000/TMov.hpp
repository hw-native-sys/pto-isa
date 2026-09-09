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
#include "pto/npu/kirin9030/common.hpp"
#include "pto/npu/kirinDev0000/TExtract.hpp"

namespace pto {

PTO_INTERNAL void SetLoop3Para()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
}

// Configure FIXP_NZ_PARA SPR for fix_cbuf_to_ubuf on dav-l510.
// [15:0]  loop4_size (= ndNum for NZ2ND/NZ2DN)
// [31:16] loop2_src_stride (in unit of C0_SIZE=32B) for NORMAL_DMA/NZ2ND/NZ2DN;
//         for LOOP_ENHANCE it is loop1_src_stride instead.
// [47:32] loop3_src_stride (in unit of C0_SIZE=32B)
// [63:48] loop4_src_stride (in unit of C0_SIZE=32B)
// Defined in TExtract.hpp (included below) — forward-declare here.
__tf__ PTO_INTERNAL void SetFixpNzPara(
    uint16_t loop4Size, uint16_t loop2SrcStride, uint16_t loop3SrcStride, uint16_t loop4SrcStride);

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

template <typename DstTile, typename SrcTile>
PTO_INTERNAL constexpr void CommonCheck()
{
    using T = typename DstTile::DType;
    using U = typename SrcTile::DType;
    static_assert(std::is_same_v<T, U>, "Fix: TMov Destination and Source tile data types must be the same.");

    if constexpr (DstTile::Loc == TileType::Left) {
        static_assert(
            std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
            "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(
            DstTile::SFractal == SLayout::RowMajor && !DstTile::isRowMajor,
            "Fix: TMov: Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    }
    if constexpr (DstTile::Loc == TileType::Right) {
        static_assert(
            std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
            "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(
            DstTile::SFractal == SLayout::ColMajor && DstTile::isRowMajor,
            "Fix: TMov: Dst fractal format should be (BFractal: RowMajor, SFractal: ColMajor).");
    }

    static_assert(
        (SrcTile::SFractal == SLayout::ColMajor && SrcTile::isRowMajor) ||
            (SrcTile::SFractal == SLayout::RowMajor && !SrcTile::isRowMajor) || (SrcTile::isRowMajor),
        "TMov: SrcTile Invalid Fractal.");
}

template <typename U>
PTO_INTERNAL void TMovToVecNd2NzFillZero(__ubuf__ U* dstPtr, uint32_t totalDstBytes)
{
    __VEC_SCOPE__
    {
        RegTensor<uint8_t> vregZero;
        MaskReg pgAll = pset_b8(PAT_ALL);
        vdup(vregZero, (uint8_t)0, pgAll, MODE_ZEROING);
        __ubuf__ uint8_t* dstBytes = (__ubuf__ uint8_t*)dstPtr;
        uint32_t fillLen = totalDstBytes;
        uint16_t fillRepeats = CeilDivision(totalDstBytes, static_cast<uint32_t>(ELE_CNT_B8));
        for (uint16_t fi = 0; fi < fillRepeats; ++fi) {
            MaskReg pregFill = CreatePredicate<uint8_t>(fillLen);
            vsts(vregZero, dstBytes, fi * ELE_CNT_B8, NORM_B8, pregFill);
        }
    }
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToVecNd2Nz(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint32_t validRow,
    uint32_t validCol, uint32_t srcValidRow, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    static_assert(
        (std::is_same<T, half>::value) || (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
            (std::is_same<T, int8_t>::value),
        "Dst and src must be float/int32_t/half/int8_t/.");

    using U = std::conditional_t<sizeof(T) == 1, uint8_t, T>;
    __ubuf__ U* dstPtr = (__ubuf__ U*)__cce_get_tile_ptr(dst);
    __ubuf__ U* srcPtr = (__ubuf__ U*)__cce_get_tile_ptr(src);
    constexpr int32_t srcRow = SrcTile::Rows;
    constexpr int32_t dstByteSize = DstTile::Rows * DstTile::Cols * sizeof(U);

    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(U);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr bool isOptForConflict = DstTile::Compact == CompactMode::RowPlusOne;
    uint32_t alignRow = (srcRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = isOptForConflict ? ((alignRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
                                              (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t virtualRow = isOptForConflict ? alignRow + 1 : alignRow;
    uint16_t innerLoopNum = srcValidRow - 1;
    uint32_t cfgVsstb = (blockStride << 16u) | (1 & 0xFFFFU);
    uint32_t repeatStrideLast = (CCE_VL * virtualRow - innerLoopNum * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE;
    uint32_t cfgVsstbLast = (blockStride << 16u) | (repeatStrideLast & 0xFFFFU);
    uint32_t srcOffset = innerLoopNum * SrcTile::RowStride;

    TMovToVecNd2NzFillZero<U>(dstPtr, static_cast<uint32_t>(dstByteSize));

    __VEC_SCOPE__
    {
        RegTensor<U> vreg;
        MaskReg preg;
        uint32_t cols = validCol;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            uint32_t count = cols - static_cast<uint32_t>(cols > elementsPerRepeat) * (cols - elementsPerRepeat);
            preg = CreatePredicate<U>(count);
            for (uint16_t i = 0; i < innerLoopNum; ++i) {
                vlds(vreg, srcPtr, SrcTile::RowStride, NORM, POST_UPDATE);
                vsstb(vreg, dstPtr, cfgVsstb, preg, POST_UPDATE);
            }
            vlds(vreg, srcPtr, elementsPerRepeat, NORM, POST_UPDATE);
            vsstb(vreg, dstPtr, cfgVsstbLast, preg, POST_UPDATE);
            srcPtr -= srcOffset;
            cols -= elementsPerRepeat;
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TMOV) OP_TYPE(element_wise) void TMovVecToVec(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
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
PTO_INTERNAL void TMovToVec(DstTile& dst, SrcTile& src)
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
AICORE void TMovToRight(DstTile& dst, SrcTile& src)
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
            TExtractToBTransCompact<DstTile, SrcTile>(
                dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovCbufToCbuf(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using T = typename SrcTile::DType;
    __cbuf__ T* srcAddr = (__cbuf__ T*)__cce_get_tile_ptr(src);
    __cbuf__ T* dstAddr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    uint32_t totalBytes = SrcTile::Rows * SrcTile::Cols * sizeof(T);
    uint64_t loop2DstStride = static_cast<uint64_t>(totalBytes);
    uint32_t loop3Size = totalBytes;
    fix_cbuf_to_cbuf(dstAddr, srcAddr, loop2DstStride, loop3Size, fixp_trans_mode_t::NORMAL_DMA, 0, 1);
}

template <typename DstTile, typename SrcTile, typename DstPtr, typename SrcPtr>
PTO_INTERNAL void TMovCbufToCbufAccDispatch(
    DstPtr dstAddr, SrcPtr srcData, uint16_t validRow, uint16_t validCol, uint16_t sizeMul)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();

    if constexpr (enableNz2Nd) {
        validCol = CeilAlignment(validCol, c0Size);
        uint16_t alignedRow = CeilAlignment(validRow, FRACTAL_NZ_ROW);
        uint64_t loop2DstStride = static_cast<uint64_t>(dstStride * sizeof(dstType));
        uint32_t loop3Size = static_cast<uint32_t>(validCol * sizeMul);
        SetFixpNzPara(1, 1, alignedRow, 0);
        fix_cbuf_to_cbuf(dstAddr, srcData, loop2DstStride, loop3Size, fixp_trans_mode_t::NZ2ND, 0, alignedRow);
        SetFixpNzPara(0, 0, 0, 0);
    } else if constexpr (enableNz2Dn) {
        SetLoop3Para();
        uint16_t alignedRow = CeilAlignment(validRow, c0Size);
        uint64_t loop2DstStride = static_cast<uint64_t>(dstStride * sizeof(dstType));
        uint32_t loop3Size = static_cast<uint32_t>(alignedRow * c0Size * sizeof(dstType) / BLOCK_BYTE_SIZE);
        uint16_t loop2Size = static_cast<uint16_t>(validCol / FRACTAL_NZ_ROW * sizeMul);
        SetFixpNzPara(1, static_cast<uint16_t>(loop3Size), 0, 0);
        fix_cbuf_to_cbuf(dstAddr, srcData, loop2DstStride, loop3Size, fixp_trans_mode_t::NZ2DN, 0, loop2Size);
        SetFixpNzPara(0, 0, 0, 0);
    } else {
        uint32_t totalBytes = static_cast<uint32_t>(SrcTile::Rows * SrcTile::Cols * sizeof(srcType));
        uint32_t totalBlocks = totalBytes / BLOCK_BYTE_SIZE;
        SetFixpNzPara(1, static_cast<uint16_t>(totalBlocks), 1, 0);
        fix_cbuf_to_cbuf(
            dstAddr, srcData, static_cast<uint64_t>(totalBytes), totalBytes, fixp_trans_mode_t::NORMAL_DMA, 0, 1);
        SetFixpNzPara(0, 0, 0, 0);
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovCbufToCbufAcc(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    static_assert(
        std::is_same_v<srcType, dstType>,
        "TMovCbufToCbufAcc: source and destination data types must be the same (no quantization on kirinDev0000).");

    __cbuf__ dstType* dstAddr = (__cbuf__ dstType*)__cce_get_tile_ptr(dst);
    __cbuf__ srcType* srcData = (__cbuf__ srcType*)__cce_get_tile_ptr(src);

    if constexpr (sizeof(dstType) == 4) {
        using b16Type = uint16_t;
        constexpr uint16_t sizeMul = sizeof(dstType) / sizeof(b16Type);
        TMovCbufToCbufAccDispatch<DstTile, SrcTile>(
            (__cbuf__ b16Type*)dstAddr, (__cbuf__ b16Type*)srcData, validRow, validCol, sizeMul);
    } else {
        TMovCbufToCbufAccDispatch<DstTile, SrcTile>(dstAddr, srcData, validRow, validCol, 1);
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToBt(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t cvtMode = 0,
    uint8_t fixVal = 0)
{
    using DstType = typename DstTile::DType;
    using SrcType = typename SrcTile::DType;
    static_assert(
        (std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
            (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>),
        "Fix: TMOV: Bias data type only supports int32_t or half.");

    constexpr const int BIAS_TABLE_UNIT = 64;
    static_assert(SrcTile::Rows == 1, "TMov: When TileType is Bias, row must be 1.");
    static_assert(
        DstTile::Cols * sizeof(DstType) % BIAS_TABLE_UNIT == 0,
        "TMov: When TileType is Bias, col * sizeof(Dtype) must be aligned to 64.");
    static_assert(
        DstTile::Cols * sizeof(DstType) <= PTO_BIAS_SIZE_BYTES,
        "TMov: The memory occupation of BiasTile exceeds bias table size.");

    __cbuf__ SrcType* srcAddrP = (__cbuf__ SrcType*)__cce_get_tile_ptr(src);
    uint64_t dstAddrP = (uint64_t)dst;

    constexpr uint32_t srcBytes = SrcTile::Numel * sizeof(SrcType);
    constexpr uint16_t burstLen = static_cast<uint16_t>(srcBytes / BLOCK_BYTE_SIZE);
    constexpr uint16_t burstNum = 1;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    // Xm register layout per ISA:
    // [1:0]   reserved = 0
    // [3:2]   CVT_MODE: 00=no conversion, 01=bf16/f16→f32, 10=f16/bf16/f32→s32, 11=reserved
    // [15:4]  burst number
    // [31:16] burst length of source data in 32B units
    // [47:32] source gap between bursts in 32B units
    // [58:48] dst gap between bursts in 32B units (must be multiple of 2)
    // [63:59] FIX_VAL: u5 fixed point decimal bits, valid when CVT_MODE=10, range [15,24]
    uint64_t xm = (static_cast<uint64_t>(cvtMode & 0x3) << 2) | (static_cast<uint64_t>(burstNum & 0xFFF) << 4) |
                  (static_cast<uint64_t>(burstLen & 0xFFFF) << 16) | (static_cast<uint64_t>(srcGap & 0xFFFF) << 32) |
                  (static_cast<uint64_t>(dstGap & 0x7FF) << 48) | (static_cast<uint64_t>(fixVal & 0x1F) << 59);

    copy_cbuf_to_bt(dstAddrP, srcAddrP, xm);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToFb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using SrcType = typename SrcTile::DType;
    using DstType = typename DstTile::DType;
    static_assert(SrcTile::Rows == 1, "TMov: When TileType is Scaling, row must be 1.");
    static_assert(
        DstTile::Cols * sizeof(DstType) % BLOCK_BYTE_SIZE == 0,
        "TMov: When TileType is Scaling, col * sizeof(Dtype) must be aligned to 32B.");
    static_assert(
        DstTile::Cols * sizeof(DstType) <= PTO_FBUF_SIZE_BYTES,
        "TMov: The memory occupation of FbTile exceeds fixpipe buffer size.");

    __cbuf__ SrcType* srcAddrP = (__cbuf__ SrcType*)__cce_get_tile_ptr(src);
    __fbuf__ DstType* dstAddrP = (__fbuf__ DstType*)__cce_get_tile_ptr(dst);

    constexpr uint32_t srcBytes = SrcTile::Numel * sizeof(SrcType);
    constexpr uint16_t burstLen = static_cast<uint16_t>(srcBytes / BLOCK_BYTE_SIZE);
    constexpr uint16_t burstNum = 1;
    constexpr uint16_t srcStride = 0;
    constexpr uint16_t dstStride = 0;

    uint64_t dstAddr = reinterpret_cast<uint64_t>(dstAddrP);
    uint8_t dstMemBlk = 0;
    __fbuf__ DstType* blockDst = dstAddrP;
    if (dstAddr >= 0xE00) {
        dstMemBlk = 4;
        blockDst = reinterpret_cast<__fbuf__ DstType*>(dstAddr - 0xE00);
    } else if (dstAddr >= 0xC00) {
        dstMemBlk = 3;
        blockDst = reinterpret_cast<__fbuf__ DstType*>(dstAddr - 0xC00);
    } else if (dstAddr >= 0x800) {
        dstMemBlk = 1;
        blockDst = reinterpret_cast<__fbuf__ DstType*>(dstAddr - 0x800);
    }

    copy_cbuf_to_fbuf_v2(blockDst, srcAddrP, dstMemBlk, burstNum, burstLen, srcStride, dstStride);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovCbufToUbNz2Nd(
    __ubuf__ uint8_t* dstBytes, __cbuf__ uint8_t* srcBytes, uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr uint32_t innerRows = SrcTile::InnerRows;
    constexpr uint32_t innerCols = SrcTile::InnerCols;
    constexpr uint32_t innerNumel = SrcTile::InnerNumel;
    constexpr uint32_t blockNumRow = SrcTile::Rows / innerRows;
    constexpr uint32_t copyBytes = c0Size * sizeof(dstType);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
    validCol = CeilAlignment(validCol, c0Size);
    SetFixpNzPara(1, 1, 1, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(validRow); ++row) {
        uint32_t srcBlockRow = row / innerRows;
        uint32_t srcInnerRow = row % innerRows;
        for (uint32_t colBlk = 0; colBlk < static_cast<uint32_t>(validCol) / c0Size; ++colBlk) {
            uint32_t srcCol = colBlk * c0Size;
            uint32_t srcBlockCol = srcCol / innerCols;
            uint32_t srcInnerCol = srcCol % innerCols;
            uint32_t srcElemOff =
                (blockNumRow * srcBlockCol + srcBlockRow) * innerNumel + srcInnerRow * innerCols + srcInnerCol;
            uint32_t srcByteOff = srcElemOff * sizeof(dstType);
            uint32_t dstByteOff = (row * dstStride + colBlk * c0Size) * sizeof(dstType);
            fix_cbuf_to_ubuf(
                dstBytes + dstByteOff, srcBytes + srcByteOff, static_cast<uint64_t>(copyBytes),
                static_cast<uint32_t>(copyBytes), fixp_trans_mode_t::NORMAL_DMA, 0, 1);
        }
    }
    SetFixpNzPara(0, 0, 0, 0);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovCbufToUb(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    static_assert(
        std::is_same_v<srcType, dstType>,
        "TMovCbufToUb: source and destination data types must be the same (no quantization on kirinDev0000).");

    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();

    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    __cbuf__ srcType* srcData = (__cbuf__ srcType*)__cce_get_tile_ptr(src);
    __ubuf__ uint8_t* dstBytes = reinterpret_cast<__ubuf__ uint8_t*>(dstAddr);
    __cbuf__ uint8_t* srcBytes = reinterpret_cast<__cbuf__ uint8_t*>(srcData);

    if constexpr (enableNz2Nd) {
        TMovCbufToUbNz2Nd<DstTile, SrcTile>(dstBytes, srcBytes, validRow, validCol);
    } else if constexpr (enableNz2Dn) {
        SetLoop3Para();
        uint16_t alignedRow = CeilAlignment(validRow, c0Size);
        uint64_t loop2DstStride = static_cast<uint64_t>(dstStride * sizeof(dstType));
        uint32_t loop3Size = static_cast<uint32_t>(alignedRow * c0Size * sizeof(dstType) / BLOCK_BYTE_SIZE);
        uint16_t loop2Size = static_cast<uint16_t>(validCol / FRACTAL_NZ_ROW);
        SetFixpNzPara(1, static_cast<uint16_t>(loop3Size), 0, 0);
        fix_cbuf_to_ubuf(dstBytes, srcBytes, loop2DstStride, loop3Size, fixp_trans_mode_t::NZ2DN, 0, loop2Size);
        SetFixpNzPara(0, 0, 0, 0);
    } else {
        uint32_t totalBytes = static_cast<uint32_t>(SrcTile::Rows * SrcTile::Cols * sizeof(srcType));
        uint32_t totalBlocks = totalBytes / BLOCK_BYTE_SIZE;
        SetFixpNzPara(1, static_cast<uint16_t>(totalBlocks), 1, 0);
        fix_cbuf_to_ubuf(
            dstBytes, srcBytes, static_cast<uint64_t>(totalBytes), totalBytes, fixp_trans_mode_t::NORMAL_DMA, 0, 1);
        SetFixpNzPara(0, 0, 0, 0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovToBias(DstTile& dst, SrcTile& src)
{
    using SrcT = typename SrcTile::DType;
    if constexpr (std::is_same_v<SrcT, half>) {
        TMovToBt<DstTile, SrcTile>(dst.data(), src.data(), 2, 16);
    } else {
        TMovToBt<DstTile, SrcTile>(dst.data(), src.data(), 0, 0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovImplFromAcc(DstTile& dst, SrcTile& src)
{
    if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCbufToCbufAcc<DstTile, SrcTile>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCbufToUb<DstTile, SrcTile>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (DstTile::Loc == TileType::Right) {
        TMovToRight(dst, src);
    } else if constexpr (DstTile::Loc == TileType::Bias) {
        TMovToBias<DstTile, SrcTile>(dst, src);
    } else if constexpr (DstTile::Loc == TileType::Scaling) {
        TMovToFb<DstTile, SrcTile>(dst.data(), src.data());
    } else {
        static_assert(sizeof(DstTile::DType) == 0, "TMov: unsupported Acc->X path on kirinDev0000.");
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovImplFromMat(DstTile& dst, SrcTile& src)
{
    static_assert(
        !(SrcTile::Loc == TileType::Mat && DstTile::Loc == TileType::Left),
        "TMov: Mat->Left is not supported on kirinDev0000.");
    static_assert(
        (SrcTile::Rows == DstTile::Rows) && ((SrcTile::Cols == DstTile::Cols)),
        "TMov: The shape of destination and source tile must be the same.");
    if constexpr (DstTile::Loc == TileType::Right) {
        TMovToRight(dst, src);
    } else if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCbufToUb<DstTile, SrcTile>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (DstTile::Loc == TileType::Bias) {
        TMovToBias<DstTile, SrcTile>(dst, src);
    } else if constexpr (DstTile::Loc == TileType::Left || DstTile::Loc == TileType::Acc) {
        TMovCbufToCbuf<DstTile, SrcTile>(dst.data(), src.data());
    } else if constexpr (DstTile::Loc == TileType::ScaleLeft || DstTile::Loc == TileType::ScaleRight) {
        static_assert(sizeof(DstTile::DType) == 0, "TMov: ScaleLeft/ScaleRight tile type is not supported.");
    } else if constexpr (DstTile::Loc == TileType::Scaling) {
        TMovToFb<DstTile, SrcTile>(dst.data(), src.data());
    } else {
        static_assert(sizeof(DstTile::DType) == 0, "TMov: unsupported Mat->X path on kirinDev0000.");
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovImplFromVec(DstTile& dst, SrcTile& src)
{
    if constexpr (DstTile::Loc == TileType::Vec) {
        if constexpr (
            (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) &&
            (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
            TMovToVecNd2Nz<typename DstTile::DType, DstTile, SrcTile>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow());
        } else {
            TMovToVec<DstTile, SrcTile>(dst, src);
        }
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        CommonCheck<DstTile, SrcTile>();
        TExtractVecToMat<DstTile, SrcTile>(
            dst.data(), src.data(), 0, 0, src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (
        DstTile::Loc == TileType::Left || DstTile::Loc == TileType::Acc || DstTile::Loc == TileType::Scaling) {
        static_assert(
            sizeof(DstTile::DType) == 0,
            "TMov: Vec->Left/Acc/Scaling is not supported on kirinDev0000 (no fractal transform path).");
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    static_assert(SrcTile::Loc != TileType::Left, "TMov: Left tile as source is not supported on kirinDev0000.");
    static_assert(SrcTile::Loc != TileType::Scaling, "TMov: Scaling tile as source is not supported on kirinDev0000.");
    if constexpr (SrcTile::Loc == TileType::Acc) {
        TMovImplFromAcc<DstTile, SrcTile>(dst, src);
    } else if constexpr (SrcTile::Loc == TileType::Mat) {
        TMovImplFromMat<DstTile, SrcTile>(dst, src);
    } else if constexpr (SrcTile::Loc == TileType::Vec) {
        TMovImplFromVec<DstTile, SrcTile>(dst, src);
    } else {
        static_assert(sizeof(SrcTile::DType) == 0, "TMov: unsupported source tile type on kirinDev0000.");
    }
}

template <typename DstTile, typename SrcTile, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    TMOV_IMPL(dst, src);
}

template <
    typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src)
{
    TMOV_IMPL(dst, src);
}

template <
    typename DstTile, typename SrcTile, typename FpTile, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src, FpTile& fp)
{
    TMOV_IMPL(dst, src);
}

template <
    typename DstTile, typename SrcTile, typename FpTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src, FpTile& fp)
{
    TMOV_IMPL(dst, src);
}

template <
    typename DstTile, typename SrcTile, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src, uint64_t preQuantScalar)
{
    TMOV_IMPL(dst, src);
}

template <
    typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTile& dst, SrcTile& src, uint64_t preQuantScalar)
{
    TMOV_IMPL(dst, src);
}
} // namespace pto
#endif
