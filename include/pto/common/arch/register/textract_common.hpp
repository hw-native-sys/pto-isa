/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file textract_common.hpp
 * @brief Common TEXTRACT implementation for Kirin9030, KirinDev0000
 *
 * This file contains only the parts shared by all consuming architectures.
 * Architecture-specific pieces are provided by each arch shell (e.g.
 * kirin9030/TExtract.hpp) via the interface contracts declared below:
 *   - TEXTRACT_TILE_IMPL: tile-level dispatcher (L0A/Acc/quant paths differ)
 *   - TCopyUbToCbufFractal: NZ-layout UB->CBUF burst copy (burst capability differs)
 *
 * The arch shell must include its own arch common.hpp before this file.
 */

#ifndef TEXTRACT_COMMON_REGISTER_HPP
#define TEXTRACT_COMMON_REGISTER_HPP

#include <pto/common/utils.hpp>

namespace pto {

//---------------------------------------------------------------------------------------
// Interface contracts: each arch shell must provide these definitions
//---------------------------------------------------------------------------------------

// Tile-level dispatcher. Kirin9030 routes Left/Acc/quant paths through L0A;
// kirinDev0000 has no L0A and routes Mat/Acc through TMov paths instead.
// Declared here because TEXTRACT_IMPL below calls it with explicit template
// arguments (no ADL).
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol);

// NZ-layout UB->CBUF copy for TExtractVecToMat. Kirin9030 uses a single
// multi-burst copy_ubuf_to_cbuf call; kirinDev0000 hardware requires one
// 32-byte-unit burst per block. Declared here because TExtractVecToMat below
// calls it with explicit template arguments (no ADL).
template <typename T>
__tf__ PTO_INTERNAL void TCopyUbToCbufFractal(
    __cbuf__ T* dstPtr, __ubuf__ T* srcPtr, uint16_t blockCout, uint16_t blockLen, uint16_t srcStride);

//---------------------------------------------------------------------------------------
// Common constants
//---------------------------------------------------------------------------------------

constexpr const int SHIFT_M_STEP_B8 = 1;   // 2^1 = 2
constexpr const int M_STEP_MIN_VAL_B8 = 2; // m_step per loop for fp8

//---------------------------------------------------------------------------------------
// CBUF -> L0B (Right operand) helpers
//---------------------------------------------------------------------------------------

template <typename DstTile, typename SrcTile, bool Transpose>
__tf__ PTO_INTERNAL void TExtractToB(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t srcRow = SrcTile::Rows;
    constexpr int32_t srcCol = SrcTile::Cols;
    constexpr int32_t dstRow = DstTile::Rows;
    constexpr int32_t dstCol = DstTile::Cols;
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __cb__ DataType* dstAddr = (__cb__ DataType*)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        pto_load_cbuf_to_cb<false>(
            dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcRow must be aligned");
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        pto_load_cbuf_to_cb<true>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToBCompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __cb__ DataType* dstAddr = (__cb__ DataType*)__cce_get_tile_ptr(dst);
    uint16_t madNAlign = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madNAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTile::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    pto_load_cbuf_to_cb<false>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToBTransCompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __cb__ DataType* dstAddr = (__cb__ DataType*)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madNAlign = CeilDivision(madN, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t kStep = (madNAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    constexpr uint16_t srcStride = SrcTile::Rows >> SHIFT_BLOCK_LEN;
    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            pto_load_cbuf_to_cb<true>(
                dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        pto_load_cbuf_to_cb<true>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToMat(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint32_t srcValidRow, uint32_t srcValidCol, uint32_t dstValidRow, uint32_t dstValidCol)
{
    using T = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t offset = SrcTile::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    if constexpr (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) {
        offset = indexRow * srcValidCol + indexCol;
    }

    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src) + offset;
    __cbuf__ T* dstPtr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    if constexpr (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) {
        uint16_t blockLen = dstValidRow * dstValidCol * sizeof(T) / BLOCK_BYTE_SIZE;
        // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0);
    } else if constexpr (!SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::RowMajor)) {
        uint16_t blockCout = CeilDivision(dstValidCol, c0Size);
        uint32_t alignRow = (dstValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        uint16_t blockLen = alignRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = SrcTile::Rows - DstTile::Rows;
        TCopyUbToCbufFractal<T>(dstPtr, srcPtr, blockCout, blockLen, srcStride);
    }
}

template <typename T>
constexpr bool is_textract_supported_type = std::disjunction_v<
    std::is_same<T, int8_t>, std::is_same<T, uint8_t>, std::is_same<T, half>, std::is_same<T, int16_t>,
    std::is_same<T, uint16_t>, std::is_same<T, int32_t>>;

template <typename DstType, typename SrcType>
PTO_INTERNAL void CheckTExtractToL0()
{
    static_assert(std::is_same_v<DstType, SrcType>, "Dst data type must be same with Src.");
    static_assert(
        (std::is_same_v<SrcType, half>) || (std::is_same_v<SrcType, int16_t>) || (std::is_same_v<SrcType, uint16_t>) ||
            (std::is_same_v<SrcType, int8_t>) || (std::is_same_v<SrcType, uint8_t>),
        "The data type must be restricted to half/(u)int8/(u)int16.");
}

template <typename DstTile, typename SrcTile>
AICORE void TExtractToRight(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        (SrcTile::SFractal == SLayout::ColMajor && SrcTile::isRowMajor) ||
            (SrcTile::SFractal == SLayout::RowMajor && !SrcTile::isRowMajor),
        "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTile::SFractal == SLayout::ColMajor && DstTile::isRowMajor, "TExtract: DstTile Invalid Fractal");
    CheckTExtractToL0<typename DstTile::DType, typename SrcTile::DType>();
    if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTile, SrcTile>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToBTransCompact<DstTile, SrcTile>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToBConv(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t srcCol,
    uint16_t dstValidRow, uint16_t dstValidCol, uint16_t indexRow, uint16_t indexCol)
{
    using DataType = typename SrcTile::DType;
    constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(DataType);

    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __cb__ DataType* dstAddr = (__cb__ DataType*)__cce_get_tile_ptr(dst);
    uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = dstValidColAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (dstValidRowAlign * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = dstValidColAlign >> SHIFT_BLOCK_LEN;
    pto_load_cbuf_to_cb<false>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TextractConvTileCheck(DstTile& dst, SrcTile& src)
{
    static_assert(
        std::is_same_v<typename DstTile::DType, int8_t> || std::is_same_v<typename DstTile::DType, uint8_t> ||
            std::is_same_v<typename DstTile::DType, int16_t> || std::is_same_v<typename DstTile::DType, uint16_t> ||
            std::is_same_v<typename DstTile::DType, int32_t> || std::is_same_v<typename DstTile::DType, uint32_t> ||
            std::is_same_v<typename DstTile::DType, half> || std::is_same_v<typename DstTile::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
    static_assert(SrcTile::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTile::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(
        sizeof(typename DstTile::DType) == sizeof(typename SrcTile::DType),
        "Fix: Source dtype must be same with dst dtype!");

    static_assert(
        (SrcTile::layout == Layout::FRACTAL_Z) || (SrcTile::layout == Layout::FRACTAL_Z_3D),
        "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(
        DstTile::SFractal == SLayout::ColMajor && DstTile::isRowMajor,
        "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol)
{
    TextractConvTileCheck<DstTile, SrcTile>(dst, src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTile::DType);
    if constexpr (SrcTile::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(
            SrcTile::staticShape[2] == FRACTAL_NZ_ROW && SrcTile::staticShape[3] == c0ElemCount,
            "Fix: The SrcTile last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTile, SrcTile>(
            dst.data(), src.data(), srcCol, dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else { //  [C1,H,W,N,C0]
        TExtractToBConv<DstTile, SrcTile>(
            dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

//---------------------------------------------------------------------------------------
// UB -> UB (Vec -> Vec) helpers
//---------------------------------------------------------------------------------------

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNDImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTile::RowStride;
    constexpr uint32_t srcRowStride = SrcTile::RowStride;

    __ubuf__ T* srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
    uint16_t rowBurstLen = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);

    if (validCol == dstRowStride && validCol == srcRowStride && totalBytes >= BLOCK_BYTE_SIZE) {
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, 1, burstLen, 0, 0);
    } else {
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, validRow, rowBurstLen, srcGap, dstGap);
    }
}

// For 1-byte non-int8/uint8 dtypes (hifloat8/float8_*), vector intrinsics (vlds/vsts/...) have no
// overload, so reinterpret the UB pointers as int8 and operate via int8 intrinsics. Semantics match
// since each element occupies exactly one byte. fp4 (sub-byte) is handled separately via byte-DMA.
template <typename T>
using TExtractRegT =
    std::conditional_t<sizeof(T) == 1 && !std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>, int8_t, T>;

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNDAlignedImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    using RegT = TExtractRegT<T>;
    __ubuf__ RegT* dstAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(dst);
    __ubuf__ RegT* srcAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTile::RowStride;
    constexpr uint32_t srcRowStride = SrcTile::RowStride;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(RegT);

    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
        RegTensor<RegT> vreg;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(validCol), elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t sreg = static_cast<uint32_t>(validCol);
            uint32_t srcRowOff = (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
            uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                preg = CreatePredicate<RegT>(sreg);
                vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, NORM);
                vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNDVectorImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    using RegT = TExtractRegT<T>;
    __ubuf__ RegT* dstAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(dst);
    __ubuf__ RegT* srcAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTile::RowStride;
    constexpr uint32_t srcRowStride = SrcTile::RowStride;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(RegT);

    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
        RegTensor<RegT> vreg;
        UnalignReg ureg;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(validCol), elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t sreg = static_cast<uint32_t>(validCol);
            __ubuf__ RegT* psrc = srcAddr + (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
            uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                preg = CreatePredicate<RegT>(sreg);
                vldas(ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                vldus(vreg, ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNDScalarImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTile::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[0] = srcAddr[indexRow * srcRowStride + indexCol];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTile, typename SrcTile>
PTO_INTERNAL void TExtractVecToVecNDDispatch(DstTile& dst, SrcTile& src, uint32_t indexRow, uint32_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());

    PTO_ASSERT(
        indexRow + DstTile::ValidRow <= SrcTile::Rows, "TEXTRACT ND_VEC : indexRow + dstValidRows exceeds srcRows!");
    PTO_ASSERT(
        indexCol + DstTile::ValidCol <= SrcTile::Cols, "TEXTRACT ND_VEC : indexCol + dstValidCols exceeds srcCols!");

    constexpr bool kStridesAligned = (SrcTile::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                                     (DstTile::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0);
    constexpr bool kValidColAligned = (DstTile::ValidCol * sizeof(T) % BLOCK_BYTE_SIZE == 0);

    if constexpr (kStridesAligned) {
        if (indexCol * sizeof(T) % BLOCK_BYTE_SIZE == 0) {
            if constexpr (kValidColAligned) {
                TExtractVecToVecNDImpl<T, DstTile, SrcTile>(
                    dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
            } else {
                TExtractVecToVecNDAlignedImpl<T, DstTile, SrcTile>(
                    dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
            }
        } else {
            TExtractVecToVecNDVectorImpl<T, DstTile, SrcTile>(
                dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
        }
    } else {
        TExtractVecToVecNDVectorImpl<T, DstTile, SrcTile>(
            dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
    }
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNZScalarImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRows = SrcTile::Rows;
    uint32_t srcOffset = (indexCol / c0Size) * srcRows * c0Size + indexRow * c0Size + (indexCol % c0Size);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[0] = srcAddr[srcOffset];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToVecNZImpl(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t srcRow, uint16_t indexRow, uint16_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint32_t byteValidCol = validCol;
    uint32_t byteIndexCol = indexCol;
    uint16_t burstNum = static_cast<uint16_t>(CeilDivision(byteValidCol, c0Size));
    uint16_t burstLen = (validRow * c0Size * typeSize) / BLOCK_BYTE_SIZE;
    uint32_t colBlockOffset = (byteIndexCol / c0Size) * srcRow * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (byteIndexCol % c0Size);
    uint32_t srcOffset = colBlockOffset + rowOffset;
    uint16_t srcGap = static_cast<uint16_t>(srcRow - validRow);
    uint16_t dstGap = static_cast<uint16_t>(DstTile::Rows - validRow);
    __ubuf__ T* srcStart = srcAddr + srcOffset;
    pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, burstNum, burstLen, srcGap, dstGap);
}

//---------------------------------------------------------------------------------------
// Top-level dispatcher
//---------------------------------------------------------------------------------------

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXTRACT_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec) {
        using T = typename DstTile::DType;
        static_assert(
            std::is_same<typename DstTile::DType, typename SrcTile::DType>::value,
            "TEXTRACT Vec→Vec : Source and destination data types must match");
        static_assert(
            (std::is_same<T, half>::value) || (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, int8_t>::value),
            "TEXTRACT Vec→Vec : Unsupported data type.");
        if constexpr (DstTile::isRowMajor && SrcTile::isRowMajor) {
            static_assert(
                DstTile::Rows <= SrcTile::Rows, "TEXTRACT ND Vec→Vec : Destination rows must not exceed source rows");
            static_assert(
                DstTile::Cols <= SrcTile::Cols, "TEXTRACT ND Vec→Vec : Destination cols must not exceed source cols");
            uint32_t idxRow = static_cast<uint32_t>(indexRow);
            uint32_t idxCol = static_cast<uint32_t>(indexCol);
            if constexpr (DstTile::ValidRow == 1 && DstTile::ValidCol == 1) {
                PTO_ASSERT(idxRow < SrcTile::Rows, "TEXTRACT ND Vec→Vec : indexRow exceeds srcRows!");
                PTO_ASSERT(idxCol < SrcTile::Cols, "TEXTRACT ND Vec→Vec : indexCol exceeds srcCols!");
                TExtractVecToVecNDScalarImpl<T, DstTile, SrcTile>(dst.data(), src.data(), idxRow, idxCol);
            } else {
                TExtractVecToVecNDDispatch<T, DstTile, SrcTile>(dst, src, idxRow, idxCol);
            }
        } else if constexpr (
            !DstTile::isRowMajor && !SrcTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor &&
            SrcTile::SFractal == SLayout::RowMajor) {
            static_assert(
                DstTile::Cols <= SrcTile::Cols, "TEXTRACT NZ Vec→Vec : Destination cols must not exceed source cols");
            if constexpr (DstTile::ValidRow == 1 && DstTile::ValidCol == 1) {
                PTO_ASSERT(indexRow < SrcTile::Rows, "TEXTRACT NZ Vec→Vec : indexRow exceeds srcRows!");
                PTO_ASSERT(indexCol < SrcTile::Cols, "TEXTRACT NZ Vec→Vec : indexCol exceeds srcCols!");
                TExtractVecToVecNZScalarImpl<T, DstTile, SrcTile>(
                    dst.data(), src.data(), static_cast<uint32_t>(indexRow), static_cast<uint32_t>(indexCol));
            } else {
                uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
                uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());
                PTO_ASSERT(
                    indexRow + validRow <= SrcTile::Rows,
                    "TEXTRACT NZ Vec→Vec : indexRow + validRow exceeds source rows!");
                PTO_ASSERT(
                    indexCol + validCol <= SrcTile::Cols,
                    "TEXTRACT NZ Vec→Vec : indexCol + validCol exceeds source cols!");
                TExtractVecToVecNZImpl<T, DstTile, SrcTile>(
                    dst.data(), src.data(), validRow, validCol, static_cast<uint16_t>(SrcTile::Rows), indexRow,
                    indexCol);
            }
        } else {
            static_assert(
                DstTile::isRowMajor == SrcTile::isRowMajor,
                "TEXTRACT Vec→Vec : Source and destination layout must match (both ND or both NZ)");
        }
    } else if constexpr (is_conv_tile_v<SrcTile>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL<DstTile, SrcTile>(dst, src, indexRow, indexCol);
    }
}

} // namespace pto
#endif // TEXTRACT_COMMON_REGISTER_HPP
