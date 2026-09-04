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
 * @file tinsert_common.hpp
 * @brief Common TINSERT Vec→Vec helpers for A5, Kirin9030, KirinDev0000
 *
 * The arch shell must include its own arch common.hpp / utils.hpp (for
 * MaskReg, CreatePredicate, etc.) before this file.
 */

#ifndef TINSERT_COMMON_REGISTER_HPP
#define TINSERT_COMMON_REGISTER_HPP

#include <pto/common/utils.hpp>

namespace pto {

//---------------------------------------------------------------------------------------
// Interface contracts: each arch shell must provide these definitions
//---------------------------------------------------------------------------------------

// GetDistVst is arch-utils-provided (defined in a5/utils.hpp, shared by the
// kirin9030/kirinDev0000 utils chain). Declared here because the helpers below
// call it with explicit template arguments (no ADL).
template <typename T, DistVST dist>
PTO_INTERNAL constexpr DistVST GetDistVst();

// vlds+vstus path: strides or indexCol NOT 32B-aligned. KirinDev0000 unrolls
// the repeat loop with compile-time ValidCol; a5 uses the runtime parameter.
// Declared here because TInsertVecToVecNDDispatch below calls it with explicit
// template arguments (no ADL).
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL OP_NAME(TINSERT) OP_TYPE(element_wise) void TInsertVecToVecNDVectorImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT);

//---------------------------------------------------------------------------------------
// Common helpers
//---------------------------------------------------------------------------------------

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void ComputeNZBlockParams(
    uint32_t validRow, uint32_t validCol, uint32_t dstRow, uint16_t& burstNum, uint16_t& burstLen, uint16_t& srcGap,
    uint16_t& dstGap, uint32_t& dstOffset, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint32_t byteValidCol = isFp4Type ? validCol / 2 : validCol;
    uint32_t byteIndexCol = isFp4Type ? indexCol / 2 : indexCol;
    burstNum = static_cast<uint16_t>(CeilDivision(byteValidCol, c0Size));
    burstLen = (validRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    uint32_t colBlockOffset = (byteIndexCol / c0Size) * dstRow * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (byteIndexCol % c0Size);
    dstOffset = colBlockOffset + rowOffset;
    uint32_t srcStrideRows;
    if constexpr (SrcTileData::Compact == CompactMode::Null) {
        srcStrideRows = SrcTileData::Rows;
    } else if constexpr (SrcTileData::Compact == CompactMode::RowPlusOne) {
        srcStrideRows = CeilDivision(validRow, static_cast<uint32_t>(FRACTAL_NZ_ROW)) * FRACTAL_NZ_ROW + 1;
    } else {
        srcStrideRows = CeilDivision(validRow, static_cast<uint32_t>(FRACTAL_NZ_ROW)) * FRACTAL_NZ_ROW;
    }
    srcGap = static_cast<uint16_t>(srcStrideRows - validRow);
    dstGap = static_cast<uint16_t>(dstRow - validRow);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertVecToVecNDImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);

    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;

    uint32_t dstOffset = indexRow * dstRowStride + indexCol;
    __ubuf__ T* dstStart = dstAddr + dstOffset;

    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
    uint16_t rowBurstLen = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);

    if (validCol == srcRowStride && validCol == dstRowStride && totalBytes >= BLOCK_BYTE_SIZE) {
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstStart, (__ubuf__ void*)srcAddr, 1, burstLen, 0, 0);
    } else {
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstStart, (__ubuf__ void*)srcAddr, validRow, rowBurstLen, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertVecToVecNZImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t dstRow, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    uint16_t burstNum, burstLen, srcGap, dstGap;
    uint32_t dstOffset;
    ComputeNZBlockParams<T, DstTileData, SrcTileData>(
        validRow, validCol, dstRow, burstNum, burstLen, srcGap, dstGap, dstOffset, indexRow, indexCol);
    __ubuf__ T* dstStart = dstAddr + dstOffset;
    pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstStart, (__ubuf__ void*)srcAddr, burstNum, burstLen, srcGap, dstGap);
}

// vlds+vsts path: strides + indexCol are 32B-aligned, ValidCol may not be.
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL OP_NAME(TINSERT) OP_TYPE(element_wise) void TInsertVecToVecNDAlignedImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr int32_t kStaticValidCol = SrcTileData::ValidCol;
    constexpr bool kSingleChunkStatic =
        (kStaticValidCol > 0) && (static_cast<uint32_t>(kStaticValidCol) <= elementsPerRepeat);

    if constexpr (kSingleChunkStatic) {
        uint32_t kTail = static_cast<uint32_t>(kStaticValidCol);
        __VEC_SCOPE__
        {
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            RegTensor<T> vreg;
            MaskReg pregTail = CreatePredicate<T>(kTail);
            for (uint16_t i = 0; i < validRow; ++i) {
                uint32_t srcRowOff = static_cast<uint32_t>(i) * srcRowStride;
                uint32_t dstRowOff = (indexRow + static_cast<uint32_t>(i)) * dstRowStride + indexCol;
                vlds(vreg, srcAddr, srcRowOff, NORM);
                vsts(vreg, dstAddr, dstRowOff, distValue, pregTail);
            }
        }
    } else {
        uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(validCol), elementsPerRepeat);
        uint32_t tailEleNum = static_cast<uint32_t>(validCol) % elementsPerRepeat;
        if (tailEleNum == 0) {
            tailEleNum = elementsPerRepeat;
        }
        uint32_t fullEleNum = elementsPerRepeat;
        uint16_t lastRepeat = repeatTimes - 1;

        __VEC_SCOPE__
        {
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            RegTensor<T> vreg;
            MaskReg pregFull = CreatePredicate<T>(fullEleNum);
            MaskReg pregTail = CreatePredicate<T>(tailEleNum);

            for (uint16_t i = 0; i < validRow; ++i) {
                uint32_t srcRowOff = static_cast<uint32_t>(i) * srcRowStride;
                uint32_t dstRowOff = (indexRow + static_cast<uint32_t>(i)) * dstRowStride + indexCol;
                for (uint16_t j = 0; j < lastRepeat; ++j) {
                    vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, NORM);
                    vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, pregFull);
                }
                vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(lastRepeat) * elementsPerRepeat, NORM);
                vsts(
                    vreg, dstAddr, dstRowOff + static_cast<uint32_t>(lastRepeat) * elementsPerRepeat, distValue,
                    pregTail);
            }
        }
    }
}

// Scalar path: ValidRow==1, ValidCol==1 — Scalar array element copy.
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertVecToVecNDScalarImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[indexRow * dstRowStride + indexCol] = srcAddr[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecNDDispatch(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());

    PTO_ASSERT(
        indexRow + SrcTileData::ValidRow <= DstTileData::Rows,
        "TINSERT ND_VEC : indexRow + srcValidRows exceeds dstRows!");
    PTO_ASSERT(
        indexCol + SrcTileData::ValidCol <= DstTileData::Cols,
        "TINSERT ND_VEC : indexCol + srcValidCols exceeds dstCols!");

    constexpr bool kStridesAligned = (SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                                     (DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0);
    constexpr bool kValidColAligned = (SrcTileData::ValidCol * sizeof(T) % BLOCK_BYTE_SIZE == 0);

    if constexpr (kStridesAligned) {
        if (indexCol * sizeof(T) % BLOCK_BYTE_SIZE == 0) {
            if constexpr (kValidColAligned) {
                TInsertVecToVecNDImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
            } else {
                TInsertVecToVecNDAlignedImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
            }
        } else {
            TInsertVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
        }
    } else {
        TInsertVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecImpl(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor) {
        static_assert(
            SrcTileData::Rows <= DstTileData::Rows,
            "TINSERT ND Vec→Vec : Source rows must not exceed destination rows");
        static_assert(
            SrcTileData::Cols <= DstTileData::Cols,
            "TINSERT ND Vec→Vec : Source cols must not exceed destination cols");

        if constexpr (SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1) {
            PTO_ASSERT(indexRow < DstTileData::Rows, "TINSERT : indexRow exceeds dstRows!");
            PTO_ASSERT(indexCol < DstTileData::Cols, "TINSERT : indexCol exceeds dstCols!");
            TInsertVecToVecNDScalarImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol);
        } else {
            TInsertVecToVecNDDispatch<T>(dst, src, indexRow, indexCol);
        }
    } else if constexpr (
        !DstTileData::isRowMajor && !SrcTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor &&
        SrcTileData::SFractal == SLayout::RowMajor) {
        static_assert(
            SrcTileData::Cols <= DstTileData::Cols,
            "TINSERT NZ Vec→Vec : Source cols must not exceed destination cols");
        uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
        PTO_ASSERT(
            indexRow + validRow <= DstTileData::Rows,
            "TINSERT NZ Vec→Vec : indexRow + validRow exceeds destination rows!");
        PTO_ASSERT(
            indexCol + validCol <= DstTileData::Cols,
            "TINSERT NZ Vec→Vec : indexCol + validCol exceeds destination cols!");
        TInsertVecToVecNZImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), validRow, validCol, static_cast<uint16_t>(DstTileData::Rows), indexRow, indexCol);
    } else {
        static_assert(
            DstTileData::isRowMajor == SrcTileData::isRowMajor,
            "TINSERT Vec→Vec : Source and destination layout must match (both ND or both NZ)");
    }
}

} // namespace pto
#endif // TINSERT_COMMON_REGISTER_HPP
