/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include "common.hpp"
#include "utils.hpp"
#include "TExtractCommon.hpp"

namespace pto {
template <typename T>
constexpr bool is_textract_supported_type = std::disjunction_v<
    std::is_same<T, int8_t>, std::is_same<T, float8_e4m3_t>, std::is_same<T, float8_e5m2_t>,
    std::is_same<T, hifloat8_t>, std::is_same<T, half>, std::is_same<T, bfloat16_t>, std::is_same<T, float>,
    std::is_same<T, float4_e2m1x2_t>, std::is_same<T, float4_e1m2x2_t>, std::is_same<T, float8_e8m0_t>>;

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractToLeft(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        (SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
            (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor) ||
            (SrcTileData::Rows == 1 && SrcTileData::isRowMajor),
        "TExtract: SrcTile Invalid Fractal");
    static_assert(
        DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor, "TExtract: DstTile Invalid Fractal");
    constexpr bool isFp4Type = std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value;
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData, isFp4Type>(
            dst.data(), src.data(), indexRow, indexCol, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, isFp4Type>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, false, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToATransCompact<DstTileData, SrcTileData, isFp4Type>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, true, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractToRight(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        (SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
            (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
        "TExtract: SrcTile Invalid Fractal");
    static_assert(
        DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor, "TExtract: DstTile Invalid Fractal");
    constexpr bool isFp4Type = std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value;
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, isFp4Type>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBTransCompact<DstTileData, SrcTileData, isFp4Type>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        is_textract_supported_type<typename DstTileData::DType>,
        "TExtract: Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
        half, bfloat16_t, float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t");
    static_assert(
        (SrcTileData::Loc == TileType::Acc) ||
            std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TExtract: Destination and Source tile data types must be the same");

    if constexpr (DstTileData::Loc == TileType::Left) {
        TExtractToLeft(dst, src, indexRow, indexCol);
    } else if constexpr (DstTileData::Loc == TileType::Right) {
        TExtractToRight(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        TExtractVecToMat<DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
            dst.GetValidCol());
    } else if constexpr (DstTileData::Loc == TileType::ScaleLeft) {
        TExtractToAmx<DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (DstTileData::Loc == TileType::ScaleRight) {
        TExtractToBmx<DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (
        SrcTileData::Loc == TileType::Acc && (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec)) {
        static_assert(
            (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor) ||
                (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
            "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
            "SFractal: NoneBox).");
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        if constexpr ((DstTileData::Loc == TileType::Mat)) {
            TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
        } else {
            TExtractAccToVec<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(),
                indexRow, indexCol);
        }
    }
}

// For 1-byte non-int8/uint8 dtypes (hifloat8/float8_*), vector intrinsics (vlds/vsts/...) have no
// overload, so reinterpret the UB pointers as int8 and operate via int8 intrinsics. Semantics match
// since each element occupies exactly one byte. fp4 (sub-byte) is handled separately via byte-DMA.
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL OP_NAME(TEXTRACT) OP_TYPE(element_wise) void TExtractVecToVecNDAlignedImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol, uint16_t validRow, uint16_t validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using RegT = TExtractRegT<T>;
    __ubuf__ RegT* dstAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(dst);
    __ubuf__ RegT* srcAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(RegT);
    constexpr int32_t kStaticValidCol = DstTileData::ValidCol;
    constexpr bool kSingleChunkStatic =
        (kStaticValidCol > 0) && (static_cast<uint32_t>(kStaticValidCol) <= elementsPerRepeat);

    if constexpr (kSingleChunkStatic) {
        uint32_t kTail = static_cast<uint32_t>(kStaticValidCol);
        __VEC_SCOPE__
        {
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
            RegTensor<RegT> vreg;
            MaskReg pregTail = CreatePredicate<RegT>(kTail);
            for (uint16_t i = 0; i < validRow; ++i) {
                uint32_t srcRowOff = (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
                uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
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
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
            RegTensor<RegT> vreg;
            MaskReg pregFull = CreatePredicate<RegT>(fullEleNum);
            MaskReg pregTail = CreatePredicate<RegT>(tailEleNum);

            for (uint16_t i = 0; i < validRow; ++i) {
                uint32_t srcRowOff = (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
                uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
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

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL OP_NAME(TEXTRACT) OP_TYPE(element_wise) void TExtractVecToVecNDVectorImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol, uint16_t validRow, uint16_t validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using RegT = TExtractRegT<T>;
    __ubuf__ RegT* srcAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(src);
    __ubuf__ RegT* dstAddr = (__ubuf__ RegT*)__cce_get_tile_ptr(dst);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr int32_t kStaticValidCol = DstTileData::ValidCol;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(RegT);
    constexpr bool kSingleChunkStatic =
        (kStaticValidCol > 0) && (static_cast<uint32_t>(kStaticValidCol) <= elementsPerRepeat);

    if constexpr (kSingleChunkStatic) {
        uint32_t kTail = static_cast<uint32_t>(kStaticValidCol);
        __VEC_SCOPE__
        {
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
            RegTensor<RegT> vreg;
            UnalignReg ureg;
            MaskReg pregTail = CreatePredicate<RegT>(kTail);
            for (uint16_t i = 0; i < validRow; ++i) {
                __ubuf__ RegT* psrc = srcAddr + (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
                uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
                vldas(ureg, psrc);
                vldus(vreg, ureg, psrc);
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
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<RegT, DistVST::DIST_NORM>())>();
            RegTensor<RegT> vreg;
            UnalignReg ureg;
            MaskReg pregFull = CreatePredicate<RegT>(fullEleNum);
            MaskReg pregTail = CreatePredicate<RegT>(tailEleNum);

            for (uint16_t i = 0; i < validRow; ++i) {
                __ubuf__ RegT* psrc = srcAddr + (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
                uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
                for (uint16_t j = 0; j < lastRepeat; ++j) {
                    vldas(ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                    vldus(vreg, ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                    vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, pregFull);
                }
                vldas(ureg, psrc + static_cast<uint32_t>(lastRepeat) * elementsPerRepeat);
                vldus(vreg, ureg, psrc + static_cast<uint32_t>(lastRepeat) * elementsPerRepeat);
                vsts(
                    vreg, dstAddr, dstRowOff + static_cast<uint32_t>(lastRepeat) * elementsPerRepeat, distValue,
                    pregTail);
            }
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractVecToVecNDDispatch(DstTileData& dst, SrcTileData& src, uint32_t indexRow, uint32_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());

    PTO_ASSERT(indexRow + validRow <= SrcTileData::Rows, "TEXTRACT ND_VEC : indexRow + dstValidRows exceeds srcRows!");
    PTO_ASSERT(indexCol + validCol <= SrcTileData::Cols, "TEXTRACT ND_VEC : indexCol + dstValidCols exceeds srcCols!");

    // fp4 (float4_e2m1x2_t / float4_e1m2x2_t) is sub-byte: each T packs 2 elements into 1 byte.
    // Vector intrinsics cannot address individual fp4 elements, so only the byte-DMA path
    // (TExtractVecToVecNDImpl) is valid. The DMA path treats T as one packed unit (1 byte),
    // so callers must use packed-unit counts: indexCol/validCol/RowStride must be in T units
    // (not individual fp4 elements), and DMA further requires indexCol to be 32-byte aligned
    // for the aligned-stride fast path.
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    if constexpr (isFp4Type) {
        static_assert(
            SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TEXTRACT ND Vec\u2192Vec fp4: SrcTile RowStride must be 32-byte aligned.");
        static_assert(
            DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TEXTRACT ND Vec\u2192Vec fp4: DstTile RowStride must be 32-byte aligned.");
        static_assert(
            DstTileData::ValidCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TEXTRACT ND Vec\u2192Vec fp4: DstTile ValidCol must be 32-byte aligned.");
        PTO_ASSERT(
            indexCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TEXTRACT ND Vec\u2192Vec fp4: indexCol must be 32-byte aligned.");
        TExtractVecToVecNDImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
        return;
    }

    constexpr bool kStridesAligned = (SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                                     (DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0);
    constexpr bool kValidColAligned = (DstTileData::ValidCol * sizeof(T) % BLOCK_BYTE_SIZE == 0);

    if constexpr (kStridesAligned) {
        if (indexCol * sizeof(T) % BLOCK_BYTE_SIZE == 0) {
            if constexpr (kValidColAligned) {
                TExtractVecToVecNDImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
            } else {
                TExtractVecToVecNDAlignedImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
            }
        } else {
            TExtractVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
        }
    } else {
        TExtractVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TExtractVecToVecNZImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t srcRow, uint16_t indexRow, uint16_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint32_t byteValidCol = isFp4Type ? validCol / 2 : validCol;
    uint32_t byteIndexCol = isFp4Type ? indexCol / 2 : indexCol;
    uint16_t burstNum = static_cast<uint16_t>(CeilDivision(byteValidCol, c0Size));
    uint16_t burstLen = (validRow * c0Size * typeSize) / BLOCK_BYTE_SIZE;
    uint32_t colBlockOffset = (byteIndexCol / c0Size) * srcRow * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (byteIndexCol % c0Size);
    uint32_t srcOffset = colBlockOffset + rowOffset;
    uint16_t srcGap = static_cast<uint16_t>(srcRow - validRow);
    uint16_t dstGap = static_cast<uint16_t>(DstTileData::Rows - validRow);
    __ubuf__ T* srcStart = srcAddr + srcOffset;
    pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, burstNum, burstLen, srcGap, dstGap);
}

template <typename WorkT, uint32_t SrcRowStride, bool Aligned>
PTO_INTERNAL void TExtractNd2NzWindowLoop(
    __ubuf__ WorkT* srcBase, __ubuf__ WorkT* dstPtr, uint16_t repeatTimes, uint16_t innerLoopNum, uint32_t validCol,
    uint32_t cfgVsstb, uint32_t cfgVsstbLast)
{
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(WorkT);
    RegTensor<WorkT> vreg;
    MaskReg preg;
    if constexpr (Aligned) {
        uint32_t cols = validCol;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            uint32_t count = cols - static_cast<uint32_t>(cols > elementsPerRepeat) * (cols - elementsPerRepeat);
            preg = CreatePredicate<WorkT>(count);
            uint32_t colOff = static_cast<uint32_t>(j) * elementsPerRepeat;
            for (uint16_t i = 0; i <= innerLoopNum; ++i) {
                __ubuf__ WorkT* psrc = srcBase + static_cast<uint32_t>(i) * SrcRowStride + colOff;
                vlds(vreg, psrc, 0, NORM);
                vsstb(vreg, dstPtr, (i < innerLoopNum) ? cfgVsstb : cfgVsstbLast, preg, POST_UPDATE);
            }
            cols -= elementsPerRepeat;
        }
    } else {
        constexpr uint32_t c0Elems = BLOCK_BYTE_SIZE / sizeof(WorkT);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<WorkT, DistVST::DIST_NORM>())>();
        UnalignReg ureg;
        uint32_t colBlkStrideElems = (cfgVsstb >> 16u) * c0Elems;
        uint32_t totalColBlk = CeilDivision(validCol, c0Elems);
        for (uint16_t i = 0; i <= innerLoopNum; ++i) {
            __ubuf__ WorkT* rowBase = srcBase + static_cast<uint32_t>(i) * SrcRowStride;
            uint32_t remaining = validCol;
            for (uint32_t cb = 0; cb < totalColBlk; ++cb) {
                uint32_t colsThis = remaining < c0Elems ? remaining : c0Elems;
                preg = CreatePredicate<WorkT>(colsThis);
                __ubuf__ WorkT* psrc = rowBase + cb * c0Elems;
                vldas(ureg, psrc);
                vldus(vreg, ureg, psrc);
                vsts(vreg, dstPtr, cb * colBlkStrideElems + static_cast<uint32_t>(i) * c0Elems, distValue, preg);
                remaining -= colsThis;
            }
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TExtractNdToNzScalar(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[0] = srcAddr[static_cast<uint32_t>(indexRow) * srcRowStride + static_cast<uint32_t>(indexCol)];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TExtractNdToNz(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol)
{
    using WorkT = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t c0Elems = BLOCK_BYTE_SIZE / sizeof(WorkT);

    uint32_t workValidCol = isFp4Type ? static_cast<uint32_t>(validCol) / 2u : static_cast<uint32_t>(validCol);
    uint32_t workIndexCol = isFp4Type ? static_cast<uint32_t>(indexCol) / 2u : static_cast<uint32_t>(indexCol);
    uint16_t slideOff = static_cast<uint16_t>(workIndexCol % c0Elems);

    __ubuf__ WorkT* dstPtr = (__ubuf__ WorkT*)__cce_get_tile_ptr(dst);
    __ubuf__ WorkT* srcPtr = (__ubuf__ WorkT*)__cce_get_tile_ptr(src);
    __ubuf__ WorkT* srcBase = srcPtr + static_cast<uint32_t>(indexRow) * srcRowStride + workIndexCol;

    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(WorkT);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(workValidCol, elementsPerRepeat));
    constexpr bool isOptForConflict = DstTileData::Compact == CompactMode::RowPlusOne;
    uint32_t alignRow = (static_cast<uint32_t>(validRow) + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = isOptForConflict ? ((alignRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
                                              (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t virtualRow = isOptForConflict ? alignRow + 1 : alignRow;
    uint16_t innerLoopNum = static_cast<uint16_t>(validRow - 1);
    uint32_t cfgVsstb = (blockStride << 16u) | (1u & 0xFFFFu);
    uint32_t repeatStrideLast =
        (CCE_VL * virtualRow - static_cast<uint32_t>(innerLoopNum) * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE;
    uint32_t cfgVsstbLast = (blockStride << 16u) | (repeatStrideLast & 0xFFFFU);

    __VEC_SCOPE__
    {
        if (slideOff == 0) {
            TExtractNd2NzWindowLoop<WorkT, srcRowStride, true>(
                srcBase, dstPtr, repeatTimes, innerLoopNum, workValidCol, cfgVsstb, cfgVsstbLast);
        } else {
            TExtractNd2NzWindowLoop<WorkT, srcRowStride, false>(
                srcBase, dstPtr, repeatTimes, innerLoopNum, workValidCol, cfgVsstb, cfgVsstbLast);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractNdToNz()
{
    static_assert(
        SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec,
        "TEXTRACT A5 ND->2xNZ : Source and destinations must be Vec (UB) tiles.");
    static_assert(
        SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox),
        "TEXTRACT A5 ND->2xNZ : Source must be ND (RowMajor, NoneBox).");
    static_assert(
        !DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor),
        "TEXTRACT A5 ND->2xNZ : Destination must be NZ (ColMajor, RowMajor fractal).");
    static_assert(
        std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TEXTRACT A5 ND->2xNZ : Source and destination data types must match.");
    static_assert(
        (std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) || (std::is_same<T, float>::value) ||
            (std::is_same<T, int32_t>::value) || (std::is_same<T, int8_t>::value) ||
            (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, float8_e4m3_t>::value) ||
            (std::is_same<T, float8_e5m2_t>::value) || (std::is_same<T, float8_e8m0_t>::value) ||
            (std::is_same<T, float4_e2m1x2_t>::value) || (std::is_same<T, float4_e1m2x2_t>::value),
        "TEXTRACT A5 ND->2xNZ : Unsupported data type.");
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(DstTileData::Cols % c0Size == 0, "TEXTRACT ND->2xNZ : Destination cols must be c0-aligned.");
    static_assert(
        (SrcTileData::RowStride * sizeof(T)) % BLOCK_BYTE_SIZE == 0,
        "TEXTRACT A5 ND->2xNZ : Source row stride must be 32-byte aligned.");
}

template <typename Dst0TileData, typename Dst1TileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_ND2XNZ_IMPL(
    Dst0TileData& dst0, Dst1TileData& dst1, SrcTileData& src, uint16_t indexRow0, uint16_t indexCol0,
    uint16_t indexRow1, uint16_t indexCol1)
{
    using T = typename SrcTileData::DType;
    CheckTExtractNdToNz<T, Dst0TileData, SrcTileData>();
    CheckTExtractNdToNz<T, Dst1TileData, SrcTileData>();

    uint16_t validRow0 = static_cast<uint16_t>(dst0.GetValidRow());
    uint16_t validCol0 = static_cast<uint16_t>(dst0.GetValidCol());
    uint16_t validRow1 = static_cast<uint16_t>(dst1.GetValidRow());
    uint16_t validCol1 = static_cast<uint16_t>(dst1.GetValidCol());

    PTO_ASSERT(
        indexRow0 + validRow0 <= SrcTileData::Rows,
        "TEXTRACT A5 ND->2xNZ : window0 indexRow + validRow exceeds srcRows!");
    PTO_ASSERT(
        indexCol0 + validCol0 <= SrcTileData::Cols,
        "TEXTRACT A5 ND->2xNZ : window0 indexCol + validCol exceeds srcCols!");
    PTO_ASSERT(
        indexRow1 + validRow1 <= SrcTileData::Rows,
        "TEXTRACT A5 ND->2xNZ : window1 indexRow + validRow exceeds srcRows!");
    PTO_ASSERT(
        indexCol1 + validCol1 <= SrcTileData::Cols,
        "TEXTRACT A5 ND->2xNZ : window1 indexCol + validCol exceeds srcCols!");

    if (validRow0 == 1 && validCol0 == 1) {
        TExtractNdToNzScalar<T, Dst0TileData, SrcTileData>(dst0.data(), src.data(), indexRow0, indexCol0);
    } else {
        TExtractNdToNz<T, Dst0TileData, SrcTileData>(
            dst0.data(), src.data(), indexRow0, indexCol0, validRow0, validCol0);
    }
    if (validRow1 == 1 && validCol1 == 1) {
        TExtractNdToNzScalar<T, Dst1TileData, SrcTileData>(dst1.data(), src.data(), indexRow1, indexCol1);
    } else {
        TExtractNdToNz<T, Dst1TileData, SrcTileData>(
            dst1.data(), src.data(), indexRow1, indexCol1, validRow1, validCol1);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        using T = typename DstTileData::DType;
        static_assert(
            std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TEXTRACT Vec→Vec : Source and destination data types must match");
        static_assert(
            (std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) || (std::is_same<T, float>::value) ||
                (std::is_same<T, int32_t>::value) || (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, float8_e4m3_t>::value) ||
                (std::is_same<T, float8_e5m2_t>::value) || (std::is_same<T, float8_e8m0_t>::value) ||
                (std::is_same<T, float4_e2m1x2_t>::value) || (std::is_same<T, float4_e1m2x2_t>::value),
            "TEXTRACT Vec→Vec : Unsupported data type.");
        if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor) {
            static_assert(
                DstTileData::Rows <= SrcTileData::Rows,
                "TEXTRACT ND Vec→Vec : Destination rows must not exceed source rows");
            static_assert(
                DstTileData::Cols <= SrcTileData::Cols,
                "TEXTRACT ND Vec→Vec : Destination cols must not exceed source cols");
            uint32_t idxRow = static_cast<uint32_t>(indexRow);
            uint32_t idxCol = static_cast<uint32_t>(indexCol);
            if constexpr (DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1) {
                PTO_ASSERT(idxRow < SrcTileData::Rows, "TEXTRACT ND Vec→Vec : indexRow exceeds srcRows!");
                PTO_ASSERT(idxCol < SrcTileData::Cols, "TEXTRACT ND Vec→Vec : indexCol exceeds srcCols!");
                TExtractVecToVecNDScalarImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
            } else {
                TExtractVecToVecNDDispatch<T, DstTileData, SrcTileData>(dst, src, idxRow, idxCol);
            }
        } else if constexpr (
            !DstTileData::isRowMajor && !SrcTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor &&
            SrcTileData::SFractal == SLayout::RowMajor) {
            static_assert(
                DstTileData::Cols <= SrcTileData::Cols,
                "TEXTRACT NZ Vec→Vec : Destination cols must not exceed source cols");
            if constexpr (DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1) {
                PTO_ASSERT(indexRow < SrcTileData::Rows, "TEXTRACT NZ Vec→Vec : indexRow exceeds srcRows!");
                PTO_ASSERT(indexCol < SrcTileData::Cols, "TEXTRACT NZ Vec→Vec : indexCol exceeds srcCols!");
                TExtractVecToVecNZScalarImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), static_cast<uint32_t>(indexRow), static_cast<uint32_t>(indexCol));
            } else {
                uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
                uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());
                PTO_ASSERT(
                    indexRow + validRow <= SrcTileData::Rows,
                    "TEXTRACT NZ Vec→Vec : indexRow + validRow exceeds source rows!");
                PTO_ASSERT(
                    indexCol + validCol <= SrcTileData::Cols,
                    "TEXTRACT NZ Vec→Vec : indexCol + validCol exceeds source cols!");
                TExtractVecToVecNZImpl<T, DstTileData, SrcTileData>(
                    dst.data(), src.data(), validRow, validCol, static_cast<uint16_t>(SrcTileData::Rows), indexRow,
                    indexCol);
            }
        } else {
            static_assert(
                DstTileData::isRowMajor == SrcTileData::isRowMajor,
                "TEXTRACT Vec→Vec : Source and destination layout must match (both ND or both NZ)");
        }
    } else if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL(dst, src, indexRow, indexCol);
    }
}

} // namespace pto

#include "TExtractImpls.hpp"

#endif
