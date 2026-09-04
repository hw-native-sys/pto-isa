/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_A6_HPP
#define TEXTRACT_A6_HPP

#include "pto/npu/a5/common.hpp"
#include "pto/npu/a5/utils.hpp"
#include "pto/npu/a5/TExtractCommon.hpp"

namespace pto {
template <typename T>
constexpr bool is_textract_supported_type = std::disjunction_v<
    std::is_same<T, int8_t>, std::is_same<T, uint8_t>, std::is_same<T, float8_e4m3_t>, std::is_same<T, float8_e5m2_t>,
    std::is_same<T, hifloat8_t>, std::is_same<T, int4b_t>, std::is_same<T, half>, std::is_same<T, bfloat16_t>,
    std::is_same<T, float>, std::is_same<T, float4_e2m1x2_t>, std::is_same<T, float4_e1m2x2_t>,
    std::is_same<T, float8_e8m0_t>
#if defined(PTO_NPU_ARCH_A6)
    ,
    std::is_same<T, hifloat4x2_t>
#endif
    >;

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
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, int4b_t>::value
#if defined(PTO_NPU_ARCH_A6)
                               || std::is_same<typename SrcTileData::DType, hifloat4x2_t>::value
#endif
        ;
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
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, int4b_t>::value
#if defined(PTO_NPU_ARCH_A6)
                               || std::is_same<typename SrcTileData::DType, hifloat4x2_t>::value
#endif
        ;
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
        "TExtract: Unsupported data type! Supported types: int8_t, uint8_t, hifloat8_t, fp8_e5m2_t, \
        fp8_e4m3fn_t, int4b_t, half, bfloat16_t, float, float4_e2m1x2_t, float4_e1m2x2_t, \
        float8_e8m0_t, hifloat4x2_t");
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
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(RegT);
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
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(RegT);
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

    PTO_ASSERT(
        indexRow + DstTileData::ValidRow <= SrcTileData::Rows,
        "TEXTRACT ND_VEC : indexRow + dstValidRows exceeds srcRows!");
    PTO_ASSERT(
        indexCol + DstTileData::ValidCol <= SrcTileData::Cols,
        "TEXTRACT ND_VEC : indexCol + dstValidCols exceeds srcCols!");

    // fp4 (float4_e2m1x2_t / float4_e1m2x2_t) is sub-byte: each T packs 2 elements into 1 byte.
    // Vector intrinsics cannot address individual fp4 elements, so only the byte-DMA path
    // (TExtractVecToVecNDImpl) is valid. The DMA path treats T as one packed unit (1 byte),
    // so callers must use packed-unit counts: indexCol/validCol/RowStride must be in T units
    // (not individual fp4 elements), and DMA further requires indexCol to be 32-byte aligned
    // for the aligned-stride fast path.
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t> ||
                               std::is_same_v<T, int4b_t>
#if defined(PTO_NPU_ARCH_A6)
                               || std::is_same_v<T, hifloat4x2_t>
#endif
        ;
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
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t> ||
                               std::is_same_v<T, int4b_t>
#if defined(PTO_NPU_ARCH_A6)
                               || std::is_same_v<T, hifloat4x2_t>
#endif
        ;
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
                (std::is_same<T, float4_e2m1x2_t>::value) || (std::is_same<T, float4_e1m2x2_t>::value)
#if defined(PTO_NPU_ARCH_A6)
                || (std::is_same<T, hifloat4x2_t>::value)
#endif
                ,
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

#include "pto/npu/a5/TExtractImpls.hpp"

#endif // TEXTRACT_A6_HPP
