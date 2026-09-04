/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_TEXTRACT_IMPLS_HPP
#define PTO_TEXTRACT_IMPLS_HPP

namespace pto {
// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(
        (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec),
        "Destination TileType only support Mat and Vec.");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert(
        (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor) ||
            (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    if constexpr ((DstTileData::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(),
            indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert((DstTileData::Loc == TileType::Vec), "Destination TileType only support Mat and Vec.");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert(
        (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TExtractAccToVec<DstTileData, SrcTileData, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(), indexRow,
        indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert(
        (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec),
        "Destination TileType only support Mat.");
    static_assert(
        (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor) ||
            (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr ((DstTileData::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(),
            indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((DstTileData::Loc == TileType::Vec), "Destination TileType only support Mat.");
    static_assert(
        (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToVec<DstTileData, SrcTileData, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(), indexRow,
        indexCol);
}

// fp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert(
        (DstTileData::Loc == TileType::Mat || DstTileData::Loc == TileType::Vec),
        "Destination TileType only support Mat and Vec.");
    static_assert(
        (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor) ||
            (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    if constexpr ((DstTileData::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(),
            indexRow, indexCol);
    }
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((DstTileData::Loc == TileType::Vec), "Destination TileType only support Mat and Vec.");
    static_assert(
        (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);

    TExtractAccToVec<DstTileData, SrcTileData, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), src.GetValidCol(), indexRow,
        indexCol);
}

} // namespace pto
#endif // PTO_TEXTRACT_IMPLS_HPP
