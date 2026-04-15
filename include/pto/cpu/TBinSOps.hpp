/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <algorithm>
#include <cmath>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
constexpr int CONSTRAINT_VEC_ROWMAJOR = 1;
constexpr int CONSTRAINT_VEC = 2;
constexpr int CONSTRAINT_ROWMAJOR = 3;
constexpr int NO_CONSTRAINT = 4;

template <ElementOp op>
struct CategoryBinSOps : std::false_type {};

template <>
struct CategoryBinSOps<ElementOp::OP_ADDS> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};
template <>
struct CategoryBinSOps<ElementOp::OP_DIVS> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};
template <>
struct CategoryBinSOps<ElementOp::OP_RDIVS> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};
template <>
struct CategoryBinSOps<ElementOp::OP_MULS> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};
template <>
struct CategoryBinSOps<ElementOp::OP_MAXS> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};
template <>
struct CategoryBinSOps<ElementOp::OP_LRELU> : std::integral_constant<int, CONSTRAINT_VEC_ROWMAJOR> {};

template <>
struct CategoryBinSOps<ElementOp::OP_SUBS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_REMS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_MINS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_ANDS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_ORS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_FMODS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_SHLS> : std::integral_constant<int, CONSTRAINT_VEC> {};
template <>
struct CategoryBinSOps<ElementOp::OP_SHRS> : std::integral_constant<int, CONSTRAINT_VEC> {};

template <>
struct CategoryBinSOps<ElementOp::OP_SELS> : std::integral_constant<int, CONSTRAINT_ROWMAJOR> {};

template <>
struct CategoryBinSOps<ElementOp::OP_XORS> : std::integral_constant<int, NO_CONSTRAINT> {};

template <typename TileData, ElementOp op>
PTO_INTERNAL void CheckBinSOpTileData()
{
    static_assert(CategoryBinSOps<op>::value, "UnaryTileScalarOpImpl: invalid ElementOp value");

    if constexpr (CategoryBinSOps<op>::value == CONSTRAINT_ROWMAJOR ||
                  CategoryBinSOps<op>::value == CONSTRAINT_VEC_ROWMAJOR) {
        static_assert(TileData::isRowMajor, "UnaryTileScalarOpImpl: TileType of src and dst tiles must be Row Major.");
    }

    if constexpr (CategoryBinSOps<op>::value == CONSTRAINT_VEC ||
                  CategoryBinSOps<op>::value == CONSTRAINT_VEC_ROWMAJOR) {
        static_assert(TileData::Loc == TileType::Vec,
                      "UnaryTileScalarOpImpl: TileType of src and dst tiles must be TileType::Vec.");
    }
}

template <typename TileDst, typename TileSrc, ElementOp op>
PTO_INTERNAL void CheckDstSrcTileData(TileDst &dst, TileSrc &src)
{
    static_assert(std::is_same_v<typename TileDst::DType, typename TileSrc::DType>,
                  "UnaryTileScalarOpImpl: The data type of dst must be consistent with src.");

    CheckBinSOpTileData<TileDst, op>();
    CheckBinSOpTileData<TileSrc, op>();

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of cols of src and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
}

template <typename TileDst, typename TileSrc, ElementOp op>
PTO_INTERNAL void UnaryTileScalarOpImpl(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, size_t extra = 0)
{
    using T = typename TileDst::DType;
    CheckDstSrcTileData<TileDst, TileSrc, op>(dst, src);

    unsigned rows = dst.GetValidRow();
    unsigned cols = dst.GetValidCol();

    if constexpr (TileDst::SFractal == SLayout::NoneBox) {
        if constexpr (TileDst::isRowMajor) {
            cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
                const std::size_t baseDst = r * TileDst::Cols;
                const std::size_t baseSrc = r * TileSrc::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < cols; ++c) {
                    const std::size_t idxDst = baseDst + c;
                    const std::size_t idxSrc = baseSrc + c;
                    ElementOpCal<T, op>::apply(dst.data()[idxDst], src.data()[idxSrc], scalar, extra);
                }
            });
        } else {
            cpu::parallel_for_rows(cols, rows, [&](std::size_t c) {
                const std::size_t baseDst = c * TileDst::Rows;
                const std::size_t baseSrc = c * TileSrc::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t idxDst = baseDst + c;
                    const std::size_t idxSrc = baseSrc + c;
                    ElementOpCal<T, op>::apply(dst.data()[idxDst], src.data()[idxSrc], scalar, extra);
                }
            });
        }
    } else {
        if constexpr (TileDst::isRowMajor) {
            cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    const std::size_t idxDst = GetTileElementOffset<TileDst>(r, c);
                    const std::size_t idxSrc = GetTileElementOffset<TileSrc>(r, c);
                    ElementOpCal<T, op>::apply(dst.data()[idxDst], src.data()[idxSrc], scalar, extra);
                }
            });
        } else {
            cpu::parallel_for_rows(cols, rows, [&](std::size_t c) {
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t idxDst = GetTileElementOffset<TileDst>(r, c);
                    const std::size_t idxSrc = GetTileElementOffset<TileSrc>(r, c);
                    ElementOpCal<T, op>::apply(dst.data()[idxDst], src.data()[idxSrc], scalar, extra);
                }
            });
        }
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TADDS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ADDS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TSUBS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_SUBS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMULS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MULS>(dst, src, scalar);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    if (scalar == static_cast<typename TileSrc::DType>(0)) {
        PTO_ASSERT(false, "TDIVS: illegal scalar is zero");
    }
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_DIVS>(dst, src, scalar);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDst &dst, typename TileSrc::DType scalar, TileSrc &src)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_RDIVS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMINS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MINS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TREMS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_REMS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TREMS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, TileDst &tmp)
{
    (void)tmp;
    if (scalar != static_cast<TileSrc::DType>(0)) {
        TREMS_IMPL(dst, src, scalar);
    } else {
        PTO_ASSERT(false, "illegal src is zero");
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TMAXS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_MAXS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TANDS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ANDS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_ORS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TXORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_XORS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TXORS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar, TileDst &tmp)
{
    (void)tmp;
    TXORS_IMPL(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TLRELU_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_LRELU>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TFMODS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    if (scalar != static_cast<TileSrc::DType>(0)) {
        UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_FMODS>(dst, src, scalar);
    } else {
        PTO_ASSERT(false, "illegal src is zero");
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TSHLS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_SHLS>(dst, src, scalar);
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TSHRS_IMPL(TileDst &dst, TileSrc &src, typename TileSrc::DType scalar)
{
    UnaryTileScalarOpImpl<TileDst, TileSrc, ElementOp::OP_SHRS>(dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TAXPY_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    for (size_t c = 0; c < col; ++c) {
        for (size_t r = 0; r < row; ++r) {
            const size_t dstIdx = GetTileElementOffset<TileDataDst>(r, c);
            const size_t srcIdx = GetTileElementOffset<TileDataSrc>(r, c);
            dst.data()[dstIdx] = static_cast<typename TileDataDst::DType>(
                dst.data()[dstIdx] + static_cast<typename TileDataDst::DType>(src.data()[srcIdx]) *
                                         static_cast<typename TileDataDst::DType>(scalar));
        }
    }
}
} // namespace pto

#endif
