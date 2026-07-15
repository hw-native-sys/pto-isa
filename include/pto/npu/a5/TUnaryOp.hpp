/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "custom/TExp_Custom.hpp"
#include "custom/TLog_Custom.hpp"
#include "custom/TSqrtHp.hpp"

namespace pto {
template <typename Op, typename T, unsigned nRepeatElem>
PTO_INTERNAL void TUnaryOps_1D_NoPostUpdate(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validRow * validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        unsigned sReg = validRow * validCol;
        MaskReg pReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pReg = CreatePredicate<T>(sReg);
            vlds(srcReg, src, i * nRepeatElem, NORM);
            Op::UnaryInstr(dstReg, srcReg, pReg);
            vsts(dstReg, dst, i * nRepeatElem, distValue, pReg);
        }
    }
}

template <typename Op, typename T, unsigned nRepeatElem>
PTO_INTERNAL void TUnaryOps_1D_PostUpdate(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validRow * validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        MaskReg pReg;
        unsigned sReg = validRow * validCol;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pReg = CreatePredicate<T>(sReg);
            vlds(srcReg, src, nRepeatElem, NORM, POST_UPDATE);
            Op::UnaryInstr(dstReg, srcReg, pReg);
            vsts(dstReg, dst, nRepeatElem, distValue, pReg, POST_UPDATE);
        }
    }
}

template <typename Op, typename T, unsigned DstRowStride, unsigned SrcRowStride, unsigned nRepeatElem>
PTO_INTERNAL void TUnaryOps_2D(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        MaskReg pReg;
        unsigned sReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                pReg = CreatePredicate<T>(sReg);
                vlds(srcReg, src, i * SrcRowStride + j * nRepeatElem, NORM);
                Op::UnaryInstr(dstReg, srcReg, pReg);
                vsts(dstReg, dst, i * DstRowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void TUnaryOps_1D_Switch(
    __ubuf__ T* dst, __ubuf__ T* src, unsigned validRow, unsigned validCol, VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE: {
            TUnaryOps_1D_NoPostUpdate<Op, T, nRepeatElem>(dst, src, validRow, validCol);
            break;
        }
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE: {
            TUnaryOps_2D<Op, T, DstTile::RowStride, SrcTile::RowStride, nRepeatElem>(dst, src, validRow, validCol);
            break;
        }
        default: {
            TUnaryOps_1D_PostUpdate<Op, T, nRepeatElem>(dst, src, validRow, validCol);
            break;
        }
    }
}

template <typename DstTile, typename SrcTile, typename Op>
PTO_INTERNAL void TUnaryOp(
    __ubuf__ typename DstTile::DType* dst, __ubuf__ typename SrcTile::DType* src, unsigned validRow, unsigned validCol,
    VFImplKind version)
{
    using T = typename DstTile::DType;
    constexpr unsigned nRepeatElem = CCE_VL / sizeof(T);
    if constexpr (
        ((DstTile::ValidCol == DstTile::Cols) && (SrcTile::ValidCol == SrcTile::Cols)) ||
        ((DstTile::Rows == 1) && (SrcTile::Rows == 1))) {
        TUnaryOps_1D_Switch<Op, T, DstTile, SrcTile, nRepeatElem>(dst, src, validRow, validCol, version);
    } else {
        TUnaryOps_2D<Op, T, DstTile::RowStride, SrcTile::RowStride, nRepeatElem>(dst, src, validRow, validCol);
    }
}

template <typename DstTile, typename SrcTile, bool needDataTypeCheck = true>
PTO_INTERNAL void TUnaryCheck()
{
    static_assert(DstTile::isRowMajor && SrcTile::isRowMajor, "TUnaryOp: Not supported Layout type");
    static_assert(
        DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec,
        "TUnaryOp: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(
        DstTile::ValidCol <= DstTile::Cols,
        "TUnaryOp: Number of dst's valid columns must not be greater than number of tile columns.");
    static_assert(
        DstTile::ValidRow <= DstTile::Rows,
        "TUnaryOp: Number of dst's valid rows must not be greater than number of tile rows.");
    static_assert(
        SrcTile::ValidCol <= SrcTile::Cols,
        "TUnaryOp: Number of src's valid columns must not be greater than number of tile columns.");
    static_assert(
        SrcTile::ValidRow <= SrcTile::Rows,
        "TUnaryOp: Number of src's valid rows must not be greater than number of tile rows.");
    static_assert(
        std::is_same_v<typename DstTile::DType, typename SrcTile::DType>,
        "TUnaryOp: The data type of dst must be consistent with of src");
    static_assert(
        !needDataTypeCheck || std::is_same_v<typename DstTile::DType, float32_t> ||
            std::is_same_v<typename DstTile::DType, float> || std::is_same_v<typename DstTile::DType, float16_t> ||
            std::is_same_v<typename DstTile::DType, half>,
        "TUnaryOp: Invalid data type.");
}

/* TEXP */
template <ExpAlgorithm PrecisionType, typename T>
struct ExpOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        if constexpr (
            PrecisionType == ExpAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, half>)) {
            ExpPrecisionImpl(dstReg, srcReg, pReg);
        } else {
            vexp(dstReg, srcReg, pReg, MODE_ZEROING);
        }
    }
};
template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TEXP) OP_TYPE(element_wise) void TExp(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, ExpOp<PrecisionType, T>>(dst, src, validRow, validCol, version);
}
template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXP_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TEXP: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TEXP: Number of rows of src and dst must be the same.");
    TExp<PrecisionType, DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TNOT */
template <typename T>
struct NotOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        vnot(dstReg, srcReg, pReg, MODE_ZEROING);
    }
};
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TNOT) OP_TYPE(element_wise) void TNot(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, NotOp<T>>(dst, src, validRow, validCol, version);
}
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNOT_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile, false>();
    static_assert(
        std::is_same_v<typename DstTile::DType, uint32_t> || std::is_same_v<typename DstTile::DType, int32_t> ||
            std::is_same_v<typename DstTile::DType, uint16_t> || std::is_same_v<typename DstTile::DType, int16_t> ||
            std::is_same_v<typename DstTile::DType, uint8_t> || std::is_same_v<typename DstTile::DType, int8_t>,
        "TNOT: Invalid data type.");

    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TNOT: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TNOT: Number of rows of src and dst must be the same.");
    TNot<DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TRELU */
template <typename T>
struct ReluOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        vrelu(dstReg, srcReg, pReg, MODE_ZEROING);
    }
};
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TRELU) OP_TYPE(element_wise) void TRelu(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, ReluOp<T>>(dst, src, validRow, validCol, version);
}
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRELU_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile, false>();
    static_assert(
        std::is_same_v<typename DstTile::DType, half> || std::is_same_v<typename DstTile::DType, float> ||
            std::is_same_v<typename DstTile::DType, int32_t>,
        "TRELU: Invalid data type.");

    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TRELU: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TRELU: Number of rows of src and dst must be the same.");
    TRelu<DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TSQRT */
template <SqrtAlgorithm PrecisionType, typename T>
struct SqrtOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        if constexpr (PrecisionType == SqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            SqrtFloatImpl<T, RegTensor<T>>(dstReg, srcReg, pReg);
        } else if constexpr (PrecisionType == SqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            SqrtPrecisionImpl<T, RegTensor<T>>(dstReg, srcReg, pReg);
        } else {
            vsqrt(dstReg, srcReg, pReg, MODE_ZEROING);
        }
    }
};
template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TSQRT) OP_TYPE(element_wise) void TSqrt(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, SqrtOp<PrecisionType, T>>(dst, src, validRow, validCol, version);
}
template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TSQRT_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TSQRT: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TSQRT: Number of rows of src and dst must be the same.");
    TSqrt<PrecisionType, DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TABS */
template <typename T>
struct AbsOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        vabs(dstReg, srcReg, pReg, MODE_ZEROING);
    }
};
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TABS) OP_TYPE(element_wise) void TAbs(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, AbsOp<T>>(dst, src, validRow, validCol, version);
}
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TABS_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile, false>();
    static_assert(
        std::is_same_v<typename DstTile::DType, float32_t> || std::is_same_v<typename DstTile::DType, float> ||
            std::is_same_v<typename DstTile::DType, float16_t> || std::is_same_v<typename DstTile::DType, half> ||
            std::is_same_v<typename DstTile::DType, int8_t> || std::is_same_v<typename DstTile::DType, int16_t> ||
            std::is_same_v<typename DstTile::DType, int32_t>,
        "TABS: Invalid data type.");
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TABS: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TABS: Number of rows of src and dst must be the same.");
    TAbs<DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TLOG */
template <LogAlgorithm PrecisionType, typename T>
struct LogOp {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& pReg)
    {
        if constexpr (
            PrecisionType == LogAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, half>)) {
            LogPrecisionImpl(dstReg, srcReg, pReg);
        } else {
            vln(dstReg, srcReg, pReg, MODE_ZEROING);
        }
    }
};
template <auto PrecisionType = LogAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TLOG) OP_TYPE(element_wise) void TLog(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    TUnaryOp<DstTile, SrcTile, LogOp<PrecisionType, T>>(dst, src, validRow, validCol, version);
}
template <auto PrecisionType = LogAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TLOG_IMPL(DstTile& dst, SrcTile& src)
{
    TUnaryCheck<DstTile, SrcTile>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TLOG: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TLOG: Number of rows of src and dst must be the same.");
    TLog<PrecisionType, DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* TNEG */
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNEG_IMPL(DstTile& dst, SrcTile& src)
{
    TMULS_IMPL(dst, src, -1);
}
} // namespace pto
#endif
