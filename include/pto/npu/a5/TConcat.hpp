/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename T>
struct IndexConcat {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "Fix: Unsupported DType size for index vector.");
    using Scalar = std::conditional_t<sizeof(T) == sizeof(int32_t), int32_t,
                                      std::conditional_t<sizeof(T) == sizeof(int16_t), int16_t, int8_t>>;
    using type = RegTensor<Scalar>;
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TCONCAT)
    OP_TYPE(element_wise) void TConcat(typename TileDataDst::TileDType __out__ dst,
                                       typename TileDataSrc0::TileDType __in__ src0,
                                       typename TileDataSrc1::TileDType __in__ src1, unsigned validRows,
                                       unsigned validCols0, unsigned validCols1)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    unsigned validColsDst = validCols0 + validCols1;
    unsigned repeatTimes0 = CeilDivision(validCols0, ElementsPerRepeat);
    unsigned repeatTimes1 = CeilDivision(validCols1, ElementsPerRepeat);

    using IndexScalar = typename IndexConcat<T>::Scalar;
    typename IndexConcat<T>::type vreg_idx;
    using UnsignedIndexScalar = typename std::make_unsigned<IndexScalar>::type;

    __VEC_SCOPE__
    {
        RegTensor<T> vreg_0;
        RegTensor<T> vreg_1;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg0 = validCols0;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes0; ++j) {
                preg = CreatePredicate<T>(sreg0);
                vlds(vreg_0, src0Ptr, i * TileDataSrc0::RowStride + j * ElementsPerRepeat, NORM);
                vsts(vreg_0, dstPtr, i * TileDataDst::RowStride + j * ElementsPerRepeat, distValue, preg);
            }

            mem_bar(VST_VLD);
            uint32_t sreg1 = validCols1;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes1; ++j) {
                preg = CreatePredicate<T>(sreg1);
                vlds(vreg_1, src1Ptr, i * TileDataSrc1::RowStride + j * ElementsPerRepeat, NORM);
                vci((RegTensor<IndexScalar> &)vreg_idx,
                    (IndexScalar)(i * TileDataDst::RowStride + validCols0 + j * ElementsPerRepeat), INC_ORDER);
                vscatter(vreg_1, dstPtr, (RegTensor<UnsignedIndexScalar> &)vreg_idx, preg);
            }
        }
    }
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TConcatCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                      std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                  "Fix: TCONCAT has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TCONCAT only support row major layout.");
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Fix: TCONCAT input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols0 = src0.GetValidCol();
    unsigned validCols1 = src1.GetValidCol();
    unsigned validColsDst = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows,
               "Fix: TCONCAT input tile src0 valid row mismatch with output tile dst row.");
    PTO_ASSERT(src1.GetValidRow() == validRows,
               "Fix: TCONCAT input tile src1 valid row mismatch with output tile dst row.");
    PTO_ASSERT(validCols0 + validCols1 == validColsDst,
               "Fix: TCONCAT output tile valid col should be equal to sum of input tiles cols.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCONCAT_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    TConcatCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    TConcat<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), src0.GetValidCol(), src1.GetValidCol());
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx>
__tf__ PTO_INTERNAL void TConcatIdx(typename TileDst::TileDType __out__ dst, typename TileSrc0::TileDType __in__ src0,
                                    typename TileSrc1::TileDType __in__ src1,
                                    typename TileSrc0Idx::TileDType __in__ idx0,
                                    typename TileSrc1Idx::TileDType __in__ idx1, unsigned validRow,
                                    unsigned dstValidCol)
{
    using dataType = typename TileDst::DType;
    using idxType = typename TileSrc0Idx::DType;

    __ubuf__ dataType *dstPtr = (__ubuf__ dataType *)__cce_get_tile_ptr(dst);
    __ubuf__ dataType *src0Ptr = (__ubuf__ dataType *)__cce_get_tile_ptr(src0);
    __ubuf__ dataType *src1Ptr = (__ubuf__ dataType *)__cce_get_tile_ptr(src1);
    __ubuf__ idxType *idx0Ptr = (__ubuf__ idxType *)__cce_get_tile_ptr(idx0);
    __ubuf__ idxType *idx1Ptr = (__ubuf__ idxType *)__cce_get_tile_ptr(idx1);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(dataType);
    constexpr unsigned dstStride = TileDst::RowStride;
    constexpr unsigned src0Stride = TileSrc0::RowStride;
    constexpr unsigned src1Stride = TileSrc1::RowStride;
    constexpr unsigned idx0Stride = TileSrc0Idx::RowStride;
    constexpr unsigned idx1Stride = TileSrc1Idx::RowStride;

    __VEC_SCOPE__
    {
        RegTensor<dataType> vreg_0;
        RegTensor<dataType> vreg_1;
        using IndexScalar = typename IndexConcat<dataType>::Scalar;
        typename IndexConcat<dataType>::type vreg_idx;
        using UnsignedIndexScalar = typename std::make_unsigned<IndexScalar>::type;
        MaskReg preg0, preg1;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<dataType, DistVST::DIST_NORM>())>();

        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            unsigned idx0Num = *(idx0 + i * idx0Stride);
            unsigned idx1Num = *(idx1 + i * idx1Stride);
            unsigned sreg0 = idx0Num < dstValidCol ? idx0Num : dstValidCol;
            unsigned src1Col = dstValidCol > sreg0 ? dstValidCol - sreg0 : 0;
            unsigned sreg1 = idx1Num < src1Col ? idx1Num : src1Col;
            unsigned src1Offset = i * dstStride + sreg0;
            uint16_t repeatTimes0 = CeilDivision(sreg0, elementsPerRepeat);
            uint16_t repeatTimes1 = CeilDivision(sreg1, elementsPerRepeat);

            for (uint16_t j = 0; j < repeatTimes0; ++j) {
                preg0 = CreatePredicate<dataType>(sreg0);
                vlds(vreg_0, src0Ptr, i * src0Stride + j * elementsPerRepeat, NORM);
                vsts(vreg_0, dstPtr, i * dstStride + j * elementsPerRepeat, distValue, preg0);
            }

            mem_bar(VST_VLD);
            for (uint16_t j = 0; j < repeatTimes1; ++j) {
                preg1 = CreatePredicate<dataType>(sreg1);
                vlds(vreg_1, src1Ptr, i * src1Stride + j * elementsPerRepeat, NORM);
                vci((RegTensor<IndexScalar> &)vreg_idx, (IndexScalar)(src1Offset + j * elementsPerRepeat), INC_ORDER);
                vscatter(vreg_1, dstPtr, (RegTensor<UnsignedIndexScalar> &)vreg_idx, preg1);
            }
        }
    }
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx)
{
    using dataType = typename TileDst::DType;
    using idxType = typename TileSrc0Idx::DType;

    static_assert(std::is_same<dataType, typename TileSrc0::DType>::value &&
                      std::is_same<dataType, typename TileSrc0::DType>::value,
                  "TCONCAT: Data type of dst, src0 and src1 must be the same.");
    static_assert(std::is_same<idxType, typename TileSrc1Idx::DType>::value,
                  "TCONCAT: Data type of src0Idx and src1Idx must be the same.");
    static_assert(std::is_same<dataType, int32_t>::value || std::is_same<dataType, int16_t>::value ||
                      std::is_same<dataType, int8_t>::value || std::is_same<dataType, uint32_t>::value ||
                      std::is_same<dataType, uint16_t>::value || std::is_same<dataType, uint8_t>::value ||
                      std::is_same<dataType, half>::value || std::is_same<dataType, float32_t>::value ||
                      std::is_same<dataType, bfloat16_t>::value,
                  "TCONCAT: Invalid data type.");
    static_assert(TileDst::Loc == TileType::Vec && TileSrc0::Loc == TileType::Vec && TileSrc1::Loc == TileType::Vec,
                  "TCONCAT: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDst::ValidRow <= TileDst::Rows && TileSrc0::ValidRow <= TileSrc0::Rows &&
                      TileSrc1::ValidRow <= TileSrc1::Rows,
                  "TCONCAT: Number of valid rows must not be greater than number of tile rows.");
    static_assert(std::is_same<idxType, int32_t>::value || std::is_same<idxType, int16_t>::value ||
                      std::is_same<idxType, int8_t>::value || std::is_same<idxType, uint32_t>::value ||
                      std::is_same<idxType, uint16_t>::value || std::is_same<idxType, uint8_t>::value,
                  "TCONCAT: Invalid data type of src0Idx.");

    unsigned validRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    PTO_ASSERT(validRow == src0.GetValidRow(), "TCONCAT: validRow of src0 must match dst.");
    PTO_ASSERT(validRow == src1.GetValidRow(), "TCONCAT: validRow of src1 must match dst.");

    TConcatIdx<TileDst, TileSrc0, TileSrc1, TileSrc0Idx, TileSrc1Idx>(
        dst.data(), src0.data(), src1.data(), src0Idx.data(), src1Idx.data(), validRow, dstValidCol);
}
} // namespace pto
#endif
