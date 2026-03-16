/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRANS_HPP
#define TTRANS_HPP

#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB32ColWise(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc::TileDType __in__ src, unsigned numRows,
                                          unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataSrc::Rows - 1) * srcStride + (TileDataSrc::Cols - 1) <= 0xFFFFFFFFULL,
                  "Fix: TTRANS gather index may overflow uint32_t register");
    if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, float>) {
        uint16_t repeatTimes = CeilDivision(numRows, elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM_B32>())>();
            for (uint16_t col = 0; col < (uint16_t)numCols; ++col) {
                uint32_t sreg = (uint32_t)numRows;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<T>(sreg);
                    vci((RegTensor<int32_t> &)vreg0, (int32_t)(chunk * elementsPerRepeat), INC_ORDER);
                    vmins(vreg0, vreg0, (uint32_t)(numRows - 1), preg);
                    vmuls(vreg0, vreg0, srcStride, preg);
                    vadds(vreg0, vreg0, col, preg);
                    vgather2(vreg1, srcPtr, (RegTensor<uint32_t> &)vreg0, preg);
                    vsts(vreg1, dstPtr, (col * dstStride + chunk * elementsPerRepeat), distValue, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 4, "Fix: TTRANS has Invalid b32 data type.");
    }
}

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB32RowWise(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc::TileDType __in__ src, unsigned numRows,
                                          unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataDst::Rows - 1) * dstStride + (TileDataDst::Cols - 1) <= 0xFFFFFFFFULL,
                  "Fix: TTRANS scatter index may overflow uint32_t register");
    if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, float>) {
        uint16_t repeatTimes = CeilDivision(numCols, elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            for (uint16_t row = 0; row < (uint16_t)numRows; ++row) {
                uint32_t sreg = (uint32_t)numCols;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(vreg1, srcPtr, (row * srcStride + chunk * elementsPerRepeat), NORM);
                    vci((RegTensor<int32_t> &)vreg0, (int32_t)(chunk * elementsPerRepeat), INC_ORDER);
                    vmins(vreg0, vreg0, (uint32_t)(numCols - 1), preg);
                    vmuls(vreg0, vreg0, dstStride, preg);
                    vadds(vreg0, vreg0, row, preg);
                    vscatter(vreg1, dstPtr, (RegTensor<uint32_t> &)vreg0, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 4, "Fix: TTRANS has Invalid b32 data type.");
    }
}

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB16ColWise(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc::TileDType __in__ src, unsigned numRows,
                                          unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataSrc::Rows - 1) * srcStride + (TileDataSrc::Cols - 1) <= 0xFFFFULL,
                  "Fix: TTRANS gather index may overflow uint16_t register");
    if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, half>) {
        uint16_t repeatTimes = CeilDivision(numRows, elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<uint16_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM_B16>())>();
            for (uint16_t col = 0; col < (uint16_t)numCols; ++col) {
                uint32_t sreg = (uint32_t)numRows;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<T>(sreg);
                    vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * elementsPerRepeat), INC_ORDER);
                    vmins(vreg0, vreg0, (uint16_t)(numRows - 1), preg);
                    vmuls(vreg0, vreg0, srcStride, preg);
                    vadds(vreg0, vreg0, col, preg);
                    vgather2(vreg1, srcPtr, (RegTensor<uint16_t> &)vreg0, preg);
                    vsts(vreg1, dstPtr, (col * dstStride + chunk * elementsPerRepeat), distValue, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 2, "Fix: TTRANS has invalid b16 data type.");
    }
}

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB16RowWise(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc::TileDType __in__ src, unsigned numRows,
                                          unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataDst::Rows - 1) * dstStride + (TileDataDst::Cols - 1) <= 0xFFFFULL,
                  "Fix: TTRANS scatter index may overflow uint16_t register");
    if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, half>) {
        uint16_t repeatTimes = CeilDivision(numCols, elementsPerRepeat);
        __VEC_SCOPE__
        {
            RegTensor<uint16_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            for (uint16_t row = 0; row < (uint16_t)numRows; ++row) {
                uint32_t sreg = (uint32_t)numCols;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(vreg1, srcPtr, (row * srcStride + chunk * elementsPerRepeat), NORM);
                    vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * elementsPerRepeat), INC_ORDER);
                    vmins(vreg0, vreg0, (uint16_t)(numCols - 1), preg);
                    vmuls(vreg0, vreg0, dstStride, preg);
                    vadds(vreg0, vreg0, row, preg);
                    vscatter(vreg1, dstPtr, (RegTensor<uint16_t> &)vreg0, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 2, "Fix: TTRANS has invalid b16 data type.");
    }
}

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB8ColWise(typename TileDataDst::TileDType __out__ dst,
                                         typename TileDataSrc::TileDType __in__ src, unsigned numRows, unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataSrc::Rows - 1) * srcStride + (TileDataSrc::Cols - 1) <= 0xFFFFULL,
                  "Fix: TTRANS gather index may overflow uint16_t register");
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
        constexpr uint32_t sregLower = elementsPerRepeat >> 1;
        uint16_t repeatTimes = CeilDivision(numRows, sregLower);
        __VEC_SCOPE__
        {
            RegTensor<uint16_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_PK_B16>())>();
            for (uint16_t col = 0; col < (uint16_t)numCols; ++col) {
                uint32_t sreg = (uint32_t)numRows;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<uint16_t>(sreg);
                    vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * sregLower), INC_ORDER);
                    vmins(vreg0, vreg0, (uint16_t)(numRows - 1), preg);
                    vmuls(vreg0, vreg0, srcStride, preg);
                    vadds(vreg0, vreg0, col, preg);
                    vgather2((RegTensor<uint16_t> &)vreg1, (__ubuf__ uint8_t *)srcPtr, (RegTensor<uint16_t> &)vreg0,
                             preg);
                    vsts(vreg1, dstPtr, (col * dstStride + chunk * sregLower), distValue, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 1, "Fix: TTRANS has invalid b8 data type.");
    }
}

template <typename TileDataSrc, typename TileDataDst, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransB8RowWise(typename TileDataDst::TileDType __out__ dst,
                                         typename TileDataSrc::TileDType __in__ src, unsigned numRows, unsigned numCols)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;

    static_assert((unsigned long long)(TileDataDst::Rows - 1) * dstStride + (TileDataDst::Cols - 1) <= 0xFFFFULL,
                  "Fix: TTRANS scatter index may overflow uint16_t register");
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
        constexpr uint32_t sregLower = elementsPerRepeat >> 1;
        uint16_t repeatTimes = CeilDivision(numCols, sregLower);
        __VEC_SCOPE__
        {
            RegTensor<uint16_t> vreg0;
            RegTensor<T> vreg1;
            MaskReg preg;
            for (uint16_t row = 0; row < (uint16_t)numRows; ++row) {
                uint32_t sreg = (uint32_t)numCols;
                for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                    preg = CreatePredicate<uint16_t>(sreg);
                    vlds(vreg1, srcPtr, (row * srcStride + chunk * sregLower), UNPK_B8);
                    vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * sregLower), INC_ORDER);
                    vmins(vreg0, vreg0, (uint16_t)(numCols - 1), preg);
                    vmuls(vreg0, vreg0, dstStride, preg);
                    vadds(vreg0, vreg0, row, preg);
                    vscatter(vreg1, dstPtr, (RegTensor<uint16_t> &)vreg0, preg);
                }
            }
        }
    } else {
        static_assert(sizeof(T) == 1, "Fix: TTRANS has invalid b8 data type.");
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataDst::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Fix: TTRANS has unsupported data type.");
    static_assert(sizeof(T) == sizeof(U), "Fix: TTRANS has inconsistent input and output data types.");
    static_assert(TileDataSrc::isRowMajor, "Fix: TTRANS has not supported layout type.");

    if constexpr (TileDataSrc::isRowMajor) {
        static_assert(TileDataSrc::Cols * sizeof(T) % 32 == 0, "Fix: TTRANS has inconsistent input shape.");
        static_assert(TileDataDst::Cols * sizeof(U) % 32 == 0, "Fix: TTRANS has inconsistent output shape.");
    } else {
        static_assert(TileDataSrc::Rows * sizeof(T) % 32 == 0, "Fix: TTRANS has inconsistent input shape.");
        static_assert(TileDataDst::Rows * sizeof(U) % 32 == 0, "Fix: TTRANS has inconsistent output shape.");
    }

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);

    unsigned numRows = (unsigned)src.GetValidRow();
    unsigned numCols = (unsigned)src.GetValidCol();

    if constexpr (sizeof(T) == 4) {
        if constexpr (TileDataSrc::Rows < TileDataSrc::Cols) {
            TTransB32RowWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(),
                                                                                         numRows, numCols);
        } else {
            TTransB32ColWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(),
                                                                                         numRows, numCols);
        }
    } else if constexpr (sizeof(T) == 2) {
        if constexpr (TileDataSrc::Rows < TileDataSrc::Cols) {
            TTransB16RowWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(),
                                                                                         numRows, numCols);
        } else {
            TTransB16ColWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(),
                                                                                         numRows, numCols);
        }
    } else if constexpr (sizeof(T) == 1) {
        if constexpr (TileDataSrc::Rows < TileDataSrc::Cols) {
            TTransB8RowWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(), numRows,
                                                                                        numCols);
        } else {
            TTransB8ColWise<TileDataSrc, TileDataDst, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(), numRows,
                                                                                        numCols);
        }
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Fix: TTRANS has invalid data type.");
    }
}
} // namespace pto

#endif // TTRANS_HPP