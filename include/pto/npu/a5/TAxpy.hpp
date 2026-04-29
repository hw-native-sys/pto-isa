/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TAXPY_HPP
#define TAXPY_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <typename T, typename U>
PTO_INTERNAL static void CallAxpy(RegTensor<T> &reg_dst, RegTensor<U> &reg_src0, U scalar, MaskReg &preg)
{
    if constexpr (std::is_same_v<T, U>) {
        vaxpy(reg_dst, reg_src0, scalar, preg);
    } else {
        // fp32 + fp16 * fp16
        RegTensor<T> reg_src_tmp;
        vcvt(reg_src_tmp, reg_src0, preg, PART_EVEN);
        vaxpy(reg_dst, reg_src_tmp, (T)(scalar), preg);
    }
}

template <typename T, typename U, unsigned elementsPerRepeat, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void AxpyInstr(__ubuf__ T *dstPtr, __ubuf__ U *src0Ptr, U scalar, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<U> vreg0;
        RegTensor<T> vreg2;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                if constexpr (std::is_same_v<T, U>) {
                    vlds(vreg0, src0Ptr, i * srcRowStride + j * elementsPerRepeat, NORM);
                } else {
                    vlds(vreg0, src0Ptr, i * srcRowStride + j * elementsPerRepeat, UNPK_B16);
                }
                vlds(vreg2, dstPtr, i * dstRowStride + j * elementsPerRepeat, NORM);
                uint32_t count =
                    ((j + 1) * elementsPerRepeat >= validCol ? validCol - j * elementsPerRepeat : elementsPerRepeat);
                preg = CreatePredicate<T>(count);
                CallAxpy<T, U>(vreg2, vreg0, scalar, preg);
                vsts(vreg2, dstPtr, i * dstRowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned dstRowStride,
          unsigned src0RowStride>
__tf__ PTO_INTERNAL void TAxpy(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src0,
                               typename TileDataSrc::DType scalar, unsigned validRow, unsigned validCol)
{
    using T = typename TileDataDst::DType;
    using U = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ U *src0Ptr = (__ubuf__ U *)__cce_get_tile_ptr(src0);
    AxpyInstr<T, U, elementsPerRepeat, dstRowStride, src0RowStride>(dstPtr, src0Ptr, scalar, validRow, validCol);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TAXPY_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    using U = typename TileDataSrc::DType;
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,
                  "TAXPY: Invalid data type");
    static_assert(std::is_same_v<T, U> || (std::is_same_v<T, float> && std::is_same_v<U, half>),
                  "TAXPY: The data type of dst must be consistent with src or dst is float while src is half.");

    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tiles must be TileType::Vec.");

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc::RowStride;

    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TAxpy<TileDataDst, TileDataSrc, elementsPerRepeat, dstRowStride, src0RowStride>(dst.data(), src0.data(), scalar,
                                                                                    validRow, validCol);
}
} // namespace pto
#endif
