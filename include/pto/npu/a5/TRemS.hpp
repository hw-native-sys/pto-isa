/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"
#include "custom/TFmodRemHp.hpp"

namespace pto {

template <RemSAlgorithm PrecisionType, typename T>
struct RemSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T scalar, MaskReg &preg)
    {
        RegTensor<T> reg_src1;
        vdup(reg_src1, scalar, preg, MODE_ZEROING);
        if constexpr (PrecisionType == RemSAlgorithm::HIGH_PRECISION && std::is_same<T, float>::value) {
            TFmodRemHP<false>(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (std::is_same<T, float>::value) {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_F, preg);
            vmuls(reg_dst, reg_dst, scalar, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<float> reg_src0_even, reg_src1_even, reg_even, reg_src0_odd, reg_src1_odd, reg_odd;
            RegTensor<T> reg_dst_even, reg_dst_odd;
            vcvt(reg_src0_even, reg_src0, preg, PART_EVEN);
            vcvt(reg_src1_even, reg_src1, preg, PART_EVEN);
            vcvt(reg_src0_odd, reg_src0, preg, PART_ODD);
            vcvt(reg_src1_odd, reg_src1, preg, PART_ODD);

            vdiv(reg_even, reg_src0_even, reg_src1_even, preg, MODE_ZEROING);
            vdiv(reg_odd, reg_src0_odd, reg_src1_odd, preg, MODE_ZEROING);

            vtrc(reg_even, reg_even, ROUND_F, preg);
            vtrc(reg_odd, reg_odd, ROUND_F, preg);

            vmuls(reg_even, reg_even, (float)scalar, preg, MODE_ZEROING);
            vmuls(reg_odd, reg_odd, (float)scalar, preg, MODE_ZEROING);

            vsub(reg_even, reg_src0_even, reg_even, preg, MODE_ZEROING);
            vsub(reg_odd, reg_src0_odd, reg_odd, preg, MODE_ZEROING);

            vcvt(reg_dst_even, reg_even, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(reg_dst_odd, reg_odd, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <RemSAlgorithm PrecisionType, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL OP_NAME(TREMS)
    OP_TYPE(element_wise) void TRemS(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataSrc::TileDType __in__ src, typename TileDataSrc::DType scalar,
                                     typename TileDataTmp::TileDType __in__ tmp, unsigned kValidRows,
                                     unsigned kValidCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    BinaryInstr<RemSOp<PrecisionType, T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                srcRowStride>(dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TRemSCheck(unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol,
                             unsigned tmpValidRow, unsigned tmpValidCol)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc::DType>::value, "The data type must be same of src and dst");
    static_assert((sizeof(T) == 2) || (sizeof(T) == 4), "TREMS: Invalid data type");
    static_assert((TileDataDst::Loc == TileType::Vec) && (TileDataSrc::Loc == TileType::Vec),
                  "TileType of dst and src tiles must be TileType::Vec.");
    static_assert((TileDataDst::ValidCol <= TileDataDst::Cols) && (TileDataDst::ValidRow <= TileDataDst::Rows) &&
                      (TileDataSrc::ValidCol <= TileDataSrc::Cols) && (TileDataSrc::ValidRow <= TileDataSrc::Rows),
                  "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    PTO_ASSERT((src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
               "Number of validColumns and validRows of src and dst must be the same.");

    TRemSCheck<TileDataDst, TileDataSrc, TileDataTmp>(src.GetValidRow(), src.GetValidCol(), validRow, validCol,
                                                      tmp.GetValidRow(), tmp.GetValidCol());
    TRemS<PrecisionType, TileDataDst, TileDataSrc, TileDataTmp, dstRowStride, srcRowStride>(
        dst.data(), src.data(), scalar, tmp.data(), validRow, validCol);
}
} // namespace pto
#endif
