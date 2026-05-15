/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFMODS_HPP
#define TFMODS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"
#include "custom/TFmodRemHp.hpp"

namespace pto {

template <FmodSAlgorithm PrecisionType, typename T>
struct FModSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T scalar, MaskReg &preg)
    {
        RegTensor<T> reg_src1;
        vdup(reg_src1, scalar, preg, MODE_ZEROING);
        if constexpr (PrecisionType == FmodSAlgorithm::HIGH_PRECISION && std::is_same<T, float>::value) {
            TFmodRemHP<true>(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (std::is_same<T, float>::value) {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_Z, preg);
            vmuls(reg_dst, reg_dst, scalar, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<T> reg_dst_even, reg_dst_odd;
            RegTensor<float> reg_src0_even, reg_src0_odd, reg_src1_even, reg_src1_odd, reg_even, reg_odd;
            vcvt(reg_src0_even, reg_src0, preg, PART_EVEN);
            vcvt(reg_src1_even, reg_src1, preg, PART_EVEN);
            vdiv(reg_even, reg_src0_even, reg_src1_even, preg, MODE_ZEROING);
            vtrc(reg_even, reg_even, ROUND_Z, preg);
            vmuls(reg_even, reg_even, (float)scalar, preg, MODE_ZEROING);
            vsub(reg_even, reg_src0_even, reg_even, preg, MODE_ZEROING);
            vcvt(reg_dst_even, reg_even, preg, ROUND_Z, RS_ENABLE, PART_EVEN);

            vcvt(reg_src0_odd, reg_src0, preg, PART_ODD);
            vcvt(reg_src1_odd, reg_src1, preg, PART_ODD);
            vdiv(reg_odd, reg_src0_odd, reg_src1_odd, preg, MODE_ZEROING);
            vtrc(reg_odd, reg_odd, ROUND_Z, preg);
            vmuls(reg_odd, reg_odd, (float)scalar, preg, MODE_ZEROING);
            vsub(reg_odd, reg_src0_odd, reg_odd, preg, MODE_ZEROING);
            vcvt(reg_dst_odd, reg_odd, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vmuls(reg_dst, reg_dst, scalar, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        }
    }
};

template <FmodSAlgorithm PrecisionType, typename TileDataDst, typename TileDataSrc, unsigned dstRowStride,
          unsigned srcRowStride>
__tf__ PTO_INTERNAL OP_NAME(TFMODS)
    OP_TYPE(element_wise) void TFModS(typename TileDataDst::TileDType __out__ dst,
                                      typename TileDataSrc::TileDType __in__ src, typename TileDataSrc::DType scalar,
                                      unsigned kValidRows, unsigned kValidCols,
                                      VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    BinaryInstr<FModSOp<PrecisionType, T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                srcRowStride>(dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFModSCheck(unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc::DType>::value, "The data type must be same of src and dst");
    static_assert((sizeof(T) == 2) || (sizeof(T) == 4), "TFMODS: Invalid data type");
    static_assert((TileDataDst::Loc == TileType::Vec) && (TileDataSrc::Loc == TileType::Vec),
                  "TileType of dst and src tiles must be TileType::Vec.");
    static_assert((TileDataDst::ValidCol <= TileDataDst::Cols) && (TileDataDst::ValidRow <= TileDataDst::Rows) &&
                      (TileDataSrc::ValidCol <= TileDataSrc::Cols) && (TileDataSrc::ValidRow <= TileDataSrc::Rows),
                  "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <auto PrecisionType = FmodSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFMODS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    PTO_ASSERT((src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
               "Number of validColumns and validRows of src and dst must be the same.");

    TFModSCheck<TileDataDst, TileDataSrc>(src.GetValidRow(), src.GetValidCol(), validRow, validCol);
    TFModS<PrecisionType, TileDataDst, TileDataSrc, dstRowStride, srcRowStride>(dst.data(), src.data(), scalar,
                                                                                validRow, validCol);
}
} // namespace pto
#endif
