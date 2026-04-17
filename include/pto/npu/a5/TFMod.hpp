/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFMOD_HPP
#define TFMOD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>
#include "custom/TFmodRemHp.hpp"

namespace pto {

template <typename T>
struct FModOp {
    PTO_INTERNAL static void BinInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                      MaskReg &preg)
    {
        if constexpr (std::is_same<T, float>::value) {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_Z, preg);
            vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<float> reg_tmp_even0, reg_tmp_even1, reg_tmp_even2, reg_tmp_odd0, reg_tmp_odd1, reg_tmp_odd2;
            RegTensor<T> reg_dst_even, reg_dst_odd;
            vcvt(reg_tmp_even0, reg_src0, preg, PART_EVEN);
            vcvt(reg_tmp_even1, reg_src1, preg, PART_EVEN);
            vdiv(reg_tmp_even2, reg_tmp_even0, reg_tmp_even1, preg, MODE_ZEROING);
            vtrc(reg_tmp_even2, reg_tmp_even2, ROUND_Z, preg);
            vmul(reg_tmp_even2, reg_tmp_even2, reg_tmp_even1, preg, MODE_ZEROING);
            vsub(reg_tmp_even2, reg_tmp_even0, reg_tmp_even2, preg, MODE_ZEROING);
            vcvt(reg_dst_even, reg_tmp_even2, preg, ROUND_Z, RS_ENABLE, PART_EVEN);

            vcvt(reg_tmp_odd0, reg_src0, preg, PART_ODD);
            vcvt(reg_tmp_odd1, reg_src1, preg, PART_ODD);
            vdiv(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd1, preg, MODE_ZEROING);
            vtrc(reg_tmp_odd2, reg_tmp_odd2, ROUND_Z, preg);
            vmul(reg_tmp_odd2, reg_tmp_odd2, reg_tmp_odd1, preg, MODE_ZEROING);
            vsub(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd2, preg, MODE_ZEROING);
            vcvt(reg_dst_odd, reg_tmp_odd2, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        }
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
          unsigned BlockSizeElem, auto PrecisionType = FmodAlgorithm::DEFAULT>
__tf__ PTO_INTERNAL void TFMod(typename TileDataDst::TileDType __out__ dst,
                               typename TileDataSrc0::TileDType __in__ src0,
                               typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
                               VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    if constexpr (PrecisionType == FmodAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
        constexpr uint32_t FMOD_INTERATION_NUM_MAX = 11;
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
        uint32_t mainRepeatTimes = validCols / ElementsPerRepeat;
        uint32_t tailCount = validCols - mainRepeatTimes * ElementsPerRepeat;
        for (uint16_t i = 0; i < validRows; i++) {
            ComputeIterationF32<FMOD_INTERATION_NUM_MAX>(dstPtr + i * dstRowStride, src0Ptr + i * src0RowStride,
                                                         src1Ptr + i * src1RowStride, mainRepeatTimes,
                                                         ElementsPerRepeat, tailCount, true);
        }
    } else {
        BinaryInstr<FModOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFModCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value || std::is_same<T, uint16_t>::value ||
                      std::is_same<T, int16_t>::value || std::is_same<T, uint32_t>::value ||
                      std::is_same<T, int32_t>::value,
                  "Fix: TFMOD has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TFMOD only support row major layout.");
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TFMOD input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TFMOD input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TFMOD input tile src1 valid shape mismatch with output tile dst shape.");
}

template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1>
PTO_INTERNAL void TFMOD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    TFModCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    TFMod<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, PrecisionType>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
