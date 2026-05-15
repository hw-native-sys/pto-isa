/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TXOR_HPP
#define TXOR_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename T>
struct XorOp {
    using U = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
    PTO_INTERNAL static void BinInstr(RegTensor<T> &dstReg, RegTensor<T> &src0Reg, RegTensor<T> &src1Reg, MaskReg &pReg)
    {
        vxor((RegTensor<U> &)dstReg, (RegTensor<U> &)src0Reg, (RegTensor<U> &)src1Reg, pReg);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
          unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TXOR)
    OP_TYPE(element_wise) void TXor(typename TileDataDst::TileDType __out__ dst,
                                    typename TileDataSrc0::TileDType __in__ src0,
                                    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows,
                                    unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<XorOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
        dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TXorCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
                      std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
                  "Fix: TXOR has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TXOR only support row major layout.");
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Fix: TXOR input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TXOR input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TXOR input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TXOR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    TXorCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    TXor<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
