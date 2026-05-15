/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TXORS_HPP
#define TXORS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"

namespace pto {
template <typename T>
struct XorSOp {
    static constexpr bool isDynFunc = true;
    RegTensor<T> reg_src1;

    template <typename S>
    PTO_INTERNAL XorSOp(S scalar)
    {
        vbr(reg_src1, scalar);
    };

    using U = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
    PTO_INTERNAL void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T src1, MaskReg &preg)
    {
        vxor((RegTensor<U> &)reg_dst, (RegTensor<U> &)reg_src0, (RegTensor<U> &)reg_src1, preg);
    }
};

template <typename T, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL OP_NAME(TXORS)
    OP_TYPE(element_wise) void TXorS(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataSrc::TileDType __in__ src0, T src1, unsigned validRows,
                                     unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc::RowStride;
    BinaryInstr<XorSOp<T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride>(
        dstPtr, src0Ptr, src1, validRows, validCols, version);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TXORS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Fix: TXORS has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TXORS only support row major layout.");
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "Fix: TXORS input tile src0, src1 and dst tile data type mismatch.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol,
               "Fix: TXORS input tile src0 valid shape mismatch with output tile dst shape.");

    TXorS<T, TileDataDst, TileDataSrc>(dst.data(), src0.data(), src1, validRow, validCol);
}
} // namespace pto
#endif
