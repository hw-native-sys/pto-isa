/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDSUB_HPP
#define TROWEXPANDSUB_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TRowExpandBinOp.hpp"

namespace pto {

template <typename T>
struct RowExpandSubOp {
    PTO_INTERNAL static void RowExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        vsub(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
    }
};

template <typename T>
struct RowExpandSubOp2 {
    PTO_INTERNAL static void RowExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        vsub(reg_dst, reg_src1, reg_src0, preg, MODE_ZEROING);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem>
__tf__ AICORE OP_NAME(TROWEXPANDSUB)
    OP_TYPE(broadcast) void TRowExpandSub(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc0::TileDType __in__ src0,
                                          typename TileDataSrc1::TileDType __in__ src1, unsigned validRow,
                                          unsigned validCol, bool src0eqdst,
                                          unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    if (src0eqdst) {
        RowExpandBinaryInstr<RowExpandSubOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat,
                             blockSizeElem>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        RowExpandBinaryInstr<RowExpandSubOp2<T>, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat,
                             blockSizeElem>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
                  "Fix: TROWEXPANDSUB src and dst data type is different!");
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                      std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                  "Fix: TROWEXPANDSUB Invalid data type.");
    static_assert(TileDataDst::isRowMajor, "Fix: TROWEXPANDSUB Invalid tile shape.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    bool src0eqdst = (validRow == src0ValidRow) && (validCol == src0ValidCol);
    bool src1eqdst = (validRow == src1ValidRow) && (validCol == src1ValidCol);
    PTO_ASSERT((src0eqdst && TileDataSrc0::isRowMajor) || (src1eqdst && TileDataSrc1::isRowMajor),
               "TROWEXPANDSUB: the validShape of src0 or src1 should be equal to dst");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);

    if (src0eqdst) {
        unsigned src1ValidCol = src1.GetValidCol();
        PTO_ASSERT(((TileDataSrc1::isRowMajor && src1ValidCol == 32 / sizeof(T)) ||
                    (!TileDataSrc1::isRowMajor && src1ValidCol == 1)) &&
                       src1.GetValidRow() == validRow,
                   "TROWEXPANDSUB: invalid src1 shape.");
        TRowExpandSub<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
            dst.data(), src0.data(), src1.data(), validRow, validCol, src0eqdst);
    } else {
        unsigned src0ValidCol = src0.GetValidCol();
        PTO_ASSERT(((TileDataSrc0::isRowMajor && src0ValidCol == 32 / sizeof(T)) ||
                    (!TileDataSrc0::isRowMajor && src0ValidCol == 1)) &&
                       src0.GetValidRow() == validRow,
                   "TROWEXPANDSUB: invalid src0 shape.");
        TRowExpandSub<TileDataDst, TileDataSrc1, TileDataSrc0, elementsPerRepeat, blockSizeElem>(
            dst.data(), src1.data(), src0.data(), validRow, validCol, src0eqdst);
    }
}
} // namespace pto
#endif