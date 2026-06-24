/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "pto/npu/a2a3/TBinOp.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TXorCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1,
                            const TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc0::DType>::value &&
                      std::is_same<T, typename TileDataSrc1::DType>::value &&
                      std::is_same<T, typename TileDataTmp::DType>::value,
                  "Fix: TXOR the data type of dst must be consistent with of src0 and src1.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor && TileDataTmp::isRowMajor,
        "Fix: TXOR only support row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TXOR input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TXOR input tile src1 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(tmp.GetValidRow() == validRows && tmp.GetValidCol() == validCols,
               "Fix: TXOR input tile tmp valid shape mismatch with output tile dst shape.");
#ifndef __PTO_AUTO__
    PTO_ASSERT(dst.data() != src0.data() && dst.data() != src1.data() && dst.data() != tmp.data() &&
                   src0.data() != tmp.data() && src1.data() != tmp.data(),
               "dst, src0, src1, tmp must in different memory.");
#endif
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TXOR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    TXorCheck(dst, src0, src1, tmp);
    TOR_IMPL(tmp, src0, src1);
    pipe_barrier(PIPE_V);
    TAND_IMPL(dst, src0, src1);
    pipe_barrier(PIPE_V);
    TNOT_IMPL(dst, dst);
    pipe_barrier(PIPE_V);
    TAND_IMPL(dst, dst, tmp);
}
} // namespace pto

#endif
