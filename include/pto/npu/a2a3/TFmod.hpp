/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TFMOD_HPP
#define TFMOD_HPP

#include <pto/common/constants.hpp>

namespace pto {
// Formula: fmod(a, b) = a - trunc(a/b) * b
struct FmodOp {
    PTO_INTERNAL static void FmodF32Instr(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1)
    {
        vdiv(dst, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322f32z(dst, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmul(dst, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src0, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride = dstRowStride,
          unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TFmod(typename TileDataDst::TileDType __out__ dst,
                               typename TileDataSrc0::TileDType __in__ src0,
                               typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        unsigned colsFmodaining = validCols;
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T *s1Next = src1Ptr + i * src1RowStride;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            FmodOp::FmodF32Instr(dstNext, s0Next, s1Next);
        } else {
            static_assert(sizeof(T) == 0, "Fix: Unsupported element type for TFMOD.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFmodCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float32_t>,
                  "Fix: TFMOD supports only float element type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TFMOD support only row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TFMOD input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TFMOD input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFMOD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    TFmodCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TFmod<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride,
          src1RowStride>(dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}

} // namespace pto

#endif
