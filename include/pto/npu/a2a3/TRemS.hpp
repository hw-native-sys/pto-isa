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

namespace pto {
// Formula: remainder(a, b) = a - floor(a/b) * b
struct RemSOp {
    PTO_INTERNAL static void RemSF32Instr(__ubuf__ float *dst, __ubuf__ float *src, float x)
    {
        vector_dup(dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vdiv(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322f32f(dst, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmuls(dst, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemSInt32Instr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t x)
    {
        __ubuf__ float *dst_f = reinterpret_cast<__ubuf__ float *>(dst);
        __ubuf__ float *src_f = reinterpret_cast<__ubuf__ float *>(src);

        vconv_s322f32(src_f, src, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        RemSF32Instr(dst_f, src_f, (float)x);

        vconv_f322s32r(dst, dst_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src, src_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL void TRemS(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                               typename TileDataDst::DType x, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;

    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = srcPtr + i * srcRowStride;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemSOp::RemSF32Instr(dstNext, s0Next, x);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemSOp::RemSInt32Instr(dstNext, s0Next, x);
        } else {
            static_assert(sizeof(T) == 0, "TREMS: Unsupported tile DType.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TREMS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;

    // static assertions
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "TREMS: The data types of dst and src must be the same.");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
                  "Fix: TREMS supports only float and int32 element types.");
    static_assert(TileDataDst::Loc == TileType::Vec && TileDataSrc::Loc == TileType::Vec,
                  "TREMS: TileType of dst and src tiles must be TileType::Vec.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "TREMS: Only support row major layout.");

    // dynamic checks
    PTO_ASSERT(dst.GetValidRow() == src.GetValidRow() && dst.GetValidRow() > 0,
               "TREMS: Number of valid rows of dst and src must be the same, and both greater than 0.");
    PTO_ASSERT(dst.GetValidCol() == src.GetValidCol() && dst.GetValidCol() > 0,
               "TREMS: Number of valid columns of dst and src must be the same, and all greater than 0.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    TRemS<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dst.data(), src.data(), scalar, dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif