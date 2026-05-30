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
// Note: For fp32, after computing remainder, we check if result * scalar < 0.
//       If signs differ, we add scalar to result to ensure the result has the same sign as scalar.
struct RemSOp {
    PTO_INTERNAL static void RemSF32Instr(__ubuf__ float *dst, __ubuf__ float *src, float x, __ubuf__ float *tmp)
    {
        // Step 1: tmp = x (broadcast scalar)
        vector_dup(tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 2: tmp = src / x
        vdiv(tmp, src, tmp, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 3: tmp = floor(tmp) - truncate towards zero
        vconv_f322f32f(tmp, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 4: dst = tmp * x = floor(src/x) * x
        vmuls(dst, tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 5: dst = src - dst = src - floor(src/x) * x (this is the remainder)
        vsub(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Sign correction: if dst * scalar < 0, then dst += scalar
        // Step 6: tmp = dst * x (check if signs differ)
        vmuls(tmp, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 7: Compare tmp < 0 using vcmpvs_lt, result goes to cmpmask
        __ubuf__ uint8_t *cmpMask = reinterpret_cast<__ubuf__ uint8_t *>(tmp);
        vcmpvs_lt(cmpMask, tmp, 0.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 8: Set the cmpmask for vsel
        set_cmpmask(cmpMask);
        pipe_barrier(PIPE_V);

        // Step 9: Compute dst + x into tmp
        vadds(tmp, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 10: vsel with selectMode 0
        // If cmpmask bit is set (tmp < 0), select tmp (dst + x), else keep dst
        vsel(dst, tmp, dst, 1, 1, 1, 1, 8, 8, 8, 0);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemSInt32Instr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t x,
                                            __ubuf__ int32_t *tmp)
    {
        __ubuf__ float *dst_f = reinterpret_cast<__ubuf__ float *>(dst);
        __ubuf__ float *src_f = reinterpret_cast<__ubuf__ float *>(src);
        __ubuf__ float *tmp_f = reinterpret_cast<__ubuf__ float *>(tmp);

        vconv_s322f32(src_f, src, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        RemSF32Instr(dst_f, src_f, (float)x, tmp_f);

        vconv_f322s32r(dst, dst_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src, src_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL void TRemS(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                               typename TileDataDst::DType x, typename TileDataTmp::TileDType __in__ tmp,
                               unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;

    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = srcPtr + i * srcRowStride;
        // Note: tmp buffer is reused for each row iteration to save memory
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemSOp::RemSF32Instr(dstNext, s0Next, x, tmpPtr);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemSOp::RemSInt32Instr(dstNext, s0Next, x, tmpPtr);
        } else {
            static_assert(sizeof(T) == 0, "TREMS: Unsupported tile DType.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp)
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
    PTO_ASSERT(tmp.GetValidCol() >= dst.GetValidCol(), "TREMS: tmp tile must have at least dst.GetValidCol() columns.");
    PTO_ASSERT(tmp.GetValidRow() >= 1, "TREMS: tmp tile must have at least 1 row.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    TRemS<TileDataDst, TileDataSrc, TileDataTmp, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dst.data(), src.data(), scalar, tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif