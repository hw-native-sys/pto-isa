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

namespace pto {
// Formula: fmod(a, b) = a - trunc(a/b) * b
struct FmodSOp {
    PTO_INTERNAL static void FmodSF32Instr(__ubuf__ float *dst, __ubuf__ float *src, float x)
    {
        vector_dup(dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vdiv(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322f32z(dst, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmuls(dst, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void FmodSF16Instr(__ubuf__ half *dst, __ubuf__ half *src, half x)
    {
        vector_dup(dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vdiv(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f162s16z((__ubuf__ int16_t *)dst, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_s162f16(dst, (__ubuf__ int16_t *)dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmuls(dst, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void FmodSInt32Instr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t x)
    {
        __ubuf__ float *dst_f = reinterpret_cast<__ubuf__ float *>(dst);
        __ubuf__ float *src_f = reinterpret_cast<__ubuf__ float *>(src);

        vconv_s322f32(src_f, src, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        FmodSF32Instr(dst_f, src_f, (float)x);

        vconv_f322s32r(dst, dst_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src, src_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void FmodSInt16Instr(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int16_t x)
    {
        __ubuf__ half *dst_f = reinterpret_cast<__ubuf__ half *>(dst);
        __ubuf__ half *src_f = reinterpret_cast<__ubuf__ half *>(src);

        vconv_s162f16(src_f, src, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        FmodSF16Instr(dst_f, src_f, (half)x);

        vconv_f162s16r(dst, dst_f, 1, 1, 1, 8, 8);
        vconv_f162s16r(src, src_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL void TFmodS(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                                typename TileDataSrc::DType x, unsigned validRows, unsigned validCols)
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
            FmodSOp::FmodSF32Instr(dstNext, s0Next, x);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float16_t>) {
            FmodSOp::FmodSF16Instr(dstNext, s0Next, x);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            FmodSOp::FmodSInt32Instr(dstNext, s0Next, x);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            FmodSOp::FmodSInt16Instr(dstNext, s0Next, x);
        } else {
            static_assert(sizeof(T) == 0, "TFMODS: Unsupported tile DType.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFMODS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;

    // static assertions
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "TFMODS: The data types of dst and src must be the same.");
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, float>::value || std::is_same<T, float32_t>::value,
                  "Fix: TFMODS currently supports float and signed 16/32-bit integer data types.");
    static_assert(TileDataDst::Loc == TileType::Vec && TileDataSrc::Loc == TileType::Vec,
                  "TFMODS: TileType of dst and src tiles must be TileType::Vec.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "TFMODS: Only support row major layout.");

    // dynamic checks
    PTO_ASSERT(dst.GetValidRow() == src.GetValidRow() && dst.GetValidRow() > 0,
               "TFMODS: Number of valid rows of dst and src must be the same, and both greater than 0.");
    PTO_ASSERT(dst.GetValidCol() == src.GetValidCol() && dst.GetValidCol() > 0,
               "TFMODS: Number of valid columns of dst and src must be the same, and all greater than 0.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    TFmodS<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dst.data(), src.data(), scalar, dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif