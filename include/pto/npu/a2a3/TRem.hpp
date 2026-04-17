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

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>

namespace pto {
// Formula: remainder(a, b) = a - floor(a/b) * b
// Note: For fp32, after computing remainder, we check if result * divider < 0.
//       If signs differ, we add divider to result to ensure the result has the same sign as divider.
struct RemOp {
    PTO_INTERNAL static void RemF32Instr(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                                         __ubuf__ float *tmp)
    {
        // Step 1: tmp = src0 / src1 (division result before floor)
        vdiv(tmp, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 2: tmp = floor(tmp) - truncate towards zero
        vconv_f322f32f(tmp, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 3: dst = tmp * src1 = floor(a/b) * b
        vmul(dst, tmp, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 4: dst = src0 - dst = a - floor(a/b) * b (this is the remainder)
        vsub(dst, src0, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Sign correction: if dst * src1 < 0, then dst += src1
        // Step 5: tmp = dst * src1 (check if signs differ)
        vmul(tmp, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 6: Compare tmp < 0 using vcmpvs_lt, result goes to cmpmask
        // Use tmp buffer cast to uint8_t for comparison result storage
        __ubuf__ uint8_t *cmpMask = reinterpret_cast<__ubuf__ uint8_t *>(tmp);
        vcmpvs_lt(cmpMask, tmp, 0.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 7: Set the cmpmask for vsel
        set_cmpmask(cmpMask);

        // Step 8: Compute dst + src1 into tmp
        vadd(tmp, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 9: vsel with selectMode 0
        // If cmpmask bit is set (tmp < 0), select tmp (dst + src1), else keep dst
        vsel(dst, tmp, dst, 1, 1, 1, 1, 8, 8, 8, 0);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemInt32Instr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
                                           __ubuf__ int32_t *tmp)
    {
        __ubuf__ float *dst_f = reinterpret_cast<__ubuf__ float *>(dst);
        __ubuf__ float *src0_f = reinterpret_cast<__ubuf__ float *>(src0);
        __ubuf__ float *src1_f = reinterpret_cast<__ubuf__ float *>(src1);
        __ubuf__ float *tmp_f = reinterpret_cast<__ubuf__ float *>(tmp);

        vconv_s322f32(src0_f, src0, 1, 1, 1, 8, 8);
        vconv_s322f32(src1_f, src1, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        RemF32Instr(dst_f, src0_f, src1_f, tmp_f);

        vconv_f322s32r(dst, dst_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src0, src0_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src1, src1_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
          unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TRem(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
                              typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp,
                              unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        unsigned colsRemaining = validCols;
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T *s1Next = src1Ptr + i * src1RowStride;
        // Note: tmp buffer is reused for each row iteration to save memory
        // Only needs space for one row (validCols elements)
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemOp::RemF32Instr(dstNext, s0Next, s1Next, tmpPtr);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemOp::RemInt32Instr(dstNext, s0Next, s1Next, tmpPtr);
        } else {
            static_assert(sizeof(T) == 0, "Fix: Unsupported element type for TREM.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1,
                            const TileDataTmp &tmp)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
                  "Fix: TREM supports only float and int32 element types.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TREM support only row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
    // tmp buffer only needs space for one row (reused across iterations)
    PTO_ASSERT(tmp.GetValidCol() >= validCols, "Fix: TREM tmp tile must have at least validCols columns.");
    PTO_ASSERT(tmp.GetValidRow() >= 1, "Fix: TREM tmp tile must have at least 1 row.");
}

template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    using T = typename TileDataDst::DType;
    TRemCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TRem<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, elementsPerRepeat, blockSizeElem, dstRowStride,
         src0RowStride, src1RowStride>(dst.data(), src0.data(), src1.data(), tmp.data(), dst.GetValidRow(),
                                       dst.GetValidCol());
}

} // namespace pto

#endif
