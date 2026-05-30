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
template <typename T, unsigned TmpStride>
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
        __ubuf__ uint8_t *cmpMask = reinterpret_cast<__ubuf__ uint8_t *>(tmp + TmpStride);
        vcmpvs_lt(cmpMask, tmp, 0.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 7: Compute dst + src1 into tmp
        vadd(tmp, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 8: Set the cmpmask for vsel
        set_cmpmask(cmpMask);

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

    PTO_INTERNAL static void RemInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *tmp)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemF32Instr(dst, src0, src1, tmp);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemInt32Instr(dst, src0, src1, tmp);
        }
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

    constexpr unsigned tmpRowStride = TileDataTmp::RowStride;
    uint16_t repeatTimes = validCols / elementsPerRepeat;
    uint16_t repeatRemain = validCols % elementsPerRepeat;

    set_mask_norm();
    set_vector_mask(-1, -1);
    for (uint16_t i = 0; i < validRows; i++) {
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T *s1Next = src1Ptr + i * src1RowStride;
        // Note: tmp buffer needs space for two rows:
        //   - Row 0: intermediate computation results
        //   - Row 1: comparison mask storage
        for (uint16_t j = 0; j < repeatTimes; j++) {
            RemOp<T, tmpRowStride>::RemInstr(dstNext, s0Next, s1Next, tmpPtr);
            dstNext += elementsPerRepeat;
            s0Next += elementsPerRepeat;
            s1Next += elementsPerRepeat;
        }
    }

    if (repeatRemain) {
        SetContinuousMask(repeatRemain);
        for (uint16_t i = 0; i < validRows; i++) {
            __ubuf__ T *dstNext = dstPtr + i * dstRowStride + repeatTimes * elementsPerRepeat;
            __ubuf__ T *s0Next = src0Ptr + i * src0RowStride + repeatTimes * elementsPerRepeat;
            __ubuf__ T *s1Next = src1Ptr + i * src1RowStride + repeatTimes * elementsPerRepeat;
            RemOp<T, tmpRowStride>::RemInstr(dstNext, s0Next, s1Next, tmpPtr);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1,
                            const TileDataTmp &tmp)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
                  "Fix: TREM supports only float and int32 element types.");
    static_assert(std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc0::DType>,
                  "Fix: TREM type of dst must be same with src0 and src1.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TREM support only row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
    // tmp buffer needs space for two rows: row 0 for intermediate results, row 1 for comparison mask
    PTO_ASSERT(tmp.GetValidCol() >= validCols, "Fix: TREM tmp tile must have at least validCols columns.");
    PTO_ASSERT(tmp.GetValidRow() >= 2, "Fix: TREM tmp tile must have at least 2 rows.");
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
