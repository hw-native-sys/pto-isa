/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TAddDeqRelu.hpp
 * @brief Fused Add + Dequantize + ReLU (TADDDEQRELU) for NPU A2/A3.
 *
 * Computes, per element:
 *   dst[i] = max(0, (src0[i] + src1[i]) * deqScale)  converted to half
 *
 * The dequantization uses precision-compensated scaling:
 *   (x >> 17) * deqScale << 17  which is mathematically equivalent to x * deqScale
 *   but avoids precision loss for large int32_t intermediate values.
 *
 * Supported types:
 *   src0, src1 : int32_t
 *   dst        : half
 *   tmp        : int32_t (temporary buffer, same shape as src0/src1)
 *   deqScale   : float scalar
 */

#ifndef TADDDEQRELU_HPP
#define TADDDEQRELU_HPP

#include "common.hpp"

namespace pto {

constexpr float DEQ_SHIFT_RIGHT_17_BIT = 1.0f / 131072.0f;
constexpr float DEQ_SHIFT_LEFT_17_BIT = 131072.0f;

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void ComputeAddDeqReluConvConfig(
    unsigned& elementsPerRepeat, unsigned& dstRepeatStride, unsigned& srcRepeatStride)
{
    constexpr unsigned dstSize = sizeof(typename TileDataDst::DType);
    constexpr unsigned srcSize = sizeof(typename TileDataSrc::DType);
    constexpr unsigned repeatWidth = dstSize > srcSize ? dstSize : srcSize;
    dstRepeatStride = (repeatWidth == dstSize) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / srcSize * dstSize);
    srcRepeatStride = (repeatWidth == srcSize) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / dstSize * srcSize);
    elementsPerRepeat = REPEAT_BYTE / repeatWidth;
}

PTO_INTERNAL void AddDeqReluComputeBlock(
    __ubuf__ half* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, __ubuf__ int32_t* tmp, float deqScale,
    uint8_t repeatCount, uint8_t convDstRepStride, uint8_t convSrcRepStride)
{
    vadd(tmp, src0, src1, repeatCount, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    __ubuf__ float* floatBuf = reinterpret_cast<__ubuf__ float*>(src0);
    vconv_s322f32(floatBuf, tmp, repeatCount, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    vmuls(floatBuf, floatBuf, static_cast<float>(DEQ_SHIFT_RIGHT_17_BIT), repeatCount, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    vmuls(floatBuf, floatBuf, static_cast<float>(deqScale), repeatCount, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    vmuls(floatBuf, floatBuf, static_cast<float>(DEQ_SHIFT_LEFT_17_BIT), repeatCount, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    __ubuf__ float* zeroBuf = reinterpret_cast<__ubuf__ float*>(tmp);
    vector_dup(zeroBuf, 0.0f, repeatCount, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);

    __ubuf__ float* reluBuf = reinterpret_cast<__ubuf__ float*>(src1);
    vmax(reluBuf, floatBuf, zeroBuf, repeatCount, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    vconv_f322f16(dst, reluBuf, repeatCount, 1, 1, convDstRepStride, convSrcRepStride);
    pipe_barrier(PIPE_V);
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, unsigned SS0,
    unsigned SS1, unsigned TS, unsigned DS>
__tf__ PTO_INTERNAL void TAddDeqRelu(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc0::TileDType __in__ src0Data,
    typename TileDataSrc1::TileDType __in__ src1Data, typename TileDataTmp::TileDType __in__ tmpData, float deqScale,
    unsigned validRow, unsigned validCol)
{
    __ubuf__ half* dstPtr = (__ubuf__ half*)__cce_get_tile_ptr(dstData);
    __ubuf__ int32_t* src0Ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(src0Data);
    __ubuf__ int32_t* src1Ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(src1Data);
    __ubuf__ int32_t* tmpPtr = (__ubuf__ int32_t*)__cce_get_tile_ptr(tmpData);

    unsigned elementsPerRepeat, convDstRepStride, convSrcRepStride;
    ComputeAddDeqReluConvConfig<TileDataDst, TileDataSrc0>(elementsPerRepeat, convDstRepStride, convSrcRepStride);

    unsigned numRepeatPerLine = validCol / elementsPerRepeat;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            for (unsigned j = 0; j < numLoop; j++) {
                unsigned span = j * elementsPerRepeat * REPEAT_MAX;
                AddDeqReluComputeBlock(
                    dstPtr + i * DS + span, src0Ptr + i * SS0 + span, src1Ptr + i * SS1 + span, tmpPtr + i * TS + span,
                    deqScale, (uint8_t)REPEAT_MAX, (uint8_t)convDstRepStride, (uint8_t)convSrcRepStride);
            }
            if (remainAfterLoop > 0) {
                unsigned span = numLoop * elementsPerRepeat * REPEAT_MAX;
                AddDeqReluComputeBlock(
                    dstPtr + i * DS + span, src0Ptr + i * SS0 + span, src1Ptr + i * SS1 + span, tmpPtr + i * TS + span,
                    deqScale, (uint8_t)remainAfterLoop, (uint8_t)convDstRepStride, (uint8_t)convSrcRepStride);
            }
        }
    }

    if (numRemainPerLine > 0) {
        unsigned base = numRepeatPerLine * elementsPerRepeat;
        SetContinuousMask(numRemainPerLine);
        for (unsigned i = 0; i < validRow; i++) {
            AddDeqReluComputeBlock(
                dstPtr + i * DS + base, src0Ptr + i * SS0 + base, src1Ptr + i * SS1 + base, tmpPtr + i * TS + base,
                deqScale, 1, (uint8_t)convDstRepStride, (uint8_t)convSrcRepStride);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TAddDeqReluCheck(
    const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1, const TileDataTmp& tmp)
{
    static_assert(std::is_same<typename TileDataSrc0::DType, int32_t>::value, "Fix: TADDDEQRELU src0 must be int32_t.");
    static_assert(std::is_same<typename TileDataSrc1::DType, int32_t>::value, "Fix: TADDDEQRELU src1 must be int32_t.");
    static_assert(std::is_same<typename TileDataDst::DType, half>::value, "Fix: TADDDEQRELU dst must be half.");
    static_assert(std::is_same<typename TileDataTmp::DType, int32_t>::value, "Fix: TADDDEQRELU tmp must be int32_t.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor && TileDataTmp::isRowMajor,
        "Fix: TADDDEQRELU only supports row major layout.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc0::Loc == TileType::Vec && TileDataSrc1::Loc == TileType::Vec &&
            TileDataTmp::Loc == TileType::Vec,
        "Fix: TADDDEQRELU tiles must live in TileType::Vec.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(validRow > 0 && validCol > 0, "Fix: TADDDEQRELU valid rows and columns must be greater than 0.");
    PTO_ASSERT(
        src0.GetValidRow() == validRow && src0.GetValidCol() == validCol,
        "Fix: TADDDEQRELU src0 valid shape mismatch with dst.");
    PTO_ASSERT(
        src1.GetValidRow() == validRow && src1.GetValidCol() == validCol,
        "Fix: TADDDEQRELU src1 valid shape mismatch with dst.");
    PTO_ASSERT(
        tmp.GetValidRow() >= validRow && tmp.GetValidCol() >= validCol,
        "Fix: TADDDEQRELU tmp must be at least as large as dst.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TADDDEQRELU_IMPL(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, float deqScale, TileDataTmp& tmp)
{
    TAddDeqReluCheck<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    constexpr unsigned DS = TileDataDst::RowStride;
    constexpr unsigned SS0 = TileDataSrc0::RowStride;
    constexpr unsigned SS1 = TileDataSrc1::RowStride;
    constexpr unsigned TS = TileDataTmp::RowStride;
    TAddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, SS0, SS1, TS, DS>(
        dst.data(), src0.data(), src1.data(), tmp.data(), deqScale, validRow, validCol);
}

} // namespace pto
#endif
