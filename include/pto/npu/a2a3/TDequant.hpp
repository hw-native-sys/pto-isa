/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDEQUANT_HPP
#define TDEQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/npu/a2a3/TBinSOp.hpp>
#include <pto/npu/a2a3/TSubS.hpp>
#include <pto/npu/a2a3/TMulS.hpp>
namespace pto {

template <typename DstDType, typename SrcDType>
PTO_INTERNAL void ConvertToDstDtype(__ubuf__ DstDType *dst, __ubuf__ SrcDType *src, uint8_t repeatNum,
                                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                    uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DstDType, float>::value && std::is_same<SrcDType, int16_t>::value) {
        vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
    } else if constexpr (std::is_same<DstDType, half>::value && std::is_same<SrcDType, int8_t>::value) {
        vconv_s82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);

    } else if constexpr (std::is_same<DstDType, float>::value && std::is_same<SrcDType, half>::value) {
        vconv_f162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        pipe_barrier(PIPE_V);
    }
}

template <typename DstDType, typename SrcDType, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void ConvertCompleteRepeats(__ubuf__ DstDType *dstPtr, __ubuf__ SrcDType *srcPtr, unsigned dstValidRows,
                                         unsigned numRepeatPerLine, unsigned elementsPerRepeat,
                                         unsigned dstRepeatStride, unsigned srcRepeatStride)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (uint32_t i = 0; i < dstValidRows; i++) {
            if (numLoop > 0) {
                for (uint32_t j = 0; j < numLoop; j++) {
                    ConvertToDstDtype<DstDType, SrcDType>(
                        dstPtr + i * dstRowStride + j * elementsPerRepeat * REPEAT_MAX,
                        srcPtr + i * srcRowStride + j * elementsPerRepeat * REPEAT_MAX, (uint8_t)REPEAT_MAX, 1, 1,
                        (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride);
                }
            }
            if (remainAfterLoop > 0) {
                ConvertToDstDtype<DstDType, SrcDType>(
                    dstPtr + i * dstRowStride + numLoop * elementsPerRepeat * REPEAT_MAX,
                    srcPtr + i * srcRowStride + numLoop * elementsPerRepeat * REPEAT_MAX, (uint8_t)remainAfterLoop, 1,
                    1, (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride);
            }
        }
    }
}

template <typename DstDType, typename SrcDType, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void ConvertRemainRegion(__ubuf__ DstDType *dstPtr, __ubuf__ SrcDType *srcPtr, unsigned dstValidRows,
                                      unsigned numRemainPerLine, unsigned dstNElemPerBlock, unsigned srcNElemPerBlock)
{
    if (numRemainPerLine > 0) {
        unsigned numLoop = dstValidRows / REPEAT_MAX;
        unsigned remainAfterLoop = dstValidRows % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                ConvertToDstDtype<DstDType, SrcDType>(
                    dstPtr + j * dstRowStride * REPEAT_MAX, srcPtr + j * srcRowStride * REPEAT_MAX, (uint8_t)REPEAT_MAX,
                    1, 1, (uint16_t)dstRowStride / dstNElemPerBlock, (uint16_t)srcRowStride / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop > 0) {
            ConvertToDstDtype<DstDType, SrcDType>(
                dstPtr + numLoop * dstRowStride * REPEAT_MAX, srcPtr + numLoop * srcRowStride * REPEAT_MAX,
                (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRowStride / dstNElemPerBlock,
                (uint16_t)srcRowStride / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename DstDType, typename SrcDType, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void ConvertForDequant(__ubuf__ DstDType *dstPtr, __ubuf__ SrcDType *srcPtr, unsigned dstValidRows,
                                    unsigned dstValidCols)
{
    uint64_t repeatWidth = static_cast<uint64_t>(max(sizeof(DstDType), sizeof(SrcDType)));
    unsigned dstRepeatStride = repeatWidth == sizeof(DstDType) ?
                                   BLOCK_MAX_PER_REPEAT :
                                   (BLOCK_MAX_PER_REPEAT / sizeof(SrcDType) * sizeof(DstDType));
    unsigned srcRepeatStride = repeatWidth == sizeof(SrcDType) ?
                                   BLOCK_MAX_PER_REPEAT :
                                   (BLOCK_MAX_PER_REPEAT / sizeof(DstDType) * sizeof(SrcDType));

    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = dstValidCols / elementsPerRepeat; // Complete repeats
    unsigned numRemainPerLine = dstValidCols % elementsPerRepeat; // Remainder elements
    constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(DstDType);
    constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(SrcDType);

    ConvertCompleteRepeats<DstDType, SrcDType, dstRowStride, srcRowStride>(
        dstPtr, srcPtr, dstValidRows, numRepeatPerLine, elementsPerRepeat, dstRepeatStride, srcRepeatStride);

    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    ConvertRemainRegion<DstDType, SrcDType, dstRowStride, srcRowStride>(dstPtr, srcPtr, dstValidRows, numRemainPerLine,
                                                                        dstNElemPerBlock, srcNElemPerBlock);
}

template <typename T, unsigned dstRowStride, unsigned scaleRowStride>
PTO_INTERNAL void ApplyScaleAndOffset(__ubuf__ T *dstPtr, __ubuf__ T *scalePtr, __ubuf__ T *offsetPtr,
                                      unsigned dstValidRows, unsigned dstValidCols)
{
    set_mask_count();
    set_vector_mask(0, dstValidCols);
    for (int i = 0; i < dstValidRows; ++i) {
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        T offsetValue = *(offsetPtr + i * scaleRowStride);
        T scaleValue = *(scalePtr + i * scaleRowStride);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vadds(dstNext, dstNext, -offsetValue, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(dstNext, dstNext, scaleValue, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, unsigned dstRowStride,
          unsigned srcRowStride, unsigned scaleRowStride>
__tf__ PTO_INTERNAL void TDequant(typename TileDataDst::TileDType __out__ dst,
                                  typename TileDataSrc::TileDType __in__ src,
                                  typename TileDataPara::TileDType __in__ scale,
                                  typename TileDataPara::TileDType __in__ offset, unsigned dstValidRows,
                                  unsigned dstValidCols)
{
    __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src);

    if constexpr (std::is_same_v<typename TileDataDst::DType, float> &&
                  std::is_same_v<typename TileDataSrc::DType, int16_t>) {
        ConvertForDequant<float, int16_t, dstRowStride, srcRowStride>(dstPtr, srcPtr, dstValidRows, dstValidCols);
    } else if constexpr (std::is_same_v<typename TileDataDst::DType, float> &&
                         std::is_same_v<typename TileDataSrc::DType, int8_t>) {
        __ubuf__ half *tempDstHalfPtr = (__ubuf__ half *)(dstPtr) + dstRowStride;
        ConvertForDequant<half, int8_t, dstRowStride * 2, srcRowStride>(tempDstHalfPtr, srcPtr, dstValidRows,
                                                                        dstValidCols);
        ConvertForDequant<float, half, dstRowStride, dstRowStride * 2>(dstPtr, tempDstHalfPtr, dstValidRows,
                                                                       dstValidCols);
    }

    using T = typename TileDataPara::DType;
    __ubuf__ T *scalePtr = (__ubuf__ T *)__cce_get_tile_ptr(scale);
    __ubuf__ T *offsetPtr = (__ubuf__ T *)__cce_get_tile_ptr(offset);

    ApplyScaleAndOffset<T, dstRowStride, TileDataPara::RowStride>(dstPtr, scalePtr, offsetPtr, dstValidRows,
                                                                  dstValidCols);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDequantCheck(const TileDataDst &dst, const TileDataSrc &src, const TileDataPara &scale,
                                const TileDataPara &offset)
{
    static_assert(TileDataDst::RowStride == TileDataSrc::RowStride,
                  "Fix: TDEQUANT src and dst tile must have the same rowStride.");
    static_assert(std::is_same<typename TileDataDst::DType, float>::value ||
                      std::is_same<typename TileDataDst::DType, float32_t>::value,
                  "Fix: TDEQUANT dst tile currently supports float data type.");
    static_assert(std::is_same<typename TileDataPara::DType, float>::value ||
                      std::is_same<typename TileDataPara::DType, float32_t>::value,
                  "Fix: TDEQUANT parameter tile scale and offset currently support float data type.");
    static_assert(std::is_same<typename TileDataSrc::DType, int8_t>::value ||
                      std::is_same<typename TileDataSrc::DType, int16_t>::value,
                  "Fix: TDEQUANT source tile currently supports int8_t and int16_t data types.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
                  "Fix: TDEQUANT src and dst tile only support row major layout.");
    unsigned dstValidRows = dst.GetValidRow();
    unsigned dstValidCols = dst.GetValidCol();
    PTO_ASSERT(src.GetValidRow() == dstValidRows && src.GetValidCol() == dstValidCols,
               "Fix: TDEQUANT source tile valid shape mismatch with dst tile valid shape.");
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{
    TDequantCheck<TileDataDst, TileDataSrc, TileDataPara>(dst, src, scale, offset);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    constexpr unsigned scaleRowStride = TileDataPara::RowStride;
    TDequant<TileDataDst, TileDataSrc, TileDataPara, dstRowStride, srcRowStride, scaleRowStride>(
        dst.data(), src.data(), scale.data(), offset.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
