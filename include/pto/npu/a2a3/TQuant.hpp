/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TQUANT_HPP
#define TQUANT_HPP

#include "pto/npu/a2a3/TRowExpandMul.hpp"
#include "pto/npu/a2a3/TRowExpandAdd.hpp"
#include "pto/npu/a2a3/TCvt.hpp"
#include "pto/npu/a2a3/TAssign.hpp"

namespace pto {

enum class QuantType
{
    INT8_SYM,
    INT8_ASYM
};

#ifdef __PTO_AUTO__

template <typename TileDataOut, typename TileDataSrc>
PTO_INTERNAL void TQuant_TCvt_Impl(__ubuf__ typename TileDataOut::DType *dstPtr,
                                   __ubuf__ typename TileDataSrc::DType *srcPtr, unsigned srcValidRow,
                                   unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    // TCVT
    static_assert(std::is_same<typename TileDataSrc::DType, float32_t>::value, "Fix: Input has to be float 32");
    __ubuf__ typename TileDataSrc::DType *srcPtrBase = srcPtr;

    uint64_t repeatWidth = static_cast<uint64_t>(max(sizeof(half), sizeof(float)));
    unsigned dstRepeatStride =
        repeatWidth == sizeof(half) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / sizeof(float) * sizeof(half));
    unsigned srcRepeatStride =
        repeatWidth == sizeof(float) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / sizeof(half) * sizeof(float));
    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = srcValidCol / elementsPerRepeat;
    unsigned numRemainPerLine = srcValidCol % elementsPerRepeat;
    constexpr unsigned SS = TileDataSrc::RowStride;
    constexpr unsigned DS = TileDataOut::RowStride;

    uint64_t originalCtrl = get_ctrl();
    bool originalSatMode = (originalCtrl & (1ULL << SAT_MODE_BIT)) == 0;

    // Apply saturation mode
    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT)); // Turn on saturation (default)

    __ubuf__ half *srcPtrHalf = (__ubuf__ half *)srcPtrBase;
    using TileDataCvtF16 = Tile<TileType::Vec, half, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;

    constexpr unsigned srcHalfNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(half);
    constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(float);
    constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataOut::DType);

    // Process main aligned region with complete repeat units
    if (numRepeatPerLine > 0) {
        TCvtHead<TileDataCvtF16, TileDataSrc, SS, SS>(srcPtrHalf, srcPtr, RoundMode::CAST_RINT, numRepeatPerLine,
                                                      srcValidRow, elementsPerRepeat, dstRepeatStride, srcRepeatStride);
    }
    // Advance pointers to unaligned remainder region
    srcPtrHalf += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    // Process remainder region with partial repeats (requires vector masking)
    if (numRemainPerLine > 0) {
        unsigned numLoop = srcValidRow / REPEAT_MAX;
        unsigned remainAfterLoop = srcValidRow % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataCvtF16, TileDataSrc>(
                    srcPtrHalf + j * SS * REPEAT_MAX, srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX,
                    RoundMode::CAST_RINT, 1, 1, (uint16_t)SS / srcHalfNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataCvtF16, TileDataSrc>(
                srcPtrHalf + numLoop * SS * REPEAT_MAX, srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop,
                RoundMode::CAST_RINT, 1, 1, (uint16_t)SS / srcHalfNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }

    srcPtrHalf = (__ubuf__ half *)srcPtrBase;
    repeatWidth = static_cast<uint64_t>(max(sizeof(typename TileDataOut::DType), sizeof(half)));
    dstRepeatStride = repeatWidth == sizeof(typename TileDataOut::DType) ?
                          BLOCK_MAX_PER_REPEAT :
                          (BLOCK_MAX_PER_REPEAT / sizeof(half) * sizeof(typename TileDataOut::DType));
    srcRepeatStride = repeatWidth == sizeof(half) ?
                          BLOCK_MAX_PER_REPEAT :
                          (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataOut::DType) * sizeof(half));
    elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    numRepeatPerLine = dstValidCol / elementsPerRepeat;
    numRemainPerLine = dstValidCol % elementsPerRepeat;

    // Process main aligned region with complete repeat units
    if (numRepeatPerLine > 0) {
        TCvtHead<TileDataOut, TileDataCvtF16, SS, DS>(dstPtr, srcPtrHalf, RoundMode::CAST_RINT, numRepeatPerLine,
                                                      dstValidRow, elementsPerRepeat, dstRepeatStride, srcRepeatStride);
    }
    // Advance pointers to unaligned remainder region
    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtrHalf += numRepeatPerLine * elementsPerRepeat;

    // Process remainder region with partial repeats (requires vector masking)
    if (numRemainPerLine > 0) {
        unsigned numLoop = dstValidRow / REPEAT_MAX;
        unsigned remainAfterLoop = dstValidRow % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataOut, TileDataCvtF16>(
                    dstPtr + j * DS * REPEAT_MAX, srcPtrHalf + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX,
                    RoundMode::CAST_RINT, 1, 1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcHalfNElemPerBlock);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataOut, TileDataCvtF16>(
                dstPtr + numLoop * DS * REPEAT_MAX, srcPtrHalf + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop,
                RoundMode::CAST_RINT, 1, 1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcHalfNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }

    // Restore original saturation mode to avoid affecting subsequent instructions
    if (originalSatMode) {
        set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
    } else {
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
    }
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ AICORE void TQuant_Sym(typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                              typename TileDataPara::TileDType __in__ scale, unsigned srcValidRow, unsigned srcValidCol,
                              unsigned dstValidRow, unsigned dstValidCol)
{
    __ubuf__ int8_t *dstPtr = (__ubuf__ int8_t *)__cce_get_tile_ptr(dst);
    __ubuf__ float *srcPtr = (__ubuf__ float *)__cce_get_tile_ptr(src);
    __ubuf__ float *scalePtr = (__ubuf__ float *)__cce_get_tile_ptr(scale);

    // TROWEXPANDMUL
    if constexpr (TileDataPara::isRowMajor) {
        TRowExpandBinaryInstr32B<RowExpandMulOp<float>, float, TileDataSrc::Rows, TileDataSrc::RowStride,
                                 TileDataSrc::RowStride, TileDataPara::RowStride>(srcPtr, srcPtr, scalePtr, srcValidRow,
                                                                                  srcValidCol);
    } else {
        __ubuf__ uint32_t *src1Ptr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(scale);
        __ubuf__ float *tmpPtr = (__ubuf__ float *)(TMP_UB_OFFSET);        // 8KB tmpbuf address
        __ubuf__ uint32_t *tmpPtr_ = (__ubuf__ uint32_t *)(TMP_UB_OFFSET); // 8KB tmpbuf address
        TRowExpandBinaryInstr<RowExpandMulOp<float>, float, uint32_t, TileDataSrc::Rows, TileDataSrc::RowStride,
                              TileDataSrc::RowStride>(srcPtr, srcPtr, src1Ptr, tmpPtr, tmpPtr_, srcValidRow,
                                                      srcValidCol);
    }

    // TCVT
    TQuant_TCvt_Impl<TileDataOut, TileDataSrc>(dstPtr, srcPtr, srcValidRow, srcValidCol, dstValidRow, dstValidCol);
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara> // uint8_t float
__tf__ AICORE void TQuant_Asym(typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                               typename TileDataPara::TileDType __in__ scale,
                               typename TileDataPara::TileDType __in__ offset, unsigned srcValidRow,
                               unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    __ubuf__ uint8_t *dstPtr = (__ubuf__ uint8_t *)__cce_get_tile_ptr(dst);
    __ubuf__ float *srcPtr = (__ubuf__ float *)__cce_get_tile_ptr(src);
    __ubuf__ float *scalePtr = (__ubuf__ float *)__cce_get_tile_ptr(scale);
    __ubuf__ float *offsetPtr = (__ubuf__ float *)__cce_get_tile_ptr(offset);

    // TROWEXPANDMUL
    if constexpr (TileDataPara::isRowMajor) {
        TRowExpandBinaryInstr32B<RowExpandMulOp<float>, float, TileDataSrc::Rows, TileDataSrc::RowStride,
                                 TileDataSrc::RowStride, TileDataPara::RowStride>(srcPtr, srcPtr, scalePtr, srcValidRow,
                                                                                  srcValidCol);
    } else {
        __ubuf__ uint32_t *src1Ptr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(scale);
        __ubuf__ float *tmpPtr = (__ubuf__ float *)(TMP_UB_OFFSET);        // 8KB tmpbuf address
        __ubuf__ uint32_t *tmpPtr_ = (__ubuf__ uint32_t *)(TMP_UB_OFFSET); // 8KB tmpbuf address
        TRowExpandBinaryInstr<RowExpandMulOp<float>, float, uint32_t, TileDataSrc::Rows, TileDataSrc::RowStride,
                              TileDataSrc::RowStride>(srcPtr, srcPtr, src1Ptr, tmpPtr, tmpPtr_, srcValidRow,
                                                      srcValidCol);
    }

    // TROWEXPANDADD
    if constexpr (TileDataPara::isRowMajor) {
        TRowExpandBinaryInstr32B<RowExpandAddOp<float>, float, TileDataSrc::Rows, TileDataSrc::RowStride,
                                 TileDataSrc::RowStride, TileDataPara::RowStride>(srcPtr, srcPtr, offsetPtr,
                                                                                  srcValidRow, srcValidCol);
    } else {
        __ubuf__ uint32_t *src1Ptr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(offset);
        __ubuf__ float *tmpPtr = (__ubuf__ float *)(TMP_UB_OFFSET);        // 8KB tmpbuf address
        __ubuf__ uint32_t *tmpPtr_ = (__ubuf__ uint32_t *)(TMP_UB_OFFSET); // 8KB tmpbuf address
        TRowExpandBinaryInstr<RowExpandAddOp<float>, float, uint32_t, TileDataSrc::Rows, TileDataSrc::RowStride,
                              TileDataSrc::RowStride>(srcPtr, srcPtr, src1Ptr, tmpPtr, tmpPtr_, srcValidRow,
                                                      srcValidCol);
    }

    // TCVT
    TQuant_TCvt_Impl<TileDataOut, TileDataSrc>(dstPtr, srcPtr, srcValidRow, srcValidCol, dstValidRow, dstValidCol);
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataOut::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");
    if constexpr (quant_type == QuantType::INT8_SYM) {
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
    }
    using TileDataCvtF16 = Tile<TileType::Vec, half, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;

    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    if constexpr (quant_type == QuantType::INT8_ASYM) {
        TQuant_Asym<quant_type, TileDataOut, TileDataSrc, TileDataPara>(
            dst.data(), src.data(), scale.data(), offset->data(), srcValidRow, srcValidCol, dstValidRow, dstValidCol);
    } else {
        TQuant_Sym<quant_type, TileDataOut, TileDataSrc, TileDataPara>(
            dst.data(), src.data(), scale.data(), srcValidRow, srcValidCol, dstValidRow, dstValidCol);
    }
}

#else // ifdef __PTO_AUTO__

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataOut::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");
    if constexpr (quant_type == QuantType::INT8_SYM) {
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
    }
    using TileDataCvtF16 = Tile<TileType::Vec, half, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;
    using TileDataCvtS32 =
        Tile<TileType::Vec, int32_t, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;

    TROWEXPANDMUL_IMPL(src, src, scale);
    if constexpr (quant_type == QuantType::INT8_ASYM) {
        TROWEXPANDADD_IMPL(src, src, *offset);
    }

    TileDataCvtF16 src_f16(src.GetValidRow(), src.GetValidCol());
    TileDataCvtS32 src_s32(src.GetValidRow(), src.GetValidCol());
    TASSIGN_IMPL(src_f16, reinterpret_cast<uintptr_t>(src.data()));
    TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(src.data()));
    TCVT_IMPL(src_s32, src, RoundMode::CAST_RINT);     // fp32->s32
    TCVT_IMPL(src_f16, src_s32, RoundMode::CAST_RINT); // s32->fp16 (exact since values are now integers)
    TCVT_IMPL(dst, src_f16, RoundMode::CAST_RINT, SaturationMode::ON);
}
#endif
} // namespace pto
#endif // TQUANT_HPP