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
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU A2/A3 Architecture
 *
 * FILE ORGANIZATION (for easy navigation):
 * =======================================
 *
 * SUPPORTED CONVERSIONS (quick lookup):
 * ====================================
 * FP32:  -> FP16, FP32 (rounding only), BF16, I16, I32, I64
 * FP16:  -> FP32, I32, I16, I8, U8, S4 (int4b_t)
 * BF16:  -> FP32, I32
 * I16:   -> FP16, FP32
 * I32:   -> FP32, I16, I64, FP16 (deq)
 * I64:   -> FP32, I32
 * U8:    -> FP16
 * I8:    -> FP16
 * S4:    -> FP16
 *
 * 1. GenCastCall* helpers (lines ~20-360)
 *    - fp32 -> fp16/fp32/int64/int32/int16/bf16
 *    - fp16 -> int32/int16/int8/uint8
 *    - bf16 -> int32
 *    - int16/int32/int64 -> fp16/fp32
 *
 * 2. GenCastCallSpecialCases (lines ~360-450)
 *    - half<->fp32, bf16->fp32, int8/uint8->half
 *    - int64<->int32, int32->int16, int32->half (deq)
 *
 * 3. GenCastCall Dispatcher (lines ~450-530)
 *    - Compile-time type routing to the correct GenCastCall* helper
 *    - Overload with tmpPtr forwards scratch buffer to NonSatTorch paths
 *
 * 4. TCvtHead (lines ~540-600)
 *    - Processes aligned repeat blocks for main data region
 *    - Overload with tmpPtr passes scratch buffer through to GenCastCall
 *
 * 5. TCvt Kernel (lines ~610-700)
 *    - Handles aligned region and remainder with vector masking
 *    - Overload with TmpTileData uses user-supplied tile instead of TMP_UB_OFFSET
 *
 * 6. TCVT_IMPL (lines ~710-end)
 *    - High-level entry point computing repeat configuration
 *    - Overloads with TmpTileData mirror TSort32's with-tmp interface
 *
 * QUICK FIND: Search for the conversion function name (e.g., "GenCastCallFp32ToFp16")
 * or the dispatcher "GenCastCall" to locate the relevant section.
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include "common.hpp"

namespace pto {
// ============================================================================
// Type Conversion Functions
// ============================================================================
// Specialized data type conversions with support for multiple rounding modes:
// RINT, ROUND, FLOOR, CEIL, TRUNC, ODD, NONE
// ============================================================================
inline namespace TCvtInternel {
// CTRL[59] controls saturation mode for FP to INT conversions:
// - 0 (ON):  Clamp to datatype range (e.g., 300.0f -> int8 = 127)
// - 1 (OFF): Truncate via bit masking (e.g., 300.0f -> int8 = 44 from 300 & 0xFF)
constexpr const int SAT_MODE_BIT = 59;

// Temporary buffer size for non-saturation conversions (REPEAT_MAX * 256 bytes)
constexpr const size_t FP16_INT8_TEMP_BUFFER_SIZE = REPEAT_MAX * 256;
} // namespace TCvtInternel

// PyTorch alignment for edge cases (inf, -inf, nan, overflow)
// 1 = PyTorch-compatible (uses NonSatTorch), 0 = standard (faster)
#define EDGE_CASE_ALIGN_ENABLE 1

// FP32 -> FP16 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToFp16(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ODD:
            vconv_f322f16o(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP32 -> FP32 conversion with rounding (normalization)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP32 -> INT64 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt64(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s64r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s64a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s64f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s64c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s64z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s64z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP32 -> INT32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP32 -> INT16 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP32 -> INT16 conversion (PyTorch-compatible for inf/-inf)
// Two-step: fp32 -> int32 -> int16
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt16_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                     __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                     RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                     uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                     __ubuf__ int32_t *tempInt32Buf)
{
    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT)); // Turn on saturation for int32 conversion
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s32r(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s32a(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s32f(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s32c(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        default:
            vconv_f322s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
    }

    pipe_barrier(PIPE_V);
    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
    vconv_s322s16(dst, tempInt32Buf, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

// FP32 -> BF16 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToBf16(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp32 to bf16 - Convert floating point to bfloat16 format
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322bf16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322bf16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322bf16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322bf16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP16 -> INT32 conversion
PTO_INTERNAL void GenCastCallFp16ToInt32ByRoundMode(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeatNum,
                                                    RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                    uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
        default:
            vconv_f162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    GenCastCallFp16ToInt32ByRoundMode((__ubuf__ int32_t *)dst, (__ubuf__ half *)src, repeatNum, mode, dstBlockStride,
                                      srcBlockStride, dstRepeatStride, srcRepeatStride);
}

// FP16 -> INT16 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP16 -> INT16 conversion (PyTorch-compatible for inf/-inf): fp16 -> int32 -> int16
//
// Per-repeat hardware element capacity:
//   int32 ops (vconv_f162s32*, vconv_s322s16) : 64
//   fp16/int16 native ops                     : 128
//
// Strategy: iterate row-by-row, splitting each row into sub-chunks of <=64 elements so every
// instruction stays within the int32 capacity.  Each sub-chunk sets its own continuous mask,
// runs one fp16->int32 repeat and one int32->int16 repeat.
//
// Params:
//   repeatNum        logical repeats (rows).
//   src/dstRepeatStride  block-stride between rows (in src fp16 / dst int16 blocks).
//   numElemsPerRow   valid elements per row.  0 means "head default" (= 128);
//                    tail callers pass numRemainPerLine in [1, 128].
//
// Caller contexts:
//   HEAD (TCvtHead): numElemsPerRow=0, caller mask full.
//   TAIL (TCvtTail): numElemsPerRow=numRemainPerLine; caller's mask is overridden per sub-chunk.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt16_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                     __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                     RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                     uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                     __ubuf__ int32_t *tempInt32Buf, uint16_t numElemsPerRow = 0)
{
    constexpr uint16_t fp16ElemsPerBlock = BLOCK_BYTE_SIZE / sizeof(half);     // 16
    constexpr uint16_t int16ElemsPerBlock = BLOCK_BYTE_SIZE / sizeof(int16_t); // 16
    constexpr uint16_t maxChunkElems = 64;                                     // int32 hw cap per repeat

    // elemsPerRow: head path leaves numElemsPerRow=0 → full 128; tail forwards actual count.
    const uint16_t elemsPerRow = (numElemsPerRow == 0) ? static_cast<uint16_t>(128) : numElemsPerRow;
    const uint16_t numSubChunks = (elemsPerRow + maxChunkElems - 1) / maxChunkElems; // 1 or 2
    const uint16_t lastChunkMask = elemsPerRow - (numSubChunks - 1) * maxChunkElems; // [1, 64]

    for (uint16_t r = 0; r < repeatNum; r++) {
        __ubuf__ half *rowSrc = src + static_cast<uint32_t>(r) * srcRepeatStride * fp16ElemsPerBlock;
        __ubuf__ int16_t *rowDst = dst + static_cast<uint32_t>(r) * dstRepeatStride * int16ElemsPerBlock;

        for (uint16_t c = 0; c < numSubChunks; c++) {
            __ubuf__ int16_t *chunkDst = rowDst + static_cast<uint32_t>(c) * maxChunkElems;
            __ubuf__ half *chunkSrc = rowSrc + static_cast<uint32_t>(c) * maxChunkElems;
            const uint16_t chunkMask = (c + 1 == numSubChunks) ? lastChunkMask : maxChunkElems;

            SetContinuousMask(chunkMask);

            // Step 1: fp16 -> int32 with saturation (clamps inf/overflow into int32 range).
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
            GenCastCallFp16ToInt32ByRoundMode(tempInt32Buf, chunkSrc, 1, mode, srcBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 2: int32 -> int16 (non-saturating wrap-around — PyTorch low-16-bit semantics).
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
            vconv_s322s16(chunkDst, tempInt32Buf, 1, dstBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }

    // Leave mask in a predictable (full) state so downstream vector ops are unaffected.
    set_vector_mask(-1, -1);
}

// FP16 -> INT8 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt8(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp16 to int8 - Convert half-precision float to 8-bit signed integer
    // Note: Saturation mode is now controlled globally by TCvt kernel
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s8a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s8f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s8c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP16 -> INT8 conversion, PyTorch-compatible for inf/-inf/overflow (wrap-around semantics).
// Pipeline: fp16 --sat--> int32 --narrow--> int16 --AND 0xFF--> int16 --> fp16 --> int8.
// (vand only accepts short* on this arch, hence the int32->int16 narrow before masking.)
//
// Per-repeat hardware element capacity:
//   int32 ops (vconv_f162s32*, vconv_s322s16) : 64
//   fp16/int16/int8 ops (vand, vconv_s162f16, vconv_f162s8z) : 128
//
// Strategy: iterate row-by-row, and within each row split into sub-chunks of <=64 elements so
// every instruction below stays within the int32 capacity.  All vector ops use repeat=1 with
// pipe_barrier between steps — correctness-first, immune to multi-repeat in-place hazards.
//
// Params:
//   repeatNum        logical repeats (rows).
//   src/dstRepeatStride  block-stride between rows (in src fp16 / dst int8 blocks).
//   numElemsPerRow   valid elements per row.  0 means "head default" (= 128);
//                    tail callers pass numRemainPerLine in [1, 128].
//
// Caller contexts:
//   HEAD (TCvtHead): numElemsPerRow=0, caller mask full.
//   TAIL (TCvtTail): numElemsPerRow=numRemainPerLine; caller's mask is overridden per sub-chunk.
//
// Temp buffer layout (256 B, reused every sub-chunk):
//   bytes   [0..127]  : int32 (lo 32) then int16 after in-place narrow (same 64 lanes)
//   bytes   [128..255]: mask of 0x00FF, then fp16 step-5 output
// (Upper 128 bytes of the int32 region are freed by the narrow and repurposed as scratch.)
//
// NOTE: src must NOT be reused as scratch — the saturation test kernel invokes TCVT three times
// on the same srcTile (ON/OFF/default), so clobbering src would corrupt later invocations.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt8_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                    __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                    RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                    uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                    __ubuf__ int32_t *tempInt32Buf, uint16_t numElemsPerRow)
{
    constexpr uint16_t fp16ElemsPerBlock = BLOCK_BYTE_SIZE / sizeof(half);   // 16
    constexpr uint16_t int8ElemsPerBlock = BLOCK_BYTE_SIZE / sizeof(int8_t); // 32
    constexpr uint16_t maxChunkElems = 64;                                   // int32 hw cap per repeat

    // Temp aliases over the shared 256-byte tempInt32Buf.
    //   tempAndBuf   : bytes [0..127]   -- int16 view (overlays the int32 lanes after narrow).
    //   tempMaskBuf  : bytes [128..255] -- scratch for the 0x00FF mask and step-5 fp16 output.
    __ubuf__ int16_t *tempAndBuf = (__ubuf__ int16_t *)tempInt32Buf;
    __ubuf__ int16_t *tempMaskBuf = tempAndBuf + maxChunkElems;
    __ubuf__ half *tempFp16Out = (__ubuf__ half *)tempMaskBuf;

    // elemsPerRow: head path leaves numElemsPerRow=0 → full 128; tail forwards actual count.
    const uint16_t elemsPerRow = (numElemsPerRow == 0) ? static_cast<uint16_t>(128) : numElemsPerRow;
    const uint16_t numSubChunks = (elemsPerRow + maxChunkElems - 1) / maxChunkElems; // 1 or 2
    const uint16_t lastChunkMask = elemsPerRow - (numSubChunks - 1) * maxChunkElems; // [1, 64]

    for (uint16_t r = 0; r < repeatNum; r++) {
        __ubuf__ half *rowSrc = src + static_cast<uint32_t>(r) * srcRepeatStride * fp16ElemsPerBlock;
        __ubuf__ int8_t *rowDst = dst + static_cast<uint32_t>(r) * dstRepeatStride * int8ElemsPerBlock;

        for (uint16_t c = 0; c < numSubChunks; c++) {
            __ubuf__ half *chunkSrc = rowSrc + static_cast<uint32_t>(c) * maxChunkElems;
            __ubuf__ int8_t *chunkDst = rowDst + static_cast<uint32_t>(c) * maxChunkElems;
            const uint16_t chunkMask = (c + 1 == numSubChunks) ? lastChunkMask : maxChunkElems;

            SetContinuousMask(chunkMask);

            // Step 1: fp16 -> int32 with saturation (clamps inf/overflow into int32 range).
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
            GenCastCallFp16ToInt32ByRoundMode(tempInt32Buf, chunkSrc, 1, mode, srcBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);

            // Switch to non-saturating (wrap-around) mode for the remaining narrowing stages —
            // this is what produces PyTorch's low-8-bit behaviour on overflow.
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));

            // Step 2: int32 -> int16, in-place over the same 64 lanes.
            vconv_s322s16(tempAndBuf, tempInt32Buf, 1, srcBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 3: broadcast 0x00FF mask into the upper scratch region.
            vector_dup(tempMaskBuf, static_cast<int16_t>(255), 1, srcBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 4: keep low 8 bits (int16 & 0xFF) in place.
            vand(tempAndBuf, tempAndBuf, tempMaskBuf, 1, srcBlockStride, srcBlockStride, srcBlockStride, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 5: int16 -> fp16 into the mask's old slot (the mask has been consumed).
            vconv_s162f16(tempFp16Out, tempAndBuf, 1, srcBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 6: fp16 -> int8 into the caller's destination for this sub-chunk.
            vconv_f162s8z(chunkDst, tempFp16Out, 1, dstBlockStride, srcBlockStride, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }

    // Leave mask in a predictable (full) state so downstream vector ops are unaffected.
    set_vector_mask(-1, -1);
}

// FP16 -> UINT8 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToUint8(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162u8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162u8a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162u8f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162u8c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162u8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162u8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// FP16 -> S4 (int4b_t) conversion
// int4 is a packed type: 2 elements per byte. The vconv_f162s4* intrinsics accept void* for the
// int4 destination. Each repeat processes 128 fp16 elements into 64 bytes of packed int4.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToS4(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeatNum, RoundMode mode,
                                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                      uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s4r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s4a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s4f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s4c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s4z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s4(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// S4 (int4b_t) -> FP16 conversion
// No rounding mode variants — only a single intrinsic (vconv_s42f16).
// Note: vconv_s42f16 uses uint8_t for repeat strides.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallS4ToFp16(__ubuf__ half *dst, __ubuf__ void *src, uint8_t repeatNum, RoundMode mode,
                                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                      uint16_t srcRepeatStride)
{
    vconv_s42f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, static_cast<uint8_t>(dstRepeatStride),
                 static_cast<uint8_t>(srcRepeatStride));
}

// BF16 -> INT32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallBf16ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_bf162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_bf162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_bf162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_bf162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// INT16 -> FP16 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt16ToFp16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s162f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s162f16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s162f16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s162f16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s162f16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s162f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// INT32 -> FP32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt32ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s322f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s322f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s322f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s322f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s322f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// INT64 -> FP32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt64ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s642f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s642f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s642f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s642f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s642f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s642f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Special case conversions: half<->fp32, bf16<->fp32, int/uint 8<->half,
// int32<->int64, int32<->int16, int32->half (deq)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallSpecialCases(__ubuf__ typename TileDataD::DType *dst,
                                          __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                          uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                  std::is_same<typename TileDataS::DType, half>::value) { // half to fp32
        vconv_f162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, bfloat16_t>::value) { // bfloat16 to float
        vconv_bf162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, uint8_t>::value) { // uint8 to half
        vconv_u82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int8_t>::value) { // int8 to half
        vconv_s82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int16_t>::value) { // int16 to float32
        vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, int64_t>::value) { // int64 to int32
        vconv_s642s32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to int64
        vconv_s322s64(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to int16
        vconv_s322s16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to half
        set_deqscale(static_cast<half>(1.0));
        pipe_barrier(PIPE_V);
        vconv_deq(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

// ============================================================================
// Type Conversion Dispatcher
// ============================================================================
// The trailing `numElemsPerRow` parameter is only consumed by NonSatTorch paths that need
// explicit sub-chunk splitting (currently fp16 -> int8).  TCvtTail forwards its
// numRemainPerLine as this argument so the NonSatTorch path can split rows wider than the
// int32 hardware capacity (64 elements) into multiple sub-chunks.  TCvtHead leaves it at the
// default 0 which the NonSatTorch path interprets as "full logical repeat" (= 128 elements).
template <typename TileDataD, typename TileDataS>
AICORE void GenCastCall(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
                        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                        uint16_t dstRepeatStride, uint16_t srcRepeatStride, uint16_t /*numElemsPerRow*/ = 0)
{
    if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                  std::is_same<typename TileDataS::DType, float>::value) {
        GenCastCallFp32ToFp16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to fp32
        GenCastCallFp32ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int64
        GenCastCallFp32ToInt64<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int32
        GenCastCallFp32ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int16
        GenCastCallFp32ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, bfloat16_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to bf16
        GenCastCallFp32ToBf16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int32
        GenCastCallFp16ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int16
        GenCastCallFp16ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int8_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int8
        GenCastCallFp16ToInt8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, uint8_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to uint8
        GenCastCallFp16ToUint8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int4b_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int4
        GenCastCallFp16ToS4<TileDataD, TileDataS>((__ubuf__ void *)dst, src, repeatNum, mode, dstBlockStride,
                                                  srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int4b_t>::value) { // int4 to half
        GenCastCallS4ToFp16<TileDataD, TileDataS>(dst, (__ubuf__ void *)src, repeatNum, mode, dstBlockStride,
                                                  srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, bfloat16_t>::value) { // bfloat16 to int32
        GenCastCallBf16ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int16_t>::value) { // int16 to half
        GenCastCallInt16ToFp16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to float
        GenCastCallInt32ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int64_t>::value) { // int64 to float
        GenCastCallInt64ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else {
        GenCastCallSpecialCases<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                      dstRepeatStride, srcRepeatStride);
    }
}

// GenCastCall overload with explicit temporary buffer pointer.
// Mirrors the no-tmp GenCastCall but forwards tmpPtr to NonSatTorch paths
// instead of using the fixed TMP_UB_OFFSET global scratch area.
//
// The trailing `numElemsPerRow` parameter (default 0) is forwarded to NonSatTorch paths that
// need explicit sub-chunk splitting.  TCvtTail passes numRemainPerLine here; TCvtHead leaves
// it at 0 (interpreted as "full logical repeat" by the fp16 -> int8 NonSatTorch path).
template <typename TileDataD, typename TileDataS>
AICORE void GenCastCall(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
                        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                        uint16_t dstRepeatStride, uint16_t srcRepeatStride, __ubuf__ int32_t *tmpPtr,
                        uint16_t numElemsPerRow = 0)
{
    if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                  std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int16
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            GenCastCallFp32ToInt16_NonSatTorch<TileDataD, TileDataS>(
                dst, src, repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride, tmpPtr);
        } else {
            GenCastCallFp32ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                         dstRepeatStride, srcRepeatStride);
        }
#else
        GenCastCallFp32ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
#endif
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int16
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            GenCastCallFp16ToInt16_NonSatTorch<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride,
                                                                     srcBlockStride, dstRepeatStride, srcRepeatStride,
                                                                     tmpPtr, numElemsPerRow);
        } else {
            GenCastCallFp16ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                         dstRepeatStride, srcRepeatStride);
        }
#else
        GenCastCallFp16ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
#endif
    } else if constexpr (std::is_same<typename TileDataD::DType, int8_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int8
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            GenCastCallFp16ToInt8_NonSatTorch<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride,
                                                                    srcBlockStride, dstRepeatStride, srcRepeatStride,
                                                                    tmpPtr, numElemsPerRow);
        } else {
            GenCastCallFp16ToInt8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                        dstRepeatStride, srcRepeatStride);
        }
#else
        GenCastCallFp16ToInt8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
#endif
    } else {
        GenCastCall<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride,
                                          srcRepeatStride);
    }
}

// ============================================================================
// Tile Conversion Helper: Process Main Data Block
// ============================================================================
// TCvtHead processes the primary aligned portion of data in complete repeat units.
// This handles data that fits evenly into repeat boundaries.
//
// @param dstPtr: Destination buffer pointer
// @param srcPtr: Source buffer pointer
// @param mode: Rounding mode for type conversions
// @param numRepeatPerLine: Number of complete repeats per line
// @param validRow: Number of valid rows to process
// @param elementsPerRepeat: Number of elements per repeat unit
// @param dstRepeatStride: Stride between repeats in destination
// @param srcRepeatStride: Stride between repeats in source
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
PTO_INST void TCvtHead(__ubuf__ typename TileDataD::DType *dstPtr, __ubuf__ typename TileDataS::DType *srcPtr,
                       RoundMode mode, unsigned numRepeatPerLine, unsigned validRow, unsigned elementsPerRepeat,
                       unsigned dstRepeatStride, unsigned srcRepeatStride)
{
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
    for (uint32_t i = 0; i < validRow; i++) {
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                                  srcPtr + i * SS + j * elementsPerRepeat * REPEAT_MAX,
                                                  (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)dstRepeatStride,
                                                  (uint16_t)srcRepeatStride);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              srcPtr + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)dstRepeatStride,
                                              (uint16_t)srcRepeatStride);
        }
    }
}

// TCvtHead overload with explicit temporary buffer pointer (forwarded to GenCastCall with tmp).
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
PTO_INST void TCvtHead(__ubuf__ typename TileDataD::DType *dstPtr, __ubuf__ typename TileDataS::DType *srcPtr,
                       RoundMode mode, unsigned numRepeatPerLine, unsigned validRow, unsigned elementsPerRepeat,
                       unsigned dstRepeatStride, unsigned srcRepeatStride, __ubuf__ int32_t *tmpPtr)
{
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
    for (uint32_t i = 0; i < validRow; i++) {
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                                  srcPtr + i * SS + j * elementsPerRepeat * REPEAT_MAX,
                                                  (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)dstRepeatStride,
                                                  (uint16_t)srcRepeatStride, tmpPtr);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              srcPtr + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)dstRepeatStride,
                                              (uint16_t)srcRepeatStride, tmpPtr);
        }
    }
}

// ============================================================================
// Saturation Mode Helpers
// ============================================================================

template <bool NeedSetCtrl>
PTO_INST bool ApplySatMode(SaturationMode satMode)
{
    if constexpr (NeedSetCtrl) {
        uint64_t originalCtrl = get_ctrl();
        bool originalSatMode = (originalCtrl & (1ULL << SAT_MODE_BIT)) == 0;
        if (satMode == SaturationMode::OFF) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
        }
        return originalSatMode;
    }
    return false;
}

template <bool NeedSetCtrl>
PTO_INST void RestoreSatMode(bool originalSatMode)
{
    if constexpr (NeedSetCtrl) {
        if (originalSatMode) {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
        } else {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
        }
    }
}

// ============================================================================
// Tile Conversion Helper: Process Remainder Data Block
// ============================================================================
// TCvtTail processes the remainder (unaligned) portion of data that doesn't
// fit evenly into repeat boundaries, using vector masking.
//
// numRemainPerLine is forwarded as the trailing `numElemsPerRow` arg of GenCastCall so that
// NonSatTorch paths which need explicit sub-chunk splitting (fp16 -> int8) can honor the valid
// column count even when it exceeds the int32 hardware capacity (64 elements) per repeat.
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS, typename... Args>
PTO_INST void TCvtTail(__ubuf__ typename TileDataD::DType *dstPtr, __ubuf__ typename TileDataS::DType *srcPtr,
                       RoundMode mode, unsigned validRow, unsigned numRemainPerLine, Args... args)
{
    constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
    constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    SetContinuousMask(numRemainPerLine);
    const uint16_t numElemsPerRow = static_cast<uint16_t>(numRemainPerLine);
    if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
            GenCastCall<TileDataD, TileDataS>(dstPtr + j * DS * REPEAT_MAX, srcPtr + j * SS * REPEAT_MAX,
                                              (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                              (uint16_t)SS / srcNElemPerBlock, args..., numElemsPerRow);
        }
    }
    if (remainAfterLoop > 0) {
        GenCastCall<TileDataD, TileDataS>(dstPtr + numLoop * DS * REPEAT_MAX, srcPtr + numLoop * SS * REPEAT_MAX,
                                          (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                          (uint16_t)SS / srcNElemPerBlock, args..., numElemsPerRow);
    }
    set_vector_mask(-1, -1);
}

// ============================================================================
// Core Tile Conversion Kernel
// ============================================================================
// TCvt orchestrates the complete tile conversion process by handling both:
//   1. Aligned region: Complete repeat units processed via TCvtHead
//   2. Remainder region: Partial repeats processed with vector masking
//
// Template parameters:
//   SS: Source row stride
//   DS: Destination row stride
//
// @param dst: Destination tile (output) - contains data after conversion
// @param src: Source tile (input) - contains original data to be converted
// @param mode: Rounding mode (RINT/ROUND/FLOOR/CEIL/TRUNC/NONE/ODD)
// @param satMode: Saturation mode for float-to-int conversions:
//                 ON  = Clamp to datatype range [min, max]
//                 OFF = Convert to int64, extract least significant N bits
// @param numRepeatPerLine: Number of complete repeats per line
// @param numRemainPerLine: Remaining elements per line (not aligned to repeat)
// @param validRow: Number of rows containing valid data
// @param elementsPerRepeat: Number of elements per repeat operation
// @param dstRepeatStride: Stride between repeats in destination buffer
// @param srcRepeatStride: Stride between repeats in source buffer
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS, bool NeedSetCtrl>
__tf__ AICORE void TCvt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src,
                        RoundMode mode, SaturationMode satMode, unsigned numRepeatPerLine, unsigned numRemainPerLine,
                        unsigned validRow, unsigned elementsPerRepeat, unsigned dstRepeatStride,
                        unsigned srcRepeatStride)
{
    bool originalSatMode = ApplySatMode<NeedSetCtrl>(satMode);

    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *srcPtr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);

    if (numRepeatPerLine > 0) {
        TCvtHead<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, numRepeatPerLine, validRow, elementsPerRepeat,
                                               dstRepeatStride, srcRepeatStride);
    }
    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine > 0) {
        TCvtTail<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, validRow, numRemainPerLine);
    }

    RestoreSatMode<NeedSetCtrl>(originalSatMode);
}

// TCvt overload with explicit TmpTileData parameter.
// Mirrors TSort32Impl's with-tmp overload: passes a user-supplied scratch tile
// through to GenCastCallFp16ToInt8_NonSatTorch instead of using TMP_UB_OFFSET.
template <typename TileDataD, typename TileDataS, typename TmpTileData, unsigned SS, unsigned DS, bool NeedSetCtrl>
__tf__ AICORE void TCvt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src,
                        typename TmpTileData::TileDType __in__ tmp, RoundMode mode, SaturationMode satMode,
                        unsigned numRepeatPerLine, unsigned numRemainPerLine, unsigned validRow,
                        unsigned elementsPerRepeat, unsigned dstRepeatStride, unsigned srcRepeatStride)
{
    bool originalSatMode = ApplySatMode<NeedSetCtrl>(satMode);

    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *srcPtr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
    __ubuf__ int32_t *tmpPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(tmp);

    if (numRepeatPerLine > 0) {
        TCvtHead<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, numRepeatPerLine, validRow, elementsPerRepeat,
                                               dstRepeatStride, srcRepeatStride, tmpPtr);
    }
    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine > 0) {
        TCvtTail<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, validRow, numRemainPerLine, tmpPtr);
    }

    RestoreSatMode<NeedSetCtrl>(originalSatMode);
}

// ============================================================================
// TCVT Helper: Compute Repeat Configuration
// ============================================================================
// Computes repeat stride and element count based on source/destination types.
// Handles int4b_t packing (2 elements per byte) as a special case.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void ComputeTCvtRepeatConfig(unsigned &elementsPerRepeat, unsigned &dstRepeatStride,
                                          unsigned &srcRepeatStride)
{
    constexpr bool isDstInt4 = std::is_same<typename TileDataD::DType, int4b_t>::value;
    constexpr bool isSrcInt4 = std::is_same<typename TileDataS::DType, int4b_t>::value;

    if constexpr (isDstInt4) {
        elementsPerRepeat = REPEAT_BYTE / sizeof(half); // 128
        dstRepeatStride = BLOCK_MAX_PER_REPEAT / 4;     // 2 (64 bytes = 2 blocks for 128 packed int4 elements)
        srcRepeatStride = BLOCK_MAX_PER_REPEAT;         // 8
    } else if constexpr (isSrcInt4) {
        elementsPerRepeat = REPEAT_BYTE / sizeof(half); // 128
        dstRepeatStride = BLOCK_MAX_PER_REPEAT;         // 8
        srcRepeatStride = BLOCK_MAX_PER_REPEAT / 4;     // 2 (64 bytes = 2 blocks for 128 packed int4 elements)
    } else {
        uint64_t repeatWidth =
            static_cast<uint64_t>(max(sizeof(typename TileDataD::DType), sizeof(typename TileDataS::DType)));
        dstRepeatStride =
            repeatWidth == sizeof(typename TileDataD::DType) ?
                BLOCK_MAX_PER_REPEAT :
                (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataS::DType) * sizeof(typename TileDataD::DType));
        srcRepeatStride =
            repeatWidth == sizeof(typename TileDataS::DType) ?
                BLOCK_MAX_PER_REPEAT :
                (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataD::DType) * sizeof(typename TileDataS::DType));
        elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    }
}

// Helper: Check if this is a narrowing conversion that defaults to non-saturating mode
template <typename TileDataD, typename TileDataS>
constexpr bool kIsNarrowingCvt =
    (std::is_same<typename TileDataD::DType, uint8_t>::value && std::is_same<typename TileDataS::DType, half>::value) ||
    (std::is_same<typename TileDataD::DType, int8_t>::value && std::is_same<typename TileDataS::DType, half>::value) ||
    (std::is_same<typename TileDataD::DType, int16_t>::value &&
     std::is_same<typename TileDataS::DType, float>::value) ||
    (std::is_same<typename TileDataD::DType, int16_t>::value && std::is_same<typename TileDataS::DType, half>::value) ||
    (std::is_same<typename TileDataD::DType, int32_t>::value &&
     std::is_same<typename TileDataS::DType, int64_t>::value) ||
    (std::is_same<typename TileDataD::DType, int16_t>::value &&
     std::is_same<typename TileDataS::DType, int32_t>::value);

// ============================================================================
// High-Level Tile Conversion Interface
// ============================================================================
// TCVT_IMPL is the main entry point for tile data type conversion.
// Calculates optimal repeat configuration and delegates to TCvt kernel.
//
// This is the main implementation with explicit satMode parameter.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl = true)
{
    unsigned dstRepeatStride, srcRepeatStride, elementsPerRepeat;
    ComputeTCvtRepeatConfig<TileDataD, TileDataS>(elementsPerRepeat, dstRepeatStride, srcRepeatStride);

    unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
    unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
    constexpr unsigned SS = TileDataS::RowStride;
    constexpr unsigned DS = TileDataD::RowStride;
    unsigned validRow = dst.GetValidRow();
    if (needSetCtrl) {
        if constexpr (kIsNarrowingCvt<TileDataD, TileDataS>) {
            TCvt<TileDataD, TileDataS, SS, DS, true>(dst.data(), src.data(), mode, satMode, numRepeatPerLine,
                                                     numRemainPerLine, validRow, elementsPerRepeat, dstRepeatStride,
                                                     srcRepeatStride);
        } else {
            TCvt<TileDataD, TileDataS, SS, DS, true>(dst.data(), src.data(), mode, SaturationMode::ON, numRepeatPerLine,
                                                     numRemainPerLine, validRow, elementsPerRepeat, dstRepeatStride,
                                                     srcRepeatStride);
        }
    } else {
        if constexpr (kIsNarrowingCvt<TileDataD, TileDataS>) {
            TCvt<TileDataD, TileDataS, SS, DS, false>(dst.data(), src.data(), mode, satMode, numRepeatPerLine,
                                                      numRemainPerLine, validRow, elementsPerRepeat, dstRepeatStride,
                                                      srcRepeatStride);
        } else {
            TCvt<TileDataD, TileDataS, SS, DS, false>(dst.data(), src.data(), mode, SaturationMode::ON,
                                                      numRepeatPerLine, numRemainPerLine, validRow, elementsPerRepeat,
                                                      dstRepeatStride, srcRepeatStride);
        }
    }
}

// TCVT_IMPL overload with explicit TmpTileData and explicit satMode.
// Mirrors TSORT32_IMPL's with-tmp overload: uses user-supplied scratch tile
// instead of TMP_UB_OFFSET for conversions that need temporary storage.
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl = true)
{
    unsigned dstRepeatStride, srcRepeatStride, elementsPerRepeat;
    ComputeTCvtRepeatConfig<TileDataD, TileDataS>(elementsPerRepeat, dstRepeatStride, srcRepeatStride);

    unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
    unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
    constexpr unsigned SS = TileDataS::RowStride;
    constexpr unsigned DS = TileDataD::RowStride;
    unsigned validRow = dst.GetValidRow();
    if (needSetCtrl) {
        TCvt<TileDataD, TileDataS, TmpTileData, SS, DS, true>(dst.data(), src.data(), tmp.data(), mode, satMode,
                                                              numRepeatPerLine, numRemainPerLine, validRow,
                                                              elementsPerRepeat, dstRepeatStride, srcRepeatStride);
    } else {
        TCvt<TileDataD, TileDataS, TmpTileData, SS, DS, false>(dst.data(), src.data(), tmp.data(), mode, satMode,
                                                               numRepeatPerLine, numRemainPerLine, validRow,
                                                               elementsPerRepeat, dstRepeatStride, srcRepeatStride);
    }
}

// TCVT_IMPL overload with explicit TmpTileData and type-specific default satMode.
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, bool needSetCtrl = true)
{
    if constexpr (kIsNarrowingCvt<TileDataD, TileDataS>) {
        TCVT_IMPL(dst, src, tmp, mode, SaturationMode::OFF, needSetCtrl);
    } else {
        TCVT_IMPL(dst, src, tmp, mode, SaturationMode::ON, needSetCtrl);
    }
}

// ============================================================================
// TCVT_IMPL Overload with Type-Specific Defaults
// ============================================================================
// This overload provides conversion-specific default saturation modes:
// - FP16→UINT8, FP16→INT8: defaults to OFF (PyTorch-compatible truncation)
// - INT64→INT32, INT32→INT16: defaults to OFF (truncation behavior)
// - All others: defaults to ON (native TCVT saturation)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, bool needSetCtrl = true)
{
    if constexpr (kIsNarrowingCvt<TileDataD, TileDataS>) {
        TCVT_IMPL(dst, src, mode, SaturationMode::OFF, needSetCtrl);
    } else {
        TCVT_IMPL(dst, src, mode, SaturationMode::ON, needSetCtrl);
    }
}
} // namespace pto
#endif
