/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU KirinDev0000 Architecture
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include "pto/npu/kirin9030/common.hpp"
#include "pto/npu/kirin9030/utils.hpp"
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <array>

// Architecture-specific saturation mode
#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define ARCH_RS_SAT RS_ENABLE // Workaround for CTRL issues
#else
#define ARCH_RS_SAT RS_DISABLE // Standard
#endif

namespace pto {

/**
 * CTRL[60]: Primary hardware control bit for saturation operations
 * - Used in combination with CTRL[59] to control saturation mode
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::ON
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::OFF
 * - Used for: float→integer, integer→integer, float→float (wider→narrower, dst≠fp32)
 */
constexpr const int SAT_MODE_BIT_60 = 60;

/**
 * CTRL[59]: Secondary hardware control bit for saturation operations
 * - Used in combination with CTRL[60] to control saturation mode
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::ON
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::OFF
 * - Used for: float→integer, integer→integer, float→float (wider→narrower, dst≠fp32)
 */
constexpr const int SAT_MODE_BIT_59 = 59;

/**
 * CTRL[48]: Saturation control bit for narrower→wider float conversions
 * - Used for: float→float (narrower→wider, dst≠fp32)
 * - CTRL[48]=1: Non-saturation mode
 * - CTRL[48]=0: Saturation mode
 */
constexpr const int SAT_MODE_BIT_48 = 48;

/**
 * Unified enum for all type conversion modes
 * Describes the vcvt intrinsic parameter pattern used for conversion
 */
enum class CastMode {
    EXPAND,         // vcvt(..., PART_EVEN) - Type expansion only, no conversion
    ROUND,          // vcvt(..., R()) - Conversion with rounding only
    ROUND_SAT,      // vcvt(..., R(), RS_DISABLE) - Conversion with rounding and saturation
    ROUND_PART,     // vcvt(..., R(), PART_EVEN) - Conversion with rounding and part operation
    ROUND_SAT_PART, // vcvt(..., R(), RS_DISABLE, PART_EVEN) - Rounding, saturation, and part
    SAT_PART,       // vcvt(..., RS_DISABLE, PART_EVEN) - Saturation and part (no rounding)
    SAT_ROUND       // vcvt(..., RS_DISABLE, R()) - Saturation then rounding (reversed order)
};

// PyTorch alignment for edge cases (inf, -inf, nan, overflow)
// 1 = PyTorch-compatible (uses NonSatTorch), 0 = standard (faster)
#define EDGE_CASE_ALIGN_ENABLE 1

#define FOR_ROWS                                     \
    for (uint16_t row = 0; row < validRows; row++) { \
        int32_t dstOffset = row * dstCols;           \
        int32_t srcOffset = row * srcCols;           \
        uint32_t sreg = validCols;

#define FOR_ELEMENTS(elNum)                                 \
    constexpr uint16_t elementsNum = (elNum);               \
    uint16_t repeatTimes = CeilDivision(sreg, elementsNum); \
    for (uint16_t idx = 0; idx < repeatTimes; idx++) {
#define END_FOR_ELEMENTS      \
    srcOffset += elementsNum; \
    dstOffset += elementsNum; \
    }

#define END_FOR_ROWS }

//=============================================================================================
// 1D Helper Templates - For contiguous data (optimized fast path)
//=============================================================================================
// These templates handle conversions when data is laid out contiguously in memory without
// padding. They process data in a single pass without row/column iteration overhead.
//
// PERFORMANCE NOTE: 1D versions are significantly faster than 2D versions when applicable,
// as they avoid the FOR_ROWS/FOR_ELEMENTS loop overhead and process data in bulk.

/**
 * Cast 64-bit integer to 32-bit (signed/float) - 1D version
 * Handles: s64 -> s32 #sat #part, s64 -> f32 #rnd #part
 */
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    vector_s64 v_input_0;
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B64);
    uint32_t sReg = totalElements;
    uint32_t len64 = sReg * 2;
    uint32_t len_even = sReg * 2;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<DST> v_output;
        MaskReg preg_b64 = CreatePredicate<float>(len64);
        MaskReg preg_b32 = CreatePredicate<float>(len_even);

        vlds(v_input_0, src, i * ELE_CNT_B64, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_b64, ARCH_RS_SAT, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b64, R(), PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B64, PK_B64, preg_b32);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// FP32 -> INT16 (PyTorch-compatible for inf/-inf)
// Two-step: fp32 -> int32 -> int16 (uses registers, no UB temp)
template <typename R>
inline AICORE void cast32to16_NonSatTorch_1D(
    __ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<float> v_input_fp32;
        RegTensor<int32_t> v_temp_int32;
        RegTensor<int16_t> v_output_int16;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_fp32, src, i * ELE_CNT_B32, NORM);
        vcvt(v_temp_int32, v_input_fp32, preg_b32, R(), RS_DISABLE);
        vcvt(v_output_int16, v_temp_int32, preg_b32, ARCH_RS_SAT, PART_EVEN);
        vsts(v_output_int16, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
    }
}

// Cast 32-bit -> 16-bit (1D)
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output_even;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output_even, v_input_0, preg_b32, ARCH_RS_SAT, PART_EVEN);
        } else {
            vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
        }
        vsts(v_output_even, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast between 32-bit types - 1D version
 * Handles: f32 -> s32 #rnd #sat, s32 -> f32 #rnd, f32 -> f32 #rnd (same-type rounding)
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast32to32_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, NORM);
        if constexpr (std::is_same<DST, SRC>::value) {
            vtrc(v_output, v_input_0, R(), preg_b32_st);
        } else if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE);
        } else {
            vcvt(v_output, v_input_0, preg_b32, R());
        }
        vsts(v_output, dst, i * ELE_CNT_B32, NORM_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// Cast 32-bit -> s64 (1D)
template <typename R, typename SRC>
inline AICORE void cast32toS64_1D_NoPostUpdate(
    __ubuf__ int64_t* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B64);
    uint32_t sReg = totalElements;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    uint32_t len64 = sReg * 2;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        vector_s64 v_output;
        MaskReg preg_b64 = CreatePredicate<float>(len64);

        vlds(v_input_0, src, i * ELE_CNT_B64, UNPK_B32);
        if constexpr (std::is_same<R, void>::value) {
            // For type expansion s32->s64 without rounding, no saturation control
            vcvt(v_output, v_input_0, preg_b32, PART_EVEN);
        } else {
            // For conversions with rounding (e.g., f32->s64), saturation mode is controllable
            vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B64, NORM_B32, preg_b64);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// Float16 (half) to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned)
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp16 -> int32
// 2. int32 -> int16
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast16to16_NonSatTorch_1D(
    __ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp16 -> int32 -> int16)
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<half> v_input_fp16;
        RegTensor<int32_t> v_temp_int32;
        RegTensor<int16_t> v_output_int16;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        // Step 1: Load fp16 and convert to int32 (stays in register)
        vlds(v_input_fp16, src, i * ELE_CNT_B32, UNPK_B16);
        vcvt(v_temp_int32, v_input_fp16, preg_b16, R(), PART_EVEN);

        // Step 2: Convert int32 to int16 with non-saturation and store
        vcvt(v_output_int16, v_temp_int32, preg_b32, ARCH_RS_SAT, PART_EVEN);
        vsts(v_output_int16, dst, i * ELE_CNT_B32, PK_B32, preg_b32_st);
    }
}

/**
 * Cast between 16-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to16_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE);
        } else if constexpr (MODE == CastMode::SAT_ROUND) {
            vcvt(v_output, v_input_0, preg_b16, RS_DISABLE, R());
        } else {
            vcvt(v_output, v_input_0, preg_b16, R());
        }
        vsts(v_output, dst, i * ELE_CNT_B16, NORM_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 16-bit to 32-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to32_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b32_st = CreatePredicate<float>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B32, UNPK_B16);
        if constexpr (MODE == CastMode::EXPAND) {
            vcvt(v_output, v_input_0, preg_b16, PART_EVEN);
        } else if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b16, R(), PART_EVEN);
        }
        vsts(v_output, dst, i * ELE_CNT_B32, NORM_B32, preg_b32_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

// Float16 (half) to signed 8-bit integer conversion for non-saturation mode (PyTorch-aligned)
// This version matches PyTorch behavior for inf/-inf and performs a multi-step conversion:
// 1. fp16 -> int16 (direct conversion)
// 2. bitwise AND with 255 using int16
// 3. int16 -> fp16
// 4. fp16 -> int8
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast16to8_NonSatTorch_1D(
    __ubuf__ int8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg pg = pset_b16(PAT_ALL);

    // Perform four-step conversion using registers (fp16 -> int16 -> AND -> fp16 -> int8)
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<half> v_input_fp16, v_temp_fp16;
        RegTensor<int16_t> v_temp_int16, v_temp_and, v_mask;
        vector_s8 v_output_int8;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        // Step 1: Load fp16 and convert to int16 (stays in register)
        vlds(v_input_fp16, src, i * ELE_CNT_B16, NORM);
        vcvt(v_temp_int16, v_input_fp16, preg_b16, R(), RS_DISABLE);

        // Step 2: Bitwise AND with 255 (stays in register)
        vdup(v_mask, static_cast<int16_t>(255), pg, MODE_ZEROING);
        vand(v_temp_and, v_temp_int16, v_mask, preg_b16_st);

        // Step 3: Convert int16 back to fp16 (stays in register)
        vcvt(v_temp_fp16, v_temp_and, preg_b16, R());

        // Step 4: Convert fp16 to int8 (no saturation) and store
        vcvt(v_output_int8, v_temp_fp16, preg_b16, R(), RS_DISABLE, PART_EVEN);
        vsts(v_output_int8, dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
    }
}

/**
 * Cast 16-bit to 8-bit types - 1D version
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input_0;
        DST_VEC v_output_even;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            // Saturation controlled by CTRL register - always use RS_DISABLE
            vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
        } else {
            // SAT_PART mode for int-to-int
            vcvt(v_output_even, v_input_0, preg_b16, ARCH_RS_SAT, PART_EVEN);
        }
        vsts(v_output_even, dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 8-bit to 16-bit types - 1D version
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        SRC_VEC v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b16 = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, UNPK_B8);
        vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
        vsts(v_output, dst, i * ELE_CNT_B16, NORM_B16, preg_b16);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 8-bit to 32-bit types - 1D version
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to32_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t next_len = (sReg > ELE_CNT_B32) ? sReg - ELE_CNT_B32 : 0;
    MaskReg pg = pset_b8(PAT_ALL);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t>&)v_zero, 0, pg, MODE_ZEROING);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        SRC_VEC v_input_0, v_input_1, v_input_2;
        RegTensor<DST> v_output_0, v_output_1;
        MaskReg preg_b16_cur = CreatePredicate<half>(sReg);
        MaskReg preg_b16_next = CreatePredicate<half>(next_len);
        MaskReg preg_b32, preg_b32_next;
        punpack(preg_b32, preg_b16_cur, LOWER);
        punpack(preg_b32_next, preg_b16_next, LOWER);

        vlds((RegTensor<uint8_t>&)v_input_0, (__ubuf__ uint8_t*)src, i * ELE_CNT_B16, UNPK_B8);
        vintlv(
            (RegTensor<uint8_t>&)v_input_1, (RegTensor<uint8_t>&)v_input_2, (RegTensor<uint8_t>&)v_input_0,
            (RegTensor<uint8_t>&)v_zero);
        vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
        vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
        vsts(v_output_0, dst, ELE_CNT_B32 * (i * 2), NORM_B32, preg_b32);
        vsts(v_output_1, dst, ELE_CNT_B32 * (i * 2 + 1), NORM_B32, preg_b32_next);
    }
}

/**
 * Cast 32-bit to 8-bit types - 1D version
 *
 * IMPLEMENTATION NOTE: Uses vselr with index vector to extract bytes from 32-bit words.
 * The conversion happens in two steps:
 *   1. vcvt: Convert 32-bit source to target type (PART_P0 extracts low byte)
 *   2. vselr: Gather bytes using index vector for proper byte packing
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8_1D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        RegTensor<SRC> v_input;
        DST_VEC v_output_p0;
        MaskReg preg_b32 = CreatePredicate<float>(sReg);

        vlds(v_input, src, i * ELE_CNT_B32, NORM);

        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output_p0, v_input, preg_b32, ROUND_R, RS_DISABLE, PART_P0);
        } else {
            vcvt(v_output_p0, v_input, preg_b32, ARCH_RS_SAT, PART_P0);
        }

        vsts(v_output_p0, dst, i * ELE_CNT_B32, PK4_B32, preg_b32);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 32-bit to hifloat8 - 1D version
 *
 * SPECIAL HANDLING: H8 (hifloat8) requires ROUND_A (round away from zero) instead of
 * ROUND_R (round to nearest even) for correct IEEE-like behavior with 8-bit precision.
 * This is a hardware requirement specific to the hifloat8 format.
 */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void cast32toH8_1D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B32);
    uint32_t sReg = totalElements;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_f32 v_input;
        vector_hif8 v_output_p0;
        MaskReg preg_b32 = CreatePredicate<float>(sReg);

        vlds(v_input, src, i * ELE_CNT_B32, NORM);
        vcvt(v_output_p0, v_input, preg_b32, ROUND_A, RS_DISABLE, PART_P0);
        vsts((vector_u8&)v_output_p0, (__ubuf__ uint8_t*)dst, i * ELE_CNT_B32, PK4_B32, preg_b32);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}

/**
 * Cast 16-bit to hifloat8 - 1D version
 * Special version for H8 that uses ROUND_A instead of template parameter
 */
template <typename R>
inline AICORE void cast16toH8_1D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    uint32_t totalElements = validRows * validCols;
    uint16_t repeatTimes = CeilDivision(totalElements, ELE_CNT_B16);
    uint32_t sReg = totalElements;
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_f16 v_input_0;
        vector_hif8 v_output_even;
        MaskReg preg_b16_st = CreatePredicate<half>(sReg);

        vlds(v_input_0, src, i * ELE_CNT_B16, NORM);
        vcvt(v_output_even, v_input_0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
        vsts((RegTensor<uint8_t>&)v_output_even, (__ubuf__ uint8_t*)dst, i * ELE_CNT_B16, PK_B16, preg_b16_st);
        // sReg is decremented by CreatePredicate with POST_UPDATE
    }
}
#endif // PTO_NPU_ARCH_A5

//=============================================================================================
// 2D Helper Templates - For non-contiguous data with padding
//=============================================================================================

/**
 * Cast 64-bit integer to 32-bit (signed/float) - 2D version
 * Handles: s64 -> s32 #sat #part, s64 -> f32 #rnd #part
 * Intrinsics:
 *   vcvt(output, input, preg, ARCH_RS_SAT, PART_EVEN)  // s64 -> s32 with saturation
 *   vcvt(output, input, preg, R(), PART_EVEN)        // s64 -> f32 with rounding
 */
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    vector_s64 v_input_0;

    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;

    FOR_ROWS
    uint32_t len64 = sreg * 2; // As we operate with 64bit blocks using 32bit operations
    MaskReg preg_b64 = CreatePredicate<float>(len64);
    uint32_t len_even = sreg * 2; // As only the even part is taken

    FOR_ELEMENTS(ELE_CNT_B64)
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(len_even);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (std::is_same<R, void>::value) {
        // For type expansion (s64->s32/f32), saturation mode is controllable
        vcvt(v_output, v_input_0, preg_b64, ARCH_RS_SAT, PART_EVEN);
    } else {
        vcvt(v_output, v_input_0, preg_b64, R(), PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, PK_B64, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 16-bit types
 * Handles: f32 -> f16 #rnd #sat #part, f32 -> bf16 #rnd #sat #part, f32 -> s16 #rnd #sat #part
 * Intrinsics:
 *   vcvt(out_odd, in_1, preg, ARCH_RS_SAT, PART_ODD/EVEN)       // No rounding mode (saturation only)
 *   vcvt(out_odd, in_1, preg, R(), RS_DISABLE, PART_ODD/EVEN)  // With rounding mode
 */
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
#ifdef PTO_NPU_ARCH_KIRINX90
    RegTensor<SRC> v_input_0, v_input_1, v_input_even, v_input_odd;
    RegTensor<DST> v_output_odd, v_output_even, v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    vlds(v_input_1, src, srcOffset + ELE_CNT_B32, NORM);
    vdintlv(v_input_even, v_input_odd, v_input_0, v_input_1);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output_even, v_input_even, preg_b32, ARCH_RS_SAT, PART_EVEN);
        vcvt(v_output_odd, v_input_odd, preg_b32, ARCH_RS_SAT, PART_ODD);
    } else {
        vcvt(v_output_even, v_input_even, preg_b32, R(), RS_DISABLE, PART_EVEN);
        vcvt(v_output_odd, v_input_odd, preg_b32, R(), RS_DISABLE, PART_ODD);
    }
#else
    RegTensor<SRC> v_input_0, v_input_1;
    RegTensor<DST> v_output_odd, v_output_even, v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B32);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output_odd, v_input_1, preg_b32, ARCH_RS_SAT, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b32, ARCH_RS_SAT, PART_EVEN);
    } else {
        vcvt(v_output_odd, v_input_1, preg_b32, R(), RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
#endif
    vor(v_output, v_output_even, v_output_odd, preg_b16);
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 16-bit types 2D without interleave version for better fusion
 * Handles: f32 -> f16 #rnd #sat #part, f32 -> bf16 #rnd #sat #part, f32 -> s16 #rnd #sat #part
 * Intrinsics:
 *   vcvt(out_odd, in_1, preg, ARCH_RS_SAT, PART_ODD/EVEN)       // No rounding mode (saturation only)
 *   vcvt(out_odd, in_1, preg, R(), RS_DISABLE, PART_ODD/EVEN)  // With rounding mode
 */
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16_2D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0, v_input_1;
    RegTensor<DST> v_output_odd, v_output_even, v_output;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output_even, v_input_0, preg_b32, ARCH_RS_SAT, PART_EVEN);
    } else {
        vcvt(v_output_even, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
    vsts(v_output_even, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// Float32 to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp32 -> int32
// 2. int32 -> int16
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast32to16_NonSatTorch_2D(
    __ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp32 -> int32 -> int16)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<float> v_input_fp32;
    RegTensor<int32_t> v_temp_int32;
    RegTensor<int16_t> v_output_int16;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    // Step 1: Load fp32 and convert to int32 (stays in register)
    vlds(v_input_fp32, src, srcOffset, NORM);
    vcvt(v_temp_int32, v_input_fp32, preg_b32, R(), RS_DISABLE);

    // Step 2: Convert int32 to int16 with non-saturation and store
    vcvt(v_output_int16, v_temp_int32, preg_b32, ARCH_RS_SAT, PART_EVEN);
    vsts(v_output_int16, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast between 32-bit types (float <-> int)
 * Modes:
 *   ROUND_SAT: f32 -> s32 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE)
 *   ROUND:     s32 -> f32 #rnd     → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast32to32(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT) {
        vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE);
    } else {
        vcvt(v_output, v_input_0, preg_b32, R());
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 64-bit signed integer
 * Handles: s32 -> s64 #part, f32 -> s64 #rnd #sat #part
 * Intrinsics:
 *   vcvt(output, input, preg, PART_EVEN)                    // s32 -> s64 (type expansion)
 *   vcvt(output, input, preg, R(), RS_DISABLE, PART_EVEN)    // f32 -> s64 (with rounding and saturation)
 */
template <typename R, typename SRC>
inline AICORE void cast32toS64(
    __ubuf__ int64_t* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    uint32_t len64 = sreg * 2; // As we operate with 64bit blocks using 32bit operations
    FOR_ELEMENTS(ELE_CNT_B64)
    RegTensor<SRC> v_input_0;
    vector_s64 v_output;

    MaskReg preg_b64 = CreatePredicate<float>(len64);

    vlds(v_input_0, src, srcOffset, UNPK_B32);
    if constexpr (std::is_same<R, void>::value) {
        vcvt(v_output, v_input_0, preg_b32, PART_EVEN);
    } else {
        // For conversions with rounding (e.g., f32->s64), saturation mode controlled by CTRL register
        vcvt(v_output, v_input_0, preg_b32, R(), RS_DISABLE, PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b64);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast between 16-bit types
 * Modes:
 *   ROUND_SAT:  f16 -> s16 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE)
 *   SAT_ROUND:  bf16 -> f16 #sat #rnd → vcvt(output, input, preg, RS_DISABLE, R()) [reversed order]
 *   ROUND:      s16 -> f16 #rnd      → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to16(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT) {
        vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE);
    } else if constexpr (MODE == CastMode::SAT_ROUND) {
        vcvt(v_output, v_input_0, preg_b16, RS_DISABLE, R());
    } else {
        vcvt(v_output, v_input_0, preg_b16, R());
    }
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// Float16 (half) to signed 16-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp16 -> int32
// 2. int32 -> int16
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast16to16_NonSatTorch_2D(
    __ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t len16 = ELE_CNT_B16;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    // Perform two-step conversion using registers (fp16 -> int32 -> int16)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<half> v_input_fp16;
    RegTensor<int32_t> v_temp_int32;
    RegTensor<int16_t> v_output_int16;
    MaskReg preg_b32_st = CreatePredicate<float>(sreg);

    // Step 1: Load fp16 and convert to int32 (stays in register)
    vlds(v_input_fp16, src, srcOffset, UNPK_B16);
    vcvt(v_temp_int32, v_input_fp16, preg_b16, R(), PART_EVEN);

    // Step 2: Convert int32 to int16 with non-saturation and store
    vcvt(v_output_int16, v_temp_int32, preg_b32, ARCH_RS_SAT, PART_EVEN);
    vsts(v_output_int16, dst, dstOffset, PK_B32, preg_b32_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 32-bit types
 * Modes:
 *   EXPAND:          Type expansion (f16/bf16/s16 -> f32/u32/s32 #part) → vcvt(output, input, preg, PART_EVEN)
 *   ROUND_PART:      f16 -> s32 #rnd #part                             → vcvt(output, input, preg, R(), PART_EVEN)
 *   ROUND_SAT_PART:  bf16 -> s32 #rnd #sat #part                       → vcvt(output, input, preg, R(), RS_DISABLE,
 * PART_EVEN)
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast16to32(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, UNPK_B16);
    if constexpr (MODE == CastMode::EXPAND) {
        vcvt(v_output, v_input_0, preg_b16, PART_EVEN);
    } else if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        vcvt(v_output, v_input_0, preg_b16, R(), PART_EVEN);
    }
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 8-bit types
 * Modes:
 *   ROUND_SAT_PART: f16 -> s8/u8 #rnd #sat #part → vcvt(..., R(), RS_DISABLE, PART_*)
 *   SAT_PART:       s16 -> u8 #sat #part         → vcvt(..., ARCH_RS_SAT, PART_*)
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B8)
    RegTensor<SRC> v_input_0, v_input_1;
    DST_VEC v_output_odd, v_output_even, v_output;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        // Use rounding with saturation controlled by CTRL register
        vcvt(v_output_odd, v_input_1, preg_b16, R(), RS_DISABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        // SAT_PART mode: saturation without rounding (integer->integer)
        vcvt(v_output_odd, v_input_1, preg_b16, ARCH_RS_SAT, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b16, ARCH_RS_SAT, PART_EVEN);
    }
    vor(v_output, v_output_even, v_output_odd, preg_b8);
    vsts(v_output, dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 16-bit to 8-bit types 2D without interleave version for better fusion
 * Modes:
 *   ROUND_SAT_PART: f16 -> s8/u8 #rnd #sat #part → vcvt(..., R(), RS_DISABLE, PART_EVEN)
 *   SAT_PART:       s16 -> u8 #sat #part         → vcvt(..., ARCH_RS_SAT, PART_EVEN)
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8_2D_NoPostUpdate(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<SRC> v_input_0;
    DST_VEC v_output_even;
    MaskReg preg_b16_st = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output_even, v_input_0, preg_b16, R(), RS_DISABLE, PART_EVEN);
    } else {
        // SAT_PART mode: s16 -> u8
        vcvt(v_output_even, v_input_0, preg_b16, ARCH_RS_SAT, PART_EVEN);
    }
    vsts(v_output_even, dst, dstOffset, PK_B16, preg_b16_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

// Float16 (half) to signed 8-bit integer conversion for non-saturation mode (PyTorch-aligned) - 2D version
// This version matches PyTorch behavior for inf/-inf and performs a multi-step conversion:
// 1. fp16 -> int16 (direct conversion)
// 2. bitwise AND with 255 using int16
// 3. int16 -> fp16
// 4. fp16 -> int8
// Uses register-based conversion (no UB temp buffers needed for A5 architecture)
template <typename R>
inline AICORE void cast16to8_NonSatTorch_2D(
    __ubuf__ int8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols)
{
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg pg = pset_b16(PAT_ALL);

    // Perform four-step conversion using registers (fp16 -> int16 -> AND -> fp16 -> int8)
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    RegTensor<half> v_input_fp16, v_temp_fp16;
    RegTensor<int16_t> v_temp_int16, v_temp_and, v_mask;
    vector_s8 v_output_int8;
    MaskReg preg_b16_st = CreatePredicate<half>(sreg);

    // Step 1: Load fp16 and convert to int16 (stays in register)
    vlds(v_input_fp16, src, srcOffset, NORM);
    vcvt(v_temp_int16, v_input_fp16, preg_b16, R(), RS_DISABLE);

    // Step 2: Bitwise AND with 255 (stays in register)
    vdup(v_mask, static_cast<int16_t>(255), pg, MODE_ZEROING);
    vand(v_temp_and, v_temp_int16, v_mask, preg_b16_st);

    // Step 3: Convert int16 back to fp16 (stays in register)
    vcvt(v_temp_fp16, v_temp_and, preg_b16, R());

    // Step 4: Convert fp16 to int8 (no saturation) and store
    vcvt(v_output_int8, v_temp_fp16, preg_b16, R(), RS_DISABLE, PART_EVEN);
    vsts(v_output_int8, dst, dstOffset, PK_B16, preg_b16_st);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 8-bit to 16-bit types
 * Handles: u8/s8 -> f16/u16/s16 #part (type expansion)
 * Intrinsic: vcvt(output, input, preg, PART_EVEN)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B16)
    SRC_VEC v_input_0;
    RegTensor<DST> v_output;
    MaskReg preg_b16 = CreatePredicate<half>(sreg);

    vlds(v_input_0, src, srcOffset, UNPK_B8);
    vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
    vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 8-bit to 32-bit types (FP8 to FP32 type expansion)
 * Handles: e4m3/e5m2/h8 -> f32 #part
 * Intrinsic: vcvt(output, input, preg, PART_*)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to32(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    MaskReg pg = pset_b8(PAT_ALL);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t>&)v_zero, 0, pg, MODE_ZEROING);

    FOR_ROWS
    int32_t rowDstOffset = row * dstCols;
    uint32_t next_len = (sreg > ELE_CNT_B32) ? sreg - ELE_CNT_B32 : 0;
    FOR_ELEMENTS(ELE_CNT_B16)
    SRC_VEC v_input_0, v_input_1, v_input_2;
    RegTensor<DST> v_output_0, v_output_1;
    MaskReg preg_b16_cur = CreatePredicate<half>(sreg);
    MaskReg preg_b16_next = CreatePredicate<half>(next_len);
    MaskReg preg_b32;
    MaskReg preg_b32_next;
    punpack(preg_b32, preg_b16_cur, LOWER);
    punpack(preg_b32_next, preg_b16_next, LOWER);

    vlds((RegTensor<uint8_t>&)v_input_0, (__ubuf__ uint8_t*)src, srcOffset, UNPK_B8);
    vintlv(
        (RegTensor<uint8_t>&)v_input_1, (RegTensor<uint8_t>&)v_input_2, (RegTensor<uint8_t>&)v_input_0,
        (RegTensor<uint8_t>&)v_zero); // interleave with zero
    vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
    vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
    vsts(v_output_0, dst, rowDstOffset + ELE_CNT_B32 * (idx * 2), NORM_B32, preg_b32);
    vsts(v_output_1, dst, rowDstOffset + ELE_CNT_B32 * (idx * 2 + 1), NORM_B32, preg_b32_next);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * Cast 32-bit to 8-bit types (both floating point and integer)
 * Handles:
 *   - f32 -> e4m3/e5m2/h8 #rnd #sat #part (ROUND_SAT_PART mode)
 *   - u32/s32 -> u8/s8 #sat #part (SAT_PART mode)
 * Intrinsics:
 *   vcvt(..., R(), RS_DISABLE, PART_P0) for floating point with rounding
 *   vcvt(..., ARCH_RS_SAT, PART_P0) for integer without rounding
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8(
    __ubuf__ DST* dst, __ubuf__ SRC* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    uint32_t preg_len_tail = (sreg % ELE_CNT_B32 == 0) ? ELE_CNT_B32 : (sreg % ELE_CNT_B32);

    FOR_ELEMENTS(ELE_CNT_B32)
    RegTensor<SRC> v_input;
    DST_VEC v_output_p0;

    vlds(v_input, src, srcOffset, NORM);

    // Convert with or without rounding based on mode - saturation controlled by CTRL register
    if constexpr (MODE == CastMode::ROUND_SAT_PART) {
        vcvt(v_output_p0, v_input, preg_b32, ROUND_R, RS_DISABLE, PART_P0);
    } else {
        vcvt(v_output_p0, v_input, preg_b32, ARCH_RS_SAT, PART_P0);
    }

    vsts(v_output_p0, dst, dstOffset, PK4_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

//=============================================================================================
// castData Overloads (2D - with row/column iteration for non-contiguous data)
//=============================================================================================
// These are the main conversion functions organized by source type for easy navigation.
// Each source type section contains conversions to all supported destination types.
//
// ORGANIZATION: Grouped by source type in ascending bit-width order (8→16→32→64-bit)
// WHY: This ordering provides quick lookup - if you know the source type, you can
// jump directly to its section and find all target conversions in one place.

//---------------------------------------------------------------------------------------------
// Source: FP32 (float) - 2D versions
//---------------------------------------------------------------------------------------------

/**
 * FP32 to FP32 - Applies rounding mode without type conversion
 * Intrinsic: vtrc(output, input, R(), preg)
 *
 * NOTE: Same-type conversions like FP32→FP32 are useful for applying rounding modes
 * to existing data without changing the underlying type (e.g., rounding to nearest even).
 */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input_0, v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    vtrc(v_output, v_input_0, R(), preg_b32);
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input_0, v_output;
    MaskReg preg_b32 = CreatePredicate<float>(sreg);

    vlds(v_input_0, src, srcOffset, NORM);
    vtrc(v_output, v_input_0, R(), preg_b32);
    vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

/**
 * FP32 to FP16
 * Conversion: f32 -> f16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(
    __ubuf__ float16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to BF16
 * Conversion: f32 -> bf16 #rnd #sat #part
 * Uses cast32to16 helper
 */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

/**
 * FP32 to I16
 * Conversion: f32 -> s16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(
    __ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast32to16_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled - saturation controlled by CTRL register
    cast32to16_2D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

/**
 * FP32 to I32
 * Conversion: f32 -> s32 #rnd #sat
 * Intrinsic: vcvt(output, input, preg, R(), RS_DISABLE)
 */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to I64
 * Conversion: f32 -> s64 #rnd #sat #part
 * Uses cast32toS64 helper
 */
template <typename R>
inline AICORE void castData(
    __ubuf__ int64_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int64_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/**
 * FP32 to FP8_E4M3
 * Conversion: f32 -> e4m3 #rnd #sat #pp
 * Uses cast32to8 helper
 */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ float8_e4m3_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float8_e4m3_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

/**
 * FP32 to FP8_E5M2
 * Conversion: f32 -> e5m2 #rnd #sat #pp
 * Uses cast32to8 helper
 */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ float8_e5m2_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float8_e5m2_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

/**
 * FP32 to H8
 * Conversion: f32 -> h8 #rnd #sat #part
 * Note: H8 conversion requires ROUND_A mode
 */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ hifloat8_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B32)
    vector_f32 v_input;
    vector_hif8 v_output_p0;

    vlds(v_input, src, srcOffset, NORM);
    vcvt(v_output_p0, v_input, preg_b32, ROUND_A, RS_DISABLE, PART_P0);
    vsts((vector_u8&)v_output_p0, (__ubuf__ uint8_t*)dst, dstOffset, PK4_B32, preg_b32);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    // Same complex logic as castData - just reuse it
    castData<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

//---------------------------------------------------------------------------------------------
// Source: FP16 (half) - 2D versions
//---------------------------------------------------------------------------------------------

/** FP16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** FP16 -> I32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** FP16 -> I16 #rnd #sat → vcvt(output, input, preg, R(), RS_DISABLE) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to16_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to16<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

/** FP16 -> I8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ int8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to8_NonSatTorch_2D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(
            dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

/** FP16 -> U8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8_2D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_u8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** FP16 -> H8 #rnd #sat #part */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ hifloat8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    // FP16->H8 conversion only supports ROUND_A or ROUND_H modes
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ROWS
    FOR_ELEMENTS(ELE_CNT_B8)
    vector_f16 v_input_0, v_input_1;
    vector_hif8 v_output_odd, v_output_even, v_output;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(sreg);

    vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
    vcvt(v_output_odd, v_input_1, preg_b16, ROUND_A, RS_DISABLE, PART_ODD);
    vcvt(v_output_even, v_input_0, preg_b16, ROUND_A, RS_DISABLE, PART_EVEN);
    vor((RegTensor<uint8_t>&)v_output, (RegTensor<uint8_t>&)v_output_even, (RegTensor<uint8_t>&)v_output_odd, preg_b8);
    vsts((RegTensor<uint8_t>&)v_output, (__ubuf__ uint8_t*)dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
    END_FOR_ROWS
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    // Same complex logic as castData - just reuse it
    castData<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

//---------------------------------------------------------------------------------------------
// Source: BFloat16 - 2D versions
//---------------------------------------------------------------------------------------------

/** BF16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** BF16 -> I32 #rnd #sat #part → vcvt(output, input, preg, R(), RS_DISABLE, PART_EVEN) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** BF16 -> F16 #sat #rnd → vcvt(output, input, preg, RS_DISABLE, R()) [reversed order] */
template <typename R>
inline AICORE void castData(
    __ubuf__ half* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

//---------------------------------------------------------------------------------------------
// Source: U8, I8 (8-bit integers) - 2D versions
//---------------------------------------------------------------------------------------------

/** U8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ half* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U8 -> U16 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint16_t* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ half* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> I16 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int16_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I8 -> I32 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: I16 (signed 16-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I16 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> FP16 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(
    __ubuf__ half* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> FP32 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> U32 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I16 -> I32 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: I32 (signed 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I32 -> FP32 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ int16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> I64 #part (type expansion) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int64_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int64_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I32 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint8_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: U32 (unsigned 32-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** U32 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint8_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ uint16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** U32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(
    __ubuf__ int16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_2D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: I64 (signed 64-bit integer) - 2D versions
//---------------------------------------------------------------------------------------------

/** I64 -> FP32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** I64 -> I32 #sat #part → vcvt(output, input, preg, ARCH_RS_SAT, PART_EVEN) */
template <typename R>
inline AICORE void castData(
    __ubuf__ int32_t* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: FP8 variants (float8_e4m3_t, float8_e5m2_t, hifloat8_t) - 2D versions
//---------------------------------------------------------------------------------------------
// FP8 formats are specialized 8-bit floating-point types with different exponent/mantissa splits:
//   - E4M3: 4 exponent bits, 3 mantissa bits (higher precision, smaller range)
//   - E5M2: 5 exponent bits, 2 mantissa bits (lower precision, larger range)
//   - HIF8: Hardware-specific 8-bit float format

/** E4M3 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ float8_e4m3_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float8_e4m3_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** E5M2 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ float8_e5m2_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float8_e5m2_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

/** H8 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_P0) */
template <typename R>
inline AICORE void castData(
    __ubuf__ float* dst, __ubuf__ hifloat8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_2D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ hifloat8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

//=============================================================================================
// castData_1D_NoPostUpdate Overloads - Organized by Source Type (8→16→32→64-bit sources)
//=============================================================================================
// Optimized 1D versions for contiguous data without padding
// Each section contains conversions FROM a specific source type TO all supported destination types

//---------------------------------------------------------------------------------------------
// Source: 8-bit types (uint8_t, int8_t, float8_e4m3_t, float8_e5m2_t, hifloat8_t) - 1D versions
//---------------------------------------------------------------------------------------------

// Source: U8 (unsigned 8-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ uint8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_u8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: I8 (signed 8-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to16_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_s8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: FP8_E4M3
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float8_e4m3_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_f8e4m3>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: FP8_E5M2
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float8_e5m2_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_f8e5m2>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: Hifloat8
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ hifloat8_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast8to32_1D_NoPostUpdate<vector_hif8>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

//---------------------------------------------------------------------------------------------
// Source: 16-bit types (half/fp16, bfloat16, int16_t) - 1D versions
//---------------------------------------------------------------------------------------------
// 16-bit conversions are commonly used for mixed-precision training and inference:
//   - FP16 (half): Standard IEEE 754 half-precision (1 sign, 5 exp, 10 mantissa)
//   - BF16 (bfloat16): Brain Float16 (1 sign, 8 exp, 7 mantissa) - FP32-compatible exponent
//   - I16: Signed 16-bit integer

// Source: FP16 (half)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols, uint32_t srcCols,
    SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<R, CastMode::ROUND_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to16_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to16_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to16_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast16to8_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(
            dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_s8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_u8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Note: FP16 -> FP8_E5M2 and FP16 -> FP8_E4M3 conversions are NOT supported
// Only FP16 -> Hifloat8 (H8) conversion is supported

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ half* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16toH8_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: BFloat16
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ bfloat16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16_1D_NoPostUpdate<R, CastMode::SAT_ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

#endif // PTO_NPU_ARCH_A5 || PTO_NPU_ARCH_KIRINDEV0000

// Source: I16 (signed 16-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ half* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to16_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int16_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast16to32_1D_NoPostUpdate<void, CastMode::EXPAND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 32-bit types (float, int32_t, uint32_t) - 1D versions
//---------------------------------------------------------------------------------------------
// Note: Keep FP32/I32/U32 together for quick lookup of all 32-bit source conversions.

// Source: FP32 (float)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ bfloat16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
#if EDGE_CASE_ALIGN_ENABLE
    if (satMode == SaturationMode::OFF) {
        // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
        cast32to16_NonSatTorch_1D<R>(dst, src, validRows, validCols, dstCols, srcCols);
    } else {
        cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
    }
#else
    // Use default implementation when edge case alignment is disabled
    cast32to16_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
#endif
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND_SAT>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int64_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINDEV0000)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float8_e4m3_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float8_e5m2_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ hifloat8_t* dst, __ubuf__ float* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toH8_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}
#endif // PTO_NPU_ARCH_A5

// Source: I32 (signed 32-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to32_1D_NoPostUpdate<R, CastMode::ROUND>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int64_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32toS64_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ int32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

// Source: U32 (unsigned 32-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint8_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to8_1D_NoPostUpdate<void, CastMode::SAT_PART, vector_u8>(
        dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ uint16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int16_t* dst, __ubuf__ uint32_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    cast32to16_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//---------------------------------------------------------------------------------------------
// Source: 64-bit types (int64_t)
//---------------------------------------------------------------------------------------------

// Source: I64 (signed 64-bit integer)
template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ float* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32_1D_NoPostUpdate<R>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

template <typename R>
inline AICORE void castData_1D_NoPostUpdate(
    __ubuf__ int32_t* dst, __ubuf__ int64_t* src, uint32_t validRows, uint32_t validCols, uint32_t dstCols,
    uint32_t srcCols, SaturationMode satMode)
{
    castS64to32_1D_NoPostUpdate<void>(dst, src, validRows, validCols, dstCols, srcCols, satMode);
}

//=============================================================================================
// Main TCVT Implementation
//=============================================================================================

/**
 * Main TCVT implementation function
 * Converts tile data from source type to destination type using specified rounding mode
 * Iterates over rows and calls appropriate castData specialization
 *
 * @param satMode: Saturation mode control (A5-specific):
 *                 In A5, saturation is controlled by both:
 *                 1. CTRL register bits [60] and [48] - set by TCVT_IMPL based on conversion type
 *                 2. RS_DISABLE/RS_DISABLE parameters in vcvt intrinsics
 *
 *                 The satMode parameter works in conjunction with CTRL bits:
 *                 - CTRL[60]: Used for float→int, int→int, float→float (wider→narrower, dst≠fp32)
 *                 - CTRL[48]: Used for float→float (narrower→wider, dst≠fp32), VTRC.fp16/bf16
 *
 *                 The actual saturation behavior is determined by both the CTRL bit setting
 *                 and the CastMode used in castData template instantiations.
 */
template <typename TileDataD, typename TileDataS, typename R>
__tf__ PTO_INTERNAL OP_NAME(TCVT) OP_TYPE(element_wise) void implTCVT(
    typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src, SaturationMode satMode,
    unsigned validRows, unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    // Saturation is controlled by:
    // 1. CTRL[60]/CTRL[48] register bits (set by caller TCVT_IMPL based on conversion type)
    // 2. RS_DISABLE/RS_DISABLE in vcvt intrinsics (determined by CastMode in castData templates)
    // The satMode parameter is passed through to castData functions which use it to select
    // between RS_DISABLE and RS_DISABLE in the vcvt intrinsic calls.

    using T1 = typename TileDataD::DType;
    using T2 = typename TileDataS::DType;
    __ubuf__ T1* dstPtr = (__ubuf__ T1*)__cce_get_tile_ptr(dst);
    __ubuf__ T2* srcPtr = (__ubuf__ T2*)__cce_get_tile_ptr(src);
    __VEC_SCOPE__
    {
        // Compile-time check: Use 1D optimization if:
        // 1. ValidCol == Cols (no column padding) for both src and dst, OR
        // 2. Both tiles have Rows == 1 (single row case)
        if constexpr (
            ((TileDataD::ValidCol == TileDataD::Cols) && (TileDataS::ValidCol == TileDataS::Cols)) ||
            ((TileDataD::Rows == 1) && (TileDataS::Rows == 1))) {
            // Use 1D path: faster bulk processing without row iteration overhead
            switch (version) {
                case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
                    castData_2D_NoPostUpdate<R>(
                        dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols, satMode);
                    break;
                case VFImplKind::VFIMPL_DEFAULT:
                case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
                case VFImplKind::VFIMPL_1D_POST_UPDATE:
                case VFImplKind::VFIMPL_2D_POST_UPDATE:
                default:
                    castData_1D_NoPostUpdate<R>(
                        dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols, satMode);
                    break;
            }

        } else {
            // Use 2D path: handles strided/padded data with row-by-row iteration
            // version parameter controls predicate update strategy:
            // VFIMPL_2D_NO_POST_UPDATE: manual predicate handling
            // default: auto predicate update
            switch (version) {
                case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
                case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
                    castData_2D_NoPostUpdate<R>(
                        dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols, satMode);
                    break;
                default:
                    castData<R>(dstPtr, srcPtr, validRows, validCols, TileDataD::Cols, TileDataS::Cols, satMode);
                    break;
            }
        }
    }
}

// ============================================================================
// Saturation Control Helper
// ============================================================================
// NOTE: bit selection for a given (Src, Dst) pair is fully compile-time via
// `SaturationCtrlTraits` below — the runtime `SaturationCtrlConfig` struct
// used by other targets is intentionally absent here so all branching folds
// away under `if constexpr`.

/**
 * Compile-time traits describing which CTRL bits a given (SrcType, DstType)
 * conversion needs to touch. By exposing this as `static constexpr bool`
 * members, every `useCtrl*` check at the use-site collapses via `if constexpr`
 * so dead code is eliminated per template instantiation — eliminating the
 * scalar branches that would otherwise be emitted for every TCVT_IMPL call.
 *
 * The bit *values* (whether to set or clear each bit) still depend on the
 * runtime `satMode`, but the *selection* of bits is fully type-driven.
 *
 * Mapping summary (matches the prior runtime logic exactly):
 *   - dst == float                          : no CTRL touched
 *   - float -> integer                      : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - int -> int (wider -> narrower)        : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - int -> int (narrower -> wider)        : no CTRL touched
 *   - fp32  -> fp16/bf16                    : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 *   - fp16/bf16 <-> fp16/bf16               : CTRL[48]=(satMode==OFF)
 *   - integer -> float (sizeof src >= dst)  : CTRL[60]=1, CTRL[59]=(satMode==OFF)
 */
template <typename SrcType, typename DstType, SaturationMode SatMode, bool NeedSetCtrl>
struct SaturationCtrlTraits {
private:
    static constexpr bool dstIsFp32 = caps::IsFP32<DstType>();
    static constexpr bool srcIsFloat = caps::IsFloatingPoint<SrcType>();
    static constexpr bool dstIsFloat = caps::IsFloatingPoint<DstType>();
    static constexpr bool srcIsInt = caps::IsInteger<SrcType>();
    static constexpr bool dstIsInt = caps::IsInteger<DstType>();
    static constexpr bool srcIsFp16Bf16 = caps::IsFP16<SrcType>() || caps::IsBF16<SrcType>();
    static constexpr bool dstIsFp16Bf16 = caps::IsFP16<DstType>() || caps::IsBF16<DstType>();

    static constexpr bool floatToInt = srcIsFloat && dstIsInt;
    static constexpr bool intToIntNarrow = srcIsInt && dstIsInt && (sizeof(SrcType) >= sizeof(DstType));
    static constexpr bool fp32ToFp16Bf16 = std::is_same<SrcType, float>::value && dstIsFp16Bf16;
    static constexpr bool fp16Bf16Pair = srcIsFp16Bf16 && dstIsFp16Bf16;
    static constexpr bool intToFloat = srcIsInt && dstIsFloat && (sizeof(SrcType) >= sizeof(DstType));

public:
    // CTRL[60] and CTRL[59] always go together in the original logic.
    static constexpr bool useCtrl60 = !dstIsFp32 && (floatToInt || intToIntNarrow || fp32ToFp16Bf16 || intToFloat);
    static constexpr bool useCtrl59 = useCtrl60;
    static constexpr bool useCtrl48 = !dstIsFp32 && fp16Bf16Pair;
    // True iff this conversion needs to inspect/restore any CTRL bit at all.
    static constexpr bool any = (useCtrl60 || useCtrl59 || useCtrl48) && (NeedSetCtrl);

    bool originalCtrl60 = false;
    bool originalCtrl59 = false;
    bool originalCtrl48 = false;

    /**
     * Apply saturation control bits for a (SrcType, DstType) conversion.
     *
     * All `useCtrl*` checks are compile-time, so for instantiations where the
     * conversion does not touch a given bit the corresponding code is dropped
     * entirely — no scalar test, no register access.
     */
    PTO_INTERNAL SaturationCtrlTraits()
    {
        if constexpr (any) {
            uint64_t originalCtrl = get_ctrl();
            if constexpr (useCtrl60) {
                // CTRL[60] is always set to 1 whenever this conversion uses it.
                originalCtrl60 = (originalCtrl & (1ULL << SAT_MODE_BIT_60)) != 0;
                set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
            }
            if constexpr (useCtrl59) {
                // CTRL[59] = 1 for OFF, 0 for ON (inverted).
                originalCtrl59 = (originalCtrl & (1ULL << SAT_MODE_BIT_59)) != 0;
                if constexpr (SatMode == SaturationMode::OFF) {
                    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
                } else {
                    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
                }
            }
            if constexpr (useCtrl48) {
                // CTRL[48] = 1 for non-saturation, 0 for saturation.
                originalCtrl48 = (originalCtrl & (1ULL << SAT_MODE_BIT_48)) != 0;
                if constexpr (SatMode == SaturationMode::OFF) {
                    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_48));
                } else {
                    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_48));
                }
            }
        }
    }

    /**
     * Restore the original CTRL bit states that this conversion touched. Bits
     * that were never written are not restored (compile-time eliminated).
     */
    PTO_INTERNAL ~SaturationCtrlTraits()
    {
        if constexpr (any) {
            if constexpr (useCtrl60) {
                if (originalCtrl60) {
                    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
                } else {
                    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_60));
                }
            }
            if constexpr (useCtrl59) {
                if (originalCtrl59) {
                    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
                } else {
                    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
                }
            }
            if constexpr (useCtrl48) {
                if (originalCtrl48) {
                    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_48));
                } else {
                    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_48));
                }
            }
        }
    }
};

// ============================================================================
// High-Level Tile Conversion Interface with explicit SaturationMode
// ============================================================================
// Small dispatch helper to keep TCVT_IMPL compact (avoids repeating the
// implTCVT<...>(dst.data(), src.data(), satMode, rows, cols) boilerplate).
template <typename RoundT, typename TileDataD, typename TileDataS>
PTO_INTERNAL void tcvtDispatch(TileDataD& dst, TileDataS& src, SaturationMode satMode)
{
    implTCVT<TileDataD, TileDataS, RoundT>(dst.data(), src.data(), satMode, dst.GetValidRow(), dst.GetValidCol());
}

// Dispatch on RoundMode -> concrete RoundType. Kept as a separate helper to
// keep TCVT_IMPL under the NBNC line-count limit.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void tcvtDispatchByRound(TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode)
{
    using SrcType = typename TileDataS::DType;
    using DstType = typename TileDataD::DType;
    switch (mode) {
        case RoundMode::CAST_RINT:
            tcvtDispatch<decltype(ROUND_R)>(dst, src, satMode);
            return;
        case RoundMode::CAST_ROUND:
            tcvtDispatch<decltype(ROUND_A)>(dst, src, satMode);
            return;
        case RoundMode::CAST_FLOOR:
            tcvtDispatch<decltype(ROUND_F)>(dst, src, satMode);
            return;
        case RoundMode::CAST_CEIL:
            tcvtDispatch<decltype(ROUND_C)>(dst, src, satMode);
            return;
        case RoundMode::CAST_TRUNC:
            tcvtDispatch<decltype(ROUND_Z)>(dst, src, satMode);
            return;
        case RoundMode::CAST_ODD:
            if constexpr (caps::IsFP16<DstType>() && caps::IsFP32<SrcType>()) {
                tcvtDispatch<decltype(ROUND_O)>(dst, src, satMode);
                return;
            }
            break; // fall through to default
        default:
            break;
    }
    // PyTorch-compatible default rounding (also matches a2a3 per-(src,dst) defaults):
    //   float -> integer : truncate toward zero (decltype(ROUND_Z))
    //   everything else  : round-to-nearest-even (decltype(ROUND_R))
    if constexpr (caps::IsFloatingPoint<SrcType>() && caps::IsInteger<DstType>()) {
        tcvtDispatch<decltype(ROUND_Z)>(dst, src, satMode);
    } else {
        tcvtDispatch<decltype(ROUND_R)>(dst, src, satMode);
    }
}
/**
 * SATURATION MODE RULES:
 * ======================
 *
 * Hardware Setup:
 * - CTRL[60] and CTRL[59] work together to control saturation mode
 * - CTRL[60]=1, CTRL[59]=0: SaturationMode::ON (saturation enabled)
 * - CTRL[60]=1, CTRL[59]=1: SaturationMode::OFF (saturation disabled)
 * - CTRL[48] value determines saturation for narrower→wider float conversions
 *
 * 1. FLOAT → INTEGER or INTEGER → INTEGER conversions:
 *    - Set CTRL[60]=1 with CTRL[59]=0 for saturation mode
 *    - Set CTRL[60]=1 with CTRL[59]=1 for non-saturation mode
 *    - RS_DISABLE used consistently in vcvt intrinsics
 *
 * 2. NARROWER → WIDER dynamic range conversions (integer):
 *    - No overflow possible, saturation not applicable
 *    - CTRL bits are neglected
 *
 * 3. FLOAT → FLOAT conversions:
 *    a) WIDER → NARROWER range (dst ≠ fp32):
 *       - Set CTRL[60]=1 with CTRL[59]=0 for saturation mode
 *       - Set CTRL[60]=1 with CTRL[59]=1 for non-saturation mode
 *       - RS_DISABLE used consistently in vcvt intrinsics
 *
 *    b) NARROWER → WIDER range (dst ≠ fp32):
 *       - Set CTRL[48]=1 for non-saturation mode
 *       - Set CTRL[48]=0 for saturation mode
 *
 *    c) Where dst = fp32:
 *       - Only non-saturation supported (RS_DISABLE)
 *       - CTRL[48]/[60]/[59] are neglected
 *       - Note: vtrc (fp32→fp32) falls into this category
 */
template <bool NeedSetCtrl, SaturationMode SatMode, typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVTImpl(TileDataD& dst, TileDataS& src, RoundMode mode)
{
    using SrcType = typename TileDataS::DType;
    using DstType = typename TileDataD::DType;

    SaturationCtrlTraits<SrcType, DstType, SatMode, NeedSetCtrl> ctrlGuard;
    tcvtDispatchByRound(dst, src, mode, SatMode);
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(
    TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode, bool needSetCtrl = true)
{
    if (needSetCtrl) {
        if (satMode == SaturationMode::ON) {
            TCVTImpl<true, SaturationMode::ON>(dst, src, mode);
        } else {
            TCVTImpl<true, SaturationMode::OFF>(dst, src, mode);
        }
    } else {
        if (satMode == SaturationMode::ON) {
            TCVTImpl<false, SaturationMode::ON>(dst, src, mode);
        } else {
            TCVTImpl<false, SaturationMode::OFF>(dst, src, mode);
        }
    }
}

// ============================================================================
// TCVT_IMPL Overloads with tmp buffer (unused in A5, for API compatibility)
// ============================================================================
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(
    TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, SaturationMode satMode, bool needSetCtrl = true)
{
    if (needSetCtrl) {
        if (satMode == SaturationMode::OFF) {
            TCVTImpl<true, SaturationMode::OFF>(dst, src, mode);
        } else {
            TCVTImpl<true, SaturationMode::ON>(dst, src, mode);
        }
    } else {
        if (satMode == SaturationMode::OFF) {
            TCVTImpl<false, SaturationMode::OFF>(dst, src, mode);
        } else {
            TCVTImpl<false, SaturationMode::ON>(dst, src, mode);
        }
    }
}

// ============================================================================
// TCVT_IMPL Overload with Type-Specific Defaults
// ============================================================================
// This overload provides conversion-specific default saturation modes:
// - FP16→UINT8, FP16→INT8: defaults to OFF (PyTorch-compatible truncation)
// - FP32/FP16→INT16: defaults to OFF (truncation behavior)
// - INT64→INT32, INT32→INT16: defaults to OFF (truncation behavior)
// - All others: defaults to ON (native TCVT saturation)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, RoundMode mode, bool needSetCtrl = true)
{
    if constexpr (
        (std::is_same<typename TileDataD::DType, uint8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        (std::is_same<typename TileDataD::DType, int8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, float>::value) ||
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        (std::is_same<typename TileDataD::DType, int32_t>::value &&
         std::is_same<typename TileDataS::DType, int64_t>::value) ||
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, int32_t>::value)) {
        if (needSetCtrl) {
            TCVTImpl<true, SaturationMode::OFF>(dst, src, mode);
        } else {
            TCVTImpl<false, SaturationMode::OFF>(dst, src, mode);
        }
    } else {
        if (needSetCtrl) {
            TCVTImpl<true, SaturationMode::ON>(dst, src, mode);
        } else {
            TCVTImpl<false, SaturationMode::ON>(dst, src, mode);
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, bool needSetCtrl = true)
{
    TCVT_IMPL(dst, src, mode, needSetCtrl);
}
} // namespace pto

#undef ARCH_RS_SAT
#undef ARCH_HAS_FP8
#undef ARCH_HAS_BFLOAT16
#endif // TCVT_HPP
