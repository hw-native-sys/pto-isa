/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include <pto/npu/a2a3/custom/TFmodRemHp.hpp>
#include <pto/npu/a2a3/custom/TRemHpStaged.hpp>

namespace pto {
// Formula: remainder(a, b) = a - floor(a/b) * b
// Note: For fp32, after computing remainder, we check if result * scalar < 0.
//       If signs differ, we add scalar to result to ensure the result has the same sign as scalar.
//
// tmp layout (single row, columns measured in T elements):
//   [0, dstRowStride)                    sign-correction candidate (dst + x)
//   [dstRowStride, dstRowStride+maskF)   packed compare bitmask, 32-byte aligned
//   [dstRowStride+maskF, +8)             cmpmask address buffer (64-bit)
struct RemSOp {
    // vcmpvs_lt must run in norm mask mode with explicit repeatTimes: a single
    // repeat only emits the first 64 bits of the packed mask and leaves stale
    // UB beyond element 64, so vsel would pick stale bits for validCols > 64.
    // VSEL_TENSOR_TENSOR_MODE (2) reads the packed bitmask through the
    // two-level address indirection (same convention as TRem / TSel).
    PTO_INTERNAL static void RemSF32Instr(
        __ubuf__ float* dst, __ubuf__ float* src, float x, __ubuf__ float* tmp, unsigned dstRowStride,
        unsigned repeatTimes, unsigned validCols)
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

        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, tmp, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // set_cmpmask dereferences a full 64-bit address. Duplicating the 32-bit
        // address with vector_dup fills both halves ((addr<<32)|addr), so vsel
        // reads the packed mask from a garbage address (observed as effectively
        // random per-element selects). Store the true 64-bit address with a
        // scalar UB store, flags around it (pypto arch32 select idiom).
        __ubuf__ uint64_t* addrBuf = reinterpret_cast<__ubuf__ uint64_t*>(tmp + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *addrBuf = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0, validCols);
        // Step 7: candidate = dst + x into tmp[0..validCols)
        vadds(tmp, dst, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 8: vsel mode 2 — pick tmp (dst + x) where the packed mask bit is set
        vsel(dst, tmp, dst, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemSInt32Instr(
        __ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t x, __ubuf__ int32_t* tmp, unsigned dstRowStride,
        unsigned repeatTimes, unsigned validCols)
    {
        __ubuf__ float* dst_f = reinterpret_cast<__ubuf__ float*>(dst);
        __ubuf__ float* src_f = reinterpret_cast<__ubuf__ float*>(src);
        __ubuf__ float* tmp_f = reinterpret_cast<__ubuf__ float*>(tmp);

        vconv_s322f32(src_f, src, 1, 1, 1, 8, 8);
        vector_dup(tmp_f, (float)x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vdiv(tmp_f, src_f, tmp_f, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322s32f(tmp, tmp_f, 1, 1, 1, 8, 8); // floor quotient
        vconv_f322s32r(src, src_f, 1, 1, 1, 8, 8); // round-reconvert dividend
        pipe_barrier(PIPE_V);

        vmuls(dst, tmp, x, 1, 1, 1, 8, 8); // floor(src/x) * x (int)
        pipe_barrier(PIPE_V);

        vsub(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_s322f32(dst_f, dst, 1, 1, 1, 8, 8);
        vconv_s322f32(tmp_f, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmuls(tmp_f, dst_f, (float)x, 1, 1, 1, 8, 8); // sign probe: r * s
        pipe_barrier(PIPE_V);

        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp_f + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, tmp_f, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        __ubuf__ uint32_t* addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp_f + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBuf)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0, validCols);
        vadds(tmp_f, dst_f, (float)x, 1, 1, 1, 8, 8); // candidate r + s
        pipe_barrier(PIPE_V);

        vsel(dst_f, tmp_f, dst_f, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);

        vconv_f322s32z(dst, dst_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <
    auto PrecisionType, typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL void TRemS(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataDst::DType x, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;

    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    if constexpr (
        PrecisionType == RemSAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // High precision remainder (issue #2): staged loop inversion when the
        // tile has enough full vectors and dst does not alias src, per-vector
        // iterations otherwise (see custom/TRemHpStaged.hpp).
        unsigned capVecs = validCols / elementsPerRepeat;
        if (capVecs == 0) {
            capVecs = 1; // per-vector path scratch
        }
        if (TRemSStagedEligibleA2A3<TileDataDst, TileDataSrc>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), validRows,
                validCols)) {
            TRemSStagedA2A3<false, dstRowStride, srcRowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), (float)x,
                reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows, validCols);
        } else {
            TRemSHpRegister<false, dstRowStride, srcRowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), (float)x,
                reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows, validCols);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < (int)validRows; ++i) {
        __ubuf__ T* dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T* s0Next = srcPtr + i * srcRowStride;
        // Note: tmp buffer is reused for each row iteration to save memory
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemSOp::RemSF32Instr(dstNext, s0Next, x, tmpPtr, dstRowStride, repeatTimes, validCols);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemSOp::RemSInt32Instr(dstNext, s0Next, x, tmpPtr, dstRowStride, repeatTimes, validCols);
        } else {
            static_assert(sizeof(T) == 0, "TREMS: Unsupported tile DType.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;

    // static assertions
    static_assert(
        std::is_same_v<T, typename TileDataSrc::DType>, "TREMS: The data types of dst and src must be the same.");
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
        "Fix: TREMS supports only float and int32 element types.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc::Loc == TileType::Vec,
        "TREMS: TileType of dst and src tiles must be TileType::Vec.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "TREMS: Only support row major layout.");

    // dynamic checks
    PTO_ASSERT(
        dst.GetValidRow() == src.GetValidRow() && dst.GetValidRow() > 0,
        "TREMS: Number of valid rows of dst and src must be the same, and both greater than 0.");
    PTO_ASSERT(
        dst.GetValidCol() == src.GetValidCol() && dst.GetValidCol() > 0,
        "TREMS: Number of valid columns of dst and src must be the same, and all greater than 0.");
    // Single-row tmp layout: [0, dstRowStride) sign-correction candidate,
    // [dstRowStride, dstRowStride+maskFloats) packed bitmask (32-byte aligned),
    // then an 8-element (32-byte) address buffer.
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    // BIT_TO_BYTE bits per byte, packed mask rounded up to BLOCK_BYTE_SIZE bytes
    constexpr unsigned maskBytes = (dstRowStride + BIT_TO_BYTE - 1) / BIT_TO_BYTE;
    constexpr unsigned maskFloats = ((maskBytes + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE) * (BLOCK_BYTE_SIZE / 4);
    constexpr unsigned tmpRequiredCols = dstRowStride + maskFloats + BIT_TO_BYTE;
    // The high precision path needs the larger scratch contract from
    // custom/TFmodRemHp.hpp, sized for the staged capacity (full vectors per
    // row, at least 1 for the per-vector fallback).
    unsigned hpRequiredCols = 0;
    if constexpr (
        PrecisionType == RemSAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        constexpr unsigned elemsPerRepeatHp = REPEAT_BYTE / sizeof(float);
        unsigned capVecs = dst.GetValidCol() / elemsPerRepeatHp;
        hpRequiredCols = HpScratchRequiredCols(capVecs > 0 ? capVecs : 1);
    }
    PTO_ASSERT(
        tmp.GetValidRow() >= 1 &&
            tmp.GetValidCol() >= (tmpRequiredCols > hpRequiredCols ? tmpRequiredCols : hpRequiredCols),
        "TREMS: tmp tile must have 1 row and enough columns for candidate, mask and address buffer.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    TRemS<
        PrecisionType, TileDataDst, TileDataSrc, TileDataTmp, elementsPerRepeat, blockSizeElem, dstRowStride,
        srcRowStride>(dst.data(), src.data(), scalar, tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif
