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
#include <pto/npu/a2a3/custom/TFmodRemHp.hpp>
#include <pto/npu/a2a3/custom/TRemHpStaged.hpp>

namespace pto {
// Formula: remainder(a, b) = a - floor(a/b) * b
// Note: For fp32, after computing remainder, we check if result * divider < 0.
//       If signs differ, we add divider to result to ensure the result has the same sign as divider.
template <typename T, unsigned dstRowStride>
struct RemOp {
    PTO_INTERNAL static void RemF32Instr(
        __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* tmp, unsigned repeatTimes,
        unsigned validCols)
    {
        set_mask_count();
        set_vector_mask(0, validCols);
        vdiv(tmp, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322f32f(tmp, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmul(dst, tmp, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src0, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(tmp, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // vcmpvs_lt must run in norm mask mode with explicit repeatTimes: unlike
        // arithmetic ops, the compare op does not auto-extend a single repeat to
        // the whole count (CANN CompareScalarCompute passes ceil(count*sizeof(T)/256)
        // repeats), so repeat=1 would only emit the first 64 bits of the packed
        // mask and leave stale UB in the rest.
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, tmp, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, validCols);

        vadd(tmp, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // VSEL_TENSOR_TENSOR_MODE (2) reads the packed bitmask from UB through a
        // two-level address indirection: set_cmpmask dereferences a full 64-bit
        // address. Duplicating the 32-bit address with vector_dup fills both
        // halves ((addr<<32)|addr), so vsel reads the mask from a garbage
        // address (observed as effectively random per-element selects). Store
        // the true 64-bit address with a scalar UB store, flags around it
        // (pypto arch32 select idiom). cmpMask occupies
        // dstRowStride..dstRowStride+maskFloats, addrBuf follows it, aligned to 32 bytes.
        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint32_t* addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBuf)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, validCols);
        vsel(dst, tmp, dst, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemInt32Instr(
        __ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, __ubuf__ int32_t* tmp,
        unsigned repeatTimes, unsigned validCols)
    {
        __ubuf__ float* dst_f = reinterpret_cast<__ubuf__ float*>(dst);
        __ubuf__ float* src0_f = reinterpret_cast<__ubuf__ float*>(src0);
        __ubuf__ float* src1_f = reinterpret_cast<__ubuf__ float*>(src1);
        __ubuf__ float* tmp_f = reinterpret_cast<__ubuf__ float*>(tmp);

        set_mask_count();
        set_vector_mask(0, validCols);

        vconv_s322f32(src0_f, src0, 1, 1, 1, 8, 8);
        vconv_s322f32(src1_f, src1, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vdiv(tmp_f, src0_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322s32f(tmp, tmp_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322s32r(src0, src0_f, 1, 1, 1, 8, 8);
        vconv_f322s32r(src1, src1_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(dst, tmp, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src0, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_s322f32(dst_f, dst, 1, 1, 1, 8, 8);
        vconv_s322f32(src1_f, src1, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(tmp_f, dst_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Same norm-mode compare with explicit repeatTimes as the f32 path.
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, tmp_f, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, validCols);

        vadd(tmp_f, dst_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // VSEL_TENSOR_TENSOR_MODE (2): same true 64-bit address store as the f32 path.
        unsigned maskByte = CeilDivision(dstRowStride, BIT_TO_BYTE); // shape
        unsigned maskFloat = CeilDivision(maskByte, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint32_t* addrBufs = reinterpret_cast<__ubuf__ uint32_t*>(tmp + dstRowStride + maskFloat);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBufs)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBufs);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, validCols);
        vsel(dst_f, tmp_f, dst_f, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);

        vconv_f322s32z(dst, dst_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* tmp, unsigned repeatTimes, unsigned validCols)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemF32Instr(dst, src0, src1, tmp, repeatTimes, validCols);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemInt32Instr(dst, src0, src1, tmp, repeatTimes, validCols);
        }
    }
};

template <
    auto PrecisionType, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
    unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride = dstRowStride,
    unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TRem(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    if constexpr (
        PrecisionType == RemAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // High precision remainder (issue #2): staged loop inversion when the
        // tile has enough full vectors and dst aliases neither input;
        // per-vector iterations otherwise (see custom/TRemHpStaged.hpp).
        if (TRemVStagedEligibleA2A3<TileDataDst, TileDataSrc0, TileDataSrc1>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), validRows, validCols)) {
            TRemVStagedA2A3<false, dstRowStride, src0RowStride, src1RowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows,
                validCols);
        } else {
            TRemVHpRegister<false, dstRowStride, src0RowStride, src1RowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows,
                validCols);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    constexpr unsigned tmpRowStride = TileDataTmp::RowStride;
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    uint16_t rowLimit = static_cast<uint16_t>(validRows);
    for (uint16_t i = 0; i < rowLimit; i++) {
        __ubuf__ T* dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T* s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T* s1Next = src1Ptr + i * src1RowStride;

        RemOp<T, dstRowStride>::RemInstr(dstNext, s0Next, s1Next, tmpPtr, repeatTimes, validCols);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(
    const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1, const TileDataTmp& tmp)
{
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
        "Fix: TREM supports only float and int32 element types.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc0::DType>,
        "Fix: TREM type of dst must be same with src0 and src1.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TREM support only row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
    // Single-row tmp layout: [0, dstRowStride) candidate, [dstRowStride, dstRowStride+maskFloats)
    // packed bitmask (32-byte aligned), then an 8-element (32-byte) address buffer.
    unsigned maskBytes = CeilDivision(TileDataDst::RowStride, BIT_TO_BYTE);
    unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
    unsigned tmpRequiredCols = TileDataDst::RowStride + maskFloats + BIT_TO_BYTE;
    PTO_ASSERT(
        tmp.GetValidRow() >= 1 && tmp.GetValidCol() >= tmpRequiredCols,
        "Fix: TREM tmp tile must have at least 1 row and enough columns for candidate, mask and address buffer.");
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    TRemCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    if constexpr (
        PrecisionType == RemAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // HP scratch contract from custom/TFmodRemHp.hpp (vector divisor),
        // sized for the staged capacity (full vectors per row, >= 1)
        constexpr unsigned elemsPerRepeatHp = REPEAT_BYTE / sizeof(float);
        unsigned capVecs = dst.GetValidCol() / elemsPerRepeatHp;
        PTO_ASSERT(
            tmp.GetValidRow() >= 1 && tmp.GetValidCol() >= HpScratchRequiredColsV(capVecs > 0 ? capVecs : 1),
            "TREM HP: tmp tile too small for the high precision scratch contract (see docs/trems_hp_a2a3.md).");
    }
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TRem<
        PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, elementsPerRepeat, blockSizeElem,
        dstRowStride, src0RowStride, src1RowStride>(
        dst.data(), src0.data(), src1.data(), tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
