/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Implementation of interface adaptation layer for device-side and cloud-side compatibility
#ifndef ARCH_CCE_INTRINSIC_HPP
#define ARCH_CCE_INTRINSIC_HPP
#include <pto/common/arch_macro.hpp>
#ifndef __CPU_SIM

namespace pto {

PTO_INTERNAL void pto_copy_ubuf_to_ubuf(
    __ubuf__ void* dst, __ubuf__ void* src, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_A5) || \
    defined(PTO_NPU_ARCH_A6)
    copy_ubuf_to_ubuf(dst, src, nBurst, lenBurst, srcGap, dstGap);
#endif
}

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
using __cce_scalar::addr_cal_mode_t;
template <typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(
    __cb__ T* dst, __cbuf__ T* src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride, uint16_t dstStride)
{
#if defined(PTO_NPU_ARCH_A2A3)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, 0, false, addr_cal_mode_t(0));
#elif defined(PTO_NPU_ARCH_KIRINX90)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, false, addr_cal_mode_t(0));
#endif
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
template <bool Transpose, typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(
    __cb__ T* dst, __cbuf__ T* src, uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep, uint8_t kStep,
    uint16_t srcStride, uint16_t dstStride)
{
    load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, Transpose);
}
#endif

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
using __cce_scalar::addr_cal_mode_t;
template <typename T>
PTO_INTERNAL void pto_load_cbuf_to_ca(
    __ca__ T* dst, __cbuf__ T* src, uint16_t baseIdx, uint8_t repeat, uint16_t srcStride, uint16_t dstStride)
{
#if defined(PTO_NPU_ARCH_A2A3)
    load_cbuf_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, 0, false, addr_cal_mode_t(0));
#elif defined(PTO_NPU_ARCH_KIRINX90)
    load_cbuf_to_ca(dst, src, baseIdx, repeat, srcStride, dstStride, false, addr_cal_mode_t(0));
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A2A3)
template <typename T>
PTO_INTERNAL void pto_vgatherb(
    __ubuf__ T* dst, __ubuf__ uint32_t* src, uint32_t offsetAddr, uint16_t dstRepeatStride, uint8_t dstBlockStride,
    uint8_t repeat)
{
    vgatherb(dst, src, offsetAddr, dstRepeatStride, dstBlockStride, repeat);
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U, typename S>
PTO_INTERNAL void pto_vgatherb(T& dstReg, __ubuf__ U* base, S& idxReg, vector_bool& mask)
{
#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
    vgatherb(dstReg, base, idxReg);
#else
    vgatherb(dstReg, base, idxReg, mask);
#endif
}
#endif

template <typename T, typename U>
PTO_INTERNAL void pto_create_cbuf_matrix(__cbuf__ T* dst, int64_t repeatConfig, U value)
{
#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRIN9050)
    if constexpr (std::is_integral_v<U>) {
        set_l0_set_value_ui(value);
    } else if (std::is_same_v<U, half>) {
        set_l0_set_value_h(value);
    }
    set_l1_2d(dst, repeatConfig);
#else
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        create_cbuf_matrix_bf16(dst, repeatConfig, value);
    } else {
        create_cbuf_matrix(dst, repeatConfig, value);
    }
#endif
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U, typename S>
PTO_INTERNAL void pto_vexpdif(T& dst, U& src0, U& src1, vector_bool mask, S part)
{
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
    vexpdif(dst, src0, src1, mask, part);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
    vsub(dst, src0, src1, mask, MODE_ZEROING);
    vexp(dst, dst, mask, MODE_ZEROING);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_ubuf_align_v2(
    __ubuf__ T* dst, __gm__ T* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t leftPaddingCount,
    uint8_t rightPaddingCount, bool constantPaddingCtl, uint8_t l2CacheCtl, uint64_t burstSrcStride,
    uint32_t burstDstStride)
{
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_ubuf_align_v2(
        dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl, l2CacheCtl,
        burstSrcStride, burstDstStride);
#elif defined(PTO_NPU_ARCH_A6)
    copy_gm_to_ubuf_align_v2(
        dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
        false /* pre_allocation*/, l2CacheCtl, burstSrcStride, burstDstStride, false /* non_eod_ctrl*/);

#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_gm_to_ubuf_align_v2(
        dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl, burstSrcStride,
        burstDstStride);
#endif
}
#endif

template <TileType type>
PTO_INTERNAL void pto_set_tload_pad_val(uint64_t config)
{
    if constexpr (type == TileType::Vec) {
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRINX90)
        set_mov_pad_val(config);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050)
        set_pad_val_outtoub(config);
#endif
    } else if constexpr (type == TileType::Mat) {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
        set_pad_val_outtol1(config);
#endif
    }
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_multi_nd2nz(
    __cbuf__ T* dst, __gm__ T* src, uint8_t sid, uint64_t loop1SrcStride, uint8_t l2CacheCtl, uint16_t nValue,
    uint32_t dValue, uint64_t loop4SrcStride, bool smallc0En = false)
{
    using U = std::conditional_t<
        sizeof(T) == sizeof(uint8_t), uint8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_cbuf_multi_nd2nz(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, loop1SrcStride, l2CacheCtl, nValue,
        dValue, loop4SrcStride, smallc0En);
#elif defined(PTO_NPU_ARCH_A6)
    copy_gm_to_cbuf_multi_nd2nz(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, loop1SrcStride, l2CacheCtl, nValue,
        dValue, loop4SrcStride, smallc0En, false /* enablePreReadIgnoreSync */);
#elif defined(PTO_NPU_ARCH_KIRIN9030)
    copy_gm_to_cbuf_multi_nd2nz(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, loop1SrcStride, nValue, dValue,
        loop4SrcStride, smallc0En, false /* antiq_en */);
#elif defined(PTO_NPU_ARCH_KIRIN9050)
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ half*>(dst), reinterpret_cast<__gm__ half*>(src), sid, loop1SrcStride, nValue,
            dValue, loop4SrcStride);
    } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ half*>(dst), reinterpret_cast<__gm__ half*>(src), sid, loop1SrcStride, nValue,
            dValue * 2, loop4SrcStride);
    } else {
        copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, loop1SrcStride, nValue, dValue,
            loop4SrcStride);
    }
#endif
}
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_multi_nd2nz(
    __cbuf__ T* dst, __gm__ T* src, uint8_t sid, uint16_t ndNum, uint16_t nValue, uint16_t dValue,
    uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride, uint16_t dstNzNStride,
    uint16_t dstNzMatrixStride)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b8(
            dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride,
            dstNzMatrixStride);
    } else if constexpr (sizeof(T) == sizeof(uint16_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b16(
            dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride,
            dstNzMatrixStride);
    }
#if defined(PTO_NPU_ARCH_A2A3)
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b32s(
            dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride,
            dstNzMatrixStride);
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        uint16_t dValueb64 = dValue * sizeof(T) / sizeof(uint32_t);
        uint16_t srcDValueb64 = srcDValue * sizeof(T) / sizeof(uint32_t);
        copy_gm_to_cbuf_multi_nd2nz_b32s(
            reinterpret_cast<__cbuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src), sid, ndNum, nValue,
            dValueb64, srcNdMatrixStride, srcDValueb64, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
#elif defined(PTO_NPU_ARCH_KIRINX90)
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        constexpr uint32_t C0_ELEMENTS_B16 = 32 / sizeof(uint16_t);
        uint32_t dValueB16 = uint32_t(dValue) * 2;
        uint32_t srcDValueB16 = uint32_t(srcDValue) * 2;
        uint32_t srcMatrixStrideB16 = uint32_t(srcNdMatrixStride) * 2;
        uint32_t dstMatrixStrideB16 = uint32_t(dstNzMatrixStride) * 2;
        if (dValueB16 <= UINT16_MAX && srcDValueB16 <= UINT16_MAX &&
            (ndNum == 1 || (srcMatrixStrideB16 <= UINT16_MAX && dstMatrixStrideB16 <= UINT16_MAX))) {
            copy_gm_to_cbuf_multi_nd2nz_b16(
                reinterpret_cast<__cbuf__ uint16_t*>(dst), reinterpret_cast<__gm__ uint16_t*>(src), sid, ndNum, nValue,
                static_cast<uint16_t>(dValueB16), static_cast<uint16_t>(srcMatrixStrideB16),
                static_cast<uint16_t>(srcDValueB16), dstNzC0Stride, dstNzNStride,
                static_cast<uint16_t>(dstMatrixStrideB16));
            return;
        }
        // Split after converting element counts to the b16 instruction's units.
        constexpr uint32_t MAX_CHUNK_ELEMENTS_B16 = UINT16_MAX / C0_ELEMENTS_B16 * C0_ELEMENTS_B16;
        for (uint32_t matrix = 0; matrix < ndNum; ++matrix) {
            for (uint32_t row = 0; row < nValue; ++row) {
                auto srcRow = reinterpret_cast<__gm__ uint16_t*>(src) + uint64_t(matrix) * srcMatrixStrideB16 +
                              uint64_t(row) * srcDValueB16;
                auto dstRow = reinterpret_cast<__cbuf__ uint16_t*>(dst) + uint64_t(matrix) * dstMatrixStrideB16 +
                              uint64_t(row) * dstNzNStride * C0_ELEMENTS_B16;
                for (uint32_t column = 0; column < dValueB16; column += MAX_CHUNK_ELEMENTS_B16) {
                    uint32_t chunkElementsB16 = dValueB16 - column;
                    if (chunkElementsB16 > MAX_CHUNK_ELEMENTS_B16) {
                        chunkElementsB16 = MAX_CHUNK_ELEMENTS_B16;
                    }
                    copy_gm_to_cbuf_multi_nd2nz_b16(
                        dstRow + uint64_t(column / C0_ELEMENTS_B16) * dstNzC0Stride * C0_ELEMENTS_B16, srcRow + column,
                        sid, 1, 1, static_cast<uint16_t>(chunkElementsB16), 0, 0, dstNzC0Stride, dstNzNStride, 0);
                }
            }
        }
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        uint16_t dValueb64 = dValue * sizeof(T) / sizeof(uint16_t);
        uint16_t srcDValueb64 = srcDValue * sizeof(T) / sizeof(uint16_t);
        copy_gm_to_cbuf_multi_nd2nz_b16(
            reinterpret_cast<__cbuf__ uint16_t*>(dst), reinterpret_cast<__gm__ uint16_t*>(src), sid, ndNum, nValue,
            dValueb64, srcNdMatrixStride, srcDValueb64, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_align_v2(
    __cbuf__ T* dst, __gm__ T* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t leftPaddingCount,
    uint8_t rightPaddingCount, bool dataSelectBit, uint8_t l2CacheCtl, uint64_t burstSrcStride, uint32_t burstDstStride)
{
    using U = std::conditional_t<
        sizeof(T) == sizeof(uint8_t), uint8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_cbuf_align_v2(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, nBurst, lenBurst, leftPaddingCount,
        rightPaddingCount, dataSelectBit, l2CacheCtl, burstSrcStride, burstDstStride);
#elif defined(PTO_NPU_ARCH_A6)
    uint64_t config0 =
        static_cast<uint64_t>(sid & 0xF) | (static_cast<uint64_t>(nBurst & 0x1FFFFF) << 4) |
        (static_cast<uint64_t>(lenBurst & 0x1FFFFF) << 25) | (static_cast<uint64_t>(leftPaddingCount & 0x3F) << 46) |
        (static_cast<uint64_t>(rightPaddingCount & 0x3F) << 52) | (static_cast<uint64_t>(dataSelectBit & 0x1) << 58) |
        (static_cast<uint64_t>(l2CacheCtl & 0xF) << 60);
    uint64_t config1 = (burstSrcStride & 0xFFFFFFFFFFULL) | (static_cast<uint64_t>(burstDstStride & 0x1FFFFF) << 40);
    copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), config0, config1);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050)
    copy_gm_to_cbuf_align_v2(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, nBurst, lenBurst, leftPaddingCount,
        rightPaddingCount, dataSelectBit, burstSrcStride, burstDstStride);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf(
    __cbuf__ T* dst, __gm__ T* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
    uint16_t dstStride)
{
    copy_gm_to_cbuf(dst, src, sid, nBurst, lenBurst, srcStride, dstStride, (pad_t)0);
}
#endif

#if defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_align(
    __cbuf__ T* dst, __gm__ T* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
    uint16_t dstStride)
{
    using U = std::conditional_t<
        sizeof(T) == sizeof(uint8_t), uint8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
    copy_gm_to_cbuf_align(
        reinterpret_cast<__cbuf__ U*>(dst), reinterpret_cast<__gm__ U*>(src), sid, nBurst, lenBurst, (uint8_t)0,
        (uint8_t)(dstStride / sizeof(T)), srcStride, dstStride >> SHIFT_BLOCK_BYTE);
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
template <typename T>
PTO_INTERNAL void pto_copy_ubuf_to_gm_align_v2(
    __gm__ T* dst, __ubuf__ T* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t l2CacheCtl,
    uint64_t burstDstStride, uint32_t burstSrcStride)
{
#if defined(PTO_NPU_ARCH_A5)
    copy_ubuf_to_gm_align_v2(dst, src, sid, nBurst, lenBurst, l2CacheCtl, burstDstStride, burstSrcStride);
#elif defined(PTO_NPU_ARCH_A6)
    copy_ubuf_to_gm_align_v2(
        dst, src, sid, nBurst, lenBurst, false /* rsw_ctrl */, false /*rsw_packet_ctrl*/, 0 /* rsw_buffer_size */,
        l2CacheCtl, burstDstStride, burstSrcStride, false /* non_eod_ctrl */);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050)
    copy_ubuf_to_gm_align_v2(dst, src, sid, nBurst, lenBurst, burstDstStride, burstSrcStride);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050)
// Byte-based L1->GM move, 8/16/32-bit.
template <typename T>
PTO_INTERNAL void pto_copy_cbuf_to_gm_align_v2(
    __gm__ T* dst, __cbuf__ T* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint64_t burstDstStride,
    uint32_t burstSrcStride)
{
    using U = std::conditional_t<
        sizeof(T) == sizeof(uint8_t), uint8_t, std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
    copy_cbuf_to_gm_align_v2(
        reinterpret_cast<__gm__ U*>(dst), reinterpret_cast<__cbuf__ U*>(src), sid, nBurst, lenBurst, burstDstStride,
        burstSrcStride);
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050)
template <typename T, typename U>
PTO_INTERNAL void pto_copy_matrix_cc_to_cbuf(
    __cbuf__ T* dst, __cc__ U* src, uint8_t sid, uint16_t nSize, uint16_t mSize, uint32_t dstStride, uint16_t srcStride,
    uint8_t l2CacheCtl, uint8_t clipReluPre, uint8_t unitFlagCtl, uint64_t quantPre, uint8_t reluPre, bool splitEn,
    bool nz2ndEn, uint64_t quantPost, uint8_t reluPost, bool clipReluPost, bool loopEnhanceEn, uint8_t eltwiseOp,
    bool eltwiseAntqEn, bool loopEnhanceMergeEn, bool c0PadEn, bool winoPostEn, bool broadcastEn, bool nz2dnEn)
{
#if defined(PTO_NPU_ARCH_A5)
    copy_matrix_cc_to_cbuf(
        dst, src, sid, nSize, mSize, dstStride, srcStride, l2CacheCtl, clipReluPre, unitFlagCtl, quantPre, reluPre,
        splitEn, nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn,
        loopEnhanceMergeEn, c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#elif defined(PTO_NPU_ARCH_A6)
    // A6 (dav-9201) drops the sid argument; position 7 (l2CacheCtl in A5) is a
    // bool quant_pre_rnd slot in the 9201 signature, so it is fixed to false.
    // The remaining parameters align one-to-one with the A5 layout.
    copy_matrix_cc_to_cbuf(
        dst, src, nSize, mSize, dstStride, srcStride, false, clipReluPre, unitFlagCtl, quantPre, reluPre, splitEn,
        nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn, loopEnhanceMergeEn,
        c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRIN9050)
    copy_matrix_cc_to_cbuf(
        dst, src, sid, nSize, mSize, dstStride, srcStride, clipReluPre, unitFlagCtl, quantPre, reluPre, splitEn,
        nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn, loopEnhanceMergeEn,
        c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#endif
}
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U>
PTO_INTERNAL void pto_copy_matrix_cc_to_cbuf(
    __cbuf__ T* dst, __cc__ U* src, uint8_t sid, uint16_t nSize, uint16_t mSize, uint32_t dstStride, uint16_t srcStride,
    uint8_t unitFlagMode, uint64_t quantPre, uint8_t reluPre, bool channelSplit, bool nz2ndEn)
{
#if defined(PTO_NPU_ARCH_A2A3)
    copy_matrix_cc_to_cbuf(
        dst, src, sid, nSize, mSize, dstStride, srcStride, unitFlagMode, quantPre, reluPre, channelSplit, nz2ndEn);
#elif defined(PTO_NPU_ARCH_KIRINX90)
    copy_matrix_cc_to_cbuf(
        dst, src, sid, nSize, mSize, dstStride, srcStride, 0 /* clipReluPre */, unitFlagMode, quantPre, reluPre,
        channelSplit, nz2ndEn, false /* c0PadEn */);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U>
PTO_INTERNAL void pto_copy_matrix_cc_to_ub(
    __ubuf__ T* dst, __cc__ U* src, uint8_t sid, uint16_t nSize, uint16_t mSize, uint32_t dstStride, uint16_t srcStride,
    uint8_t dualDstCtl, bool subBlockid, uint8_t clipReluPre, uint8_t unitFlagCtl, uint64_t quantPre, uint8_t reluPre,
    bool splitEn, bool nz2ndEn, uint64_t quantPost, uint8_t reluPost, bool clipReluPost, bool loopEnhanceEn,
    uint8_t eltwiseOp, bool eltwiseAntqEn, bool loopEnhanceMergeEn, bool c0PadEn, bool winoPostEn, bool broadcastEn,
    bool nz2dnEn)
{
#if defined(PTO_NPU_ARCH_A5)
    copy_matrix_cc_to_ub(
        dst, src, sid, nSize, mSize, dstStride, srcStride, dualDstCtl, subBlockid, clipReluPre, unitFlagCtl, quantPre,
        reluPre, splitEn, nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn,
        loopEnhanceMergeEn, c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#elif defined(PTO_NPU_ARCH_A6)
    // A6 (dav-9201) copy_matrix_cc_to_ub has no dualDstCtl/subBlockid control:
    // the platform wrapper hard-fills positions 7-9 with 0/false/false and its
    // 3510-signature compat layer drops dual_dst_ctl/sub_blockid entirely.
    // Position 9 is the quant_pre_rnd slot (false = default rounding).
    copy_matrix_cc_to_ub(
        dst, src, nSize, mSize, dstStride, srcStride, 0, false, false, clipReluPre, unitFlagCtl, quantPre, reluPre,
        splitEn, nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn,
        loopEnhanceMergeEn, c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#elif defined(PTO_NPU_ARCH_KIRIN9030)
    copy_matrix_cc_to_ub(
        dst, src, sid, nSize, mSize, dstStride, srcStride, clipReluPre, unitFlagCtl, quantPre, reluPre, splitEn,
        nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn, loopEnhanceMergeEn,
        c0PadEn, winoPostEn, broadcastEn, nz2dnEn);
#elif defined(PTO_NPU_ARCH_KIRINX90)
    copy_matrix_cc_to_ub(
        dst, src, sid, nSize, mSize, dstStride, srcStride, clipReluPre, unitFlagCtl, quantPre, reluPre, splitEn,
        nz2ndEn, quantPost, reluPost, clipReluPost, loopEnhanceEn, eltwiseOp, eltwiseAntqEn, loopEnhanceMergeEn,
        c0PadEn, winoPostEn);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_KIRIN9050)
__tf__ PTO_INTERNAL void pto_copy_cbuf_to_ubuf(
    __ubuf__ void* dst, __cbuf__ void* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;

    uint32_t totalBytes = lenBurst * CBUF_UB_BURST_UNIT;
    uint64_t loop2DstStride = static_cast<uint64_t>((lenBurst + dstGap) * CBUF_UB_BURST_UNIT);
    uint32_t loop3Size = totalBytes;
    uint16_t loop2Size = nBurst;

    fix_cbuf_to_ubuf(
        reinterpret_cast<__ubuf__ uint8_t*>(dst), reinterpret_cast<__cbuf__ uint8_t*>(src), loop2DstStride, loop3Size,
        fixp_trans_mode_t::NORMAL_DMA, static_cast<uint64_t>(0), loop2Size);
}
#elif !defined(__COSTMODEL)
PTO_INTERNAL void pto_copy_cbuf_to_ubuf(
    __ubuf__ void* dst, __cbuf__ void* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    copy_cbuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap);
}
#endif

} // namespace pto

#if defined(PTO_NPU_ARCH_A6)
// A5-signature compat overload for the 13-arg dav-9201 copy_gm_to_ubuf_align_v2.
// pypto codegen emits the 11-arg A5 form directly (it bypasses the pto wrapper),
// which the A6 builtin rejects at compile time; forward the extra controls
// (pre_allocation/non_eod_ctrl) as false, mirroring pto_copy_gm_to_ubuf_align_v2.
template <typename T>
__aicore__ inline void copy_gm_to_ubuf_align_v2(
    __ubuf__ T* dst, __gm__ T* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t leftPaddingCount,
    uint8_t rightPaddingCount, bool constantPaddingCtl, uint8_t l2CacheCtl, uint64_t burstSrcStride,
    uint32_t burstDstStride)
{
    copy_gm_to_ubuf_align_v2(
        dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
        false /* pre_allocation */, l2CacheCtl, burstSrcStride, burstDstStride, false /* non_eod_ctrl */);
}
#endif

#endif // __CPU_SIM
#endif // ARCH_CCE_INTRINSIC_HPP
