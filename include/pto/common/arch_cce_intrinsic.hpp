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

PTO_INTERNAL void pto_copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint16_t nBurst, uint16_t lenBurst,
                                        uint16_t srcGap, uint16_t dstGap)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_A5)
    copy_ubuf_to_ubuf(dst, src, nBurst, lenBurst, srcGap, dstGap);
#endif
}

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
using __cce_scalar::addr_cal_mode_t;
template <typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(__cb__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
                                      uint16_t srcStride, uint16_t dstStride)
{
#if defined(PTO_NPU_ARCH_A2A3)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, 0, false, addr_cal_mode_t(0));
#elif defined(PTO_NPU_ARCH_KIRINX90)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, false, addr_cal_mode_t(0));
#endif
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
template <bool Transpose, typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(__cb__ T *dst, __cbuf__ T *src, uint16_t mStartPosition, uint16_t kStartPosition,
                                      uint8_t mStep, uint8_t kStep, uint16_t srcStride, uint16_t dstStride)
{
    load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, Transpose);
}
#endif

#if defined(PTO_NPU_ARCH_A2A3)
template <typename T>
PTO_INTERNAL void pto_vgatherb(__ubuf__ T *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr, uint16_t dstRepeatStride,
                               uint8_t dstBlockStride, uint8_t repeat)
{
    vgatherb(dst, src, offsetAddr, dstRepeatStride, dstBlockStride, repeat);
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U, typename S>
PTO_INTERNAL void pto_vgatherb(T &dstReg, __ubuf__ U *base, S &idxReg, vector_bool &mask)
{
#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
    vgatherb(dstReg, base, idxReg);
#else
    vgatherb(dstReg, base, idxReg, mask);
#endif
}
#endif

template <typename T, typename U>
PTO_INTERNAL void pto_create_cbuf_matrix(__cbuf__ T *dst, int64_t repeatConfig, U value)
{
#if defined(PTO_NPU_ARCH_KIRIN9030)
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

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U, typename S>
PTO_INTERNAL void pto_vexpdif(T &dst, U &src0, U &src1, vector_bool mask, S part)
{
#if defined(PTO_NPU_ARCH_A5)
    vexpdif(dst, src0, src1, mask, part);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
    vsub(dst, src0, src1, mask, MODE_ZEROING);
    vexp(dst, dst, mask, MODE_ZEROING);
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_ubuf_align_v2(__ubuf__ T *dst, __gm__ T *src, uint8_t sid, uint32_t nBurst,
                                               uint32_t lenBurst, uint8_t leftPaddingCount, uint8_t rightPaddingCount,
                                               bool constantPaddingCtl, uint8_t l2CacheCtl, uint64_t burstSrcStride,
                                               uint32_t burstDstStride)
{
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
                             l2CacheCtl, burstSrcStride, burstDstStride);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, lenBurst, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
                             burstSrcStride, burstDstStride);
#endif
}
#endif

template <TileType type>
PTO_INTERNAL void pto_set_tload_pad_val(uint64_t config)
{
    if constexpr (type == TileType::Vec) {
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRINX90)
        set_mov_pad_val(config);
#elif defined(PTO_NPU_ARCH_KIRIN9030)
        set_pad_val_outtoub(config);
#endif
    } else if constexpr (type == TileType::Mat) {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
        set_pad_val_outtol1(config);
#endif
    }
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_multi_nd2nz(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint64_t loop1SrcStride,
                                                  uint8_t l2CacheCtl, uint16_t nValue, uint32_t dValue,
                                                  uint64_t loop4SrcStride, bool smallc0En = false)
{
    using U = std::conditional_t<sizeof(T) == sizeof(uint8_t), uint8_t,
                                 std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ U *>(dst), reinterpret_cast<__gm__ U *>(src), sid,
                                loop1SrcStride, l2CacheCtl, nValue, dValue, loop4SrcStride, smallc0En);
#elif defined(PTO_NPU_ARCH_KIRIN9030)
    copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ U *>(dst), reinterpret_cast<__gm__ U *>(src), sid,
                                loop1SrcStride, nValue, dValue, loop4SrcStride, smallc0En, false /* antiq_en */);
#endif
}
#elif defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_multi_nd2nz(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint16_t ndNum,
                                                  uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride,
                                                  uint16_t srcDValue, uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                                  uint16_t dstNzMatrixStride)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                                       dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(T) == sizeof(uint16_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                                        dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
#if defined(PTO_NPU_ARCH_A2A3)
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                                         dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        uint16_t dValueb64 = dValue * sizeof(T) / sizeof(uint32_t);
        uint16_t srcDValueb64 = srcDValue * sizeof(T) / sizeof(uint32_t);
        copy_gm_to_cbuf_multi_nd2nz_b32s(
            reinterpret_cast<__cbuf__ uint32_t *>(dst), reinterpret_cast<__gm__ uint32_t *>(src), sid, ndNum, nValue,
            dValueb64, srcNdMatrixStride, srcDValueb64, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
#elif defined(PTO_NPU_ARCH_KIRINX90)
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        uint16_t dValueb32 = dValue * sizeof(T) / sizeof(uint16_t);
        uint16_t srcDValueb32 = srcDValue * sizeof(T) / sizeof(uint16_t);
        copy_gm_to_cbuf_multi_nd2nz_b16(
            reinterpret_cast<__cbuf__ uint16_t *>(dst), reinterpret_cast<__gm__ uint16_t *>(src), sid, ndNum, nValue,
            dValueb32, srcNdMatrixStride, srcDValueb32, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        uint16_t dValueb64 = dValue * sizeof(T) / sizeof(uint64_t);
        uint16_t srcDValueb64 = srcDValue * sizeof(T) / sizeof(uint64_t);
        copy_gm_to_cbuf_multi_nd2nz_b16(
            reinterpret_cast<__cbuf__ uint16_t *>(dst), reinterpret_cast<__gm__ uint16_t *>(src), sid, ndNum, nValue,
            dValueb64, srcNdMatrixStride, srcDValueb64, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
#endif
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
template <typename T>
PTO_INTERNAL void pto_copy_gm_to_cbuf_align_v2(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint32_t nBurst,
                                               uint32_t lenBurst, uint8_t leftPaddingCount, uint8_t rightPaddingCount,
                                               bool dataSelectBit, uint8_t l2CacheCtl, uint64_t burstSrcStride,
                                               uint32_t burstDstStride)
{
    using U = std::conditional_t<sizeof(T) == sizeof(uint8_t), uint8_t,
                                 std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
#if defined(PTO_NPU_ARCH_A5)
    copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ U *>(dst), reinterpret_cast<__gm__ U *>(src), sid, nBurst,
                             lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit, l2CacheCtl, burstSrcStride,
                             burstDstStride);
#elif defined(PTO_NPU_ARCH_KIRIN9030)
    copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ U *>(dst), reinterpret_cast<__gm__ U *>(src), sid, nBurst,
                             lenBurst, leftPaddingCount, rightPaddingCount, dataSelectBit, burstSrcStride,
                             burstDstStride);
#endif
}
#endif

} // namespace pto

#endif // __CPU_SIM
#endif // ARCH_CCE_INTRINSIC_HPP
