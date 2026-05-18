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
    set_l0_set_value_ui((uint32_t)value);
    set_l1_2d(dst, repeatConfig);
#else
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        create_cbuf_matrix_bf16(dst, repeatConfig, value);
    } else {
        create_cbuf_matrix(dst, repeatConfig, value);
    }
#endif
}
#endif // __CPU_SIM
#endif // ARCH_CCE_INTRINSIC_HPP