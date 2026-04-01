/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCI_HPP
#define TCI_HPP

#include <pto/common/constants.hpp>

namespace pto {
template <typename TileData, typename T>
PTO_INTERNAL void CheckValid()
{
    static_assert((std::is_same<typename TileData::DType, T>::value), "Fix: TCI expect src and dst same datatype");
    static_assert((sizeof(typename TileData::DType) == 4 || (sizeof(typename TileData::DType) == 2)),
                  "Fix: TCI expect datatype is b32 or b16");
    static_assert((TileData::Cols != 1), "Fix: TCI expect tile row is 1");
}

template <typename TileData, typename T, int descending>
__tf__ AICORE void TCI(typename TileData::TileDType __out__ dst, T start, unsigned validCol)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

    // scalar
    if constexpr (descending) {
        for (int32_t i = 0; i < validCol; i++) {
            *(dstPtr + i) = start - i;
        }
    } else {
        for (int32_t i = 0; i < validCol; i++) {
            *(dstPtr + i) = start + i;
        }
    }
}

template <typename TileData, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start)
{
    CheckValid<TileData, T>();

    unsigned validCol = dst.GetValidCol();

    TCI<TileData, T, descending>(dst.data(), start, validCol);
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
__tf__ AICORE void TCI_b32_repeat(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp,
                                  T S, unsigned validCol, unsigned numRepeatPerLine, unsigned numRemainPerLine)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ float *tmp0 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp);
    __ubuf__ float *tmp1 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 128);

    set_mask_count();
    set_vector_mask(0, 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vector_dup(tmp0 + i * 8, (float)float(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 64);
    vcgadd((__ubuf__ float *)tmp1, tmp0, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 8);
    vmuls(tmp0, (__ubuf__ float *)tmp1, 8.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    vbrcb((__ubuf__ uint32_t *)(tmp0), (__ubuf__ uint32_t *)(tmp0), 1, 8, 1);
    pipe_barrier(PIPE_V);
    vadd((__ubuf__ float *)tmp1, (__ubuf__ float *)tmp1, tmp0, 8, 1, 0, 1, 8, 0, 8);
    pipe_barrier(PIPE_V);
    vconv_f322s32r((__ubuf__ int32_t *)tmp1, (__ubuf__ float *)tmp1, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
#pragma unroll
    for (int i = 0; i < numRepeatPerLine; i++) {
        vadds((__ubuf__ int32_t *)(dst + 64 * i), (__ubuf__ int32_t *)tmp1, S + 64 * i, 1, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    if (numRemainPerLine) {
        set_mask_norm();
        SetContinuousMask(numRemainPerLine);
        vadds((__ubuf__ int32_t *)(dst + 64 * numRepeatPerLine), (__ubuf__ int32_t *)tmp1, S + 64 * numRepeatPerLine, 1,
              1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    if (descending) {
        set_mask_count();
        set_vector_mask(0, validCol);
        vmuls((__ubuf__ int32_t *)dst, (__ubuf__ int32_t *)dst, -1, 1, 1, 1, 8, 8);
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
__tf__ AICORE void TCI_b32_normal(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp,
                                  T S, unsigned validCol, unsigned numRepeatPerLine, unsigned numRemainPerLine)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

    __ubuf__ float *tmp1 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp);
    __ubuf__ float *tmp2 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 128);

    set_mask_count();
    set_vector_mask(0, 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vector_dup(tmp1 + i * 8, (float)float(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 64);
    vcgadd((__ubuf__ float *)tmp2, tmp1, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 8);
    vmuls(tmp1, (__ubuf__ float *)tmp2, 8.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    vbrcb((__ubuf__ uint32_t *)(tmp1), (__ubuf__ uint32_t *)(tmp1), 1, 8, 1);
    pipe_barrier(PIPE_V);
    set_mask_count();
    set_vector_mask(0, validCol);
    vadd((__ubuf__ float *)tmp2, (__ubuf__ float *)tmp2, tmp1, 8, 1, 0, 1, 8, 0, 8);
    pipe_barrier(PIPE_V);
    vconv_f322s32r((__ubuf__ int32_t *)dst, (__ubuf__ float *)tmp2, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds((__ubuf__ int32_t *)dst, (__ubuf__ int32_t *)dst, S, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    if (descending) {
        vmuls((__ubuf__ int32_t *)dst, (__ubuf__ int32_t *)dst, -1, 1, 1, 1, 8, 8);
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
__tf__ AICORE void TCI_b16_repeat(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp,
                                  T S, unsigned validCol, unsigned numRepeatPerLine, unsigned numRemainPerLine)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

    __ubuf__ float *tmp0 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp);
    __ubuf__ float *tmp1 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 128);
    __ubuf__ half *tmp2 =
        reinterpret_cast<__ubuf__ half *>((__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 256));
    __ubuf__ half *tmp3 =
        reinterpret_cast<__ubuf__ half *>((__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 384));

    set_mask_count();
    set_vector_mask(0, 8);
    for (int i = 0; i < 8; i++) {
        vector_dup(tmp0 + i * 8, (float)float(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
        vector_dup(tmp0 + 64 + i * 8, (float)float(i + 8) * 0.125f, 1, 1, 1, 1, (int64_t)0);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 64);
    vcgadd((__ubuf__ float *)tmp1, tmp0, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vcgadd((__ubuf__ float *)tmp1 + 8, tmp0 + 64, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16r((__ubuf__ half *)tmp2, (__ubuf__ float *)tmp1, 1, 1, 1, 8, 8);

    pipe_barrier(PIPE_V);
    set_vector_mask(0, 8);
    vmuls(tmp3, (__ubuf__ half *)tmp2, (half)16.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    vbrcb((__ubuf__ uint16_t *)(tmp3), (__ubuf__ uint16_t *)(tmp3), 1, 8, 1);
    pipe_barrier(PIPE_V);

    set_vector_mask(0, 65535);
    for (int i = 0; i < 8; i++) {
        vadd((__ubuf__ half *)(tmp3 + i * 16), (__ubuf__ half *)(tmp3 + i * 16), tmp2, 1, 1, 0, 1, 8, 0, 8);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    vconv_f162s16r((__ubuf__ int16_t *)tmp3, (__ubuf__ half *)tmp3, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    for (int i = 0; i < numRepeatPerLine; i++) {
        vadds((__ubuf__ int16_t *)(dst + 128 * i), (__ubuf__ int16_t *)tmp3, S + 128 * i, 1, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    if (numRemainPerLine) {
        SetContinuousMask(numRemainPerLine);
        vadds((__ubuf__ int16_t *)(dst + 128 * numRepeatPerLine), (__ubuf__ int16_t *)tmp3, S + 128 * numRepeatPerLine,
              1, 1, 1, 8, 8);
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
__tf__ AICORE void TCI_b16_normal(typename TileData::TileDType __out__ dst, typename TileDataTmp::TileDType __in__ tmp,
                                  T S, unsigned validCol, unsigned numRepeatPerLine, unsigned numRemainPerLine)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

    __ubuf__ float *tmp1 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp);
    __ubuf__ float *tmp2 = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 128);
    __ubuf__ half *tmp3 =
        reinterpret_cast<__ubuf__ half *>((__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 196));
    __ubuf__ half *tmp4 =
        reinterpret_cast<__ubuf__ half *>((__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp + 260));

    set_mask_count();
    set_vector_mask(0, 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vector_dup(tmp1 + i * 8, (float)float(i) * 0.125f, 1, 1, 1, 1, (int64_t)0);
        vector_dup(tmp1 + 64 + i * 8, (float)float(i + 8) * 0.125f, 1, 1, 1, 1, (int64_t)0);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 64);
    vcgadd((__ubuf__ float *)tmp2, tmp1, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vcgadd((__ubuf__ float *)tmp2 + 8, tmp1 + 64, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16r((__ubuf__ half *)tmp3, (__ubuf__ float *)tmp2, 1, 1, 1, 8, 8);

    pipe_barrier(PIPE_V);
    set_vector_mask(0, 8);
    vmuls(tmp4, (__ubuf__ half *)tmp3, (half)16.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    vbrcb((__ubuf__ uint16_t *)(tmp4), (__ubuf__ uint16_t *)(tmp4), 1, 8, 1);
    pipe_barrier(PIPE_V);

    set_vector_mask(0, 65535);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        vadd((__ubuf__ half *)(tmp4 + i * 16), (__ubuf__ half *)(tmp4 + i * 16), tmp3, 1, 1, 0, 1, 8, 0, 8);
    }
    pipe_barrier(PIPE_V);
    set_mask_count();
    set_vector_mask(0, validCol);
    vconv_f162s16r((__ubuf__ int16_t *)dst, (__ubuf__ half *)tmp4, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds((__ubuf__ int16_t *)dst, (__ubuf__ int16_t *)dst, S, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    if (descending) {
        vmuls((__ubuf__ int16_t *)dst, (__ubuf__ int16_t *)dst, -1, 1, 1, 1, 8, 8);
    }
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start, TileDataTmp &tmp)
{
    CheckValid<TileData, T>();

    unsigned validCol = dst.GetValidCol();
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    unsigned numRepeatPerLine = validCol / elementsPerRepeat;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;

    if (sizeof(typename TileData::DType) == 4 && numRepeatPerLine) {
        TCI_b32_repeat<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol, numRepeatPerLine,
                                                             numRemainPerLine);
    } else if (sizeof(typename TileData::DType) == 4) {
        TCI_b32_normal<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol, numRepeatPerLine,
                                                             numRemainPerLine);
    } else if (sizeof(typename TileData::DType) == 2 && numRepeatPerLine) {
        TCI_b16_repeat<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol, numRepeatPerLine,
                                                             numRemainPerLine);
        pipe_barrier(PIPE_V);
        if (descending) {
            TMULS_IMPL(dst, dst, -1);
        }
    } else {
        TCI_b16_normal<TileData, TileDataTmp, T, descending>(dst.data(), tmp.data(), start, validCol, numRepeatPerLine,
                                                             numRemainPerLine);
    }
}
} // namespace pto
#endif
