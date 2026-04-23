/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOW_HPP
#define TPOW_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"
#include "pto/npu/a2a3/TBinSOp.hpp"

namespace pto {
PTO_INTERNAL bool IsInteger(float f) noexcept
{
    FloatUnion converter;
    converter.f = f;
    const uint32_t bits = converter.i;
    const uint32_t exp = (bits >> 23) & 0xFF;

    // NaN or inf
    if (exp == 0xFF)
        return false;

    // exp is 0：处理 ±0 和次正规数（只有 ±0 是整数）
    if (exp == 0)
        return (bits & 0x7FFFFFFF) == 0; // 清除符号位后全零

    // 正规数
    const int32_t e = static_cast<int32_t>(exp) - 127;
    // |f| < 1 且非零 → 不是整数
    if (e < 0)
        return false;
    // 指数 ≥ 23 → 所有值都是整数（尾数已无法表示小数部分）
    if (e >= 23)
        return true;

    // 检查尾数的小数部分是否为零
    const uint32_t frac = bits & 0x7FFFFF;
    const uint32_t mask = (1u << (23 - e)) - 1;
    return (frac & mask) == 0;
}

PTO_INTERNAL bool IsInf(float f) noexcept
{
    FloatUnion converter;
    converter.f = f;
    return ((converter.i & 0x7FFFFFFF) == 0x7F800000);
}

PTO_INTERNAL int TPowICore(int base, int exp)
{
    if (exp == 0) {
        return 1; // 0^0 also returns 1
    } else if (base == 0) {
        return 0; // 0^positive = 0
    } else if (base == 1) {
        return 1;
    } else if (base == -1) {
        // (-1)^even = 1, (-1)^odd = -1
        return (exp & 1) ? -1 : 1;
    } else if (exp < 0) {
        return 0; // PyTorch behavior: negative integer exponent is not allowed for integer base
    }

    int result = 1;
    while (exp != 0) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride>
PTO_INTERNAL void TPowI(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, unsigned validRow, unsigned validCol)
{
    for (uint32_t i = 0; i < validRow; ++i) {
        for (uint32_t j = 0; j < validCol; ++j) {
            dst[i * DstStride + j] = TPowICore(base[i * BaseStride + j], exp[i * ExpStride + j]);
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride>
PTO_INTERNAL void TPowI(__ubuf__ T *dst, __ubuf__ T *base, T exp, unsigned validRow, unsigned validCol)
{
    for (uint32_t i = 0; i < validRow; ++i) {
        for (uint32_t j = 0; j < validCol; ++j) {
            dst[i * DstStride + j] = TPowICore(base[i * BaseStride + j], exp);
        }
    }
}

PTO_INTERNAL void SwitchPowFResult(__ubuf__ float *dst, float base, float exp, float tmp)
{
    if (exp == 0.0f) {
        *dst = 1.0f;
    } else if (base == 0.0f) {
        *dst = (exp > 0.0f) ? 0.0f : (1.0f / 0.0f);
    } else if (base == -1.0f && IsInf(exp)) {
        *dst = 1.0f;
    } else if (IsInteger(exp)) {
        if (static_cast<int>(exp) % 2 != 0 && base < 0.0f) {
            *dst = -tmp;
        } else {
            *dst = tmp;
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride, uint32_t TmpStride>
PTO_INTERNAL void ProcessSpecialCaseForPowF(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, __ubuf__ T *tmp,
                                            unsigned validRow, unsigned validCol)
{
    for (uint16_t i = 0; i < validRow; ++i) {
        for (uint16_t j = 0; j < validCol; ++j) {
            SwitchPowFResult(dst + i * DstStride + j, *(base + i * BaseStride + j), *(exp + i * ExpStride + j),
                             *(tmp + i * TmpStride + j));
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t TmpStride>
PTO_INTERNAL void ProcessSpecialCaseForPowF(__ubuf__ T *dst, __ubuf__ T *base, T exp, __ubuf__ T *tmp,
                                            unsigned validRow, unsigned validCol)
{
    for (uint16_t i = 0; i < validRow; ++i) {
        for (uint16_t j = 0; j < validCol; ++j) {
            SwitchPowFResult(dst + i * DstStride + j, *(base + i * BaseStride + j), exp, *(tmp + i * TmpStride + j));
        }
    }
}

template <typename T, bool NeedAbs>
struct PowOp {
    PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, uint8_t repeats,
                                      uint8_t dstStride = 8, uint8_t baseStride = 8, uint8_t expStride = 8)
    {
        if constexpr (NeedAbs) {
            vabs(dst, base, repeats, 1, 1, dstStride, baseStride);
            pipe_barrier(PIPE_V);
            vln(dst, dst, repeats, 1, 1, dstStride, dstStride);
        } else {
            vln(dst, base, repeats, 1, 1, dstStride, baseStride);
        }

        pipe_barrier(PIPE_V);
        vmul(dst, dst, exp, repeats, 1, 1, 1, dstStride, dstStride, expStride);
        pipe_barrier(PIPE_V);
        vexp(dst, dst, repeats, 1, 1, dstStride, dstStride);
    }
};

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride, uint32_t TmpStride>
PTO_INTERNAL void TPowF(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, __ubuf__ T *tmp, unsigned validRow,
                        unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    BinaryInstr<PowOp<T, true>, T, elementsPerRepeat, blockSizeElem, TmpStride, BaseStride, ExpStride>(
        tmp, base, exp, validRow, validCol);
    BinaryInstr<PowOp<T, false>, T, elementsPerRepeat, blockSizeElem, DstStride, BaseStride, ExpStride>(
        dst, base, exp, validRow, validCol);
    PtoSetWaitFlag<PIPE_V, PIPE_S>(EVENT_ID7, EVENT_ID7);
    ProcessSpecialCaseForPowF<T, DstStride, BaseStride, ExpStride, TmpStride>(dst, base, exp, tmp, validRow, validCol);
    PtoSetWaitFlag<PIPE_S, PIPE_V>(EVENT_ID7, EVENT_ID7);
}

template <typename T, bool NeedAbs>
struct PowSOp {
    PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *base, T exp, uint8_t repeats, uint8_t dstStride = 8,
                                       uint8_t baseStride = 8)
    {
        if constexpr (NeedAbs) {
            vabs(dst, base, repeats, 1, 1, dstStride, baseStride);
            pipe_barrier(PIPE_V);
            vln(dst, dst, repeats, 1, 1, dstStride, dstStride);
        } else {
            vln(dst, base, repeats, 1, 1, dstStride, baseStride);
        }

        pipe_barrier(PIPE_V);
        vmuls(dst, dst, exp, repeats, 1, 1, dstStride, dstStride);
        pipe_barrier(PIPE_V);
        vexp(dst, dst, repeats, 1, 1, dstStride, dstStride);
    }
};

template <typename T, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPowF(__ubuf__ T *dst, __ubuf__ T *base, T exp, __ubuf__ T *tmp, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    TBinSInstr<PowSOp<T, true>, TmpTile, BaseTile, elementsPerRepeat, blockSizeElem, TmpTile::RowStride,
               BaseTile::RowStride>(tmp, base, exp, validRow, validCol);

    TBinSInstr<PowSOp<T, false>, DstTile, BaseTile, elementsPerRepeat, blockSizeElem, DstTile::RowStride,
               BaseTile::RowStride>(dst, base, exp, validRow, validCol);

    PtoSetWaitFlag<PIPE_V, PIPE_S>(EVENT_ID7, EVENT_ID7);
    ProcessSpecialCaseForPowF<T, DstTile::RowStride, BaseTile::RowStride, TmpTile::RowStride>(dst, base, exp, tmp,
                                                                                              validRow, validCol);
    PtoSetWaitFlag<PIPE_S, PIPE_V>(EVENT_ID7, EVENT_ID7);
}

template <typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
__tf__ PTO_INTERNAL void TPow(typename DstTile::TileDType __out__ dstData, typename BaseTile::TileDType __in__ baseData,
                              typename ExpTile::TileDType __in__ expData, typename TmpTile::TileDType __in__ tmpData,
                              unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;

    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *base = (__ubuf__ T *)__cce_get_tile_ptr(baseData);
    __ubuf__ T *exp = (__ubuf__ T *)__cce_get_tile_ptr(expData);

    if constexpr (std::is_integral_v<T>) {
        PtoSetWaitFlag<PIPE_V, PIPE_S>(EVENT_ID7, EVENT_ID7);
        TPowI<T, DstTile::RowStride, BaseTile::RowStride, ExpTile::RowStride>(dst, base, exp, validRow, validCol);
        PtoSetWaitFlag<PIPE_S, PIPE_V>(EVENT_ID7, EVENT_ID7);
    } else {
        __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
        TPowF<T, DstTile::RowStride, BaseTile::RowStride, ExpTile::RowStride, TmpTile::RowStride>(dst, base, exp, tmp,
                                                                                                  validRow, validCol);
    }
}

template <typename DstTile, typename BaseTile, typename ExpTile>
PTO_INTERNAL void PowCheckType()
{
    static_assert(DstTile::Loc == TileType::Vec && BaseTile::Loc == TileType::Vec && ExpTile::Loc == TileType::Vec,
                  "TPOW: TileType of dst, base and exp tiles must be TileType::Vec.");
    static_assert(DstTile::ValidCol <= DstTile::Cols,
                  "TPOW: Number of dst's valid columns must not be greater than number of tile columns.");
    static_assert(DstTile::ValidRow <= DstTile::Rows,
                  "TPOW: Number of dst's valid rows must not be greater than number of tile rows.");
    static_assert(BaseTile::ValidCol <= BaseTile::Cols,
                  "TPOW: Number of base's valid columns must not be greater than number of tile columns.");
    static_assert(BaseTile::ValidRow <= BaseTile::Rows,
                  "TPOW: Number of base's valid rows must not be greater than number of tile rows.");
    static_assert(ExpTile::ValidCol <= ExpTile::Cols,
                  "TPOW: Number of exp's valid columns must not be greater than number of tile columns.");
    static_assert(ExpTile::ValidRow <= ExpTile::Rows,
                  "TPOW: Number of exp's valid rows must not be greater than number of tile rows.");
    static_assert(DstTile::isRowMajor && BaseTile::isRowMajor && ExpTile::isRowMajor,
                  "TPOW: Not supported Layout type");

    using T = typename DstTile::DType;
    static_assert(isSupportType<T, int32_t, int16_t, int8_t, uint32_t, uint16_t, uint8_t, float>,
                  "Fix: TPOW has invalid data type.");

    static_assert(std::is_same_v<T, typename BaseTile::DType> && std::is_same_v<T, typename ExpTile::DType>,
                  "TPOW: The data type of dst, base and exp must be consistent");
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp)
{
    using T = typename DstTile::DType;
    PowCheckType<DstTile, BaseTile, ExpTile>();

    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(validRows == base.GetValidRow() && validCols == base.GetValidCol(),
               "Fix: TPOW input tile base valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(validRows == exp.GetValidRow() && validCols == exp.GetValidCol(),
               "Fix: TPOW input tile exp valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(std::is_integral_v<T> || (validRows == tmp.GetValidRow() && validCols == tmp.GetValidCol()),
               "Fix: TPOW input tile tmp valid shape mismatch with output tile dst shape when the data type is "
               "floating point.");

    TPow<DstTile, BaseTile, ExpTile, TmpTile>(dst.data(), base.data(), exp.data(), tmp.data(), validRows, validCols);
}

template <typename DstTile, typename BaseTile, typename TmpTile>
__tf__ PTO_INTERNAL void TPows(typename DstTile::TileDType __out__ dstData,
                               typename BaseTile::TileDType __in__ baseData, typename DstTile::DType exp,
                               typename TmpTile::TileDType __in__ tmpData, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;

    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *base = (__ubuf__ T *)__cce_get_tile_ptr(baseData);

    if constexpr (std::is_integral_v<T>) {
        PtoSetWaitFlag<PIPE_V, PIPE_S>(EVENT_ID7, EVENT_ID7);
        TPowI<T, DstTile::RowStride, BaseTile::RowStride>(dst, base, exp, validRow, validCol);
        PtoSetWaitFlag<PIPE_S, PIPE_V>(EVENT_ID7, EVENT_ID7);
    } else {
        __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
        TPowF<T, DstTile, BaseTile, TmpTile>(dst, base, exp, tmp, validRow, validCol);
    }
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPOWS_IMPL(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp)
{
    using T = typename DstTile::DType;
    PowCheckType<DstTile, BaseTile, DstTile>();

    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(validRows == base.GetValidRow() && validCols == base.GetValidCol(),
               "Fix: TPOW input tile base valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(std::is_integral_v<T> || (validRows == tmp.GetValidRow() && validCols == tmp.GetValidCol()),
               "Fix: TPOW input tile tmp valid shape mismatch with output tile dst shape when the data type is "
               "floating point.");

    TPows<DstTile, BaseTile, TmpTile>(dst.data(), base.data(), exp, tmp.data(), validRows, validCols);
}

} // namespace pto

#endif
