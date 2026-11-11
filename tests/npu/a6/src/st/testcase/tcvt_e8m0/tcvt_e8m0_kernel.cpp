/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "cases.h"

using namespace pto;

namespace {
constexpr int kGuardBytes = BLOCK_BYTE_SIZE;

template <typename T, bool Dynamic>
PTO_INTERNAL T makeTile(unsigned rows, unsigned cols)
{
    if constexpr (Dynamic) {
        return T(rows, cols);
    } else {
        return T();
    }
}

template <int SR, int SC, int DR, int DC, int VR, int VC, bool ToE8m0, bool Dynamic>
__global__ AICORE void runE8m0Cvt(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, unsigned rows, unsigned cols, int round, int api)
{
    using SrcT = std::conditional_t<ToE8m0, bfloat16_t, float8_e8m0_t>;
    using DstT = std::conditional_t<ToE8m0, float8_e8m0_t, bfloat16_t>;
    constexpr int srcBytes = SR * SC * sizeof(SrcT);
    using Source = Tile<TileType::Vec, SrcT, SR, SC, BLayout::RowMajor, Dynamic ? DYNAMIC : VR, Dynamic ? DYNAMIC : VC>;
    using Destination =
        Tile<TileType::Vec, DstT, DR, DC, BLayout::RowMajor, Dynamic ? DYNAMIC : VR, Dynamic ? DYNAMIC : VC>;
    constexpr int dstBytes = Destination::StorageBytes + 2 * kGuardBytes;
    auto source = makeTile<Source, Dynamic>(rows, cols);
    auto destination = makeTile<Destination, Dynamic>(rows, cols);
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(source, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    pto_copy_gm_to_ubuf_align_v2(rawInput.data(), src, 0, 1, srcBytes, 0, 0, false, 0, 0, 0);
    pto_copy_gm_to_ubuf_align_v2(rawOutput.data(), dst, 0, 1, dstBytes, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode rounding = static_cast<RoundMode>(round);
    Tile<TileType::Vec, int32_t, 1, 8> unusedTemporary;
    switch (api) {
        case 0:
            TCVT(destination, source, rounding);
            break;
        case 1:
            TCVT(destination, source, rounding, SaturationMode::ON);
            break;
        case 2:
            TCVT(destination, source, rounding, SaturationMode::OFF);
            break;
        case 3:
            TCVT(destination, source, unusedTemporary, rounding);
            break;
        case 4:
            TCVT(destination, source, unusedTemporary, rounding, SaturationMode::OFF);
            break;
        default:
            TCVT<false>(destination, source, rounding, SaturationMode::ON);
            break;
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto_copy_ubuf_to_gm_align_v2(dst, rawOutput.data(), 0, 1, dstBytes, 0, 0, 0);
}

// Exhaustive e8m0 -> bf16 sweep over all 256 exponent codes (covers the 0xFF
// NaN encoding 0x7FFF) on a single full row.
__global__ AICORE void runE8m0ToBf16AllBytes(__gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    constexpr int columns = 256;
    constexpr int srcBytes = columns;
    Tile<TileType::Vec, float8_e8m0_t, 1, columns> source;
    Tile<TileType::Vec, bfloat16_t, 1, columns> destination;
    constexpr int dstBytes = destination.StorageBytes + 2 * kGuardBytes;
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(source, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    pto_copy_gm_to_ubuf_align_v2(rawInput.data(), src, 0, 1, srcBytes, 0, 0, false, 0, 0, 0);
    pto_copy_gm_to_ubuf_align_v2(rawOutput.data(), dst, 0, 1, dstBytes, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(destination, source, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto_copy_ubuf_to_gm_align_v2(dst, rawOutput.data(), 0, 1, dstBytes, 0, 0, 0);
}

// bf16 -> e8m0 over sign / Inf / NaN / exponent-saturation inputs. The generic
// case fixtures only use positive finite values with exponent <= 0xFD; this
// kernel exercises negative magnitudes (the conversion drops the sign bit, as
// in the CANN reference emulation), Inf/NaN (exponent 0xFF) and the 0xFE
// exponent band with truncating round modes.
__global__ AICORE void runBf16ToE8m0Specials(__gm__ uint8_t* dst, __gm__ uint8_t* src, int round)
{
    constexpr int columns = 256;
    constexpr int srcBytes = columns * 2;
    Tile<TileType::Vec, bfloat16_t, 1, columns> source;
    Tile<TileType::Vec, float8_e8m0_t, 1, columns> destination;
    constexpr int dstBytes = destination.StorageBytes + 2 * kGuardBytes;
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(source, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    pto_copy_gm_to_ubuf_align_v2(rawInput.data(), src, 0, 1, srcBytes, 0, 0, false, 0, 0, 0);
    pto_copy_gm_to_ubuf_align_v2(rawOutput.data(), dst, 0, 1, dstBytes, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(destination, source, static_cast<RoundMode>(round));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto_copy_ubuf_to_gm_align_v2(dst, rawOutput.data(), 0, 1, dstBytes, 0, 0, 0);
}
} // namespace

void launchE8m0Cvt(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream)
{
    switch (static_cast<E8m0CvtCaseId>(caseId)) {
#define E8M0_CVT_LAUNCH(Name, SR, SC, DR, DC, VR, VC, ToE8m0, Dynamic)                                                 \
    case E8m0CvtCaseId::Name:                                                                                          \
        runE8m0Cvt<SR, SC, DR, DC, VR, VC, ToE8m0, Dynamic><<<1, nullptr, stream>>>(dst, src, rows, cols, round, api); \
        break;
        E8M0_CVT_CASES(E8M0_CVT_LAUNCH)
#undef E8M0_CVT_LAUNCH
    }
}

void launchE8m0ToBf16AllBytes(unsigned char* dst, unsigned char* src, void* stream)
{
    runE8m0ToBf16AllBytes<<<1, nullptr, stream>>>(dst, src);
}

void launchBf16ToE8m0Specials(unsigned char* dst, unsigned char* src, int round, void* stream)
{
    runBf16ToE8m0Specials<<<1, nullptr, stream>>>(dst, src, round);
}
