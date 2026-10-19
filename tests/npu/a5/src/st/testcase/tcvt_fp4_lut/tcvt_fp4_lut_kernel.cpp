/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

template <int SR, int SC, int DR, int DC, int VR, int VC, bool NZ, bool Compact, bool Dynamic>
__global__ AICORE void runFp4Lut(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, unsigned rows, unsigned cols, int round, int api)
{
    constexpr int srcBytes = SR * SC / 2;
    constexpr int dstBytes = DR * DC + 2 * kGuardBytes;
    using Source =
        Tile<TileType::Vec, float4_e2m1x2_t, SR, SC, BLayout::RowMajor, Dynamic ? DYNAMIC : VR, Dynamic ? DYNAMIC : VC>;
    using Destination = Tile<
        TileType::Vec, float8_e4m3_t, DR, DC, NZ ? BLayout::ColMajor : BLayout::RowMajor, Dynamic ? DYNAMIC : VR,
        Dynamic ? DYNAMIC : VC, NZ ? SLayout::RowMajor : SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Null,
        Compact ? CompactMode::Normal : CompactMode::Null>;
    auto source = makeTile<Source, Dynamic>(rows, cols);
    auto destination = makeTile<Destination, Dynamic>(rows, cols);
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(source, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, srcBytes>, pto::Stride<1, 1, 1, srcBytes, 1>> input(src);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, dstBytes>, pto::Stride<1, 1, 1, dstBytes, 1>> output(dst);
    // Return a raw view of all physical padding and guards, independently of TSTORE NZ.
    TLOAD(rawInput, input);
    TLOAD(rawOutput, output);
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
    TSTORE(output, rawOutput);
}

// Supported pairs exercise the original generic dispatch with bit-exact goldens.
template <bool Fp4Source>
__global__ AICORE void runRegression(__gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    using Src = std::conditional_t<Fp4Source, float4_e2m1x2_t, float>;
    using Dst = std::conditional_t<Fp4Source, bfloat16_t, float8_e4m3_t>;
    constexpr int columns = 128;
    constexpr int srcBytes = Fp4Source ? columns / 2 : columns * sizeof(Src);
    constexpr int dstBytes = columns * sizeof(Dst);
    Tile<TileType::Vec, Src, 1, columns> source;
    Tile<TileType::Vec, Dst, 1, columns> destination;
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes + 2 * kGuardBytes> rawOutput;
    TASSIGN(source, 0);
    TASSIGN(rawInput, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, srcBytes>, pto::Stride<1, 1, 1, srcBytes, 1>> input(src);
    GlobalTensor<
        uint8_t, Shape<1, 1, 1, 1, dstBytes + 2 * kGuardBytes>, pto::Stride<1, 1, 1, dstBytes + 2 * kGuardBytes, 1>>
        output(dst);
    TLOAD(rawInput, input);
    TLOAD(rawOutput, output);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(destination, source, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, rawOutput);
}
} // namespace

void launchFp4Lut(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream)
{
    switch (static_cast<Fp4LutCaseId>(caseId)) {
#define FP4_LUT_LAUNCH(Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic) \
    case Fp4LutCaseId::Name:                                               \
        runFp4Lut<SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic>            \
            <<<1, nullptr, stream>>>(dst, src, rows, cols, round, api);    \
        break;
        FP4_LUT_CASES(FP4_LUT_LAUNCH)
#undef FP4_LUT_LAUNCH
    }
}

void launchFp4LutRegression(int pair, unsigned char* dst, unsigned char* src, void* stream)
{
    if (pair == 0) {
        runRegression<true><<<1, nullptr, stream>>>(dst, src);
    } else {
        runRegression<false><<<1, nullptr, stream>>>(dst, src);
    }
}
