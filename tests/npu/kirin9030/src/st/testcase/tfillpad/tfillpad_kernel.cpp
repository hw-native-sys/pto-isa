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
#include <limits>
#include <algorithm>

using namespace std;
using namespace pto;

#define type_32_aligned(T) (32 / sizeof(T))
#define align_to_32B(x, T) ((((x) + type_32_aligned(T) - 1) / type_32_aligned(T)) * (type_32_aligned(T)))

template <
    typename T, int srcRows, int srcCols, int dstRows, int dstCols, PadValue LoadPadVal_ = PadValue::Null,
    PadValue FillPadVal_ = PadValue::Null, bool inplace = false, bool expand = false>
AICORE void runTFILLPAD(__gm__ T* out, __gm__ T* src)
{
    constexpr int srcTileCols = expand ? align_to_32B(srcCols, T) : dstCols;

    using SrcShape = Shape<1, 1, 1, srcRows, srcCols>;
    using DstShape = Shape<1, 1, 1, dstRows, dstCols>;
    using SrcStride = pto::Stride<srcRows * srcCols, srcRows * srcCols, srcRows * srcCols, srcCols, 1>;
    using DstStride = pto::Stride<dstRows * dstCols, dstRows * dstCols, dstRows * dstCols, dstCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using SrcTile = Tile<
        TileType::Vec, T, srcRows, srcTileCols, BLayout::RowMajor, srcRows, srcCols, SLayout::NoneBox, 512,
        LoadPadVal_>;
    using DstTile = Tile<
        TileType::Vec, T, dstRows, dstCols, BLayout::RowMajor, dstRows, dstCols, SLayout::NoneBox, 512, FillPadVal_>;
    SrcTile srcTile;
    DstTile dstTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<inplace ? 0x0 : SrcTile::Numel * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (expand) {
        TFILLPAD_EXPAND(dstTile, srcTile);
    } else if (inplace) {
        TFILLPAD_INPLACE(dstTile, srcTile);
    } else {
        TFILLPAD(dstTile, srcTile);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTFILLPAD_1(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<float, 64, 127, 64, 128, PadValue::Max, PadValue::Max>((__gm__ float*)out, (__gm__ float*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_2(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<float, 64, 127, 64, 144, PadValue::Max, PadValue::Max>((__gm__ float*)out, (__gm__ float*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_3(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<float, 64, 127, 64, 160, PadValue::Min, PadValue::Max>((__gm__ float*)out, (__gm__ float*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_4(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<float, 260, 7, 260, 16, PadValue::Min, PadValue::Max>((__gm__ float*)out, (__gm__ float*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_5(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<float, 260, 7, 260, 16, PadValue::Min, PadValue::Max, true>((__gm__ float*)out, (__gm__ float*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_6(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<uint16_t, 260, 7, 260, 32, PadValue::Min, PadValue::Max>((__gm__ uint16_t*)out, (__gm__ uint16_t*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_7(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<int8_t, 260, 7, 260, 64, PadValue::Min, PadValue::Max>((__gm__ int8_t*)out, (__gm__ int8_t*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_8(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<uint16_t, 259, 7, 260, 32, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ uint16_t*)out, (__gm__ uint16_t*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_9(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<int8_t, 259, 7, 260, 64, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_10(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<int16_t, 260, 7, 260, 32, PadValue::Min, PadValue::Min>((__gm__ int16_t*)out, (__gm__ int16_t*)src);
}

extern "C" __global__ AICORE void launchTFILLPAD_11(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTFILLPAD<int32_t, 260, 7, 260, 32, PadValue::Min, PadValue::Min>((__gm__ int32_t*)out, (__gm__ int32_t*)src);
}

template <int32_t testKey>
void launchTFILLPAD(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTFILLPAD_1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTFILLPAD_2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTFILLPAD_3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTFILLPAD_4<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTFILLPAD_5<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTFILLPAD_6<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTFILLPAD_7<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTFILLPAD_8<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTFILLPAD_9<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 10) {
        launchTFILLPAD_10<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 11) {
        launchTFILLPAD_11<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTFILLPAD<1>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<2>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<3>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<4>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<5>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<6>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<7>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<8>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<9>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<10>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<11>(uint8_t* out, uint8_t* src, void* stream);
