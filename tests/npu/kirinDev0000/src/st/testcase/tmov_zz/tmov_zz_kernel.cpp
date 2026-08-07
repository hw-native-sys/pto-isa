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
#include "acl/acl.h"

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

namespace TMovZZTest {

// kirin9030 tmov_zz: Redesigned to only keep TMOV ND->NZ搬运流程.
// Removed TQUANT quantization and E8M0 ZZ layout conversion (not supported on kirin9030).
// Input: int8_t (FP8) ND layout data.
// Output: int8_t (FP8) NZ layout data via TMOV ND->NZ.
template <int validRows, int validCols>
AICORE void runTMovZZ(__gm__ int8_t* outFp8Nz, __gm__ int8_t* src)
{
    constexpr int paddedCols = PTO_CEIL(validCols, BLOCK_SIZE / sizeof(uint32_t));
    constexpr int paddedRows16 = PTO_CEIL(validRows, 16);

    using SrcGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstFp8GlobalNZ = GlobalTensor<
        int8_t, TileShape2D<int8_t, paddedRows16, paddedCols, Layout::NZ>,
        BaseShape2D<int8_t, paddedRows16, paddedCols, Layout::NZ>, Layout::NZ>;

    using SrcTile = Tile<
        TileType::Vec, int8_t, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    // NZ destination tile with RowPlusOne compact mode for ND->NZ conversion
    constexpr int virtualRow = PTO_CEIL(validRows, FRACTAL_NZ_ROW) + 1;
    using Fp8NZTile = Tile<
        TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, validRows, paddedCols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;

    SrcTile srcTile(validRows, validCols);
    Fp8NZTile fp8TileNZ;

    SrcGlobal srcGlobal(src);
    DstFp8GlobalNZ fp8GlobalNZ(outFp8Nz);

    TASSIGN<0x0>(srcTile);
    TASSIGN<sizeof(int8_t) * validRows * paddedCols>(fp8TileNZ);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // TMOV: ND -> NZ layout conversion
    TMOV(fp8TileNZ, srcTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(fp8GlobalNZ, fp8TileNZ);
}

template <int validRows, int validCols>
__global__ AICORE void launchTMovZZKernel(__gm__ int8_t* outFp8Nz, __gm__ int8_t* src)
{
    runTMovZZ<validRows, validCols>(outFp8Nz, src);
}

template <int validRows, int validCols>
void LaunchTMovZZ(uint8_t* dstFp8Nz, uint8_t* src, void* stream)
{
    launchTMovZZKernel<validRows, validCols>
        <<<1, nullptr, stream>>>(reinterpret_cast<int8_t*>(dstFp8Nz), reinterpret_cast<int8_t*>(src));
}

template void LaunchTMovZZ<32, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 128>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 192>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 256>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 320>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 384>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 448>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 512>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 576>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 640>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 704>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 768>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 832>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<64, 896>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<128, 128>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<128, 256>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<128, 384>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<256, 192>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<8, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<6, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<13, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<3, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<29, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<31, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<47, 64>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<31, 128>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<47, 128>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<31, 256>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);
template void LaunchTMovZZ<47, 256>(uint8_t* dstFp8Nz, uint8_t* src, void* stream);

} // namespace TMovZZTest
