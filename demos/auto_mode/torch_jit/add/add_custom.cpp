/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Modified from demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp

#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t BLOCK_DIM = 20;                    // number of vector cores(AIVs)
constexpr unsigned BLOCK_ROWS = 20;                   // number of AIVs in rows
constexpr unsigned BLOCK_COLS = 1;                    // number of AIVs in cols
constexpr unsigned UB_SIZE = 0x30000;                 // 192KB UB of A2A3
constexpr unsigned MAX_TILE_SIZE = (0x10000 - 0x100); // Maximum tile size

template <typename T, unsigned tileRows, unsigned tileCols>
AICORE void runTAdd(__gm__ T* z, __gm__ T* x, __gm__ T* y, uint32_t totalLength)
{
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);
    static_assert(BLOCK_ROWS * BLOCK_COLS == BLOCK_DIM, "Wrong block tilling!");
    // inter-vector cores block tiling
    constexpr unsigned bTileRows = tileRows / BLOCK_ROWS;
    constexpr unsigned bTileCols = tileCols / BLOCK_COLS;
    static_assert(bTileRows * bTileCols * sizeof(T) <= MAX_TILE_SIZE, "UB buffer overflow.");

    // define GlobalData on global memory with shape and stride
    using ShapeDim5 = pto::Shape<1, 1, 1, bTileRows, bTileCols>;
    using StridDim5 = pto::Stride<1, 1, 1, tileCols, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
    const unsigned offset = block_idx * bTileRows * bTileCols;
    GlobalData xGlobal(x + offset);
    GlobalData yGlobal(y + offset);
    GlobalData zGlobal(z + offset);

    using TileData = Tile<TileType::Vec, T, bTileRows, bTileCols, BLayout::RowMajor, -1, -1>;
    TileData xTile(bTileRows, bTileCols), yTile(bTileRows, bTileCols), zTile(bTileRows, bTileCols);

    TLOAD(xTile, xGlobal);
    TLOAD(yTile, yGlobal);

    TADD(zTile, xTile, yTile);

    TSTORE(zGlobal, zTile);

#else // else branch for `#if defined(__DAV_C220_VEC__)`
// do nothing for Cube branch
#endif
}

// kernel entry
__global__ AICORE void add_custom(__gm__ void* x, __gm__ void* y, __gm__ void* z, uint32_t totalLength)
{
    // Define the tile size
    constexpr unsigned tileRows = 20;
    constexpr unsigned tileCols = 2048;
    // main kernel, totalLength is dynamic input
    runTAdd<half, tileRows, tileCols>((__gm__ half*)z, (__gm__ half*)x, (__gm__ half*)y, totalLength);
}

extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* z, int N)
{
    add_custom<<<blockDim, nullptr, stream>>>(x, y, z, N);
}
