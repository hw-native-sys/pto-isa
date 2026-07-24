/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Modified from demos/baseline/add/csrc/kernel/add_custom.cpp

#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t BLOCK_DIM = 20;                     // number of vector cores(AIVs)
constexpr unsigned BLOCK_ROWS = 20;                    // number of AIVs in rows
constexpr unsigned BLOCK_COLS = 1;                     // number of AIVs in cols
constexpr uint32_t BUFFER_NUM = 2;                     // ping-pong buffer
constexpr unsigned UB_SIZE = 0x30000;                  // 192KB UB of A2A3
constexpr unsigned X_PING = 0x0;                       // ping address of input x in UB buffer
constexpr unsigned X_PONG = (X_PING + 0x8000 + 0x100); // pong address of input 1 in UB buffer
constexpr unsigned Y_PING = 0x10000;                   // ping address of input y in UB buffer
constexpr unsigned Y_PONG = (Y_PING + 0x8000 + 0x100); // pong address of input y in UB buffer
constexpr unsigned Z_PING = 0x20000;                   // ping address of output z in UB buffer
constexpr unsigned Z_PONG = (Z_PING + 0x8000 + 0x100); // pong address of output z in UB buffer
constexpr unsigned MAX_TILE_SIZE = (0x10000 - 0x100);  // Maximum tile size
constexpr uint32_t tileNum = 2;                        // tile number on one vector core

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

    // inner-vector core block tiling
    constexpr unsigned tileSRows = bTileRows;
    constexpr unsigned tileSCols = bTileCols / tileNum / BUFFER_NUM;
    // define GlobalData on global memory with shape and stride
    using ShapeDim5 = pto::Shape<1, 1, 1, tileSRows, tileSCols>;
    using StridDim5 = pto::Stride<1, 1, 1, tileCols, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
    GlobalData xGlobal(x);
    GlobalData yGlobal(y);
    GlobalData zGlobal(z);

    // define TileData on UB buffer with static shape and dynamic mask
    using TileData = Tile<TileType::Vec, T, tileSRows, tileSCols, BLayout::RowMajor, -1, -1>;
    unsigned bLength = totalLength / block_num;
    // valid mask(vRows, vCols) of each tile
    unsigned vRows = tileRows / block_num;
    unsigned vCols = bLength / tileNum / BUFFER_NUM;
    // define ping-pong buffer for related tiles
    TileData xTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};
    TileData yTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};
    TileData zTiles[BUFFER_NUM] = {TileData(vRows, vCols), TileData(vRows, vCols)};

    // assign the UB address for each tile
    TASSIGN(xTiles[0], X_PING);
    TASSIGN(xTiles[1], X_PONG);
    TASSIGN(yTiles[0], Y_PING);
    TASSIGN(yTiles[1], Y_PONG);
    TASSIGN(zTiles[0], Z_PING);
    TASSIGN(zTiles[1], Z_PONG);
    // total number of loops of one vector core
    int32_t loopCount = tileNum * BUFFER_NUM;
    // address offset between vector cores
    unsigned offset = block_idx * bTileRows * bTileCols;
    int8_t pingpong_flag = 0; // ping pong pipeline flag

    // synchronization operations between hardware pipelines
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (uint32_t i = 0; i < loopCount; i++) {
        unsigned iterOffset = offset + i * tileSRows * tileSCols;
        // update address of GlobalData
        TASSIGN(xGlobal, x + iterOffset);
        TASSIGN(yGlobal, y + iterOffset);
        TASSIGN(zGlobal, z + iterOffset);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)(pingpong_flag));
        // load data from global memory to UB buffer
        TLOAD(xTiles[pingpong_flag], xGlobal);
        TLOAD(yTiles[pingpong_flag], yGlobal);

        set_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)(pingpong_flag));

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)(pingpong_flag));
        // perform elementwise addition by vector core
        TADD(zTiles[pingpong_flag], xTiles[pingpong_flag], yTiles[pingpong_flag]);
        set_flag(PIPE_V, PIPE_MTE2, (event_t)(pingpong_flag));

        set_flag(PIPE_V, PIPE_MTE3, (event_t)(pingpong_flag));
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)(pingpong_flag));
        // store data from UB buffer to global memory
        TSTORE(zGlobal, zTiles[pingpong_flag]);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)(pingpong_flag));
        pingpong_flag = (pingpong_flag == 0) ? 1 : 0;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TASSIGN(zGlobal, z);
    z = zGlobal.data();

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
