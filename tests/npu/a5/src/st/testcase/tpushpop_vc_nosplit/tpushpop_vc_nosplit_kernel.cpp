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
#include <pto/common/fifo.hpp>

using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

/**
 * TILE_NO_SPLIT mode: a single Vec core (AIV0) handles the full TILE_K x TILE_N tile and
 * pushes it to L1 via TINSERT at row 0, col 0. No row-splitting across subcores.
 * Cube side waits/frees only one intra-block flag (vs two in TILE_UP_DOWN mode).
 */
template <typename QuantT, typename InT, typename OutT, int TOTAL_M, int TOTAL_K, int N, int CASE_TILE_K>
__global__ AICORE void runTPushPopVCNSMatmul(
    __gm__ OutT* out, __gm__ InT* srcA, __gm__ QuantT* quantB, __gm__ OutT* scale, __gm__ OutT* offset)
{
    constexpr uint32_t TILE_K = CASE_TILE_K;
    constexpr uint32_t TILE_N = N;
    constexpr uint32_t NUM_K_TILES = TOTAL_K / CASE_TILE_K;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint8_t FIFO_PERIOD = 1;

    // Vec tiles cover the full TILE_K x TILE_N (no split between subcores)
    using VecTileProd = Tile<TileType::Vec, OutT, TILE_K, TILE_N, BLayout::RowMajor, TILE_K, TILE_N>;
    using VecTileNZ =
        Tile<TileType::Vec, OutT, TILE_K, TILE_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;
    using MatTileCons =
        Tile<TileType::Mat, OutT, TILE_K, TILE_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;

    MatTileCons matFifoTile;
    // FIFO slot size = full tile (same as TILE_UP_DOWN, just written by one Vec core)
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_V2C, TILE_K * TILE_N * sizeof(OutT), FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void*)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x10000);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(InT);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(TILE_K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(TILE_N, blockAlign);

    using GlobalA = GlobalTensor<
        InT, pto::Shape<1, 1, 1, TOTAL_M, TILE_K>,
        pto::Stride<TOTAL_M * TOTAL_K, TOTAL_M * TOTAL_K, TOTAL_M * TOTAL_K, TOTAL_K, 1>>;
    using GlobalOut = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, TOTAL_M, TILE_N>,
        pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, TOTAL_M * TILE_N, TILE_N, 1>>;

    using TileMatA =
        Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, TOTAL_M, TILE_K, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, TOTAL_M, TILE_K>;
    using PopTile =
        Tile<TileType::Mat, OutT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;
    using RightTile = TileRight<OutT, ALIGNED_K, ALIGNED_N, TILE_K, TILE_N>;
    using AccTile = TileAcc<OutT, TOTAL_M, TILE_N, TOTAL_M, TILE_N>;

    // Quantization tiles sized for the full TILE_K (not half)
    using QuantTile = Tile<TileType::Vec, QuantT, TILE_K, TILE_N, BLayout::RowMajor, TILE_K, TILE_N>;
    using ScaleTile = Tile<TileType::Vec, OutT, TILE_K, 8, BLayout::RowMajor, -1, -1>;
    using OffsetTile = Tile<TileType::Vec, OutT, TILE_K, 8, BLayout::RowMajor, -1, -1>;

    if constexpr (DAV_VEC) {
        // TILE_NO_SPLIT: only subcore 0 (AIV0) loads data and pushes the full tile to L1.
        // Subcore 1 stays idle on the Vec side; the Cube only waits on a single intra-block flag.
        if (get_subblockid() == 0) {
            QuantTile quantTile;
            VecTileProd dequantTile;
            VecTileNZ dequantTileNZ;
            ScaleTile scaleTile(TILE_K, 1);
            OffsetTile offsetTile(TILE_K, 1);
            TASSIGN(quantTile, 0x0);
            TASSIGN(dequantTile, 0x10000);
            TASSIGN(dequantTileNZ, 0x18000);
            TASSIGN(scaleTile, 0x20000);
            TASSIGN(offsetTile, 0x28000);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            // Global tensor covers the full TILE_K rows (no subblock offset)
            using GlobalQuantB = GlobalTensor<
                QuantT, pto::Shape<1, 1, 1, TILE_K, TILE_N>,
                pto::Stride<TOTAL_K * TILE_N, TOTAL_K * TILE_N, TILE_K * TILE_N, TILE_N, 1>>;
            using GlobalScaleOffset =
                GlobalTensor<OutT, pto::Shape<1, 1, 1, TILE_K, 1>, pto::Stride<TOTAL_K, TOTAL_K, TILE_K, 1, 1>>;

            for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
                GlobalQuantB globalQuantB(quantB + k_tile * TILE_K * TILE_N);
                GlobalScaleOffset globalScale(scale + k_tile * TILE_K);
                GlobalScaleOffset globalOffset(offset + k_tile * TILE_K);

                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                TLOAD(quantTile, globalQuantB);
                TLOAD(scaleTile, globalScale);
                TLOAD(offsetTile, globalOffset);

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

                TDEQUANT(dequantTile, quantTile, scaleTile, offsetTile);
                TMOV(dequantTileNZ, dequantTile);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

                // push full vector tile to FIFO using TILE_NO_SPLIT:
                // TINSERT_IMPL inserts the full TILE_K x TILE_N at row 0, col 0 in L1
                TPUSH<MatPipe, VecTileNZ, TileSplitAxis::TILE_NO_SPLIT>(mPipe, dequantTileNZ);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            pipe_barrier(PIPE_ALL);
        }
    }

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        TASSIGN(aMatTile, 0x0);

        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
            GlobalA globalA(srcA + k_tile * TILE_K);

            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            TLOAD(aMatTile, globalA);
            // pop mat tile from FIFO; with TILE_NO_SPLIT the Cube waits on a single intra-block flag
            TPOP<MatPipe, MatTileCons, TileSplitAxis::TILE_NO_SPLIT>(mPipe, matFifoTile);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            TMOV(aTile, aMatTile);
            TMOV(bTile, matFifoTile);
            // free FIFO slot; with TILE_NO_SPLIT sets only one intra-block flag back to Vec
            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(mPipe);

            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            if (k_tile == 0) {
                TMATMUL(accTile, aTile, bTile);
            } else {
                TMATMUL_ACC(accTile, accTile, aTile, bTile);
            }

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        }

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        GlobalOut globalOut(out);
        TSTORE<AccTile, GlobalOut>(globalOut, accTile);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopVCNSMatmul(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopVCNSMatmul<int8_t, float, float, 16, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int8_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    } else if constexpr (tilingKey == 2) {
        runTPushPopVCNSMatmul<int8_t, float, float, 16, 128, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int8_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    } else if constexpr (tilingKey == 3) {
        runTPushPopVCNSMatmul<int8_t, float, float, 16, 256, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int8_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    } else if constexpr (tilingKey == 4) {
        runTPushPopVCNSMatmul<int16_t, float, float, 16, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int16_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    } else if constexpr (tilingKey == 5) {
        runTPushPopVCNSMatmul<int16_t, float, float, 16, 128, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int16_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    } else if constexpr (tilingKey == 6) {
        runTPushPopVCNSMatmul<int16_t, float, float, 16, 256, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<int16_t*>(quantB),
            reinterpret_cast<float*>(scale), reinterpret_cast<float*>(offset));
    }
}

template void LaunchTPushPopVCNSMatmul<1>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCNSMatmul<2>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCNSMatmul<3>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCNSMatmul<4>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCNSMatmul<5>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCNSMatmul<6>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
