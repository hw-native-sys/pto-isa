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

#define VEC_CORES 2

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

template <
    typename InT, typename OutT, int TOTAL_M, int CASE_TILE_M, int K, int N,
    TileSplitAxis SplitAxis = TileSplitAxis::TILE_UP_DOWN>
__global__ AICORE void runTPushPopMatmulAdd(__gm__ OutT* out, __gm__ InT* srcA, __gm__ InT* srcB, __gm__ OutT* bias)
{
    constexpr uint32_t TILE_K = K;
    constexpr uint32_t TILE_N = N;
    constexpr uint32_t NUM_M_TILES = TOTAL_M / CASE_TILE_M;

    // TILE_UP_DOWN splits along rows; TILE_LEFT_RIGHT splits along columns
    constexpr uint32_t VEC_M = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (CASE_TILE_M / VEC_CORES) : CASE_TILE_M;
    constexpr uint32_t VEC_N = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (TILE_N / VEC_CORES) : TILE_N;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint8_t FIFO_PERIOD = 1;

    using AccTile = TileAcc<OutT, CASE_TILE_M, TILE_N, CASE_TILE_M, TILE_N>;
    using VecTileHalf = Tile<TileType::Vec, OutT, VEC_M, VEC_N, BLayout::RowMajor, VEC_M, VEC_N>;
    using BiasTile = Tile<TileType::Vec, OutT, VEC_M, VEC_N, BLayout::RowMajor, VEC_M, VEC_N>;
    using OutTile = Tile<TileType::Vec, OutT, VEC_M, VEC_N, BLayout::RowMajor, VEC_M, VEC_N>;

    VecTileHalf vecFifoTile;
    // pip init
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * VEC_M * VEC_N, FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void*)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(InT);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(CASE_TILE_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(TILE_K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(TILE_N, blockAlign);

    using GlobalA = GlobalTensor<
        InT, pto::Shape<1, 1, 1, CASE_TILE_M, TILE_K>,
        pto::Stride<TOTAL_M * TILE_K, TOTAL_M * TILE_K, CASE_TILE_M * TILE_K, TILE_K, 1>>;
    using GlobalB = GlobalTensor<
        InT, pto::Shape<1, 1, 1, TILE_K, TILE_N>,
        pto::Stride<TILE_K * TILE_N, TILE_K * TILE_N, TILE_K * TILE_N, TILE_N, 1>>;
    // Row stride is always TILE_N (full matrix width) so TILE_LEFT_RIGHT sub-tiles are read correctly
    using GlobalBias = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, VEC_M, VEC_N>,
        pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, VEC_M * TILE_N, TILE_N, 1>>;
    using GlobalOut = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, VEC_M, VEC_N>,
        pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, VEC_M * TILE_N, TILE_N, 1>>;

    using TileMatA =
        Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, CASE_TILE_M, TILE_K, SLayout::RowMajor, 512>;
    using TileMatB =
        Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, CASE_TILE_M, TILE_K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, TILE_K, TILE_N>;

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        TileMatB bMatTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);

        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        for (int m_tile = 0; m_tile < NUM_M_TILES; m_tile++) {
            GlobalA globalA(srcA + m_tile * CASE_TILE_M * TILE_K);
            GlobalB globalB(srcB);

            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            TLOAD(aMatTile, globalA);
            TLOAD(bMatTile, globalB);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);

            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

            TMATMUL(accTile, aTile, bTile);

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            // push Acc tile to fifo
            TPUSH<MatPipe, AccTile, SplitAxis>(mPipe, accTile);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        BiasTile biasTile;
        OutTile outTile;
        TASSIGN(biasTile, 0x10000);
        TASSIGN(outTile, 0x20000);

        uint32_t subBlockIdx = get_subblockid();

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        for (int m_tile = 0; m_tile < NUM_M_TILES; m_tile++) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            // pop vector tile from fifo
            TPOP<MatPipe, VecTileHalf, SplitAxis>(mPipe, vecFifoTile);

            size_t biasOffset;
            if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
                // subblock selects upper/lower half of rows
                biasOffset = static_cast<size_t>(m_tile * CASE_TILE_M + subBlockIdx * VEC_M) * TILE_N;
            } else {
                // subblock selects left/right half of columns within each row
                biasOffset = static_cast<size_t>(m_tile * CASE_TILE_M) * TILE_N + subBlockIdx * VEC_N;
            }
            GlobalBias globalBias(bias + biasOffset);

            TLOAD(biasTile, globalBias);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TADD(outTile, vecFifoTile, biasTile);
            // free the buffer
            TFREE<MatPipe, SplitAxis>(mPipe);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            size_t outOffset;
            if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
                outOffset = static_cast<size_t>(m_tile * CASE_TILE_M + subBlockIdx * VEC_M) * TILE_N;
            } else {
                outOffset = static_cast<size_t>(m_tile * CASE_TILE_M) * TILE_N + subBlockIdx * VEC_N;
            }
            GlobalOut globalOut(out + outOffset);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopMatmulAdd(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream)
{
    // Keys 1-4: TILE_UP_DOWN (split along rows)
    if constexpr (tilingKey == 1) {
        runTPushPopMatmulAdd<half, float, 16, 16, 32, 32, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 2) {
        runTPushPopMatmulAdd<half, float, 32, 16, 32, 32, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 3) {
        runTPushPopMatmulAdd<float, float, 16, 16, 32, 32, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 4) {
        runTPushPopMatmulAdd<half, float, 64, 16, 32, 32, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
        // Keys 5-8: TILE_LEFT_RIGHT (split along columns)
    } else if constexpr (tilingKey == 5) {
        runTPushPopMatmulAdd<half, float, 16, 16, 32, 32, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 6) {
        runTPushPopMatmulAdd<half, float, 32, 16, 32, 32, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 7) {
        runTPushPopMatmulAdd<float, float, 16, 16, 32, 32, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(bias));
    } else if constexpr (tilingKey == 8) {
        runTPushPopMatmulAdd<half, float, 64, 16, 32, 32, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA), reinterpret_cast<half*>(srcB),
            reinterpret_cast<float*>(bias));
    }
}

template void LaunchTPushPopMatmulAdd<1>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<2>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<3>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<4>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<5>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<6>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<7>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);
template void LaunchTPushPopMatmulAdd<8>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, void* stream);