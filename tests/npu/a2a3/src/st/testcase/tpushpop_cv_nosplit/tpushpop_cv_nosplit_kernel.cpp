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
 * Cube-Vector mixed kernel using TILE_NO_SPLIT mode.
 *
 * The Cube core computes matmul (A x B) and pushes the full AccTile to a
 * single vector core (AIV0) via TPUSH with TileSplitAxis::TILE_NO_SPLIT.
 * AIV0 pops the received tile, loads the corresponding bias row, and writes
 * out (accTile + bias) with TADD.  AIV1 is idle throughout the FIFO loop.
 */
template <typename InT, typename OutT, int TOTAL_M, int CASE_TILE_M, int K, int N>
__global__ AICORE void runTPushPopMatmulAddNoSplit(
    __gm__ uint64_t* ffts_addr, __gm__ OutT* out, __gm__ InT* srcA, __gm__ InT* srcB, __gm__ OutT* bias,
    __gm__ OutT* fifoMem)
{
    // a2a3 cross-core sync (FFTS) requires the base address of the FFTS table.
    // Without this call ffts_cross_core_sync / wait_flag_dev use an invalid
    // address, breaking the Cube↔Vec handshake.
    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr uint32_t TILE_K = K;
    constexpr uint32_t TILE_N = N;
    constexpr uint32_t NUM_M_TILES = TOTAL_M / CASE_TILE_M;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    // Local UB base address for the Vec-side FIFO double-buffer.
    constexpr uint32_t localFiFoBase = 0x0;

    // Accumulator tile produced by Cube (full CASE_TILE_M x TILE_N).
    using AccTile = TileAcc<OutT, CASE_TILE_M, TILE_N, CASE_TILE_M, TILE_N>;

    // Vec tile: same shape as AccTile because TILE_NO_SPLIT delivers the whole
    // result to a single Vec core without any row/column splitting.
    using VecTileFull = Tile<TileType::Vec, OutT, CASE_TILE_M, TILE_N, BLayout::RowMajor, CASE_TILE_M, TILE_N>;
    using BiasTile = Tile<TileType::Vec, OutT, CASE_TILE_M, TILE_N, BLayout::RowMajor, CASE_TILE_M, TILE_N>;
    using OutTile = Tile<TileType::Vec, OutT, CASE_TILE_M, TILE_N, BLayout::RowMajor, CASE_TILE_M, TILE_N>;

    VecTileFull vecFifoTile;

    // a2a3 uses a GM-backed FIFO: TPUSH writes the AccTile to fifoMem via
    // TSTORE, and TPOP reads it back via TLOAD.  fifoMem must be a valid
    // device memory region of at least FIFO_DEPTH * SlotSize bytes.
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * CASE_TILE_M * TILE_N, FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void*)(uint64_t)fifoMem, 0x0, localFiFoBase);

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
    using GlobalBias = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, CASE_TILE_M, TILE_N>,
        pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, CASE_TILE_M * TILE_N, TILE_N, 1>>;
    using GlobalOut = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, CASE_TILE_M, TILE_N>,
        pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, CASE_TILE_M * TILE_N, TILE_N, 1>>;

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

            // Push the full AccTile to a single Vec core (AIV0) without splitting.
            TPUSH<MatPipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(mPipe, accTile);

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

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        for (int m_tile = 0; m_tile < NUM_M_TILES; m_tile++) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            // Pop the full tile (TILE_NO_SPLIT: AIV0 only).
            TPOP<MatPipe, VecTileFull, TileSplitAxis::TILE_NO_SPLIT>(mPipe, vecFifoTile);

            size_t biasOffset = static_cast<size_t>(m_tile * CASE_TILE_M) * TILE_N;
            GlobalBias globalBias(bias + biasOffset);

            TLOAD(biasTile, globalBias);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TADD(outTile, vecFifoTile, biasTile);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            size_t outOffset = static_cast<size_t>(m_tile * CASE_TILE_M) * TILE_N;
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
void LaunchTPushPopMatmulAddNoSplit(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, uint8_t* fifoMem, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopMatmulAddNoSplit<half, float, 16, 16, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA),
            reinterpret_cast<half*>(srcB), reinterpret_cast<float*>(bias), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 2) {
        runTPushPopMatmulAddNoSplit<half, float, 32, 16, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<half*>(srcA),
            reinterpret_cast<half*>(srcB), reinterpret_cast<float*>(bias), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 3) {
        runTPushPopMatmulAddNoSplit<float, float, 16, 16, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(bias), reinterpret_cast<float*>(fifoMem));
    }
}

template void LaunchTPushPopMatmulAddNoSplit<1>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMatmulAddNoSplit<2>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMatmulAddNoSplit<3>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* bias, uint8_t* fifoMem, void* stream);
