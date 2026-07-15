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

// Computation flow (DIR_BOTH pipe):
//   TILE_UP_DOWN:    each Vec core handles TOTAL_M/2 rows, full columns
//   TILE_LEFT_RIGHT: each Vec core handles full rows, K/2 or N/2 columns
//
//   Vec: tileC = tileA + tileB  (per vector core portion)
//   Vec→Cube (V2C): TPUSH tileC → combined [TOTAL_M, K] in FIFO
//   Cube: TPOP [TOTAL_M,K], TLOAD tileD[K,N], tileE = TMATMUL([TOTAL_M,K]×[K,N])
//   Cube→Vec (C2V): TPUSH tileE[TOTAL_M,N] → vectors pop their portion
//   Vec: tileG = tileE_part - tileF, TSTORE tileG
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis = TileSplitAxis::TILE_UP_DOWN>
__global__ AICORE void runTPushPopDirBoth(
    __gm__ uint64_t* ffts_addr, __gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* srcD, __gm__ T* srcF,
    __gm__ T* fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr uint32_t V2C_ROWS = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TOTAL_M / VEC_CORES) : TOTAL_M;
    constexpr uint32_t V2C_COLS = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (K / VEC_CORES) : K;
    constexpr uint32_t C2V_ROWS = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TOTAL_M / VEC_CORES) : TOTAL_M;
    constexpr uint32_t C2V_COLS = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (N / VEC_CORES) : N;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t SLOT_SIZE = TOTAL_M * N * sizeof(T);

    using BothPipe = TPipe<FLAG_ID, Direction::DIR_BOTH, SLOT_SIZE, FIFO_DEPTH>;

    constexpr uint32_t v2cL1Base = 0x0;
    constexpr uint32_t c2vUBBase = 0x0;

    BothPipe pipe((__gm__ void*)fifoMem, c2vUBBase, v2cL1Base);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    if constexpr (DAV_VEC) {
        // ======================== Phase 1: V2C ========================
        using VecTileK = Tile<TileType::Vec, T, V2C_ROWS, V2C_COLS, BLayout::RowMajor, V2C_ROWS, V2C_COLS>;
        using GlobalAB = GlobalTensor<
            T, pto::Shape<1, 1, 1, V2C_ROWS, V2C_COLS>,
            pto::Stride<TOTAL_M * K, TOTAL_M * K, V2C_ROWS * V2C_COLS, K, 1>>;

        VecTileK tileA, tileB, tileC;
        TASSIGN(tileA, 0x0);
        TASSIGN(tileB, 0x4000);
        TASSIGN(tileC, 0x8000);

        uint32_t subBlockIdx = get_subblockid();
        size_t abOffset;
        if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
            abOffset = static_cast<size_t>(subBlockIdx * V2C_ROWS) * K;
        } else {
            abOffset = static_cast<size_t>(subBlockIdx) * V2C_COLS;
        }

        GlobalAB globalA(srcA + abOffset);
        GlobalAB globalB(srcB + abOffset);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        TLOAD(tileA, globalA);
        TLOAD(tileB, globalB);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        TADD(tileC, tileA, tileB);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TPUSH<BothPipe, VecTileK, SplitAxis>(pipe, tileC);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        // ======================== Phase 2: C2V ========================
        using VecTileN = Tile<TileType::Vec, T, C2V_ROWS, C2V_COLS, BLayout::RowMajor, C2V_ROWS, C2V_COLS>;
        using GlobalFOut = GlobalTensor<
            T, pto::Shape<1, 1, 1, C2V_ROWS, C2V_COLS>,
            pto::Stride<TOTAL_M * N, TOTAL_M * N, C2V_ROWS * C2V_COLS, N, 1>>;

        VecTileN vecTileHalf, tileF, tileG;
        TASSIGN(tileF, 0x10000);
        TASSIGN(tileG, 0x18000);

        size_t fOutOffset;
        if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
            fOutOffset = static_cast<size_t>(subBlockIdx * C2V_ROWS) * N;
        } else {
            fOutOffset = static_cast<size_t>(subBlockIdx) * C2V_COLS;
        }
        GlobalFOut globalF(srcF + fOutOffset);
        GlobalFOut globalOut(out + fOutOffset);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        TPOP<BothPipe, VecTileN, SplitAxis>(pipe, vecTileHalf);
        TLOAD(tileF, globalF);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        TSUB(tileG, vecTileHalf, tileF);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(globalOut, tileG);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_CUBE) {
        using PopTileV2C =
            Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, TOTAL_M, K, SLayout::RowMajor, 512>;
        using TileMatD = Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
        using GlobalD = GlobalTensor<T, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
        using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, TOTAL_M, K>;
        using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, K, N>;
        using AccTile = TileAcc<T, TOTAL_M, N, TOTAL_M, N>;

        PopTileV2C popTile;
        TileMatD matTileD;
        TASSIGN(matTileD, 0x20000);

        LeftTile leftTile;
        RightTile rightTile;
        AccTile accTile;
        TASSIGN(leftTile, 0x0);
        TASSIGN(rightTile, 0x0);
        TASSIGN(accTile, 0x0);

        GlobalD globalD(srcD);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        TPOP<BothPipe, PopTileV2C, SplitAxis>(pipe, popTile);
        TLOAD(matTileD, globalD);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        TMOV(leftTile, popTile);
        TMOV(rightTile, matTileD);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        TMATMUL(accTile, leftTile, rightTile);

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        TPUSH<BothPipe, AccTile, SplitAxis>(pipe, accTile);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopDirBoth(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF),
            reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 2) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF),
            reinterpret_cast<float*>(fifoMem));
    }
}

template void LaunchTPushPopDirBoth<1>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    void* stream);
template void LaunchTPushPopDirBoth<2>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    void* stream);
