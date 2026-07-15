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

// Computation flow (DIR_BOTH pipe, a5 local FIFOs):
//   TILE_UP_DOWN:    split along rows    — each vector handles [M/2, K] and [M/2, N]
//   TILE_LEFT_RIGHT: split along columns — each vector handles [M, K/2] and [M, N/2]
//
//   Vec: tileC = tileA + tileB  (per vector core)
//   Vec→Cube (V2C): TPUSH tileC_NZ → Mat FIFO in L1, combined [TOTAL_M, K]
//   Cube: TPOP [TOTAL_M,K] from Mat FIFO, TLOAD tileD[K,N], tileE = TMATMUL
//   Cube→Vec (C2V): TPUSH tileE[TOTAL_M,N] → Vec FIFO in UB, vectors pop their portion
//   Vec: tileG = tileE_part - tileF, TSTORE tileG
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis = TileSplitAxis::TILE_UP_DOWN>
__global__ AICORE void runTPushPopDirBoth(__gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* srcD, __gm__ T* srcF)
{
    constexpr uint32_t VEC_M = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TOTAL_M / VEC_CORES) : TOTAL_M;
    constexpr uint32_t VEC_K = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (K / VEC_CORES) : K;
    constexpr uint32_t VEC_N = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (N / VEC_CORES) : N;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t SLOT_SIZE = TOTAL_M * K * sizeof(T);

    using BothPipe = TPipe<FLAG_ID, Direction::DIR_BOTH, SLOT_SIZE, FIFO_DEPTH>;

    constexpr uint32_t v2cL1Base = 0x20000;
    constexpr uint32_t c2vUBBase = 0x0;

    BothPipe pipe((__gm__ void*)(uint64_t)0x0, c2vUBBase, v2cL1Base);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    if constexpr (DAV_VEC) {
        // ======================== Phase 1: V2C ========================
        using VecTileK = Tile<TileType::Vec, T, VEC_M, VEC_K, BLayout::RowMajor, VEC_M, VEC_K>;
        using VecTileNZ = Tile<TileType::Vec, T, VEC_M, VEC_K, BLayout::ColMajor, VEC_M, VEC_K, SLayout::RowMajor, 512>;
        using GlobalAB =
            GlobalTensor<T, pto::Shape<1, 1, 1, VEC_M, VEC_K>, pto::Stride<TOTAL_M * K, TOTAL_M * K, VEC_M * K, K, 1>>;

        VecTileK tileA, tileB, tileC;
        VecTileNZ tileC_NZ;
        TASSIGN(tileA, 0x0);
        TASSIGN(tileB, 0x4000);
        TASSIGN(tileC, 0x8000);
        TASSIGN(tileC_NZ, 0xC000);

        uint32_t subBlockIdx = get_subblockid();
        size_t vecOffsetAB;
        if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
            vecOffsetAB = static_cast<size_t>(subBlockIdx * VEC_M) * K;
        } else {
            vecOffsetAB = static_cast<size_t>(subBlockIdx) * VEC_K;
        }

        GlobalAB globalA(srcA + vecOffsetAB);
        GlobalAB globalB(srcB + vecOffsetAB);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        TLOAD(tileA, globalA);
        TLOAD(tileB, globalB);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        TADD(tileC, tileA, tileB);
        TMOV(tileC_NZ, tileC);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TPUSH<BothPipe, VecTileNZ, SplitAxis>(pipe, tileC_NZ);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        // ======================== Phase 2: C2V ========================
        using VecTileN = Tile<TileType::Vec, T, VEC_M, VEC_N, BLayout::RowMajor, VEC_M, VEC_N>;
        using GlobalFOut =
            GlobalTensor<T, pto::Shape<1, 1, 1, VEC_M, VEC_N>, pto::Stride<TOTAL_M * N, TOTAL_M * N, VEC_M * N, N, 1>>;

        VecTileN vecTileHalf, tileF, tileG;
        TASSIGN(tileF, 0x10000);
        TASSIGN(tileG, 0x18000);

        size_t vecOffsetFOut;
        if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
            vecOffsetFOut = static_cast<size_t>(subBlockIdx * VEC_M) * N;
        } else {
            vecOffsetFOut = static_cast<size_t>(subBlockIdx) * VEC_N;
        }
        GlobalFOut globalF(srcF + vecOffsetFOut);
        GlobalFOut globalOut(out + vecOffsetFOut);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        TPOP<BothPipe, VecTileN, SplitAxis>(pipe, vecTileHalf);
        TLOAD(tileF, globalF);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        TSUB(tileG, vecTileHalf, tileF);
        TFREE<BothPipe, SplitAxis>(pipe);

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
        TASSIGN(matTileD, 0x0);

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
        TFREE<BothPipe, SplitAxis>(pipe);

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
void LaunchTPushPopDirBoth(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF));
    } else if constexpr (tilingKey == 2) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF));
    }
}

template void LaunchTPushPopDirBoth<1>(
    uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, void* stream);
template void LaunchTPushPopDirBoth<2>(
    uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, void* stream);
