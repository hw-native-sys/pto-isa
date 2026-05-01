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

template <typename InT, typename OutT, int M, int K, int N, int RepeatN>
__global__ AICORE void runTPushTpopSubtile(__gm__ uint64_t *fftsAddr, __gm__ OutT *out, __gm__ InT *srcA,
                                           __gm__ InT *srcB, __gm__ OutT *fifoMem)
{
    set_ffts_base_addr((uint64_t)fftsAddr);

    constexpr int FULL_N = N * RepeatN;
    constexpr int VEC_M = 16;
    constexpr int VEC_LOAD_TIMES = M / (VEC_CORES * VEC_M);

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t LOCAL_FIFO_BASE = 0x0;

    // slot size is [128, 512]
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, M * FULL_N * sizeof(OutT), FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void *)(uint64_t)fifoMem, LOCAL_FIFO_BASE, 0x0);

    using AccTile = TileAcc<OutT, M, N, M, N>;
    using SlotGlobal = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, FULL_N>, pto::Stride<1, 1, 1, FULL_N, 1>>;
    using PopGlobal = GlobalTensor<OutT, pto::Shape<1, 1, 1, M / VEC_CORES, FULL_N>, pto::Stride<1, 1, 1, FULL_N, 1>>;
    using VecTileData = Tile<TileType::Vec, OutT, RepeatN * VEC_M, N, BLayout::RowMajor, RepeatN * VEC_M, N>;
    using LoadGlobal3D = GlobalTensor<OutT, pto::Shape<1, 1, RepeatN, VEC_M, N>, pto::Stride<1, 1, N, FULL_N, 1>>;
    using OutGlobal3D = GlobalTensor<OutT, pto::Shape<1, 1, RepeatN, VEC_M, N>, pto::Stride<1, 1, N, FULL_N, 1>>;

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(InT);
    constexpr int ALIGNED_M = CeilAlign<int>(M, 16);
    constexpr int ALIGNED_K = CeilAlign<int>(K, blockAlign);
    constexpr int ALIGNED_N = CeilAlign<int>(N, blockAlign);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<InT, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using TileMatA = Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, K, N>;

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        TileMatB bMatTile;
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

        TLOAD(aMatTile, globalA);
        TLOAD(bMatTile, globalB);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(accTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        SlotGlobal pushGlobal;
        TALLOC<MatPipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(mPipe, pushGlobal);

        using StoreGlobal = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, N>, pto::Stride<1, 1, 1, FULL_N, 1>>;
        StoreGlobal storeGlobal;
        for (int nTile = 0; nTile < RepeatN; ++nTile) { // 4
            // store [128, 128] data into one horizontal block of the [128, 512] slot
            TASSIGN(storeGlobal, pushGlobal.data() + nTile * N);
            TSTORE(storeGlobal, accTile);
        } // [128, 512] in Global memory

        TPUSH<MatPipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(mPipe, pushGlobal);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        VecTileData vecTile[2];
        VecTileData dstTile[2];
        TASSIGN(vecTile[0], 0x0);
        TASSIGN(vecTile[1], 0x10000);
        TASSIGN(dstTile[0], 0x20000);
        TASSIGN(dstTile[1], 0x30000);

        uint32_t subBlockIdx = get_subblockid();

        PopGlobal popGlobal;
        TPOP<MatPipe, PopGlobal, TileSplitAxis::TILE_UP_DOWN>(mPipe, popGlobal);

        LoadGlobal3D loadGlobal;
        OutGlobal3D outGlobal;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        for (int rowSlice = 0; rowSlice < VEC_LOAD_TIMES; ++rowSlice) { // 4
            const uint32_t bufferIndex = static_cast<uint32_t>(rowSlice & 1);
            const size_t vecBaseRow = static_cast<size_t>(M / VEC_CORES) * static_cast<size_t>(subBlockIdx);
            const size_t localRowOffset = static_cast<size_t>(rowSlice * VEC_M);
            const size_t outRowOffset = (vecBaseRow + localRowOffset) * static_cast<size_t>(FULL_N);
            __gm__ OutT *loadPtr = popGlobal.data() + localRowOffset * static_cast<size_t>(FULL_N);

            TASSIGN(loadGlobal, loadPtr);
            wait_flag(PIPE_MTE3, PIPE_MTE2, rowSlice & 1);
            TLOAD(vecTile[bufferIndex], loadGlobal);
            set_flag(PIPE_MTE2, PIPE_V, rowSlice & 1);
            wait_flag(PIPE_MTE2, PIPE_V, rowSlice & 1);

            TADDS(dstTile[bufferIndex], vecTile[bufferIndex], static_cast<OutT>(3.14));
            set_flag(PIPE_V, PIPE_MTE3, rowSlice & 1);
            wait_flag(PIPE_V, PIPE_MTE3, rowSlice & 1);

            TASSIGN(outGlobal, out + outRowOffset);
            TSTORE(outGlobal, dstTile[bufferIndex]);
            set_flag(PIPE_MTE3, PIPE_MTE2, rowSlice & 1);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TFREE<MatPipe, PopGlobal, TileSplitAxis::TILE_UP_DOWN>(mPipe, popGlobal);

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushTpopSubtile(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *fifoMem, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTPushTpopSubtile<half, float, 128, 128, 128, 4><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<half *>(srcA),
            reinterpret_cast<half *>(srcB), reinterpret_cast<float *>(fifoMem));
    }
}

template void LaunchTPushTpopSubtile<1>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *fifoMem,
                                        void *stream);
