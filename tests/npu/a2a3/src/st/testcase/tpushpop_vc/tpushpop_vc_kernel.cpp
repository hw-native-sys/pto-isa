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

template <typename QuantT, typename InT, typename OutT, int TOTAL_M, int TOTAL_K, int N, int CASE_TILE_K>
__global__ AICORE void runTPushPopVCMatmul(__gm__ uint64_t *ffts_addr, __gm__ OutT *out, __gm__ InT *srcA,
                                           __gm__ QuantT *quantB, __gm__ OutT *scale, __gm__ OutT *offset,
                                           __gm__ OutT *fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts_addr);
    constexpr uint32_t TILE_K = CASE_TILE_K;
    constexpr uint32_t HALF_TILE_K = TILE_K / 2;
    constexpr uint32_t TILE_N = N;
    constexpr uint32_t NUM_K_TILES = TOTAL_K / CASE_TILE_K;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint8_t FIFO_PERIOD = 1;
    // fifo base used for TPOP of cube side (bMatTile)
    constexpr uint32_t localFiFoBase = 0x20000;

    using VecTileProd = Tile<TileType::Vec, OutT, HALF_TILE_K, TILE_N, BLayout::RowMajor, HALF_TILE_K, TILE_N>;
    using MatTileCons =
        Tile<TileType::Mat, OutT, TILE_K, TILE_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;

    using MatPipe = TPipe<FLAG_ID, FIFOType::GM_FIFO, FIFO_DEPTH, FIFO_PERIOD, VecTileProd, MatTileCons>;
    MatPipe mPipe(fifoMem, localFiFoBase);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(InT);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(TILE_K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(TILE_N, blockAlign);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, TOTAL_M, TILE_K>,
                                 pto::Stride<TOTAL_M * TOTAL_K, TOTAL_M * TOTAL_K, TOTAL_M * TOTAL_K, TOTAL_K, 1>>;
    using GlobalOut = GlobalTensor<OutT, pto::Shape<1, 1, 1, TOTAL_M, TILE_N>,
                                   pto::Stride<TOTAL_M * TILE_N, TOTAL_M * TILE_N, TOTAL_M * TILE_N, TILE_N, 1>>;

    using TileMatA =
        Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, TOTAL_M, TILE_K, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, TOTAL_M, TILE_K>;
    using PopTile =
        Tile<TileType::Mat, OutT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;
    using RightTile = TileRight<OutT, ALIGNED_K, ALIGNED_N, TILE_K, TILE_N>;
    using AccTile = TileAcc<OutT, TOTAL_M, TILE_N, TOTAL_M, TILE_N>;

    using QuantTile = Tile<TileType::Vec, QuantT, HALF_TILE_K, TILE_N, BLayout::RowMajor, HALF_TILE_K, TILE_N>;
    using ScaleTile = Tile<TileType::Vec, OutT, HALF_TILE_K, 8, BLayout::RowMajor, -1, -1>;
    using OffsetTile = Tile<TileType::Vec, OutT, HALF_TILE_K, 8, BLayout::RowMajor, -1, -1>;

    if constexpr (DAV_VEC) {
        QuantTile quantTile;
        VecTileProd dequantTile;
        ScaleTile scaleTile(HALF_TILE_K, 1);
        OffsetTile offsetTile(HALF_TILE_K, 1);
        TASSIGN(quantTile, 0x0);
        TASSIGN(dequantTile, 0x10000);
        TASSIGN(scaleTile, 0x20000);
        TASSIGN(offsetTile, 0x28000);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif

        using GlobalQuantB =
            GlobalTensor<QuantT, pto::Shape<1, 1, 1, HALF_TILE_K, TILE_N>,
                         pto::Stride<TOTAL_K * TILE_N, TOTAL_K * TILE_N, HALF_TILE_K * TILE_N, TILE_N, 1>>;
        using GlobalScaleOffset =
            GlobalTensor<OutT, pto::Shape<1, 1, 1, HALF_TILE_K, 1>, pto::Stride<TOTAL_K, TOTAL_K, HALF_TILE_K, 1, 1>>;

        uint32_t subBlockIdx = get_subblockid();
        size_t entryOffsetVal = subBlockIdx * HALF_TILE_K * TILE_N * sizeof(OutT);

        for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
            GlobalQuantB globalQuantB(quantB + k_tile * TILE_K * TILE_N + subBlockIdx * HALF_TILE_K * TILE_N);
            GlobalScaleOffset globalScale(scale + k_tile * TILE_K + subBlockIdx * HALF_TILE_K);
            GlobalScaleOffset globalOffset(offset + k_tile * TILE_K + subBlockIdx * HALF_TILE_K);

#ifndef __PTO_AUTO__
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
#endif

            TLOAD(quantTile, globalQuantB);
            TLOAD(scaleTile, globalScale);
            TLOAD(offsetTile, globalOffset);

#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif

            TDEQUANT(dequantTile, quantTile, scaleTile, offsetTile);

#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

            mPipe.prod.setEntryOffset(entryOffsetVal);
            TPUSH(dequantTile, mPipe);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

#ifndef __PTO_AUTO__
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
#endif
    }

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        PopTile bMatTile;
        TASSIGN(aMatTile, 0x0);

        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        typename MatPipe::Consumer cons;

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
#endif

        for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
            GlobalA globalA(srcA + k_tile * TILE_K);

#ifndef __PTO_AUTO__
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
#endif

            TLOAD(aMatTile, globalA);

            TPOP(bMatTile, mPipe);

#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
#endif

            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
#endif

            if (k_tile == 0) {
                TMATMUL(accTile, aTile, bTile);
            } else {
                TMATMUL_ACC(accTile, accTile, aTile, bTile);
            }

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        GlobalOut globalOut(out);
        TSTORE<AccTile, GlobalOut>(globalOut, accTile);

#ifndef __PTO_AUTO__
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
#endif
    }
}

template <int32_t tilingKey>
void LaunchTPushPopVCMatmul(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                            uint8_t *offset, uint8_t *fifoMem, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopVCMatmul<int8_t, float, float, 16, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int8_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    } else if constexpr (tilingKey == 2) {
        runTPushPopVCMatmul<int8_t, float, float, 16, 128, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int8_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    } else if constexpr (tilingKey == 3) {
        runTPushPopVCMatmul<int8_t, float, float, 16, 256, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int8_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    } else if constexpr (tilingKey == 4) {
        runTPushPopVCMatmul<int16_t, float, float, 16, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int16_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    } else if constexpr (tilingKey == 5) {
        runTPushPopVCMatmul<int16_t, float, float, 16, 128, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int16_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    } else if constexpr (tilingKey == 6) {
        runTPushPopVCMatmul<int16_t, float, float, 16, 256, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcA),
            reinterpret_cast<int16_t *>(quantB), reinterpret_cast<float *>(scale), reinterpret_cast<float *>(offset),
            reinterpret_cast<float *>(fifoMem));
    }
}

template void LaunchTPushPopVCMatmul<1>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);
template void LaunchTPushPopVCMatmul<2>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);
template void LaunchTPushPopVCMatmul<3>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);
template void LaunchTPushPopVCMatmul<4>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);
template void LaunchTPushPopVCMatmul<5>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);
template void LaunchTPushPopVCMatmul<6>(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                                        uint8_t *offset, uint8_t *fifoMem, void *stream);