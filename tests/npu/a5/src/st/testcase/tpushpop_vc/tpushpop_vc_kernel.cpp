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
    typename QuantT, typename InT, typename OutT, int TOTAL_M, int TOTAL_K, int N, int CASE_TILE_K,
    TileSplitAxis SplitAxis = TileSplitAxis::TILE_UP_DOWN>
__global__ AICORE void runTPushPopVCMatmul(
    __gm__ OutT* out, __gm__ InT* srcA, __gm__ QuantT* quantB, __gm__ OutT* scale, __gm__ OutT* offset)
{
    constexpr uint32_t TILE_K = CASE_TILE_K;
    constexpr uint32_t TILE_N = N;
    constexpr uint32_t NUM_K_TILES = TOTAL_K / CASE_TILE_K;

    // TILE_UP_DOWN splits along K rows; TILE_LEFT_RIGHT splits along N columns
    constexpr uint32_t PROD_K = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TILE_K / VEC_CORES) : TILE_K;
    constexpr uint32_t PROD_N = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (TILE_N / VEC_CORES) : TILE_N;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint8_t FIFO_PERIOD = 1;

    using VecTileProd = Tile<TileType::Vec, OutT, PROD_K, PROD_N, BLayout::RowMajor, PROD_K, PROD_N>;
    using VecTileNZ =
        Tile<TileType::Vec, OutT, PROD_K, PROD_N, BLayout::ColMajor, PROD_K, PROD_N, SLayout::RowMajor, 512>;
    using MatTileCons =
        Tile<TileType::Mat, OutT, TILE_K, TILE_N, BLayout::ColMajor, TILE_K, TILE_N, SLayout::RowMajor, 512>;

    // pipe init
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

    using QuantTile = Tile<TileType::Vec, QuantT, PROD_K, PROD_N, BLayout::RowMajor, PROD_K, PROD_N>;
    // For TILE_LEFT_RIGHT both vector cores work on the same K rows, so each needs all PROD_K scale/offset values
    using ScaleTile = Tile<TileType::Vec, OutT, PROD_K, 8, BLayout::RowMajor, -1, -1>;
    using OffsetTile = Tile<TileType::Vec, OutT, PROD_K, 8, BLayout::RowMajor, -1, -1>;

    if constexpr (DAV_VEC) {
        QuantTile quantTile;
        VecTileProd dequantTile;
        VecTileNZ dequantTileNZ;
        ScaleTile scaleTile(PROD_K, 1);
        OffsetTile offsetTile(PROD_K, 1);
        TASSIGN(quantTile, 0x0);
        TASSIGN(dequantTile, 0x10000);
        TASSIGN(dequantTileNZ, 0x18000);
        TASSIGN(scaleTile, 0x20000);
        TASSIGN(offsetTile, 0x28000);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        // Row stride is always TILE_N so TILE_LEFT_RIGHT sub-tiles (non-contiguous cols) are loaded correctly
        using GlobalQuantB = GlobalTensor<
            QuantT, pto::Shape<1, 1, 1, PROD_K, PROD_N>,
            pto::Stride<TOTAL_K * TILE_N, TOTAL_K * TILE_N, PROD_K * TILE_N, TILE_N, 1>>;
        using GlobalScaleOffset =
            GlobalTensor<OutT, pto::Shape<1, 1, 1, PROD_K, 1>, pto::Stride<TOTAL_K, TOTAL_K, PROD_K, 1, 1>>;

        uint32_t subBlockIdx = get_subblockid();

        for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
            size_t quantBOffset;
            size_t scaleOffsetOffset;
            if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
                // each subblock handles a different row range of quantB/scale/offset
                quantBOffset = static_cast<size_t>(k_tile * TILE_K + subBlockIdx * PROD_K) * TILE_N;
                scaleOffsetOffset = static_cast<size_t>(k_tile * TILE_K + subBlockIdx * PROD_K);
            } else {
                // each subblock handles a different column range; scale/offset are shared (same K rows)
                quantBOffset = static_cast<size_t>(k_tile * TILE_K) * TILE_N + subBlockIdx * PROD_N;
                scaleOffsetOffset = static_cast<size_t>(k_tile * TILE_K);
            }
            GlobalQuantB globalQuantB(quantB + quantBOffset);
            GlobalScaleOffset globalScale(scale + scaleOffsetOffset);
            GlobalScaleOffset globalOffset(offset + scaleOffsetOffset);

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

            // push vector tile to fifo
            TPUSH<MatPipe, VecTileNZ, SplitAxis>(mPipe, dequantTileNZ);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        PopTile matFifoTile;
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
            // pop mat tile from fifo
            TPOP<MatPipe, PopTile, SplitAxis>(mPipe, matFifoTile);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            TMOV(aTile, aMatTile);
            TMOV(bTile, matFifoTile);
            TFREE<MatPipe, SplitAxis>(mPipe);

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

template <
    typename QuantT, typename InT, typename OutT, int TOTAL_M, int TOTAL_K, int N, int CASE_TILE_K,
    TileSplitAxis SplitAxis>
void LaunchTPushPopVCMatmulImpl(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream)
{
    runTPushPopVCMatmul<QuantT, InT, OutT, TOTAL_M, TOTAL_K, N, CASE_TILE_K, SplitAxis><<<1, nullptr, stream>>>(
        reinterpret_cast<OutT*>(out), reinterpret_cast<InT*>(srcA), reinterpret_cast<QuantT*>(quantB),
        reinterpret_cast<OutT*>(scale), reinterpret_cast<OutT*>(offset));
}

template <int32_t tilingKey>
void LaunchTPushPopVCMatmul(uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream)
{
    // Keys 1-6: TILE_UP_DOWN (split along K rows)
    if constexpr (tilingKey == 1) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 64, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 2) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 128, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 3) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 256, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 4) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 64, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 5) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 128, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 6) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 256, 32, 64, TileSplitAxis::TILE_UP_DOWN>(
            out, srcA, quantB, scale, offset, stream);
        // Keys 7-12: TILE_LEFT_RIGHT (split along N columns)
        // int8 cases use N=64 so PROD_N=32, satisfying 32*sizeof(int8_t)=32 bytes alignment
    } else if constexpr (tilingKey == 7) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 64, 64, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 8) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 128, 64, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 9) {
        LaunchTPushPopVCMatmulImpl<int8_t, float, float, 16, 256, 64, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 10) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 64, 32, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 11) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 128, 32, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    } else if constexpr (tilingKey == 12) {
        LaunchTPushPopVCMatmulImpl<int16_t, float, float, 16, 256, 32, 64, TileSplitAxis::TILE_LEFT_RIGHT>(
            out, srcA, quantB, scale, offset, stream);
    }
}

template void LaunchTPushPopVCMatmul<1>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<2>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<3>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<4>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<5>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<6>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<7>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<8>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<9>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<10>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<11>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);
template void LaunchTPushPopVCMatmul<12>(
    uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, void* stream);

/**
 * V2C (UB→L1) test for TPUSH with explicit subBlockId (TILE_UP_DOWN).
 *
 * Vec:  load src0/src1 [M/2,K], TADD → resTile, TMOV NZ, TPUSH(resTile, subBlockId) into cube L1
 * Cube: TPOP resTile [M,K] from L1, TLOAD src2 [K,N] from GM, TMATMUL, TSTORE out [M,N]
 */
template <typename T, int TOTAL_M, int K, int N>
__global__ AICORE void runTPushPopVCSubBlockId(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2)
{
    constexpr uint32_t VEC_ROWS = TOTAL_M / VEC_CORES;
    constexpr uint32_t VEC_COLS = K;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t v2cL1Base = 0x10000;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_V2C, TOTAL_M * K * sizeof(T), FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void*)(uint64_t)0x0, (uint32_t)0x0, v2cL1Base);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    if constexpr (DAV_VEC) {
        using VecTile = Tile<TileType::Vec, T, VEC_ROWS, VEC_COLS, BLayout::RowMajor, VEC_ROWS, VEC_COLS>;
        using VecTileNZ =
            Tile<TileType::Vec, T, VEC_ROWS, VEC_COLS, BLayout::ColMajor, VEC_ROWS, VEC_COLS, SLayout::RowMajor, 512>;
        using GlobalSrc = GlobalTensor<
            T, pto::Shape<1, 1, 1, VEC_ROWS, VEC_COLS>,
            pto::Stride<TOTAL_M * K, TOTAL_M * K, VEC_ROWS * VEC_COLS, K, 1>>;

        VecTile src0Tile, src1Tile, resTile;
        VecTileNZ resTileNZ;
        TASSIGN(src0Tile, 0x0);
        TASSIGN(src1Tile, 0x4000);
        TASSIGN(resTile, 0x8000);
        TASSIGN(resTileNZ, 0xC000);

        int32_t subBlockId = static_cast<int32_t>(get_subblockid());
        size_t srcOffset = static_cast<size_t>(subBlockId) * VEC_ROWS * K;

        GlobalSrc globalSrc0(src0 + srcOffset);
        GlobalSrc globalSrc1(src1 + srcOffset);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        TLOAD(src0Tile, globalSrc0);
        TLOAD(src1Tile, globalSrc1);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        TADD(resTile, src0Tile, src1Tile);
        TMOV(resTileNZ, resTile);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TPUSH<MatPipe, VecTileNZ, TileSplitAxis::TILE_UP_DOWN>(mPipe, resTileNZ, subBlockId);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_CUBE) {
        using PopTile =
            Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, TOTAL_M, K, SLayout::RowMajor, 512>;
        using TileMatRight =
            Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
        using GlobalSrc2 = GlobalTensor<T, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
        using GlobalOut =
            GlobalTensor<T, pto::Shape<1, 1, 1, TOTAL_M, N>, pto::Stride<TOTAL_M * N, TOTAL_M * N, TOTAL_M * N, N, 1>>;
        using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, TOTAL_M, K>;
        using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, K, N>;
        using AccTile = TileAcc<T, TOTAL_M, N, TOTAL_M, N>;

        PopTile resTile;
        TileMatRight src2Tile;
        TASSIGN(src2Tile, 0x0);

        LeftTile leftTile;
        RightTile rightTile;
        AccTile accTile;
        TASSIGN(leftTile, 0x0);
        TASSIGN(rightTile, 0x0);
        TASSIGN(accTile, 0x0);

        GlobalSrc2 globalSrc2(src2);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        TPOP<MatPipe, PopTile, TileSplitAxis::TILE_UP_DOWN>(mPipe, resTile);
        TLOAD(src2Tile, globalSrc2);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        TMOV(leftTile, resTile);
        TMOV(rightTile, src2Tile);
        TFREE<MatPipe, TileSplitAxis::TILE_UP_DOWN>(mPipe);

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(accTile, leftTile, rightTile);

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

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
void LaunchTPushPopVCSubBlockId(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopVCSubBlockId<float, 128, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1),
            reinterpret_cast<float*>(src2));
    }
}

template void LaunchTPushPopVCSubBlockId<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);