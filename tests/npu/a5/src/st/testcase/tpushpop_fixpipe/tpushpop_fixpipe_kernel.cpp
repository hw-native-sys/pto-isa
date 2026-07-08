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
#include <pto/common/fixpipe.hpp>

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
AICORE constexpr inline T CeilAlign(T value, T align)
{
    if (align == 0) {
        return 0;
    }
    return (value + align - 1) / align * align;
}

template <typename InT, typename AccT, typename OutT, int M, int K, int N>
__global__ AICORE void RunTPushPopFixpipe(__gm__ OutT *out, __gm__ InT *srcA, __gm__ InT *srcB)
{
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t blockAlign = (sizeof(InT) == 1) ? 32U : (C0_SIZE_BYTE / sizeof(InT));
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using AccTile = TileAcc<AccT, ALIGNED_M, ALIGNED_N, M, N>;
    using VecTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using OutTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using FixpipeConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::F322F16>;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * M * N, FIFO_DEPTH, 2, true>;
    MatPipe pipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<InT, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, K, N>;

    if constexpr (DAV_CUBE) {
        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

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

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

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

        TPUSH<MatPipe, AccTile, FixpipeConfig>(pipe, accTile);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            VecTile vecTile;
            OutTile outTile;
            TASSIGN(outTile, 0x20000);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TPOP<MatPipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TADDS(outTile, vecTile, static_cast<OutT>(1.0f));

            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            GlobalOut globalOut(out);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        pipe_barrier(PIPE_ALL);
    }
}

template <typename InT, typename AccT, typename OutT, int M, int K, int N>
__global__ AICORE void RunTPushPopFixpipeDEQF16(__gm__ OutT *out, __gm__ InT *srcA, __gm__ InT *srcB)
{
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t blockAlign = (sizeof(InT) == 1) ? 32U : (C0_SIZE_BYTE / sizeof(InT));
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using AccTile = TileAcc<AccT, ALIGNED_M, ALIGNED_N, M, N>;
    using VecTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using OutTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using FixpipeConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::DEQF16>;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * M * N, FIFO_DEPTH, 2, true>;
    MatPipe pipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<InT, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, K, N>;

    if constexpr (DAV_CUBE) {
        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

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

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

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

        TPUSH<MatPipe, AccTile, FixpipeConfig>(pipe, accTile);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            VecTile vecTile;
            OutTile outTile;
            TASSIGN(outTile, 0x20000);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TPOP<MatPipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TADDS(outTile, vecTile, static_cast<OutT>(1.0f));

            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            GlobalOut globalOut(out);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        pipe_barrier(PIPE_ALL);
    }
}

template <typename InT, typename AccT, typename OutT, int M, int K, int N>
__global__ AICORE void RunTPushPopFixpipeVDEQF16(__gm__ OutT *out, __gm__ InT *srcA, __gm__ InT *srcB,
                                                 __gm__ uint64_t *srcQuant)
{
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t blockAlign = (sizeof(InT) == 1) ? 32U : (C0_SIZE_BYTE / sizeof(InT));
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using AccTile = TileAcc<AccT, ALIGNED_M, ALIGNED_N, M, N>;
    using VecTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using OutTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using FixpipeConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::VDEQF16>;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * M * N, FIFO_DEPTH, 2, true>;
    MatPipe pipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<InT, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    using GlobalQuant = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<TileType::Mat, uint64_t, 1, ALIGNED_N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;
    using FbTile = Tile<TileType::Scaling, uint64_t, 1, ALIGNED_N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, K, N>;

    if constexpr (DAV_CUBE) {
        GlobalA globalA(srcA);
        GlobalB globalB(srcB);
        GlobalQuant globalQuant(srcQuant);

        TileMatA aMatTile;
        TileMatB bMatTile;
        TileMatFbData fbMatTile;
        FbTile fbTile;
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;

        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);
        TASSIGN(fbMatTile, 0x40000);
        TASSIGN(fbTile, 0x0);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

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

        TLOAD(fbMatTile, globalQuant);
        set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID2);

        TMOV(fbTile, fbMatTile);
        TPUSH<MatPipe, AccTile, FixpipeConfig>(pipe, accTile);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            VecTile vecTile;
            OutTile outTile;
            TASSIGN(outTile, 0x20000);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TPOP<MatPipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TADDS(outTile, vecTile, static_cast<OutT>(1.0f));

            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            GlobalOut globalOut(out);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        pipe_barrier(PIPE_ALL);
    }
}

template <typename InT, typename AccT, typename OutT, int M, int K, int N>
__global__ AICORE void RunTPushPopFixpipeF322BF16(__gm__ OutT *out, __gm__ InT *srcA, __gm__ InT *srcB)
{
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t blockAlign = (sizeof(InT) == 1) ? 32U : (C0_SIZE_BYTE / sizeof(InT));
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using AccTile = TileAcc<AccT, ALIGNED_M, ALIGNED_N, M, N>;
    using VecTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using OutTile = Tile<TileType::Vec, OutT, M, N, BLayout::RowMajor, M, N>;
    using FixpipeConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::F322BF16>;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(OutT) * M * N, FIFO_DEPTH, 2, true>;
    MatPipe pipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x0);

    using GlobalA = GlobalTensor<InT, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<InT, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<OutT, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, InT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, InT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<InT, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<InT, ALIGNED_K, ALIGNED_N, K, N>;

    if constexpr (DAV_CUBE) {
        GlobalA globalA(srcA);
        GlobalB globalB(srcB);

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

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

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

        TPUSH<MatPipe, AccTile, FixpipeConfig>(pipe, accTile);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            VecTile vecTile;
            OutTile outTile;
            TASSIGN(outTile, 0x20000);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TPOP<MatPipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TADDS(outTile, vecTile, static_cast<OutT>(1.0f));

            TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            GlobalOut globalOut(out);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopFixpipe(uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTPushPopFixpipe<half, float, half, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcA), reinterpret_cast<half *>(srcB));
    } else if constexpr (tilingKey == 2) {
        RunTPushPopFixpipeF322BF16<half, float, bfloat16_t, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(srcA), reinterpret_cast<half *>(srcB));
    } else if constexpr (tilingKey == 3) {
        RunTPushPopFixpipeDEQF16<int8_t, int32_t, half, 128, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(srcA), reinterpret_cast<int8_t *>(srcB));
    } else if constexpr (tilingKey == 4) {
        RunTPushPopFixpipeVDEQF16<int8_t, int32_t, half, 128, 128, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(srcA),
                                     reinterpret_cast<int8_t *>(srcB), reinterpret_cast<uint64_t *>(srcQuant));
    }
}

template void LaunchTPushPopFixpipe<1>(uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant, void *stream);
template void LaunchTPushPopFixpipe<2>(uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant, void *stream);
template void LaunchTPushPopFixpipe<3>(uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant, void *stream);
template void LaunchTPushPopFixpipe<4>(uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant, void *stream);
