/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * Sub-block dispatch coverage for the A5 TPUSH/TPOP tile overloads.
 *
 * TPUSH/TPOP come in two forms: a two-argument overload that derives the
 * sub-block ID itself, and a three-argument overload that takes it from the
 * caller.  This testcase pins the contract of both, for both split modes, on
 * the two directions where the sub-block ID actually reaches the FIFO address
 * arithmetic:
 *
 *   - DIR_V2C  : the producer side uses it (Producer::pushVec2MatFiFo picks the
 *                L1 row window that the vector core writes into);
 *   - DIR_C2V_GM: the consumer side uses it (Consumer::popVecTileFromGMFiFo
 *                picks the GM sub-range that the vector core reads back).
 *
 * The four dispatch paths under test:
 *
 *   1. TILE_NO_SPLIT, implicit ID - the FIFO has a single logical lane, so the
 *      slot address must not depend on the hardware sub-block ID at all.
 *   2. TILE_NO_SPLIT, explicit ID - passing a non-zero ID must not move the
 *      payload either; this is what makes a compile-time literal 0 on the
 *      implicit path a semantics preserving choice.
 *   3. TILE_UP_DOWN, implicit ID - the two vector cores must keep addressing
 *      their own half through the hardware sub-block ID.
 *   4. TILE_UP_DOWN, explicit ID - the caller supplied ID must win over the
 *      hardware one.  The cases below hand each core its peer's ID, so a
 *      correct implementation produces a half-swapped result and an
 *      implementation that silently reads get_subblockid() does not.
 *
 * TILE_NO_SPLIT wires the cross-core handshake to vector sub-block 0 only
 * (Producer/Consumer::setIntraBlockBySplit skips the +VEC_CORE_ID_OFFSET flag),
 * so the no-split cases keep AIV1 out of the FIFO loop, exactly like the other
 * no-split STs in this directory.
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

/**
 * DIR_V2C: TPUSH sub-block dispatch.
 *
 * Each vector core loads its slice of B from GM, converts it to the NZ layout
 * the cube expects and pushes it into the L1 FIFO.  The cube pops the whole
 * [K, N] tile back, multiplies it with A and stores A x B to GM, so the golden
 * data pins which vector core wrote which row window of the FIFO slot.
 *
 * ExplicitId == false : two-argument TPUSH/TPOP (implicit sub-block ID).
 * ExplicitId == true  : three-argument TPUSH/TPOP, each core passing its peer's
 *                       ID (1 - get_subblockid()).
 */
template <typename T, int M, int K, int N, TileSplitAxis SplitAxis, bool ExplicitId>
__global__ AICORE void runTPushPopSubBlockDispatchV2C(__gm__ T* out, __gm__ T* srcA, __gm__ T* srcB)
{
    constexpr bool IS_NO_SPLIT = (SplitAxis == TileSplitAxis::TILE_NO_SPLIT);
    // TILE_UP_DOWN splits the K rows of the slot between the two vector cores.
    constexpr uint32_t PROD_K = IS_NO_SPLIT ? K : (K / VEC_CORES);

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_V2C, K * N * sizeof(T), FIFO_DEPTH, 2, IS_NO_SPLIT>;
    MatPipe mPipe((__gm__ void*)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x10000);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using VecTileProd = Tile<TileType::Vec, T, PROD_K, N, BLayout::RowMajor, PROD_K, N>;
    using VecTileNZ = Tile<TileType::Vec, T, PROD_K, N, BLayout::ColMajor, PROD_K, N, SLayout::RowMajor, 512>;
    using PopTile = Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using GlobalA = GlobalTensor<T, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<T, pto::Shape<1, 1, 1, PROD_K, N>, pto::Stride<K * N, K * N, PROD_K * N, N, 1>>;
    using GlobalOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;

    if constexpr (DAV_VEC) {
        const uint32_t subBlockIdx = get_subblockid();
        // TILE_NO_SPLIT only signals sub-block 0, so AIV1 must stay out of the pipe.
        bool isProducer = true;
        if constexpr (IS_NO_SPLIT) {
            isProducer = (subBlockIdx == 0);
        }

        if (isProducer) {
            VecTileProd bTile;
            VecTileNZ bTileNZ;
            TASSIGN(bTile, 0x0);
            TASSIGN(bTileNZ, 0x10000);

            GlobalB globalB(srcB + subBlockIdx * PROD_K * N);
            TLOAD(bTile, globalB);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TMOV(bTileNZ, bTile);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            if constexpr (ExplicitId) {
                // Hand each core its peer's ID: the caller supplied value must be
                // the one the FIFO address arithmetic uses.
                TPUSH<MatPipe, VecTileNZ, SplitAxis>(mPipe, bTileNZ, static_cast<int32_t>(1 - subBlockIdx));
            } else {
                TPUSH<MatPipe, VecTileNZ, SplitAxis>(mPipe, bTileNZ);
            }

            pipe_barrier(PIPE_ALL);
        }
    }

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        PopTile matFifoTile;
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        GlobalA globalA(srcA);
        TLOAD(aMatTile, globalA);

        // The Mat consumer path takes the whole slot whatever the sub-block ID is;
        // the explicit overload must behave exactly like the implicit one here.
        if constexpr (ExplicitId) {
            TPOP<MatPipe, PopTile, SplitAxis>(mPipe, matFifoTile, 0);
        } else {
            TPOP<MatPipe, PopTile, SplitAxis>(mPipe, matFifoTile);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aTile, aMatTile);
        TMOV(bTile, matFifoTile);
        TFREE<MatPipe, SplitAxis>(mPipe);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(accTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        GlobalOut globalOut(out);
        TSTORE<AccTile, GlobalOut>(globalOut, accTile);

        pipe_barrier(PIPE_ALL);
    }
}

/**
 * DIR_C2V_GM: TPOP sub-block dispatch.
 *
 * The cube computes A x B and pushes the whole [M, N] accumulator into a GM
 * FIFO slot.  Each vector core pops its own sub-range back into UB and stores
 * it to its own row window of the output, so the golden data pins which vector
 * core read which sub-range of the slot.
 */
template <typename T, int M, int K, int N, TileSplitAxis SplitAxis, bool ExplicitId>
__global__ AICORE void runTPushPopSubBlockDispatchC2VGm(
    __gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* fifoMem)
{
    constexpr bool IS_NO_SPLIT = (SplitAxis == TileSplitAxis::TILE_NO_SPLIT);
    // TILE_UP_DOWN splits the M rows of the slot between the two vector cores.
    constexpr uint32_t CONS_M = IS_NO_SPLIT ? M : (M / VEC_CORES);

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t LOCAL_FIFO_BASE = 0x0;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V_GM, M * N * sizeof(T), FIFO_DEPTH, 2, IS_NO_SPLIT>;
    MatPipe mPipe((__gm__ void*)(uint64_t)fifoMem, LOCAL_FIFO_BASE, 0x0);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using GlobalA = GlobalTensor<T, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<T, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<T, pto::Shape<1, 1, 1, CONS_M, N>, pto::Stride<M * N, M * N, CONS_M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, M, K>;
    using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;
    using VecTileCons = Tile<TileType::Vec, T, CONS_M, N, BLayout::RowMajor, CONS_M, N>;

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

        // The Acc producer path never looks at the sub-block ID; the split axis
        // only decides how many vector cores are signalled.
        TPUSH<MatPipe, AccTile, SplitAxis>(mPipe, accTile);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        const uint32_t subBlockIdx = get_subblockid();
        // TILE_NO_SPLIT only signals sub-block 0, so AIV1 must stay out of the pipe.
        bool isConsumer = true;
        if constexpr (IS_NO_SPLIT) {
            isConsumer = (subBlockIdx == 0);
        }

        if (isConsumer) {
            VecTileCons vecTile; // TPOP assigns the UB address from the local FIFO base

            if constexpr (ExplicitId) {
                // Hand each core its peer's ID: the caller supplied value must be
                // the one the FIFO address arithmetic uses.
                TPOP<MatPipe, VecTileCons, SplitAxis>(mPipe, vecTile, static_cast<int32_t>(1 - subBlockIdx));
            } else {
                TPOP<MatPipe, VecTileCons, SplitAxis>(mPipe, vecTile);
            }

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            GlobalOut globalOut(out + subBlockIdx * CONS_M * N);
            TSTORE(globalOut, vecTile);

            pipe_barrier(PIPE_ALL);
        }
    }
}

template <int32_t tilingKey>
void LaunchTPushPopSubBlockDispatchV2C(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopSubBlockDispatchV2C<float, 16, 64, 32, TileSplitAxis::TILE_NO_SPLIT, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB));
    } else if constexpr (tilingKey == 2) {
        runTPushPopSubBlockDispatchV2C<float, 16, 64, 32, TileSplitAxis::TILE_NO_SPLIT, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB));
    } else if constexpr (tilingKey == 3) {
        runTPushPopSubBlockDispatchV2C<float, 16, 64, 32, TileSplitAxis::TILE_UP_DOWN, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB));
    } else if constexpr (tilingKey == 4) {
        runTPushPopSubBlockDispatchV2C<float, 16, 64, 32, TileSplitAxis::TILE_UP_DOWN, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB));
    }
}

template <int32_t tilingKey>
void LaunchTPushPopSubBlockDispatchC2VGm(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream)
{
    if constexpr (tilingKey == 5) {
        runTPushPopSubBlockDispatchC2VGm<float, 32, 32, 64, TileSplitAxis::TILE_NO_SPLIT, false>
            <<<1, nullptr, stream>>>(
                reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
                reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 6) {
        runTPushPopSubBlockDispatchC2VGm<float, 32, 32, 64, TileSplitAxis::TILE_UP_DOWN, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 7) {
        runTPushPopSubBlockDispatchC2VGm<float, 32, 32, 64, TileSplitAxis::TILE_UP_DOWN, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA), reinterpret_cast<float*>(srcB),
            reinterpret_cast<float*>(fifoMem));
    }
}

template void LaunchTPushPopSubBlockDispatchV2C<1>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream);
template void LaunchTPushPopSubBlockDispatchV2C<2>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream);
template void LaunchTPushPopSubBlockDispatchV2C<3>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream);
template void LaunchTPushPopSubBlockDispatchV2C<4>(uint8_t* out, uint8_t* srcA, uint8_t* srcB, void* stream);
template void LaunchTPushPopSubBlockDispatchC2VGm<5>(
    uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopSubBlockDispatchC2VGm<6>(
    uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopSubBlockDispatchC2VGm<7>(
    uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
