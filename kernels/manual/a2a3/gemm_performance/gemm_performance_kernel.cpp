/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)

// Pipeline mental model (instruction-level):
// - TLOAD     (GM -> L1):   fill aMatTile/bMatTile
// - TEXTRACT  (L1 -> L0):   slice aMatTile/bMatTile into aTile/bTile for the current baseK
// - TMATMUL   (Cube):       cTile = A*B (accumulated over K)
// - TSTORE    (L0C -> GM):  write cTile back to GM
//
// The code still uses PIPE_MTE* events for synchronization because those are the underlying hardware pipes;
// comments refer to the high-level PTO instructions to make tuning easier.

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL(cTile, aTile, bTile);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);
    }
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id)
{
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}
template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id)
{
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <
    typename T, typename U, typename S, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN>
AICORE inline void InitGMOffsets(
    __gm__ U*& currentSrc0, __gm__ S*& currentSrc1, __gm__ T*& currentDst, __gm__ T* out, __gm__ U* src0,
    __gm__ S* src1)
{
    // Work partition (SPMD-style):
    // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
    // - It reads the corresponding A panel [singleCoreM, K] and B panel [K, singleCoreN].
    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentDst = out + gmOffsetC;
}

template <
    typename T, typename U, typename S, int m, int k, int n, uint32_t baseM, uint32_t baseK, uint32_t baseN,
    uint32_t stepKa, uint32_t stepKb, uint32_t singleCoreK, typename TileMatA, typename TileMatB, typename LeftTile,
    typename RightTile, typename ResTile>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t i, uint32_t j, __gm__ U* currentSrc0, __gm__ S* currentSrc1, TileMatA aMatTile[BUFFER_NUM],
    TileMatB bMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile& cTile,
    uint8_t& mte2DBFlag, uint8_t& mte1DBFlag)
{
    // A panel staged by each TLOAD (GM->L1) when kModstepKa == 0: [baseM, baseK * stepKa]
    using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
    using NDsingleCoreShapeA = BaseShape2D<U, m, k, Layout::ND>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

    // B panel staged by each TLOAD (GM->L1) when kModstepKa == 0: [baseK * stepKb, baseN]
    using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, k, n, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    const uint32_t kModstepKa = kIter % stepKa;

    // TLOAD stage:
    // - Every stepKa iterations, load a larger [baseM, baseK * stepKa] panel into L1 and then slice it with TEXTRACT.
    // - Double buffering is driven by mte2DBFlag.
    if (kModstepKa == 0) {
        GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
        GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

        // Wait until TEXTRACT is done with this L1 buffer before reusing it.
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0; // mte2DBFlag reversed
    // Wait until TMATMUL is done with the current L0A/L0B buffer before overwriting it via TEXTRACT.
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    // TEXTRACT stage: slice the loaded L1 panel into the baseK chunk we need this iteration.
    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);

    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);

    if ((kIter + 1) % stepKa == 0) {
        // Allow the next TLOAD to reuse this L1 slot.
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    // TMATMUL stage: compute (or accumulate) into cTile.
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    // Signal that TMATMUL is done, so the next iteration may TEXTRACT into the other ping-pong slot.
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <
    typename T, typename U, typename S, int m, int n, uint32_t baseM, uint32_t baseN, uint32_t singleCoreK,
    typename ResTile>
AICORE inline void StoreResult(ResTile& cTile, __gm__ T* currentDst, uint32_t i, uint32_t j)
{
    // TSTORE stage: write the finished C tile [baseM, baseN] back to GM.
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    // the data size read from L0C after single k loop is [baseM, baseN]
    using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
    using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>; // stride use global C m n
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

    GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

AICORE inline void InitSyncFlags()
{
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}

template <
    typename T, typename U, typename S, typename B, uint32_t blockDim, int m, int k, int n, int validM, int validK,
    int validN, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK,
    uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmE2E(__gm__ T* out, __gm__ U* src0, __gm__ S* src1)
{
    __gm__ U* currentSrc0 = nullptr;
    __gm__ S* currentSrc1 = nullptr;
    __gm__ T* currentDst = nullptr;
    InitGMOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>(
        currentSrc0, currentSrc1, currentDst, out, src0, src1);

    using TileMatA =
        Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
    using TileMatB =
        Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;

    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    ResTile cTile;

    // L1 staging buffers (aMatTile/bMatTile) are double-buffered for TLOAD overlap.
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U));
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) + baseK * baseN * stepKb * sizeof(U));

    // L0A/L0B ping-pong buffers (TEXTRACT destination).
    // Keep each per-buffer footprint <= 32 KiB to fit in a ping/pang slot.
    TASSIGN(aTile[0], 0x0);                     // L0A ping
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES); // L0A pang
    TASSIGN(bTile[0], 0x0);                     // L0B ping
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES); // L0B pang
    TASSIGN(cTile, 0x0);

    constexpr uint32_t mLoop = singleCoreM / baseM;
    constexpr uint32_t nLoop = singleCoreN / baseN;
    constexpr uint32_t kLoop = singleCoreK / baseK;
    uint8_t mte2DBFlag = 0, mte1DBFlag = 0;

    // supplement first sync instr for reverse sync in ProcessKIteration
    InitSyncFlags();

    for (uint32_t i = 0; i < mLoop; i++) {
        for (uint32_t j = 0; j < nLoop; j++) {
            for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
                ProcessKIteration<
                    T, U, S, m, k, n, baseM, baseK, baseN, stepKa, stepKb, singleCoreK, TileMatA, TileMatB, LeftTile,
                    RightTile, ResTile>(
                    kIter, i, j, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                    mte1DBFlag);
            }
            StoreResult<T, U, S, m, n, baseM, baseN, singleCoreK, ResTile>(cTile, currentDst, i, j);
        }
    }

    // supplement last sync instr for reverse sync in ProcessKIteration
    WaitSyncFlags();
}

template <
    typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepN>
__global__ AICORE void GemmPerformance(__gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1)
{
    RunGemmE2E<
        float, half, half, float, blockDim, m, k, n, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK,
        baseN, stepM, stepKa, stepKb, stepN>(
        reinterpret_cast<__gm__ float*>(out), reinterpret_cast<__gm__ half*>(src0),
        reinterpret_cast<__gm__ half*>(src1));
}

#ifndef __COSTMODEL
template <typename T>
void LaunchGEMME2E(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    constexpr uint32_t blockDim = 24;
    constexpr uint32_t m = 6144;
    constexpr uint32_t n = 6144;
    constexpr uint32_t k = 6144;
    constexpr uint32_t singleCoreM = 1536;
    constexpr uint32_t singleCoreN = 1024;
    constexpr uint32_t singleCoreK = 6144;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 4;
    constexpr uint32_t stepKb = 4;
    constexpr uint32_t stepN = 1;
    GemmPerformance<
        T, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa, stepKb, stepN>
        <<<blockDim, nullptr, stream>>>(out, src0, src1);
}

template void LaunchGEMME2E<uint16_t>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
#endif
