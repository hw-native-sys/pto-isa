/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include <pto/common/debug.h>

using namespace pto;
constexpr uint32_t BUFFER_NUM = 2; // ping pong, 2 buffer
constexpr uint32_t SCALE_FACTOR = 32;
constexpr uint32_t SHIFT_SCALE_FACTOR = 5;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KB per buffer)
constexpr uint32_t mxScalePara = 4;
// Pipeline mental model (instruction-level):
// - TLOAD     (GM -> L1):   fill aMatTile/bMatTile
// - TEXTRACT  (L1 -> L0):   slice aMatTile/bMatTile into aTile/bTile for the current baseK
// - TMATMUL_MX   (Cube):    cTile = A*B (accumulated over K)
// - TSTORE    (L0C -> GM):  write cTile back to GM
//
// The code still uses PIPE_MTE* events for synchronization because those are the underlying hardware pipes;
// comments refer to the high-level PTO instructions to make tuning easier.

template <typename OutTile, typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile>
AICORE inline void MatmulAcc(
    OutTile cTile, LeftTile aTile, RightTile bTile, LeftScaleTile aScaleTile, RightScaleTile bScaleTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
    } else {
        TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
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
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN>
AICORE inline void InitGMOffsets(
    __gm__ U*& currentSrc0, __gm__ U*& currentSrc1, __gm__ X*& currentSrc2, __gm__ X*& currentSrc3,
    __gm__ T*& currentDst, __gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3,
    uint32_t validsingleCoreM, uint32_t validsingleCoreN)
{
    // Work partition (SPMD-style):
    // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
    // - It reads the corresponding A panel [singleCoreM, K] and B panel [K, singleCoreN].
    uint32_t mIter = (m + singleCoreM - 1) / singleCoreM;
    uint32_t nIter = (n + singleCoreN - 1) / singleCoreN;
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint32_t rowStart = mIterIdx * singleCoreM;
    if (rowStart >= m)
        rowStart = m;

    uint64_t gmOffsetA = (rowStart * k) >> 1;
    uint64_t gmOffsetB = (nIterIdx * k * singleCoreN) >> 1;
    uint64_t gmOffsetScaleA = (rowStart * k) >> SHIFT_SCALE_FACTOR;
    uint64_t gmOffsetScaleB = (nIterIdx * k * singleCoreN) >> SHIFT_SCALE_FACTOR;
    uint64_t gmOffsetC = rowStart * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentSrc2 = src2 + gmOffsetScaleA;
    currentSrc3 = src3 + gmOffsetScaleB;
    currentDst = out + gmOffsetC;
}

template <
    typename T, typename U, typename X, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK,
    uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB, typename TileMatA, typename TileMatB,
    typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile, typename LeftScaleTile,
    typename RightScaleTile, typename ResTile>
AICORE inline void InitBuffers(
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA aScaleMatTile[BUFFER_NUM],
    TileScaleB bScaleMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
    LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM], ResTile& cTile)
{
    // L1 staging buffers (aMatTile/bMatTile) are double-buffered for TLOAD overlap.
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U) / 2);
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) / 2);
    TASSIGN(
        bMatTile[1],
        0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) / 2 + baseK * baseN * stepKb * sizeof(U) / 2);

    constexpr uint32_t baseAddr =
        baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) / 2 + baseK * baseN * stepKb * sizeof(U) / 2 * BUFFER_NUM;
    TASSIGN(aScaleMatTile[0], baseAddr);
    TASSIGN(aScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * sizeof(X));
    TASSIGN(bScaleMatTile[0], baseAddr + baseM * baseScaleK * stepKscaleA * BUFFER_NUM * sizeof(X));
    TASSIGN(
        bScaleMatTile[1], baseAddr + baseM * baseScaleK * stepKscaleA * BUFFER_NUM * sizeof(X) +
                              baseScaleK * baseN * stepKscaleB * sizeof(X));

    // L0A/L0B ping-pong buffers (TEXTRACT destination).
    // Keep each per-buffer footprint <= 32 KiB to fit in a ping/pang slot.
    TASSIGN(aTile[0], 0x0);                     // L0A ping
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES); // L0A pong
    TASSIGN(bTile[0], 0x0);                     // L0B ping
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES); // L0B pong
    TASSIGN(cTile, 0x0);

    TASSIGN(aScaleTile[0], GetScaleAddr(aTile[0].data()));
    TASSIGN(aScaleTile[1], GetScaleAddr(aTile[1].data()));
    TASSIGN(bScaleTile[0], GetScaleAddr(bTile[0].data()));
    TASSIGN(bScaleTile[1], GetScaleAddr(bTile[1].data()));
}

template <
    uint32_t baseK, uint32_t baseScaleK, uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB,
    typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile,
    typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void MacroMatmul(
    uint32_t kIter, uint8_t currMte2Idx, uint8_t currMte2mxIdx, uint8_t mte1DBFlag, TileMatA aMatTile[BUFFER_NUM],
    TileMatB bMatTile[BUFFER_NUM], TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM],
    LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], LeftScaleTile aScaleTile[BUFFER_NUM],
    RightScaleTile bScaleTile[BUFFER_NUM], ResTile& cTile, uint32_t currentM, uint32_t currentN)
{
    const uint32_t kModstepKa = kIter % stepKa;
    // Wait until TMATMUL is done with the current L0A/L0B buffer before overwriting it via TEXTRACT.
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    // TEXTRACT stage: slice the loaded L1 panel into the baseK chunk we need this iteration.
    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);
    TEXTRACT(aScaleTile[mte1DBFlag], aScaleMatTile[currMte2mxIdx], 0, (kIter % stepKscaleA) * baseScaleK);

    if (kModstepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
    TEXTRACT(bScaleTile[mte1DBFlag], bScaleMatTile[currMte2mxIdx], (kIter % stepKscaleB) * baseScaleK, 0);

    if ((kIter + 1) % stepKa == 0) {
        // Allow the next TLOAD to reuse this L1 slot.
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    // TMATMUL stage: compute (or accumulate) into cTile.
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], aScaleTile[mte1DBFlag], bScaleTile[mte1DBFlag], kIter);
    // Signal that TMATMUL is done, so the next iteration may TEXTRACT into the other ping-pong slot.
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
}

template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreK, uint32_t baseM, uint32_t baseK,
    uint32_t baseN, uint32_t baseScaleK, uint32_t stepKa, uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB,
    typename TileMatA, typename TileMatB, typename TileScaleA, typename TileScaleB, typename LeftTile,
    typename RightTile, typename LeftScaleTile, typename RightScaleTile, typename ResTile>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t i, uint32_t j, __gm__ U* currentSrc0, __gm__ U* currentSrc1, __gm__ X* currentSrc2,
    __gm__ X* currentSrc3, TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
    TileScaleA aScaleMatTile[BUFFER_NUM], TileScaleB bScaleMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM],
    RightTile bTile[BUFFER_NUM], LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM],
    ResTile& cTile, uint8_t& mte2DBFlag, uint8_t& mte2mxDBFlag, uint8_t& mte1DBFlag, uint32_t currentM,
    uint32_t currentN)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;

    using GlobalDataSrc0 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, m, k, Layout::ND>, Layout::ND>;
    using GlobalDataSrc1 = GlobalTensor<U, DynShapeDim5, BaseShape2D<U, k, n, Layout::DN>, Layout::DN>;
    using GlobalDataSrc2 =
        GlobalTensor<X, DynShapeDim5, BaseShape2D<X, m, k / SCALE_FACTOR, Layout::MX_A_ND>, Layout::MX_A_ND>;
    using GlobalDataSrc3 =
        GlobalTensor<X, DynShapeDim5, BaseShape2D<X, k / SCALE_FACTOR, n, Layout::MX_B_DN>, Layout::MX_B_DN>;

    const uint32_t kModstepKa = kIter % stepKa;
    const uint32_t kModstepKscaleA = kIter % stepKscaleA;

    // TLOAD stage:
    // - Every stepKa iterations, load a larger [baseM, baseK * stepKa] panel into L1 and then slice it with TEXTRACT.
    // - Double buffering is driven by mte2DBFlag.

    if (kModstepKa == 0) {
        GlobalDataSrc0 gmA(
            currentSrc0 + i * singleCoreK * baseM / 2 + kIter * baseK / 2, DynShapeDim5(currentM, baseK * stepKa));
        GlobalDataSrc1 gmB(
            currentSrc1 + j * singleCoreK * baseN / 2 + kIter * baseK / 2, DynShapeDim5(baseK * stepKa, currentN));

        if (kModstepKscaleA == 0) {
            GlobalDataSrc2 gmScaleA(
                currentSrc2 + i * singleCoreK * baseM / SCALE_FACTOR + kIter * baseScaleK,
                DynShapeDim5(currentM, baseScaleK * stepKscaleA));
            GlobalDataSrc3 gmScaleB(
                currentSrc3 + j * singleCoreK * baseN / SCALE_FACTOR + kIter * baseScaleK,
                DynShapeDim5(baseScaleK * stepKscaleB, currentN));
            // Wait until TEXTRACT is done with this L1 buffer before reusing it.
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
            TLOAD(aMatTile[mte2DBFlag], gmA);
            TLOAD<TileScaleA, GlobalDataSrc2>(aScaleMatTile[mte2mxDBFlag], gmScaleA);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TLOAD(bMatTile[mte2DBFlag], gmB);
            TLOAD<TileScaleB, GlobalDataSrc3>(bScaleMatTile[mte2mxDBFlag], gmScaleB);
            mte2mxDBFlag = (mte2mxDBFlag == 0) ? 1 : 0;
        } else {
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
            TLOAD(aMatTile[mte2DBFlag], gmA);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TLOAD(bMatTile[mte2DBFlag], gmB);
        }
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;     // mte2DBFlag reversed
    const uint32_t currMte2mxIdx = (mte2mxDBFlag == 0) ? 1 : 0; // mte2mxDBFlag reversed

    MacroMatmul<
        baseK, baseScaleK, stepKa, stepKb, stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB,
        LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
        kIter, currMte2Idx, currMte2mxIdx, mte1DBFlag, aMatTile, bMatTile, aScaleMatTile, bScaleMatTile, aTile, bTile,
        aScaleTile, bScaleTile, cTile, currentM, currentN);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename T, typename U, int m, int n, uint32_t singleCoreK, uint32_t baseM, uint32_t baseN, typename ResTile>
AICORE inline void StoreResult(
    ResTile& cTile, __gm__ T* currentDst, uint32_t i, uint32_t j, uint32_t currentM, uint32_t currentN)
{
    // TSTORE stage: write the finished C tile [baseM, baseN] back to GM.
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    // the data size read from L0C after single k loop is [baseM, baseN]
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStrideDim5 = pto::Stride<m * n, m * n, m * n, n, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShapeDim5, DynStrideDim5, Layout::ND>;

    uint32_t gmOffset = i * baseM * n + j * baseN;
    GlobalDataOut dstGlobal(currentDst + gmOffset, DynShapeDim5(currentM, currentN));
    TSTORE(dstGlobal, cTile);

    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}

AICORE inline void InitSyncFlags()
{
    // supplement first sync instr for reverse sync in ProcessKIteration
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);
}

AICORE inline void WaitSyncFlags()
{
    // supplement last sync instr for reverse sync in ProcessKIteration
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}

template <
    typename T, typename U, typename X, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t baseScaleK, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepKscaleA, uint32_t stepKscaleB, typename TileMatA, typename TileMatB,
    typename TileScaleA, typename TileScaleB, typename LeftTile, typename RightTile, typename LeftScaleTile,
    typename RightScaleTile, typename ResTile>
AICORE inline void Compute(
    __gm__ U* currentSrc0, __gm__ U* currentSrc1, __gm__ X* currentSrc2, __gm__ X* currentSrc3, __gm__ T*& currentDst,
    TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], TileScaleA aScaleMatTile[BUFFER_NUM],
    TileScaleB bScaleMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM],
    LeftScaleTile aScaleTile[BUFFER_NUM], RightScaleTile bScaleTile[BUFFER_NUM], ResTile& cTile,
    uint32_t currentSingleCoreM, uint32_t currentSingleCoreN)
{
    uint8_t mte2DBFlag = 0;
    uint8_t mte2mxDBFlag = 0;
    uint8_t mte1DBFlag = 0;

    const uint32_t loopsM = (singleCoreM + baseM - 1) / baseM;
    const uint32_t loopsN = (singleCoreN + baseN - 1) / baseN;
    const uint32_t loopsK = (singleCoreK + baseK - 1) / baseK;

    uint32_t remM = currentSingleCoreM % baseM;
    uint32_t remN = currentSingleCoreN % baseN;

    for (uint32_t i = 0; i < loopsM; ++i) {
        for (uint32_t j = 0; j < loopsN; ++j) {
            uint32_t currentM = (i == loopsM - 1 && remM > 0) ? remM : baseM;
            uint32_t currentN = (j == loopsN - 1 && remN > 0) ? remN : baseN;

            for (int buf = 0; buf < BUFFER_NUM; ++buf) {
                aMatTile[buf] = TileMatA(currentM, baseK * stepKa);
                bMatTile[buf] = TileMatB(baseK * stepKb, currentN);
                aScaleMatTile[buf] = TileScaleA(currentM, baseScaleK * stepKscaleA);
                bScaleMatTile[buf] = TileScaleB(baseScaleK * stepKscaleB, currentN);

                aTile[buf] = LeftTile(currentM, baseK);
                bTile[buf] = RightTile(baseK, currentN);
                aScaleTile[buf] = LeftScaleTile(currentM, baseScaleK);
                bScaleTile[buf] = RightScaleTile(baseScaleK, currentN);
            }

            ResTile outTile(currentM, currentN);
            TASSIGN(outTile, 0x0);

            for (uint32_t kIter = 0; kIter < loopsK; ++kIter) {
                ProcessKIteration<
                    T, U, X, m, k, n, singleCoreK, baseM, baseK, baseN, baseScaleK, stepKa, stepKb, stepKscaleA,
                    stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile,
                    RightScaleTile, ResTile>(
                    kIter, i, j, currentSrc0, currentSrc1, currentSrc2, currentSrc3, aMatTile, bMatTile, aScaleMatTile,
                    bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, outTile, mte2DBFlag, mte2mxDBFlag, mte1DBFlag,
                    currentM, currentN);
            }

            StoreResult<T, U, m, n, singleCoreK, baseM, baseN, ResTile>(outTile, currentDst, i, j, currentM, currentN);
        }
    }
}

template <
    typename T, typename U, typename X, uint32_t blockDim, int m, int k, int n, uint32_t baseM, uint32_t baseK,
    uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN, uint32_t singleCoreM,
    uint32_t singleCoreK, uint32_t singleCoreN>
AICORE inline void RunMxMatmulWithTail(
    __gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3, uint32_t validSingleCoreM,
    uint32_t validSingleCoreN)
{
    __gm__ U* currentSrc0 = nullptr;
    __gm__ U* currentSrc1 = nullptr;
    __gm__ X* currentSrc2 = nullptr;
    __gm__ X* currentSrc3 = nullptr;
    __gm__ T* currentDst = nullptr;

    InitGMOffsets<T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, out, src0, src1, src2, src3, validSingleCoreM,
        validSingleCoreN);

    constexpr uint32_t baseScaleK = baseK / SCALE_FACTOR;
    constexpr uint32_t stepKscaleA = stepKa * mxScalePara;
    constexpr uint32_t stepKscaleB = stepKb * mxScalePara;

    // L1 Tile - use dynamic shape
    using TileMatA = Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    using TileMatB = Tile<TileType::Mat, U, baseK * stepKb, baseN, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;

    using TileScaleA =
        Tile<TileType::Mat, X, baseM, baseScaleK * stepKscaleA, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;

    using TileScaleB =
        Tile<TileType::Mat, X, baseScaleK * stepKscaleB, baseN, BLayout::ColMajor, -1, -1, SLayout::ColMajor, 32>;

    TileMatA aMatTile[BUFFER_NUM];
    TileMatB bMatTile[BUFFER_NUM];
    TileScaleA aScaleMatTile[BUFFER_NUM];
    TileScaleB bScaleMatTile[BUFFER_NUM];
    // L0 Tile - use dynamic shape
    using LeftTile = TileLeftCompact<U, baseM, baseK, -1, -1>;
    using RightTile = TileRightCompact<U, baseK, baseN, -1, -1>;
    using LeftScaleTile = TileLeftScaleCompact<X, baseM, baseScaleK, -1, -1>;
    using RightScaleTile = TileRightScaleCompact<X, baseScaleK, baseN, -1, -1>;
    using ResTile = TileAccCompact<float, baseM, baseN, -1, -1>;

    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];
    LeftScaleTile aScaleTile[BUFFER_NUM];
    RightScaleTile bScaleTile[BUFFER_NUM];
    ResTile cTile;

    InitBuffers<
        T, U, X, baseM, baseK, baseN, baseScaleK, stepKa, stepKb, stepKscaleA, stepKscaleB, TileMatA, TileMatB,
        TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile, RightScaleTile, ResTile>(
        aMatTile, bMatTile, aScaleMatTile, bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile);

    InitSyncFlags();
    Compute<
        T, U, X, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, baseScaleK, stepKa, stepKb,
        stepKscaleA, stepKscaleB, TileMatA, TileMatB, TileScaleA, TileScaleB, LeftTile, RightTile, LeftScaleTile,
        RightScaleTile, ResTile>(
        currentSrc0, currentSrc1, currentSrc2, currentSrc3, currentDst, aMatTile, bMatTile, aScaleMatTile,
        bScaleMatTile, aTile, bTile, aScaleTile, bScaleTile, cTile, validSingleCoreM, validSingleCoreN);

    WaitSyncFlags();
}

template <
    typename T, typename U, typename X, uint32_t blockDim, int m, int k, int n, uint32_t singleCoreM,
    uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM,
    uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunMxMatmulDispatch(__gm__ T* out, __gm__ U* src0, __gm__ U* src1, __gm__ X* src2, __gm__ X* src3)
{
    constexpr uint32_t mIter = (m + singleCoreM - 1) / singleCoreM;
    constexpr uint32_t nIter = (n + singleCoreN - 1) / singleCoreN;

    uint32_t coreIdx = get_block_idx();

    if (coreIdx >= mIter * nIter) {
        return;
    }

    const uint32_t mIterIdx = coreIdx % mIter;
    const uint32_t nIterIdx = coreIdx / mIter;

    uint32_t currentSingleCoreM = singleCoreM;
    uint32_t currentSingleCoreN = singleCoreN;

    if (mIterIdx == mIter - 1 && m % singleCoreM != 0) {
        currentSingleCoreM = m % singleCoreM;
    }
    if (nIterIdx == nIter - 1 && n % singleCoreN != 0) {
        currentSingleCoreN = n % singleCoreN;
    }

    RunMxMatmulWithTail<
        T, U, X, blockDim, m, k, n, baseM, baseK, baseN, stepM, stepKa, stepKb, stepN, singleCoreM, singleCoreK,
        singleCoreN>(out, src0, src1, src2, src3, currentSingleCoreM, currentSingleCoreN);
}

template <
    uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
    uint32_t stepKb, uint32_t stepN>
__global__ AICORE void MxMatmulPerformance(
    __gm__ uint8_t* out, __gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ uint8_t* src2, __gm__ uint8_t* src3)
{
    RunMxMatmulDispatch<
        bfloat16_t, float4_e2m1x2_t, float8_e8m0_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM,
        baseK, baseN, stepM, stepKa, stepKb, stepN>(
        reinterpret_cast<__gm__ bfloat16_t*>(out), reinterpret_cast<__gm__ float4_e2m1x2_t*>(src0),
        reinterpret_cast<__gm__ float4_e2m1x2_t*>(src1), reinterpret_cast<__gm__ float8_e8m0_t*>(src2),
        reinterpret_cast<__gm__ float8_e8m0_t*>(src3));
}

void LaunchMxMatmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream)
{
    constexpr uint32_t blockDim = 32;
    constexpr uint32_t m = 2040;
    constexpr uint32_t k = 8192;
    constexpr uint32_t n = 8100;
    constexpr uint32_t singleCoreM = 512;
    constexpr uint32_t singleCoreK = 8192;
    constexpr uint32_t singleCoreN = 1024;
    constexpr uint32_t baseM = 256;
    constexpr uint32_t baseK = 256;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 2;
    constexpr uint32_t stepKb = 2;
    constexpr uint32_t stepN = 1;
    MxMatmulPerformance<
        blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa, stepKb, stepN>
        <<<blockDim, nullptr, stream>>>(out, src0, src1, src2, src3);
}

void LaunchMxMatmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);
