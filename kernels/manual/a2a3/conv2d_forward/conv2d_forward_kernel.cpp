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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024; // L0A/L0B ping-pong split (32 KiB per buffer)

template <typename T>
AICORE constexpr inline T CeilDivision(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}
template <typename T>
AICORE constexpr inline T Min(T num_1, T num_2)
{
    return num_1 < num_2 ? num_1 : num_2;
}
template <typename T>
AICORE constexpr inline T Max(T num_1, T num_2)
{
    return num_1 > num_2 ? num_1 : num_2;
}
template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t kIter)
{
    if (kIter == 0) {
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
template <typename T, typename U, typename S, int n, uint32_t singleCoreN, uint32_t cin, uint32_t hin, uint32_t win,
          uint32_t c0, uint32_t hout, uint32_t wout>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ S *&currentSrc1, __gm__ T *&currentDst, __gm__ T *out,
                                 __gm__ U *src0, __gm__ S *src1)
{
    // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
    // - It reads the corresponding A panel [singleCoreM, K] and B panel [K, singleCoreN].
    constexpr uint32_t nIdex = n / singleCoreN;
    uint32_t mCoreIdx = get_block_idx() / nIdex; // get current launch core idx
    uint32_t nCoreIdx = get_block_idx() % nIdex;
    uint64_t gmOffsetA = mCoreIdx * cin * hin * win * c0;
    uint64_t gmOffsetB = nCoreIdx * c0 * singleCoreN;
    uint64_t gmOffsetC = mCoreIdx * n * hout * wout + nCoreIdx * singleCoreN * hout * wout;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
    currentDst = out + gmOffsetC;
}

template <typename T, typename U, typename S, uint32_t n, uint32_t baseM, uint32_t baseN, uint32_t hout, uint32_t wout,
          typename OutTile>
AICORE inline void StoreResult(OutTile cTile, __gm__ T *currentDst, uint32_t mIter, uint32_t nIter)
{
    // TSTORE stage: write the finished C tile [baseM, baseN] back to GM.
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    constexpr int gStrideOut[5] = {n * hout * wout, hout * wout * 16, wout * 16, 16, 1};
    using ShapeDim5Out = pto::Shape<1, baseN / 16, CeilDivision(baseM, wout), wout, 16>;
    using StridDim5Out = pto::Stride<gStrideOut[0], gStrideOut[1], gStrideOut[2], gStrideOut[3], gStrideOut[4]>;
    using GlobalDataOut = GlobalTensor<T, ShapeDim5Out, StridDim5Out, Layout::NC1HWC0>;

    GlobalDataOut dstGlobal(currentDst + mIter * baseM * gStrideOut[3] + nIter * baseN / 16 * gStrideOut[1]);
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
template <uint32_t baseK, uint32_t stepKa, uint32_t stepKb, typename TileMatAData, typename TileMatBData,
          typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void MacroMatmul(uint32_t kIter, uint8_t currMte2Idx, uint8_t mte1DBFlag,
                               TileMatAData fmapMat[BUFFER_NUM], TileMatBData weightMat[BUFFER_NUM],
                               LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &cTile,
                               uint32_t woutStart)
{
    const uint32_t kModStepKa = kIter % stepKa;
    // Wait until TMATMUL is done with the current L0A/L0B buffer before overwriting it via TEXTRACT.
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    // TEXTRACT stage: slice the loaded L1 panel into the baseK chunk we need this iteration.
    if (kModStepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TIMG2COL(aTile[mte1DBFlag], fmapMat[currMte2Idx], woutStart, kModStepKa * baseK);
    if (kModStepKa == 0)
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], weightMat[currMte2Idx], (kIter % stepKb) * baseK, 0);

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
}
template <typename U, uint32_t cin, uint32_t hin, uint32_t win, uint32_t c0, uint32_t n, uint32_t channelSize,
          uint32_t hk, uint32_t wk, uint32_t baseK, uint32_t baseN, uint32_t stepKa, uint32_t stepKb,
          typename TileMatAData, typename TileMatBData, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void ProcessKIteration(__gm__ U *currentSrc0, __gm__ U *currentSrc1, uint32_t hinStart, uint32_t hinCount,
                                     uint32_t woutStart, uint32_t kIter, uint32_t nIter,
                                     TileMatAData fmapMat[BUFFER_NUM], TileMatBData weightMat[BUFFER_NUM],
                                     LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &outTile,
                                     uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    constexpr int gStrideSrc0[5] = {cin * hin * win * c0, hin * win * c0, win * c0, c0, 1};
    using ShapeDim5Src0 = pto::Shape<1, channelSize / c0, -1, win, c0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {cin * hk * wk * n * c0, n * c0, 16 * c0, c0, 1};
    using ShapeDim5Src1 = pto::Shape<1, stepKb * baseK / c0, baseN / 16, 16, c0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    const uint32_t kModStepKa = kIter % stepKa;

    if (kModStepKa == 0) {
        GlobalDataSrc0 fmap(currentSrc0 + (kIter * baseK) / (hk * wk * c0) * gStrideSrc0[1] + hinStart * gStrideSrc0[2],
                            pto::Shape<1, channelSize / c0, -1, win, c0>(hinCount));
        GlobalDataSrc1 weight(currentSrc1 + (kIter * baseK / c0) * gStrideSrc1[1] +
                              (nIter * baseN / 16) * gStrideSrc1[2]);
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(fmapMat[mte2DBFlag], fmap);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(weightMat[mte2DBFlag], weight);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }
    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0; // mte2DBFlag reversed

    MacroMatmul<baseK, stepKa, stepKb>(kIter, currMte2Idx, mte1DBFlag, fmapMat, weightMat, aTile, bTile, outTile,
                                       woutStart);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}
template <typename T, typename U, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM,
          uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM,
          uint32_t stepKa, uint32_t stepKb, uint32_t stepN, uint32_t batch, uint32_t cin, uint32_t hin, uint32_t win,
          uint32_t c0, uint32_t hk, uint32_t wk, uint32_t hout, uint32_t wout, uint32_t channelSize, typename ResTile,
          uint32_t strideH = 1, uint32_t padTop = 1, typename TileMatAData, typename TileMatBData, typename LeftTile,
          typename RightTile>

AICORE inline void Compute(__gm__ U *currentSrc0, __gm__ U *currentSrc1, __gm__ T *&currentDst,
                           TileMatAData fmapMat[BUFFER_NUM], TileMatBData weightMat[BUFFER_NUM],
                           LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM])
{
    constexpr uint32_t tailValidM = singleCoreM % baseM;
    constexpr uint32_t mLoop = CeilDivision(singleCoreM, baseM);
    constexpr uint32_t nLoop = CeilDivision(singleCoreN, baseN);
    constexpr uint32_t kLoop = CeilDivision(singleCoreK, baseK);
    uint8_t mte2DBFlag = 0, mte1DBFlag = 0;

    for (uint32_t mIter = 0; mIter < mLoop; mIter++) {
        int64_t mStart = mIter * baseM;
        int64_t mCount = Min<int64_t>(baseM, hout * wout - mStart);
        int64_t mEnd = mStart + mCount - 1;
        int64_t houtStart = mStart / wout;
        int64_t woutStart = mStart % wout;
        int64_t houtEnd = mEnd / wout;
        int64_t hinStart = Max<int64_t>(0, houtStart * strideH - padTop);
        int64_t hinEnd = Min<int64_t>(houtEnd * strideH - padTop + (hk - 1), hin - 1);
        int64_t hinCount = hinEnd - hinStart + 1;
        uint32_t currentM = (mIter == mLoop - 1 && tailValidM > 0) ? tailValidM : baseM;
        for (int i = 0; i < BUFFER_NUM; ++i) {
            aTile[i] = LeftTile(currentM);
            fmapMat[i] = TileMatAData(hinCount);
            fmapMat[i].SetFmapH(hinCount);
            fmapMat[i].SetFmapW(win);
            fmapMat[i].SetChannelSize(channelSize);
            fmapMat[i].SetFilterH(3);
            fmapMat[i].SetFilterW(3);
            fmapMat[i].SetPadList(0, 1);
            fmapMat[i].SetPadList(1, 1);
            fmapMat[i].SetPadList(2, Max<int64_t>(0, padTop - houtStart * strideH));
            fmapMat[i].SetPadList(3, Max<int64_t>(0, (houtEnd + hk) * strideH - hin - padTop));
        }
        ResTile outTile(currentM);
        TASSIGN(outTile, 0x0);
        SETFMATRIX(fmapMat[0]);
        TASSIGN(fmapMat[0], 0x0);
        TASSIGN(fmapMat[1], 0x0 + channelSize * win * (hinCount + (hk - 1)) * sizeof(U));
        for (uint32_t nIter = 0; nIter < nLoop; nIter++) {
            for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
                ProcessKIteration<U, cin, hin, win, c0, n, channelSize, hk, wk, baseK, baseN, stepKa, stepKb>(
                    currentSrc0, currentSrc1, hinStart, hinCount, woutStart, kIter, nIter, fmapMat, weightMat, aTile,
                    bTile, outTile, mte2DBFlag, mte1DBFlag);
            }
            StoreResult<T, U, U, n, baseM, baseN, hout, wout>(outTile, currentDst, mIter, nIter);
        }
    }
}
template <typename T, typename U, typename S, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN, uint32_t batch,
          uint32_t cin, uint32_t hin, uint32_t win, uint32_t c0, uint32_t hk, uint32_t wk, uint32_t hout, uint32_t wout,
          uint32_t strideH = 1, uint32_t strideW = 1, uint32_t dilationH = 1, uint32_t dilationW = 1,
          uint32_t padTop = 1, uint32_t padBottom = 1, uint32_t padLeft = 1, uint32_t padRight = 1>
AICORE inline void RunConv2dForward(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    __gm__ U *currentSrc0 = nullptr;
    __gm__ S *currentSrc1 = nullptr;
    __gm__ T *currentDst = nullptr;
    InitGMOffsets<T, U, S, n, singleCoreN, cin, hin, win, c0, hout, wout>(currentSrc0, currentSrc1, currentDst, out,
                                                                          src0, src1);
    constexpr uint32_t channelSize = CeilDivision(stepKa * baseK, hk * wk);
    constexpr int bufferSizeA = sizeof(U) * channelSize * win * (CeilDivision(baseM, wout) + (hk - 1));
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<1, channelSize / c0, -1, win, c0>>;
    TileMatAData fmapMat[BUFFER_NUM];

    constexpr int bufferSizeB = stepKb * baseK * baseN * sizeof(U);
    using TileMatBData = ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                                  pto::ConvTileShape<stepKb * baseK / c0, baseN / 16, 16, c0>>;
    TileMatBData weightMat[BUFFER_NUM];
    TASSIGN(weightMat[0], 0x20000);
    TASSIGN(weightMat[1], 0x20000 + baseK * baseN * stepKb * sizeof(U));

    using LeftTile = TileLeft<U, baseM, baseK, -1, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAccCompact<float, baseM, baseN, -1, baseN>;
    LeftTile aTile[BUFFER_NUM];
    RightTile bTile[BUFFER_NUM];

    TASSIGN(aTile[0], 0x0);                     // L0A ping
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES); // L0A pang
    TASSIGN(bTile[0], 0x0);                     // L0B ping
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES); // L0B pang

    InitSyncFlags();
    Compute<half, half, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
            stepKb, stepN, batch, cin, hin, win, c0, hk, wk, hout, wout, channelSize, ResTile>(
        currentSrc0, currentSrc1, currentDst, fmapMat, weightMat, aTile, bTile);
    WaitSyncFlags();
}

template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM, uint32_t singleCoreK,
          uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM, uint32_t stepKa,
          uint32_t stepKb, uint32_t stepN, uint32_t batch, uint32_t cin, uint32_t hin, uint32_t win, uint32_t c0,
          uint32_t hk, uint32_t wk, uint32_t hout, uint32_t wout, uint32_t strideH = 1, uint32_t strideW = 1,
          uint32_t dilationH = 1, uint32_t dilationW = 1, uint32_t padTop = 1, uint32_t padBottom = 1,
          uint32_t padLeft = 1, uint32_t padRight = 1>
__global__ AICORE void Conv2dForward(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    RunConv2dForward<half, half, half, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN,
                     stepM, stepKa, stepKb, stepN, batch, cin, hin, win, c0, hk, wk, hout, wout>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
template <typename T>
void launchConv2dForward(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    constexpr uint32_t blockDim = 24;
    constexpr uint32_t m = 6144;
    constexpr uint32_t n = 6144;
    constexpr uint32_t k = 4608;
    constexpr uint32_t singleCoreM = 1536;
    constexpr uint32_t singleCoreN = 1024;
    constexpr uint32_t singleCoreK = 4608;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 48;
    constexpr uint32_t baseN = 256;

    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 3;
    constexpr uint32_t stepKb = 3;
    constexpr uint32_t stepN = 1;

    constexpr uint32_t batch = 4;
    constexpr uint32_t cin = 32;
    constexpr uint32_t hin = 16;
    constexpr uint32_t win = 96;
    constexpr uint32_t c0 = 16;
    constexpr uint32_t hk = 3;
    constexpr uint32_t wk = 3;
    constexpr uint32_t strideH = 1;
    constexpr uint32_t strideW = 1;
    constexpr uint32_t dilationH = 1;
    constexpr uint32_t dilationW = 1;
    constexpr uint32_t padTop = 1;
    constexpr uint32_t padBottom = 1;
    constexpr uint32_t padLeft = 1;
    constexpr uint32_t padRight = 1;
    constexpr uint32_t hout = (hin + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1;
    constexpr uint32_t wout = (win + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1;
    Conv2dForward<T, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
                  stepKb, stepN, batch, cin, hin, win, c0, hk, wk, hout, wout>
        <<<blockDim, nullptr, stream>>>(out, src0, src1);
}
template void launchConv2dForward<uint16_t>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
