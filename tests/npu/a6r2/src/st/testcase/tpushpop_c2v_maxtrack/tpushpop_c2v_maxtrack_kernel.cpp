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

constexpr uint32_t BUFFER_NUM = 2;
// A6R2: L0A/L0B = 64 KiB (float 128x128 fills one bank) → L1 ping-pong only for A/B.
// L0C = 512 KiB → Acc can be double-buffered so TPUSH overlaps the next TMATMUL.

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

AICORE inline void InitSyncFlags()
{
    // L1 ping-pong + single L0 + Acc ping-pong reverse-sync credits.
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_FIX, PIPE_M>(0);
    SetFlag<PIPE_FIX, PIPE_M>(1);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(1);
}

// A6R2 MaxTrack C2V:
//   GM A[512,128] @ B[128,512] -> C[512,512], gMax[512,64], rMax[512,1]
//   Tile M/N by 128 (4x4): each iter A/B/C [128,128], gMax [128,16], rMax [1,128]
//   TPUSH MaxTrack: nTile==0 compare_en=false (seed), else true -> TPOP + TADD(+2) + TFREE
//   Assemble gMax by (mTile,nTile); store HW-compared rMax once per mTile after nLoop

__global__ AICORE void RunTPushPopC2VMaxTrack(
    __gm__ float* outVec, __gm__ float* outGMax, __gm__ float* outRMax, __gm__ float* srcA, __gm__ float* srcB)
{
    constexpr int GmARows = 512;
    constexpr int GmACols = 128;
    constexpr int GmBRows = 128;
    constexpr int GmBCols = 512;
    constexpr int GmCRows = 512;
    constexpr int GmCCols = 512;
    constexpr int TileM = 128;
    constexpr int TileK = 128;
    constexpr int TileN = 128;
    constexpr uint8_t kGroupNum = 8;
    constexpr int mLoops = GmCRows / TileM;
    constexpr int nLoops = GmCCols / TileN;
    constexpr int kGMaxColsFull = GmCCols / kGroupNum;
    constexpr int kGMaxColsTile = TileN / kGroupNum;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using AccTile = TileAcc<float, TileM, TileN, TileM, TileN>;
    using VecTile = Tile<TileType::Vec, float, TileM, TileN, BLayout::RowMajor, TileM, TileN>;
    using GMaxTile = Tile<TileType::Vec, float, TileM, kGMaxColsTile, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    constexpr int RMaxCols = ((TileM + static_cast<int>(C0_SIZE_BYTE / sizeof(float)) - 1) /
                              static_cast<int>(C0_SIZE_BYTE / sizeof(float))) *
                             static_cast<int>(C0_SIZE_BYTE / sizeof(float));
    using RMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using OutVecTile = Tile<TileType::Vec, float, TileM, TileN, BLayout::RowMajor, TileM, TileN>;
    using OutGMaxTile = Tile<TileType::Vec, float, TileM, kGMaxColsTile, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using OutRMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using TwosVecTile = OutVecTile;
    using TwosGMaxTile = OutGMaxTile;
    using TwosRMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, RMaxCols>;

    // nTile==0: seed MAX_BUFF (compare_en=false); later nTiles: compare with MAX_BUFF.
    using MaxTrackCfgInit = MaxTrack<true, true, true, false, true, kGroupNum>;
    using MaxTrackCfgCmp = MaxTrack<true, true, true, true, true, kGroupNum>;
    using FixpipeConfigInit = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0, AtomicType::AtomicNone,
        ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfgInit>;
    using FixpipeConfigCmp = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0, AtomicType::AtomicNone,
        ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfgCmp>;

    constexpr uint32_t kVecSlotBytes = static_cast<uint32_t>(sizeof(float) * TileM * TileN);
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, kVecSlotBytes, FIFO_DEPTH, 2, true>;

    // UB: C2V FIFO [0,0x20000), GMax/RMax sideband, then out ping-pong + twos (< 384 KiB).
    constexpr uint32_t C2V_BUF = 0x0;
    constexpr uint32_t GMAX_BUF = 0x20000;
    constexpr uint32_t RMAX_BUF = 0x24000;
    constexpr uint32_t kOutBase = 0x25000;
    MatPipe pipe((__gm__ void*)(uint64_t)0x0, C2V_BUF, (uint32_t)0x0, GMAX_BUF, RMAX_BUF);

    using GlobalAPanel = GlobalTensor<
        float, pto::Shape<1, 1, 1, TileM, TileK>,
        pto::Stride<GmARows * GmACols, GmARows * GmACols, GmARows * GmACols, GmACols, 1>>;
    using GlobalBPanel = GlobalTensor<
        float, pto::Shape<1, 1, 1, TileK, TileN>,
        pto::Stride<GmBRows * GmBCols, GmBRows * GmBCols, GmBRows * GmBCols, GmBCols, 1>>;
    using GlobalOutVec = GlobalTensor<
        float, pto::Shape<1, 1, 1, TileM, TileN>,
        pto::Stride<GmCRows * GmCCols, GmCRows * GmCCols, GmCRows * GmCCols, GmCCols, 1>>;
    // Assemble gMax into full [GmCRows, kGMaxColsFull]; rMax slice [1, TileM] per mTile
    using GlobalOutGMax = GlobalTensor<
        float, pto::Shape<1, 1, 1, TileM, kGMaxColsTile>,
        pto::Stride<GmCRows * kGMaxColsFull, GmCRows * kGMaxColsFull, GmCRows * kGMaxColsFull, kGMaxColsFull, 1>>;
    using GlobalOutRMax =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, TileM>, pto::Stride<TileM, TileM, TileM, TileM, 1>>;

    using TileMatA = Tile<TileType::Mat, float, TileM, TileK, BLayout::ColMajor, TileM, TileK, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, TileK, TileN, BLayout::ColMajor, TileK, TileN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, TileM, TileK, TileM, TileK>;
    using RightTile = TileRight<float, TileK, TileN, TileK, TileN>;

    if constexpr (DAV_CUBE) {
        constexpr uint32_t kL1ABytes = static_cast<uint32_t>(TileM * TileK * sizeof(float));
        constexpr uint32_t kL1BBytes = static_cast<uint32_t>(TileK * TileN * sizeof(float));
        constexpr uint32_t kAccBytes = static_cast<uint32_t>(AccTile::StorageBytes);

        TileMatA aMatTile[BUFFER_NUM];
        TileMatB bMatTile[BUFFER_NUM];
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile[BUFFER_NUM];

        // L1 double-buffer: A ping/pong then B ping/pong (overlap TLOAD with TMATMUL/TPUSH)
        TASSIGN(aMatTile[0], 0x0);
        TASSIGN(aMatTile[1], kL1ABytes);
        TASSIGN(bMatTile[0], kL1ABytes * BUFFER_NUM);
        TASSIGN(bMatTile[1], kL1ABytes * BUFFER_NUM + kL1BBytes);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        // Acc ping-pong on L0C (overlap TPUSH with next TMATMUL)
        TASSIGN(accTile[0], 0x0);
        TASSIGN(accTile[1], kAccBytes);

        InitSyncFlags();

        uint8_t mte2DBFlag = 0;
        uint8_t fixDBFlag = 0;
        for (int mTile = 0; mTile < mLoops; ++mTile) {
            for (int nTile = 0; nTile < nLoops; ++nTile) {
                GlobalAPanel globalA(srcA + mTile * TileM * GmACols);
                GlobalBPanel globalB(srcB + nTile * TileN);

                // TLOAD (GM -> L1): wait until TMOV finished with this L1 slot
                WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
                TLOAD(aMatTile[mte2DBFlag], globalA);
                SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
                TLOAD(bMatTile[mte2DBFlag], globalB);
                SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
                const uint8_t currMte2Idx = mte2DBFlag;
                mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;

                // TMOV (L1 -> L0): wait until previous TMATMUL released L0
                WaitFlag<PIPE_M, PIPE_MTE1>(0);
                WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
                TMOV(aTile, aMatTile[currMte2Idx]);
                WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
                TMOV(bTile, bMatTile[currMte2Idx]);
                SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx); // free L1 for next TLOAD

                SetFlag<PIPE_MTE1, PIPE_M>(0);
                WaitFlag<PIPE_MTE1, PIPE_M>(0);
                WaitFlag<PIPE_FIX, PIPE_M>(fixDBFlag); // Acc slot free after previous TPUSH
                TMATMUL(accTile[fixDBFlag], aTile, bTile);
                SetFlag<PIPE_M, PIPE_MTE1>(0); // free L0 for next TMOV

                SetFlag<PIPE_M, PIPE_FIX>(fixDBFlag);
                WaitFlag<PIPE_M, PIPE_FIX>(fixDBFlag);
                if (nTile == 0) {
                    TPUSH<MatPipe, AccTile, FixpipeConfigInit>(pipe, accTile[fixDBFlag]);
                } else {
                    TPUSH<MatPipe, AccTile, FixpipeConfigCmp>(pipe, accTile[fixDBFlag]);
                }
                SetFlag<PIPE_FIX, PIPE_M>(fixDBFlag);
                fixDBFlag = (fixDBFlag == 0) ? 1 : 0;
            }
        }

        WaitSyncFlags();
        // RowMax Notify(cons >= SLOT_NUM) posts one more credit than seed Waits; drain it.
        if (mLoops >= static_cast<int>(FIFO_DEPTH)) {
            WaitRowMaxSlot(pipe);
        }
        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            constexpr uint32_t kOutVecBytes = static_cast<uint32_t>(TileM * TileN * sizeof(float));
            constexpr uint32_t kOutGMaxBytes = static_cast<uint32_t>(TileM * kGMaxColsTile * sizeof(float));
            constexpr uint32_t kOutRMaxBytes = static_cast<uint32_t>(RMaxCols * sizeof(float));

            VecTile vecTile;
            GMaxTile gMaxTile;
            RMaxTile rMaxTile;
            OutVecTile outVecTile[BUFFER_NUM];
            OutGMaxTile outGMaxTile[BUFFER_NUM]{OutGMaxTile(TileM, kGMaxColsTile), OutGMaxTile(TileM, kGMaxColsTile)};
            OutRMaxTile outRMaxTile[BUFFER_NUM]{OutRMaxTile(TileM), OutRMaxTile(TileM)};
            TwosVecTile twosVec;
            TwosGMaxTile twosGMax(TileM, kGMaxColsTile);
            TwosRMaxTile twosRMax;

            TASSIGN(outVecTile[0], kOutBase);
            TASSIGN(outVecTile[1], kOutBase + kOutVecBytes);
            TASSIGN(outGMaxTile[0], kOutBase + 2 * kOutVecBytes);
            TASSIGN(outGMaxTile[1], kOutBase + 2 * kOutVecBytes + kOutGMaxBytes);
            TASSIGN(outRMaxTile[0], kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes);
            TASSIGN(outRMaxTile[1], kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes + kOutRMaxBytes);
            const uint32_t twosBase = kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes + 2 * kOutRMaxBytes;
            TASSIGN(twosVec, twosBase);
            TASSIGN(twosGMax, twosBase + kOutVecBytes);
            TASSIGN(twosRMax, twosBase + kOutVecBytes + kOutGMaxBytes);

            TEXPANDS(twosVec, 2.0f);
            TEXPANDS(twosGMax, 2.0f);
            TEXPANDS(twosRMax, 2.0f);

            // Out ping-pong credits: overlap TSTORE with next TADD.
            SetFlag<PIPE_MTE3, PIPE_V>(0);
            SetFlag<PIPE_MTE3, PIPE_V>(1);

            uint8_t mte3DBFlag = 0;
            for (int mTile = 0; mTile < mLoops; ++mTile) {
                uint8_t lastBuf = 0;
                for (int nTile = 0; nTile < nLoops; ++nTile) {
                    // Last nTile: EndRowMaxCmp releases this m-strip RowMax slot inside TPOP.
                    if (nTile + 1 == nLoops) {
                        TPOP<
                            MatPipe, VecTile, GMaxTile, RMaxTile, FixpipeConfigCmp, TileSplitAxis::TILE_NO_SPLIT, true>(
                            pipe, vecTile, gMaxTile, rMaxTile);
                    } else {
                        TPOP<
                            MatPipe, VecTile, GMaxTile, RMaxTile, FixpipeConfigCmp, TileSplitAxis::TILE_NO_SPLIT,
                            false>(pipe, vecTile, gMaxTile, rMaxTile);
                    }

                    WaitFlag<PIPE_MTE3, PIPE_V>(mte3DBFlag);

                    TADD(outVecTile[mte3DBFlag], vecTile, twosVec);
                    TADD(outGMaxTile[mte3DBFlag], gMaxTile, twosGMax);
                    TADD(outRMaxTile[mte3DBFlag], rMaxTile, twosRMax);

                    TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

                    SetFlag<PIPE_V, PIPE_MTE3>(mte3DBFlag);
                    WaitFlag<PIPE_V, PIPE_MTE3>(mte3DBFlag);

                    // Assemble vec / gMax into full GM by (mTile, nTile).
                    GlobalOutVec globalOutVec(outVec + mTile * TileM * GmCCols + nTile * TileN);
                    TSTORE(globalOutVec, outVecTile[mte3DBFlag]);
                    GlobalOutGMax globalOutGMax(outGMax + mTile * TileM * kGMaxColsFull + nTile * kGMaxColsTile);
                    TSTORE(globalOutGMax, outGMaxTile[mte3DBFlag]);

                    SetFlag<PIPE_MTE3, PIPE_V>(mte3DBFlag);
                    lastBuf = mte3DBFlag;
                    mte3DBFlag = (mte3DBFlag == 0) ? 1 : 0;
                }

                // Store RowMax once per mTile (HW-compared result in lastBuf).
                WaitFlag<PIPE_MTE3, PIPE_V>(lastBuf);
                SetFlag<PIPE_V, PIPE_MTE3>(lastBuf);
                WaitFlag<PIPE_V, PIPE_MTE3>(lastBuf);
                GlobalOutRMax globalOutRMax(outRMax + mTile * TileM);
                TSTORE(globalOutRMax, outRMaxTile[lastBuf]);
                SetFlag<PIPE_MTE3, PIPE_V>(lastBuf);
            }

            WaitFlag<PIPE_MTE3, PIPE_V>(0);
            WaitFlag<PIPE_MTE3, PIPE_V>(1);
        }

        pipe_barrier(PIPE_ALL);
    }
}

void LaunchTPushPopC2VMaxTrack(float* outVec, float* outGMax, float* outRMax, float* srcA, float* srcB, void* stream)
{
    RunTPushPopC2VMaxTrack<<<1, 0, stream>>>(outVec, outGMax, outRMax, srcA, srcB);
}

// -----------------------------------------------------------------------------
// Simple mLoop-only case: A[512,128] @ B[128,128] -> C[512,128]
//   Tile M by 128 (mLoops=4, nLoops=1): each Acc [128,128], gMax [128,16], rMax [1,128]
//   Only seed TPUSH (nTile always 0); every TPOP uses EndRowMaxCmp=true
// -----------------------------------------------------------------------------

__global__ AICORE void RunTPushPopC2VMaxTrackMOnly(
    __gm__ float* outVec, __gm__ float* outGMax, __gm__ float* outRMax, __gm__ float* srcA, __gm__ float* srcB)
{
    constexpr int MOnlyARows = 512;
    constexpr int MOnlyACols = 128;
    constexpr int MOnlyBRows = 128;
    constexpr int MOnlyBCols = 128;
    constexpr int MOnlyCRows = 512;
    constexpr int MOnlyCCols = 128;
    constexpr int MOnlyTileM = 128;
    constexpr int MOnlyTileK = 128;
    constexpr int MOnlyTileN = 128;
    constexpr uint8_t kGroupNum = 8;
    constexpr int MOnlyMLoops = MOnlyCRows / MOnlyTileM;
    constexpr int MOnlyGMaxCols = MOnlyCCols / kGroupNum;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using AccTile = TileAcc<float, MOnlyTileM, MOnlyTileN, MOnlyTileM, MOnlyTileN>;
    using VecTile = Tile<TileType::Vec, float, MOnlyTileM, MOnlyTileN, BLayout::RowMajor, MOnlyTileM, MOnlyTileN>;
    using GMaxTile = Tile<TileType::Vec, float, MOnlyTileM, MOnlyGMaxCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    constexpr int RMaxCols = ((MOnlyTileM + static_cast<int>(C0_SIZE_BYTE / sizeof(float)) - 1) /
                              static_cast<int>(C0_SIZE_BYTE / sizeof(float))) *
                             static_cast<int>(C0_SIZE_BYTE / sizeof(float));
    using RMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using OutVecTile = Tile<TileType::Vec, float, MOnlyTileM, MOnlyTileN, BLayout::RowMajor, MOnlyTileM, MOnlyTileN>;
    using OutGMaxTile = Tile<TileType::Vec, float, MOnlyTileM, MOnlyGMaxCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using OutRMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using TwosVecTile = OutVecTile;
    using TwosGMaxTile = OutGMaxTile;
    using TwosRMaxTile = Tile<TileType::Vec, float, 1, RMaxCols, BLayout::RowMajor, 1, RMaxCols>;

    // nLoops==1: only seed (no cross-N compare).
    using MaxTrackCfg = MaxTrack<true, true, true, false, true, kGroupNum>;
    using FixpipeConfig = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0, AtomicType::AtomicNone,
        ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfg>;

    constexpr uint32_t kVecSlotBytes = static_cast<uint32_t>(sizeof(float) * MOnlyTileM * MOnlyTileN);
    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, kVecSlotBytes, FIFO_DEPTH, 2, true>;

    constexpr uint32_t C2V_BUF = 0x0;
    constexpr uint32_t GMAX_BUF = 0x20000;
    constexpr uint32_t RMAX_BUF = 0x24000;
    constexpr uint32_t kOutBase = 0x25000;
    MatPipe pipe((__gm__ void*)(uint64_t)0x0, C2V_BUF, (uint32_t)0x0, GMAX_BUF, RMAX_BUF);

    using GlobalAPanel = GlobalTensor<
        float, pto::Shape<1, 1, 1, MOnlyTileM, MOnlyTileK>,
        pto::Stride<MOnlyARows * MOnlyACols, MOnlyARows * MOnlyACols, MOnlyARows * MOnlyACols, MOnlyACols, 1>>;
    using GlobalBPanel = GlobalTensor<
        float, pto::Shape<1, 1, 1, MOnlyTileK, MOnlyTileN>,
        pto::Stride<MOnlyBRows * MOnlyBCols, MOnlyBRows * MOnlyBCols, MOnlyBRows * MOnlyBCols, MOnlyBCols, 1>>;
    using GlobalOutVec = GlobalTensor<
        float, pto::Shape<1, 1, 1, MOnlyTileM, MOnlyTileN>,
        pto::Stride<MOnlyCRows * MOnlyCCols, MOnlyCRows * MOnlyCCols, MOnlyCRows * MOnlyCCols, MOnlyCCols, 1>>;
    using GlobalOutGMax = GlobalTensor<
        float, pto::Shape<1, 1, 1, MOnlyTileM, MOnlyGMaxCols>,
        pto::Stride<
            MOnlyCRows * MOnlyGMaxCols, MOnlyCRows * MOnlyGMaxCols, MOnlyCRows * MOnlyGMaxCols, MOnlyGMaxCols, 1>>;
    using GlobalOutRMax = GlobalTensor<
        float, pto::Shape<1, 1, 1, 1, MOnlyTileM>, pto::Stride<MOnlyTileM, MOnlyTileM, MOnlyTileM, MOnlyTileM, 1>>;

    using TileMatA = Tile<
        TileType::Mat, float, MOnlyTileM, MOnlyTileK, BLayout::ColMajor, MOnlyTileM, MOnlyTileK, SLayout::RowMajor,
        512>;
    using TileMatB = Tile<
        TileType::Mat, float, MOnlyTileK, MOnlyTileN, BLayout::ColMajor, MOnlyTileK, MOnlyTileN, SLayout::RowMajor,
        512>;
    using LeftTile = TileLeft<float, MOnlyTileM, MOnlyTileK, MOnlyTileM, MOnlyTileK>;
    using RightTile = TileRight<float, MOnlyTileK, MOnlyTileN, MOnlyTileK, MOnlyTileN>;

    if constexpr (DAV_CUBE) {
        constexpr uint32_t kL1ABytes = static_cast<uint32_t>(MOnlyTileM * MOnlyTileK * sizeof(float));
        constexpr uint32_t kL1BBytes = static_cast<uint32_t>(MOnlyTileK * MOnlyTileN * sizeof(float));
        constexpr uint32_t kAccBytes = static_cast<uint32_t>(AccTile::StorageBytes);

        TileMatA aMatTile[BUFFER_NUM];
        TileMatB bMatTile[BUFFER_NUM];
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile[BUFFER_NUM];

        TASSIGN(aMatTile[0], 0x0);
        TASSIGN(aMatTile[1], kL1ABytes);
        TASSIGN(bMatTile[0], kL1ABytes * BUFFER_NUM);
        TASSIGN(bMatTile[1], kL1ABytes * BUFFER_NUM + kL1BBytes);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile[0], 0x0);
        TASSIGN(accTile[1], kAccBytes);

        InitSyncFlags();

        uint8_t mte2DBFlag = 0;
        uint8_t fixDBFlag = 0;
        for (int mTile = 0; mTile < MOnlyMLoops; ++mTile) {
            GlobalAPanel globalA(srcA + mTile * MOnlyTileM * MOnlyACols);
            GlobalBPanel globalB(srcB);

            WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
            TLOAD(aMatTile[mte2DBFlag], globalA);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TLOAD(bMatTile[mte2DBFlag], globalB);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
            const uint8_t currMte2Idx = mte2DBFlag;
            mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;

            WaitFlag<PIPE_M, PIPE_MTE1>(0);
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TMOV(aTile, aMatTile[currMte2Idx]);
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
            TMOV(bTile, bMatTile[currMte2Idx]);
            SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);

            SetFlag<PIPE_MTE1, PIPE_M>(0);
            WaitFlag<PIPE_MTE1, PIPE_M>(0);
            WaitFlag<PIPE_FIX, PIPE_M>(fixDBFlag);
            TMATMUL(accTile[fixDBFlag], aTile, bTile);
            SetFlag<PIPE_M, PIPE_MTE1>(0);

            SetFlag<PIPE_M, PIPE_FIX>(fixDBFlag);
            WaitFlag<PIPE_M, PIPE_FIX>(fixDBFlag);
            TPUSH<MatPipe, AccTile, FixpipeConfig>(pipe, accTile[fixDBFlag]);
            SetFlag<PIPE_FIX, PIPE_M>(fixDBFlag);
            fixDBFlag = (fixDBFlag == 0) ? 1 : 0;
        }

        WaitSyncFlags();
        if (MOnlyMLoops >= static_cast<int>(FIFO_DEPTH)) {
            WaitRowMaxSlot(pipe);
        }
    }

    if constexpr (DAV_VEC) {
        if (get_subblockid() == 0) {
            constexpr uint32_t kOutVecBytes = static_cast<uint32_t>(MOnlyTileM * MOnlyTileN * sizeof(float));
            constexpr uint32_t kOutGMaxBytes = static_cast<uint32_t>(MOnlyTileM * MOnlyGMaxCols * sizeof(float));
            constexpr uint32_t kOutRMaxBytes = static_cast<uint32_t>(RMaxCols * sizeof(float));

            VecTile vecTile;
            GMaxTile gMaxTile;
            RMaxTile rMaxTile;
            OutVecTile outVecTile[BUFFER_NUM];
            OutGMaxTile outGMaxTile[BUFFER_NUM]{
                OutGMaxTile(MOnlyTileM, MOnlyGMaxCols), OutGMaxTile(MOnlyTileM, MOnlyGMaxCols)};
            OutRMaxTile outRMaxTile[BUFFER_NUM]{OutRMaxTile(MOnlyTileM), OutRMaxTile(MOnlyTileM)};
            TwosVecTile twosVec;
            TwosGMaxTile twosGMax(MOnlyTileM, MOnlyGMaxCols);
            TwosRMaxTile twosRMax;

            TASSIGN(outVecTile[0], kOutBase);
            TASSIGN(outVecTile[1], kOutBase + kOutVecBytes);
            TASSIGN(outGMaxTile[0], kOutBase + 2 * kOutVecBytes);
            TASSIGN(outGMaxTile[1], kOutBase + 2 * kOutVecBytes + kOutGMaxBytes);
            TASSIGN(outRMaxTile[0], kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes);
            TASSIGN(outRMaxTile[1], kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes + kOutRMaxBytes);
            const uint32_t twosBase = kOutBase + 2 * kOutVecBytes + 2 * kOutGMaxBytes + 2 * kOutRMaxBytes;
            TASSIGN(twosVec, twosBase);
            TASSIGN(twosGMax, twosBase + kOutVecBytes);
            TASSIGN(twosRMax, twosBase + kOutVecBytes + kOutGMaxBytes);

            TEXPANDS(twosVec, 2.0f);
            TEXPANDS(twosGMax, 2.0f);
            TEXPANDS(twosRMax, 2.0f);

            SetFlag<PIPE_MTE3, PIPE_V>(0);
            SetFlag<PIPE_MTE3, PIPE_V>(1);

            uint8_t mte3DBFlag = 0;
            for (int mTile = 0; mTile < MOnlyMLoops; ++mTile) {
                // nLoops==1: each pop ends the m-strip RowMax cmp/slot.
                TPOP<MatPipe, VecTile, GMaxTile, RMaxTile, FixpipeConfig, TileSplitAxis::TILE_NO_SPLIT, true>(
                    pipe, vecTile, gMaxTile, rMaxTile);

                WaitFlag<PIPE_MTE3, PIPE_V>(mte3DBFlag);

                TADD(outVecTile[mte3DBFlag], vecTile, twosVec);
                TADD(outGMaxTile[mte3DBFlag], gMaxTile, twosGMax);
                TADD(outRMaxTile[mte3DBFlag], rMaxTile, twosRMax);

                TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

                SetFlag<PIPE_V, PIPE_MTE3>(mte3DBFlag);
                WaitFlag<PIPE_V, PIPE_MTE3>(mte3DBFlag);

                GlobalOutVec globalOutVec(outVec + mTile * MOnlyTileM * MOnlyCCols);
                TSTORE(globalOutVec, outVecTile[mte3DBFlag]);
                GlobalOutGMax globalOutGMax(outGMax + mTile * MOnlyTileM * MOnlyGMaxCols);
                TSTORE(globalOutGMax, outGMaxTile[mte3DBFlag]);
                GlobalOutRMax globalOutRMax(outRMax + mTile * MOnlyTileM);
                TSTORE(globalOutRMax, outRMaxTile[mte3DBFlag]);

                SetFlag<PIPE_MTE3, PIPE_V>(mte3DBFlag);
                mte3DBFlag = (mte3DBFlag == 0) ? 1 : 0;
            }

            WaitFlag<PIPE_MTE3, PIPE_V>(0);
            WaitFlag<PIPE_MTE3, PIPE_V>(1);
        }
    }
    pipe_barrier(PIPE_ALL);
}

void LaunchTPushPopC2VMaxTrackMOnly(
    float* outVec, float* outGMax, float* outRMax, float* srcA, float* srcB, void* stream)
{
    RunTPushPopC2VMaxTrackMOnly<<<1, 0, stream>>>(outVec, outGMax, outRMax, srcA, srcB);
}
