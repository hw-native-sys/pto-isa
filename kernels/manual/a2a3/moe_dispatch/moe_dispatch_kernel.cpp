/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ============================================================================
// MoE Dispatch Kernel — PTO-ISA Dual-Path Implementation
//
// Two independent kernel paths for dispatch communication:
//
// 1. Direct (2-step): TLOAD remote GM → UB → TSTORE split to local GM
//    - Adaptive MOVE_NUM based on UB capacity
//    - Cross-rank continuous event-driven ping-pong pipeline
//
// 2. ViaGM (4-step): TGET remote GM → local temp GM → TLOAD → UB → TSTORE split
//    - MegaMoE style with intermediate GM buffer
//    - Uses PTO-ISA TGET instruction with built-in ping-pong
//
// Execution model: AIV-only (vector cores), launched via mpirun for multi-rank
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"

// PIPE_FIX is a CCE built-in for cube cores; define as PIPE_V for vec-only compilation
// (only used in opPipeList array which we don't access at runtime)
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_V
#endif
#include "pto/comm/pto_comm_inst.hpp"

#include "moe_dispatch_config.h"

#include "comm_context.h"

// ============================================================================
// Device-side helper: translate local shmem pointer to remote rank's address
// ============================================================================
template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T*)(ctx->windowsIn[pe] + offset);
}

// ============================================================================
// PATH 1: MoeDispatchDirect — 2-step fast path
//
// TLOAD interleaved rows from remote GM directly into UB, then TSTORE to split
// token and scale to separate compact GM destinations.
// Cross-rank continuous pipeline with event-driven ping-pong.
// ============================================================================
template <int HIDDEN_SIZE, int TILE_COLS, int MOVE_NUM>
AICORE void MoeDispatchDirect(
    __gm__ int8_t* gmA, __gm__ float* gmPerTokenScale, __gm__ int32_t* cumsumMM, __gm__ int32_t* tokenPerExpert,
    __gm__ int32_t* preSumBeforeRank, __gm__ uint8_t* shmemBase, __gm__ CommDeviceContext* hcclCtx, int32_t EP,
    int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA, int32_t tpeRowStride = 0, int32_t cumsumStride = 0)
{
    int32_t myRank = static_cast<int32_t>(hcclCtx->rankId);
    int32_t coreIdx = get_block_idx();
    int32_t coreNum = get_block_num();
    int32_t expNum = (tpeRowStride > 0) ? tpeRowStride : EP * expertPerRank;
    int32_t csStride = (cumsumStride > 0) ? cumsumStride : expertPerRank;

    constexpr int32_t copyInNum = HIDDEN_SIZE + UB_ALIGN;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using ViewTile = pto::Tile<pto::TileType::Vec, int8_t, MOVE_NUM, TILE_COLS, pto::BLayout::RowMajor, -1, -1>;

    constexpr int32_t INTERLEAVED_TILE_BYTES = MOVE_NUM * TILE_COLS;
    constexpr int32_t PING_OFFSET = 0;
    constexpr int32_t PONG_OFFSET = (INTERLEAVED_TILE_BYTES + 31) & ~31;

    ViewTile interleavedPing(MOVE_NUM, TILE_COLS);
    ViewTile tokenViewPing(MOVE_NUM, HIDDEN_SIZE);
    ViewTile scaleViewPing(MOVE_NUM, UB_ALIGN);
    TASSIGN(interleavedPing, PING_OFFSET);
    TASSIGN(tokenViewPing, PING_OFFSET);
    TASSIGN(scaleViewPing, PING_OFFSET + HIDDEN_SIZE);

    ViewTile interleavedPong(MOVE_NUM, TILE_COLS);
    ViewTile tokenViewPong(MOVE_NUM, HIDDEN_SIZE);
    ViewTile scaleViewPong(MOVE_NUM, UB_ALIGN);
    TASSIGN(interleavedPong, PONG_OFFSET);
    TASSIGN(tokenViewPong, PONG_OFFSET);
    TASSIGN(scaleViewPong, PONG_OFFSET + HIDDEN_SIZE);

    uint32_t prevGroupSum = 0;

    for (int32_t groupIdx = 0; groupIdx < expertPerRank; ++groupIdx) {
        uint32_t currentM = static_cast<uint32_t>(cumsumMM[(EP - 1) * csStride + groupIdx]);

        bool hasPending = false;
        int32_t pendingPP = 0;
        int32_t pendingRows = 0;
        __gm__ int8_t* pendTokenDstPtr = nullptr;
        __gm__ int8_t* pendScaleDstPtr = nullptr;
        int32_t globalChunkIdx = 0;

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < EP; dstEpIdx += coreNum) {
            uint32_t rowStart;
            if (dstEpIdx == 0) {
                rowStart = prevGroupSum;
            } else {
                rowStart = static_cast<uint32_t>(cumsumMM[(dstEpIdx - 1) * csStride + groupIdx]) + prevGroupSum;
            }

            if (rowStart >= static_cast<uint32_t>(maxOutputSize)) {
                continue;
            }

            int32_t tpeIdx = dstEpIdx * expNum + myRank * expertPerRank + groupIdx;
            uint32_t rows = static_cast<uint32_t>(tokenPerExpert[tpeIdx]);

            if (rowStart + rows > static_cast<uint32_t>(maxOutputSize)) {
                rows = static_cast<uint32_t>(maxOutputSize) - rowStart;
            }

            if (rows == 0) {
                continue;
            }

            int32_t rowSrc = preSumBeforeRank[dstEpIdx * expertPerRank + groupIdx];

            __gm__ uint8_t* otherRankBase = CommRemotePtr(hcclCtx, shmemBase, dstEpIdx);
            __gm__ int8_t* remoteSrcPtr =
                reinterpret_cast<__gm__ int8_t*>(otherRankBase + offsetA + static_cast<int64_t>(rowSrc) * copyInNum);

            int32_t processCount = (static_cast<int32_t>(rows) + MOVE_NUM - 1) / MOVE_NUM;

            for (int32_t p = 0; p < processCount; ++p) {
                int32_t curRows = MOVE_NUM;
                if (p == processCount - 1) {
                    int32_t rem = static_cast<int32_t>(rows) - p * MOVE_NUM;
                    if (rem < MOVE_NUM)
                        curRows = rem;
                }

                int32_t curPP = globalChunkIdx & 1;
                event_t curEvent = curPP ? EVENT_ID1 : EVENT_ID0;
                auto& loadTile = curPP ? interleavedPong : interleavedPing;

                __gm__ int8_t* chunkSrc = remoteSrcPtr + static_cast<int64_t>(p) * MOVE_NUM * copyInNum;
                int64_t srcTotalBytes = static_cast<int64_t>(curRows) * copyInNum;
                ShapeDyn srcShape(1, 1, 1, static_cast<size_t>(curRows), static_cast<size_t>(copyInNum));
                StrideDyn srcStride(srcTotalBytes, srcTotalBytes, srcTotalBytes, copyInNum, 1);
                Global remoteSrcG(chunkSrc, srcShape, srcStride);

                loadTile.RowMaskInternal = curRows;
                loadTile.ColMaskInternal = TILE_COLS;

                if (hasPending) {
                    event_t prevEvent = pendingPP ? EVENT_ID1 : EVENT_ID0;
                    wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

                    auto& prevTokenView = pendingPP ? tokenViewPong : tokenViewPing;
                    auto& prevScaleView = pendingPP ? scaleViewPong : scaleViewPing;
                    prevTokenView.RowMaskInternal = pendingRows;
                    prevTokenView.ColMaskInternal = HIDDEN_SIZE;
                    prevScaleView.RowMaskInternal = pendingRows;
                    prevScaleView.ColMaskInternal = UB_ALIGN;

                    int64_t pendTokenBytes = static_cast<int64_t>(pendingRows) * HIDDEN_SIZE;
                    ShapeDyn pendTokenShape(
                        1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(HIDDEN_SIZE));
                    StrideDyn pendTokenStride(pendTokenBytes, pendTokenBytes, pendTokenBytes, HIDDEN_SIZE, 1);
                    Global pendTokenDstG(pendTokenDstPtr, pendTokenShape, pendTokenStride);

                    int64_t pendScaleBytes = static_cast<int64_t>(pendingRows) * UB_ALIGN;
                    ShapeDyn pendScaleShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(UB_ALIGN));
                    StrideDyn pendScaleStride(pendScaleBytes, pendScaleBytes, pendScaleBytes, UB_ALIGN, 1);
                    Global pendScaleDstG(pendScaleDstPtr, pendScaleShape, pendScaleStride);

                    TSTORE(pendTokenDstG, prevTokenView);
                    TSTORE(pendScaleDstG, prevScaleView);
                    TLOAD(loadTile, remoteSrcG);

                    set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                } else {
                    TLOAD(loadTile, remoteSrcG);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                }

                hasPending = true;
                pendingPP = curPP;
                pendingRows = curRows;
                uint32_t dstRow = rowStart + static_cast<uint32_t>(p * MOVE_NUM);
                pendTokenDstPtr = gmA + static_cast<int64_t>(dstRow) * HIDDEN_SIZE;
                pendScaleDstPtr =
                    reinterpret_cast<__gm__ int8_t*>(gmPerTokenScale) + static_cast<int64_t>(dstRow) * UB_ALIGN;
                globalChunkIdx++;
            }
        }

        if (hasPending) {
            event_t lastEvent = pendingPP ? EVENT_ID1 : EVENT_ID0;
            wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);

            auto& lastTokenView = pendingPP ? tokenViewPong : tokenViewPing;
            auto& lastScaleView = pendingPP ? scaleViewPong : scaleViewPing;
            lastTokenView.RowMaskInternal = pendingRows;
            lastTokenView.ColMaskInternal = HIDDEN_SIZE;
            lastScaleView.RowMaskInternal = pendingRows;
            lastScaleView.ColMaskInternal = UB_ALIGN;

            int64_t lastTokenBytes = static_cast<int64_t>(pendingRows) * HIDDEN_SIZE;
            ShapeDyn lastTokenShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(HIDDEN_SIZE));
            StrideDyn lastTokenStride(lastTokenBytes, lastTokenBytes, lastTokenBytes, HIDDEN_SIZE, 1);
            Global lastTokenDstG(pendTokenDstPtr, lastTokenShape, lastTokenStride);

            int64_t lastScaleBytes = static_cast<int64_t>(pendingRows) * UB_ALIGN;
            ShapeDyn lastScaleShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(UB_ALIGN));
            StrideDyn lastScaleStride(lastScaleBytes, lastScaleBytes, lastScaleBytes, UB_ALIGN, 1);
            Global lastScaleDstG(pendScaleDstPtr, lastScaleShape, lastScaleStride);

            TSTORE(lastTokenDstG, lastTokenView);
            TSTORE(lastScaleDstG, lastScaleView);

            set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
            wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
        }

        prevGroupSum += currentM;
    }
}

// ============================================================================
// PATH 2: MoeDispatchViaGM — 4-step MegaMoE-style path
//
// Phase 1: TGET remote GM → tempGmBuffer (using TGET with ping-pong staging)
// Phase 2: TLOAD tempGmBuffer → UB → TSTORE split to gmA + gmPerTokenScale
// ============================================================================
template <int HIDDEN_SIZE, int TILE_COLS, int MOVE_NUM>
AICORE void MoeDispatchViaGM(
    __gm__ int8_t* gmA, __gm__ float* gmPerTokenScale, __gm__ int8_t* tempGmBuffer, __gm__ int32_t* cumsumMM,
    __gm__ int32_t* tokenPerExpert, __gm__ int32_t* preSumBeforeRank, __gm__ uint8_t* shmemBase,
    __gm__ CommDeviceContext* hcclCtx, int32_t EP, int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA)
{
    int32_t myRank = static_cast<int32_t>(hcclCtx->rankId);
    int32_t coreIdx = get_block_idx();
    int32_t coreNum = get_block_num();
    int32_t expNum = EP * expertPerRank;

    constexpr int32_t copyInNum = HIDDEN_SIZE + UB_ALIGN;
    constexpr int32_t TGET_TILE_ROWS = 2;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // TGET staging tiles (small, just for remote→local GM transfer)
    using TgetTile = pto::Tile<pto::TileType::Vec, int8_t, TGET_TILE_ROWS, TILE_COLS, pto::BLayout::RowMajor, -1, -1>;
    TgetTile tgetPing(TGET_TILE_ROWS, TILE_COLS);
    TgetTile tgetPong(TGET_TILE_ROWS, TILE_COLS);
    constexpr int32_t TGET_TILE_BYTES = TGET_TILE_ROWS * TILE_COLS;
    TASSIGN(tgetPing, 0);
    TASSIGN(tgetPong, (TGET_TILE_BYTES + 31) & ~31);

    // Phase 2 split tiles (reuse UB after TGET completes for each rank)
    using SplitTile = pto::Tile<pto::TileType::Vec, int8_t, MOVE_NUM, TILE_COLS, pto::BLayout::RowMajor, -1, -1>;

    constexpr int32_t SPLIT_TILE_BYTES = MOVE_NUM * TILE_COLS;
    constexpr int32_t SPLIT_PING_OFFSET = 0;
    constexpr int32_t SPLIT_PONG_OFFSET = (SPLIT_TILE_BYTES + 31) & ~31;

    uint32_t prevGroupSum = 0;

    for (int32_t groupIdx = 0; groupIdx < expertPerRank; ++groupIdx) {
        uint32_t currentM = static_cast<uint32_t>(cumsumMM[(EP - 1) * expertPerRank + groupIdx]);

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < EP; dstEpIdx += coreNum) {
            uint32_t rowStart;
            if (dstEpIdx == 0) {
                rowStart = prevGroupSum;
            } else {
                rowStart = static_cast<uint32_t>(cumsumMM[(dstEpIdx - 1) * expertPerRank + groupIdx]) + prevGroupSum;
            }

            if (rowStart >= static_cast<uint32_t>(maxOutputSize)) {
                continue;
            }

            int32_t tpeIdx = dstEpIdx * expNum + myRank * expertPerRank + groupIdx;
            uint32_t rows = static_cast<uint32_t>(tokenPerExpert[tpeIdx]);

            if (rowStart + rows > static_cast<uint32_t>(maxOutputSize)) {
                rows = static_cast<uint32_t>(maxOutputSize) - rowStart;
            }

            if (rows == 0) {
                continue;
            }

            int32_t rowSrc = preSumBeforeRank[dstEpIdx * expertPerRank + groupIdx];

            __gm__ uint8_t* otherRankBase = CommRemotePtr(hcclCtx, shmemBase, dstEpIdx);
            __gm__ int8_t* remoteSrcPtr =
                reinterpret_cast<__gm__ int8_t*>(otherRankBase + offsetA + static_cast<int64_t>(rowSrc) * copyInNum);

            // Phase 1: TGET remote → tempGmBuffer (interleaved format preserved)
            __gm__ int8_t* tempDst = tempGmBuffer + static_cast<int64_t>(rowStart) * copyInNum;
            int64_t totalBytes = static_cast<int64_t>(rows) * copyInNum;
            ShapeDyn srcShape(1, 1, 1, static_cast<size_t>(rows), static_cast<size_t>(copyInNum));
            StrideDyn srcStride(totalBytes, totalBytes, totalBytes, copyInNum, 1);
            Global remoteSrcG(remoteSrcPtr, srcShape, srcStride);
            Global tempDstG(tempDst, srcShape, srcStride);

            pto::comm::TGET(tempDstG, remoteSrcG, tgetPing, tgetPong);

            // Phase 2: TLOAD tempGmBuffer → UB → TSTORE split (event-driven ping-pong)
            SplitTile splitInterleavedPing(MOVE_NUM, TILE_COLS);
            SplitTile splitTokenPing(MOVE_NUM, HIDDEN_SIZE);
            SplitTile splitScalePing(MOVE_NUM, UB_ALIGN);
            TASSIGN(splitInterleavedPing, SPLIT_PING_OFFSET);
            TASSIGN(splitTokenPing, SPLIT_PING_OFFSET);
            TASSIGN(splitScalePing, SPLIT_PING_OFFSET + HIDDEN_SIZE);

            SplitTile splitInterleavedPong(MOVE_NUM, TILE_COLS);
            SplitTile splitTokenPong(MOVE_NUM, HIDDEN_SIZE);
            SplitTile splitScalePong(MOVE_NUM, UB_ALIGN);
            TASSIGN(splitInterleavedPong, SPLIT_PONG_OFFSET);
            TASSIGN(splitTokenPong, SPLIT_PONG_OFFSET);
            TASSIGN(splitScalePong, SPLIT_PONG_OFFSET + HIDDEN_SIZE);

            int32_t processCount = (static_cast<int32_t>(rows) + MOVE_NUM - 1) / MOVE_NUM;

            bool hasPending = false;
            int32_t pendingPP = 0;
            int32_t pendingRows = 0;
            __gm__ int8_t* pendTokenDstPtr = nullptr;
            __gm__ int8_t* pendScaleDstPtr = nullptr;

            for (int32_t p = 0; p < processCount; ++p) {
                int32_t curRows = MOVE_NUM;
                if (p == processCount - 1) {
                    int32_t rem = static_cast<int32_t>(rows) - p * MOVE_NUM;
                    if (rem < MOVE_NUM)
                        curRows = rem;
                }

                int32_t curPP = p & 1;
                event_t curEvent = curPP ? EVENT_ID1 : EVENT_ID0;
                auto& loadTile = curPP ? splitInterleavedPong : splitInterleavedPing;

                __gm__ int8_t* chunkSrc = tempDst + static_cast<int64_t>(p) * MOVE_NUM * copyInNum;
                int64_t chunkBytes = static_cast<int64_t>(curRows) * copyInNum;
                ShapeDyn chunkShape(1, 1, 1, static_cast<size_t>(curRows), static_cast<size_t>(copyInNum));
                StrideDyn chunkStride(chunkBytes, chunkBytes, chunkBytes, copyInNum, 1);
                Global localSrcG(chunkSrc, chunkShape, chunkStride);

                loadTile.RowMaskInternal = curRows;
                loadTile.ColMaskInternal = TILE_COLS;

                if (hasPending) {
                    event_t prevEvent = pendingPP ? EVENT_ID1 : EVENT_ID0;
                    wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

                    auto& prevTokenView = pendingPP ? splitTokenPong : splitTokenPing;
                    auto& prevScaleView = pendingPP ? splitScalePong : splitScalePing;
                    prevTokenView.RowMaskInternal = pendingRows;
                    prevTokenView.ColMaskInternal = HIDDEN_SIZE;
                    prevScaleView.RowMaskInternal = pendingRows;
                    prevScaleView.ColMaskInternal = UB_ALIGN;

                    int64_t pendTokenBytes = static_cast<int64_t>(pendingRows) * HIDDEN_SIZE;
                    ShapeDyn pendTokenShape(
                        1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(HIDDEN_SIZE));
                    StrideDyn pendTokenStride(pendTokenBytes, pendTokenBytes, pendTokenBytes, HIDDEN_SIZE, 1);
                    Global pendTokenDstG(pendTokenDstPtr, pendTokenShape, pendTokenStride);

                    int64_t pendScaleBytes = static_cast<int64_t>(pendingRows) * UB_ALIGN;
                    ShapeDyn pendScaleShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(UB_ALIGN));
                    StrideDyn pendScaleStride(pendScaleBytes, pendScaleBytes, pendScaleBytes, UB_ALIGN, 1);
                    Global pendScaleDstG(pendScaleDstPtr, pendScaleShape, pendScaleStride);

                    TSTORE(pendTokenDstG, prevTokenView);
                    TSTORE(pendScaleDstG, prevScaleView);
                    TLOAD(loadTile, localSrcG);

                    set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                } else {
                    TLOAD(loadTile, localSrcG);
                    set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                }

                hasPending = true;
                pendingPP = curPP;
                pendingRows = curRows;
                uint32_t dstRow = rowStart + static_cast<uint32_t>(p * MOVE_NUM);
                pendTokenDstPtr = gmA + static_cast<int64_t>(dstRow) * HIDDEN_SIZE;
                pendScaleDstPtr =
                    reinterpret_cast<__gm__ int8_t*>(gmPerTokenScale) + static_cast<int64_t>(dstRow) * UB_ALIGN;
            }

            if (hasPending) {
                event_t lastEvent = pendingPP ? EVENT_ID1 : EVENT_ID0;
                wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);

                auto& lastTokenView = pendingPP ? splitTokenPong : splitTokenPing;
                auto& lastScaleView = pendingPP ? splitScalePong : splitScalePing;
                lastTokenView.RowMaskInternal = pendingRows;
                lastTokenView.ColMaskInternal = HIDDEN_SIZE;
                lastScaleView.RowMaskInternal = pendingRows;
                lastScaleView.ColMaskInternal = UB_ALIGN;

                int64_t lastTokenBytes = static_cast<int64_t>(pendingRows) * HIDDEN_SIZE;
                ShapeDyn lastTokenShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(HIDDEN_SIZE));
                StrideDyn lastTokenStride(lastTokenBytes, lastTokenBytes, lastTokenBytes, HIDDEN_SIZE, 1);
                Global lastTokenDstG(pendTokenDstPtr, lastTokenShape, lastTokenStride);

                int64_t lastScaleBytes = static_cast<int64_t>(pendingRows) * UB_ALIGN;
                ShapeDyn lastScaleShape(1, 1, 1, static_cast<size_t>(pendingRows), static_cast<size_t>(UB_ALIGN));
                StrideDyn lastScaleStride(lastScaleBytes, lastScaleBytes, lastScaleBytes, UB_ALIGN, 1);
                Global lastScaleDstG(pendScaleDstPtr, lastScaleShape, lastScaleStride);

                TSTORE(lastTokenDstG, lastTokenView);
                TSTORE(lastScaleDstG, lastScaleView);

                set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
                wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
            }
        }

        prevGroupSum += currentM;
    }
}

// ============================================================================
// PATH 3: MoeDispatchWithSync — CrossRankSync + Direct dispatch
//
// Integrates CrossRankSync (Module 2) as a kernel-internal preamble:
//   Phase A: AllGather localTokenPerExpert via DataAsFlag (TSTORE remote + TWAIT)
//   Phase B: Restore values, compute cumsumMM and preSumBeforeRank
//   Phase C: SYNCALL then proceed with standard Direct dispatch loop
// ============================================================================

template <int HIDDEN_SIZE, int TILE_COLS, int MOVE_NUM>
AICORE void MoeDispatchWithSync(
    __gm__ int8_t* gmA, __gm__ float* gmPerTokenScale, __gm__ uint8_t* shmemBase, __gm__ CommDeviceContext* hcclCtx,
    __gm__ int32_t* workspace, __gm__ int32_t* syncGmWorkspace, int32_t EP, int32_t expertPerRank,
    int32_t maxOutputSize, int64_t offsetA, int64_t offsetTPE)
{
    int32_t myRank = static_cast<int32_t>(hcclCtx->rankId);
    int32_t coreIdx = get_block_idx();
    int32_t coreNum = get_block_num();

    int32_t paddedExpNum = ((EP * expertPerRank) + 7) & ~7;

    // UB workspace for software SYNCALL (needs coreNum * 32 bytes)
    constexpr int32_t SYNC_UB_ELEMS = 32;
    using SyncUbTile = pto::Tile<pto::TileType::Vec, int32_t, 1, SYNC_UB_ELEMS, pto::BLayout::RowMajor, -1, -1>;
    SyncUbTile syncUbTile(1, SYNC_UB_ELEMS);
    TASSIGN(syncUbTile, 0);

    using SyncShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using SyncStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using SyncGlobal = pto::GlobalTensor<int32_t, SyncShape, SyncStride, pto::Layout::ND>;
    int32_t syncElems = coreNum * SYNCALL_SOFT_SLOT_INT32;
    SyncShape syncGmShape(1, 1, 1, 1, static_cast<size_t>(syncElems));
    SyncStride syncGmStride(syncElems, syncElems, syncElems, syncElems, 1);
    SyncGlobal syncGmG(syncGmWorkspace, syncGmShape, syncGmStride);

    // Workspace layout (padded cumsumMM for DMA-aligned TSTORE):
    //   [0 .. EP*paddedExpNum)                                     : cumsumMM (full rows)
    //   [EP*paddedExpNum .. EP*paddedExpNum + EP*expertPerRank)     : preSumBeforeRank
    //   [EP*paddedExpNum + EP*expertPerRank .. end)                 : tokenPerExpert (padded)
    __gm__ int32_t* wsCumsumMM = workspace;
    __gm__ int32_t* wsPSBR = workspace + EP * paddedExpNum;
    __gm__ int32_t* wsTPE = workspace + EP * paddedExpNum + EP * expertPerRank;

    // Pointers into this rank's shmem TPE exchange area
    __gm__ int32_t* localTPEBase = reinterpret_cast<__gm__ int32_t*>(shmemBase + offsetTPE);

    // ========================================================================
    // Phase A: Write localTokenPerExpert + DataAsFlag to all remote ranks
    // ========================================================================
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

        using TPETile = pto::Tile<pto::TileType::Vec, int32_t, 1, 64, pto::BLayout::RowMajor, -1, -1>;
        TPETile tpeTile(1, paddedExpNum);
        TASSIGN(tpeTile, 0);

        tpeTile.RowMaskInternal = 1;
        tpeTile.ColMaskInternal = paddedExpNum;

        int64_t tpeRowBytes = static_cast<int64_t>(paddedExpNum) * sizeof(int32_t);
        ShapeDyn tpeShape(1, 1, 1, 1, static_cast<size_t>(paddedExpNum));
        StrideDyn tpeStride(tpeRowBytes / 4, tpeRowBytes / 4, tpeRowBytes / 4, tpeRowBytes / 4, 1);

        // Load my localTokenPerExpert from shmem (row myRank in the TPE area)
        __gm__ int32_t* myTPEAddr = localTPEBase + myRank * paddedExpNum;
        GlobalI32 myTPEG(myTPEAddr, tpeShape, tpeStride);
        TLOAD(tpeTile, myTPEG);

        // Add DataAsFlag offset
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(tpeTile, tpeTile, static_cast<int32_t>(DATA_AS_FLAG_OFFSET));

        // Write to each remote rank's TPE area at row[myRank]
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        for (int32_t dstRank = coreIdx; dstRank < EP; dstRank += coreNum) {
            if (dstRank == myRank)
                continue;
            __gm__ int32_t* remoteTPEBase =
                reinterpret_cast<__gm__ int32_t*>(CommRemotePtr(hcclCtx, shmemBase, dstRank) + offsetTPE);
            __gm__ int32_t* remoteDst = remoteTPEBase + myRank * paddedExpNum;
            GlobalI32 remoteDstG(remoteDst, tpeShape, tpeStride);
            TSTORE(remoteDstG, tpeTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    // ========================================================================
    // Phase B: Wait for all remote ranks' data, restore, compute routing tables
    // ========================================================================
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

        // Signal shape for TWAIT (check first element of each row)
        ShapeDyn signalShape(1, 1, 1, 1, 1);
        StrideDyn signalStride(1, 1, 1, 1, 1);

        // Wait for all remote ranks
        for (int32_t srcRank = coreIdx; srcRank < EP; srcRank += coreNum) {
            if (srcRank == myRank)
                continue;
            __gm__ int32_t* signalAddr = localTPEBase + srcRank * paddedExpNum;
            GlobalI32 signalG(signalAddr, signalShape, signalStride);
            pto::comm::TWAIT(signalG, 0, pto::comm::WaitCmp::NE);
        }

        // All data arrived. Now restore values and compute routing tables.
        // Only core 0 computes cumsumMM and preSumBeforeRank (small data, sequential).
        pto::SYNCALL<pto::SyncAllMode::Soft>(syncGmG, syncUbTile);

        if (coreIdx == 0) {
            // Phase B.1: Read all TPE rows from shmem, restore (subtract flag),
            // write to workspace TPE using TLOAD/TADDS/TSTORE (padded layout)
            pipe_barrier(PIPE_ALL);

            using TPEShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using TPEStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using TPEGlobal = pto::GlobalTensor<int32_t, TPEShape, TPEStride, pto::Layout::ND>;
            using TPETile = pto::Tile<pto::TileType::Vec, int32_t, 1, 64, pto::BLayout::RowMajor, -1, -1>;

            TPETile tpeRowTile(1, paddedExpNum);
            constexpr int32_t TPE_UB_OFFSET = SYNC_UB_ELEMS * static_cast<int32_t>(sizeof(int32_t));
            TASSIGN(tpeRowTile, TPE_UB_OFFSET);
            tpeRowTile.RowMaskInternal = 1;
            tpeRowTile.ColMaskInternal = paddedExpNum;

            int64_t tpeRowBytes = static_cast<int64_t>(paddedExpNum) * sizeof(int32_t);
            TPEShape rowShape(1, 1, 1, 1, static_cast<size_t>(paddedExpNum));
            TPEStride rowStride(tpeRowBytes / 4, tpeRowBytes / 4, tpeRowBytes / 4, tpeRowBytes / 4, 1);

            for (int32_t srcRank = 0; srcRank < EP; ++srcRank) {
                __gm__ int32_t* srcAddr = localTPEBase + srcRank * paddedExpNum;
                TPEGlobal srcG(srcAddr, rowShape, rowStride);
                TLOAD(tpeRowTile, srcG);

                if (srcRank != myRank) {
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TADDS(tpeRowTile, tpeRowTile, -static_cast<int32_t>(DATA_AS_FLAG_OFFSET));
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                } else {
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                }

                __gm__ int32_t* dstAddr = wsTPE + srcRank * paddedExpNum;
                TPEGlobal dstG(dstAddr, rowShape, rowStride);
                TSTORE(dstG, tpeRowTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }

            // Phase B.2: Compute cumsumMM (vectorized prefix sum using TADD)
            // Full-row TLOAD/TADD/TSTORE on paddedExpNum-wide rows for DMA alignment.
            // Result layout: wsCumsumMM[i * paddedExpNum + col] = prefix sum over srcRank 0..i
            pipe_barrier(PIPE_ALL);

            TPETile accumTile(1, paddedExpNum);
            TPETile tmpTile(1, paddedExpNum);
            constexpr int32_t ACCUM_UB_OFFSET = 0;
            int32_t tmpUbOffset = paddedExpNum * static_cast<int32_t>(sizeof(int32_t));
            TASSIGN(accumTile, ACCUM_UB_OFFSET);
            TASSIGN(tmpTile, tmpUbOffset);
            accumTile.RowMaskInternal = 1;
            accumTile.ColMaskInternal = paddedExpNum;
            tmpTile.RowMaskInternal = 1;
            tmpTile.ColMaskInternal = paddedExpNum;

            for (int32_t i = 0; i < EP; ++i) {
                __gm__ int32_t* srcAddr = wsTPE + i * paddedExpNum;
                TPEGlobal srcG(srcAddr, rowShape, rowStride);

                if (i == 0) {
                    TLOAD(accumTile, srcG);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                } else {
                    TLOAD(tmpTile, srcG);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TADD(accumTile, accumTile, tmpTile);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                }

                __gm__ int32_t* dstAddr = wsCumsumMM + i * paddedExpNum;
                TPEGlobal dstG(dstAddr, rowShape, rowStride);
                TSTORE(dstG, accumTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }

            // Phase B.3: Compute preSumBeforeRank
            // preSumBeforeRank[srcRank * expertPerRank + g] = sum of all tokens that
            // srcRank sends to (dst < myRank) and (dst == myRank, expert < g)
            for (int32_t srcRank = 0; srcRank < EP; ++srcRank) {
                int32_t offset = 0;
                for (int32_t dst = 0; dst < EP; ++dst) {
                    for (int32_t g = 0; g < expertPerRank; ++g) {
                        if (dst == myRank) {
                            volatile __gm__ int32_t* psbrPtr =
                                reinterpret_cast<volatile __gm__ int32_t*>(wsPSBR + srcRank * expertPerRank + g);
                            *psbrPtr = offset;
                        }
                        int32_t tpeIdx = srcRank * paddedExpNum + dst * expertPerRank + g;
                        volatile __gm__ int32_t* tpePtr = reinterpret_cast<volatile __gm__ int32_t*>(wsTPE + tpeIdx);
                        __asm__ __volatile__("");
                        dcci((__gm__ void*)tpePtr, SINGLE_CACHE_LINE);
                        __asm__ __volatile__("");
                        offset += *tpePtr;
                    }
                }
            }

            // Flush writes
            pipe_barrier(PIPE_ALL);
        }
    }

    // ========================================================================
    // Phase C: SYNCALL then dispatch using computed routing tables
    // ========================================================================
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGmG, syncUbTile);

    MoeDispatchDirect<HIDDEN_SIZE, TILE_COLS, MOVE_NUM>(
        gmA, gmPerTokenScale, wsCumsumMM + myRank * expertPerRank, wsTPE, wsPSBR, shmemBase, hcclCtx, EP, expertPerRank,
        maxOutputSize, offsetA, paddedExpNum, paddedExpNum);
}

// ============================================================================
// __global__ Entry Points — Direct Path
// ============================================================================
#define DIRECT_KERNEL_PARAMS                                                                                     \
    __gm__ int8_t *gmA, __gm__ float *gmPerTokenScale, __gm__ int32_t *cumsumMM, __gm__ int32_t *tokenPerExpert, \
        __gm__ int32_t *preSumBeforeRank, __gm__ uint8_t *shmemBase, __gm__ CommDeviceContext *hcclCtx,          \
        __gm__ int32_t *syncWorkspace, int32_t EP, int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA

extern "C" __global__ AICORE void MoeDispatchDirect_K128(DIRECT_KERNEL_PARAMS)
{
    MoeDispatchDirect<128, 160, DispatchTraits<160>::MOVE_NUM>(
        gmA, gmPerTokenScale, cumsumMM, tokenPerExpert, preSumBeforeRank, shmemBase, hcclCtx, EP, expertPerRank,
        maxOutputSize, offsetA);
}

// ============================================================================
// __global__ Entry Points — ViaGM Path
// ============================================================================
#define VIAGM_KERNEL_PARAMS                                                                                   \
    __gm__ int8_t *gmA, __gm__ float *gmPerTokenScale, __gm__ int8_t *tempGmBuffer, __gm__ int32_t *cumsumMM, \
        __gm__ int32_t *tokenPerExpert, __gm__ int32_t *preSumBeforeRank, __gm__ uint8_t *shmemBase,          \
        __gm__ CommDeviceContext *hcclCtx, __gm__ int32_t *syncWorkspace, int32_t EP, int32_t expertPerRank,  \
        int32_t maxOutputSize, int64_t offsetA

extern "C" __global__ AICORE void MoeDispatchViaGM_K128(VIAGM_KERNEL_PARAMS)
{
    MoeDispatchViaGM<128, 160, DispatchTraits<160>::MOVE_NUM>(
        gmA, gmPerTokenScale, tempGmBuffer, cumsumMM, tokenPerExpert, preSumBeforeRank, shmemBase, hcclCtx, EP,
        expertPerRank, maxOutputSize, offsetA);
}

// ============================================================================
// __global__ Entry Points — WithSync Path (CrossRankSync + Direct)
// ============================================================================
#define WITHSYNC_KERNEL_PARAMS                                                                                       \
    __gm__ int8_t *gmA, __gm__ float *gmPerTokenScale, __gm__ uint8_t *shmemBase, __gm__ CommDeviceContext *hcclCtx, \
        __gm__ int32_t *workspace, __gm__ int32_t *syncGmWorkspace, int32_t EP, int32_t expertPerRank,               \
        int32_t maxOutputSize, int64_t offsetA, int64_t offsetTPE

extern "C" __global__ AICORE void MoeDispatchWithSync_K128(WITHSYNC_KERNEL_PARAMS)
{
    MoeDispatchWithSync<128, 160, DispatchTraits<160>::MOVE_NUM>(
        gmA, gmPerTokenScale, shmemBase, hcclCtx, workspace, syncGmWorkspace, EP, expertPerRank, maxOutputSize, offsetA,
        offsetTPE);
}

// ============================================================================
// Host-callable launch wrappers
// ============================================================================
#include "acl/acl.h"
#include <cstdio>

bool LaunchMoeDispatchK128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* cumsumMM, void* tokenPerExpert,
    void* preSumBeforeRank, void* shmemBase, void* hcclCtx, void* syncWorkspace, int32_t EP, int32_t expertPerRank,
    int32_t maxOutputSize, int64_t offsetA)
{
    fprintf(
        stderr, "[KERNEL] LaunchMoeDispatchDirect_K128: blockNum=%d EP=%d expertPerRank=%d maxOutput=%d\n", blockNum,
        EP, expertPerRank, maxOutputSize);
    MoeDispatchDirect_K128<<<blockNum, nullptr, stream>>>(
        (__gm__ int8_t*)gmA, (__gm__ float*)gmPerTokenScale, (__gm__ int32_t*)cumsumMM, (__gm__ int32_t*)tokenPerExpert,
        (__gm__ int32_t*)preSumBeforeRank, (__gm__ uint8_t*)shmemBase, (__gm__ CommDeviceContext*)hcclCtx,
        (__gm__ int32_t*)syncWorkspace, EP, expertPerRank, maxOutputSize, offsetA);
    aclError err = aclrtSynchronizeStream((aclrtStream)stream);
    fprintf(stderr, "[KERNEL] aclrtSynchronizeStream returned: %d\n", (int)err);
    return (err == ACL_SUCCESS);
}

bool LaunchMoeDispatchViaGM_K128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* tempGmBuffer, void* cumsumMM,
    void* tokenPerExpert, void* preSumBeforeRank, void* shmemBase, void* hcclCtx, void* syncWorkspace, int32_t EP,
    int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA)
{
    fprintf(
        stderr, "[KERNEL] LaunchMoeDispatchViaGM_K128: blockNum=%d EP=%d expertPerRank=%d maxOutput=%d\n", blockNum, EP,
        expertPerRank, maxOutputSize);
    MoeDispatchViaGM_K128<<<blockNum, nullptr, stream>>>(
        (__gm__ int8_t*)gmA, (__gm__ float*)gmPerTokenScale, (__gm__ int8_t*)tempGmBuffer, (__gm__ int32_t*)cumsumMM,
        (__gm__ int32_t*)tokenPerExpert, (__gm__ int32_t*)preSumBeforeRank, (__gm__ uint8_t*)shmemBase,
        (__gm__ CommDeviceContext*)hcclCtx, (__gm__ int32_t*)syncWorkspace, EP, expertPerRank, maxOutputSize, offsetA);
    aclError err = aclrtSynchronizeStream((aclrtStream)stream);
    fprintf(stderr, "[KERNEL] aclrtSynchronizeStream returned: %d\n", (int)err);
    return (err == ACL_SUCCESS);
}

bool LaunchMoeDispatchWithSync_K128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* shmemBase, void* hcclCtx, void* workspace,
    void* syncGmWorkspace, int32_t EP, int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA, int64_t offsetTPE)
{
    fprintf(
        stderr, "[KERNEL] LaunchMoeDispatchWithSync_K128: blockNum=%d EP=%d expertPerRank=%d maxOutput=%d\n", blockNum,
        EP, expertPerRank, maxOutputSize);
    MoeDispatchWithSync_K128<<<blockNum, nullptr, stream>>>(
        (__gm__ int8_t*)gmA, (__gm__ float*)gmPerTokenScale, (__gm__ uint8_t*)shmemBase,
        (__gm__ CommDeviceContext*)hcclCtx, (__gm__ int32_t*)workspace, (__gm__ int32_t*)syncGmWorkspace, EP,
        expertPerRank, maxOutputSize, offsetA, offsetTPE);
    aclError err = aclrtSynchronizeStream((aclrtStream)stream);
    fprintf(stderr, "[KERNEL] aclrtSynchronizeStream returned: %d\n", (int)err);
    return (err == ACL_SUCCESS);
}
