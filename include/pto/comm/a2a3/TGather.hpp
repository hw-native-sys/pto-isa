/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGATHER_HPP
#define PTO_COMM_TGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Ping-pong double-buffering state for chunked TGATHER transfers
struct TgatherPingPongState {
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;
};

// Simple gather path: per-rank data fits in a single tile, loop over all ranks
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TgatherSimple(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                int gShape4, int nranks, int perRankRows)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

    using DynShape5D = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstViewT = GlobalTensor<T, DynShape5D, DynStride, GlobalDstData::layout>;

    DynShape5D perRankShape(gShape0, gShape1, gShape2, gShape3, gShape4);
    DynStride dstViewStride(
        dstGlobalData.GetStride(GlobalTensorDim::DIM_0), dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
        dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3, dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

    for (int r = 0; r < nranks; ++r) {
        TLOAD(stagingTileData, parallelGroup[r]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        int64_t dstOffset = static_cast<int64_t>(r) * perRankRows * dstStride3;
        DstViewT dstView(dstGlobalData.data() + dstOffset, perRankShape, dstViewStride);
        TSTORE(dstView, stagingTileData);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

// Inner row/col 2D sliding loop for chunked gather
template <typename TileData, typename T, Layout srcLayout, Layout dstLayout>
PTO_INTERNAL void TgatherChunkedRowColLoop(TileData &stagingTileData, __gm__ T *srcPtr, __gm__ T *dstPtr,
                                           int64_t srcBase, int64_t dstBase, int gShape3, int gShape4, int tileValidRow,
                                           int tileValidCol, const int (&srcStep)[5], const int (&dstStep)[5])
{
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcChunkView = GlobalTensor<T, ChunkShape, DynStride, srcLayout>;
    using DstChunkView = GlobalTensor<T, ChunkShape, DynStride, dstLayout>;
    DynStride srcChunkStride(srcStep[0], srcStep[1], srcStep[2], srcStep[3], srcStep[4]);
    DynStride dstChunkStride(dstStep[0], dstStep[1], dstStep[2], dstStep[3], dstStep[4]);

    for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
        int curRows = (gShape3 - rowOff < tileValidRow) ? (gShape3 - rowOff) : tileValidRow;
        if constexpr (isDynamicRow)
            stagingTileData.RowMaskInternal = curRows;
        for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
            int curCols = (gShape4 - colOff < tileValidCol) ? (gShape4 - colOff) : tileValidCol;
            if constexpr (isDynamicCol)
                stagingTileData.ColMaskInternal = curCols;
            const int64_t srcOff =
                srcBase + static_cast<int64_t>(rowOff) * srcStep[3] + static_cast<int64_t>(colOff) * srcStep[4];
            const int64_t dstOff =
                dstBase + static_cast<int64_t>(rowOff) * dstStep[3] + static_cast<int64_t>(colOff) * dstStep[4];
            ChunkShape chunkShape(1, 1, 1, curRows, curCols);
            SrcChunkView srcView(srcPtr + srcOff, chunkShape, srcChunkStride);
            TLOAD(stagingTileData, srcView);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            DstChunkView dstChunk(dstPtr + dstOff, chunkShape, dstChunkStride);
            TSTORE(dstChunk, stagingTileData);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

// 2D sliding chunked gather with single buffer
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TgatherChunkedSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                       int gShape4, int tileValidRow, int tileValidCol, int nranks, int perRankRows)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    auto &refSrc = parallelGroup[0];
    const int srcStep[5] = {static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_4))};
    const int dstStep[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    for (int r = 0; r < nranks; ++r) {
        const int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstStep[3];
        for (int i0 = 0; i0 < gShape0; ++i0) {
            for (int i1 = 0; i1 < gShape1; ++i1) {
                for (int i2 = 0; i2 < gShape2; ++i2) {
                    const int64_t srcBase = static_cast<int64_t>(i0) * srcStep[0] +
                                            static_cast<int64_t>(i1) * srcStep[1] +
                                            static_cast<int64_t>(i2) * srcStep[2];
                    const int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstStep[0] +
                                            static_cast<int64_t>(i1) * dstStep[1] +
                                            static_cast<int64_t>(i2) * dstStep[2];
                    TgatherChunkedRowColLoop<TileData, T, GlobalSrcData::layout, GlobalDstData::layout>(
                        stagingTileData, parallelGroup[r].data(), dstGlobalData.data(), srcBase, dstBase, gShape3,
                        gShape4, tileValidRow, tileValidCol, srcStep, dstStep);
                }
            }
        }
    }
}

// ============================================================================
// TGATHER_IMPL: Gather operation - root collects data from all ranks
//
// The calling NPU is the root and gathers data from all ranks, concatenating
// the results along DIM_3 (row dimension) into a local output buffer.
//
// Each rank r contributes data of shape (D0, D1, D2, H, W). The destination
// tensor has shape (D0, D1, D2, N*H, W), where rank r's data is placed at
// rows [r*H, (r+1)*H).
//
// When the per-rank GlobalTensor exceeds the UB tile capacity in rows and/or
// columns, the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, per-rank DIM_3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, DIM_4 must be divisible by ValidCol.
//   - All source tensors in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                               TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0, "TGATHER: ParallelGroup size must be positive");
    const int rootIdx = parallelGroup.GetRootIdx();
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "TGATHER: rootIdx out of range");

    auto &srcRef = parallelGroup[0];
    const int gShape0 = srcRef.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcRef.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcRef.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcRef.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcRef.GetShape(GlobalTensorDim::DIM_4);

    const int perRankRows = gShape3;
    const int tileValidRow = stagingTileData.GetValidRow();
    const int tileValidCol = stagingTileData.GetValidCol();
    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;

    PTO_ASSERT(tileValidRow > 0, "TGATHER: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TGATHER: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    // Simple path: per-rank data fits in UB tile
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        TgatherSimple<ParallelGroupType, GlobalDstData, TileData>(parallelGroup, dstGlobalData, stagingTileData,
                                                                  gShape0, gShape1, gShape2, gShape3, gShape4, nranks,
                                                                  perRankRows);
        return;
    }

    // 2D sliding chunked path
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TGATHER chunked: per-rank DIM_3 must be divisible by tile ValidRow when static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TGATHER chunked: DIM_4 must be divisible by tile ValidCol when static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TgatherChunkedSingle<ParallelGroupType, GlobalDstData, TileData>(parallelGroup, dstGlobalData, stagingTileData,
                                                                     gShape0, gShape1, gShape2, gShape3, gShape4,
                                                                     tileValidRow, tileValidCol, nranks, perRankRows);
}

// Process one chunk iteration with ping-pong double buffering for gather
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TgatherPingPongProcessChunk(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                              TileData &pingTile, TileData &pongTile, int64_t srcOffset,
                                              int64_t dstOffset, int currentRows, int currentCols,
                                              const DynStrideT &srcChunkStride, const DynStrideT &dstChunkStride,
                                              int rank, TgatherPingPongState &state)
{
    using SrcGlobalT = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename SrcGlobalT::RawDType;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, SrcGlobalT::layout>;
    using DstChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, GlobalDstData::layout>;

    const bool currentIsPing = state.usePing;
    constexpr bool hasDynRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool hasDynCol = (TileData::ValidCol == DYNAMIC);

    TileData &loadTile = currentIsPing ? pingTile : pongTile;
    event_t curEvent = currentIsPing ? EVENT_ID0 : EVENT_ID1;

    if constexpr (hasDynRow)
        loadTile.RowMaskInternal = currentRows;
    if constexpr (hasDynCol)
        loadTile.ColMaskInternal = currentCols;

    ChunkShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcChunkView srcView(parallelGroup[rank].data() + srcOffset, chunkShape, srcChunkStride);

    if (state.hasPending) {
        TileData &storeTile = currentIsPing ? pongTile : pingTile;
        event_t prevEvent = currentIsPing ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

        ChunkShape pendShape(1, 1, 1, state.pendingRows, state.pendingCols);
        DstChunkView pendDst(dstGlobalData.data() + state.pendingDstOffset, pendShape, dstChunkStride);

        TSTORE(pendDst, storeTile);
        TLOAD(loadTile, srcView);

        set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);

        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
    } else {
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
    }

    state.pendingDstOffset = dstOffset;
    state.pendingRows = currentRows;
    state.pendingCols = currentCols;
    state.hasPending = true;
    state.usePing = !currentIsPing;
}

// Drain the last pending TSTORE after the chunked ping-pong loop completes
template <typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TgatherPingPongEpilogue(GlobalDstData &dstGlobalData, TileData &pingTile, TileData &pongTile,
                                          const TgatherPingPongState &state, const DynStrideT &dstChunkStride)
{
    if (!state.hasPending)
        return;
    using ElemT = typename GlobalDstData::RawDType;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DstChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, GlobalDstData::layout>;

    const bool finalInPong = state.usePing;
    TileData &lastTile = finalInPong ? pongTile : pingTile;
    event_t lastEvent = finalInPong ? EVENT_ID1 : EVENT_ID0;

    wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);
    ChunkShape lastShape(1, 1, 1, state.pendingRows, state.pendingCols);
    DstChunkView dstView(dstGlobalData.data() + state.pendingDstOffset, lastShape, dstChunkStride);
    TSTORE(dstView, lastTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
    wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
}

// Process one (rank, dim0, dim1, dim2) slice with 2D row/col sliding for chunked gather ping-pong
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStride>
PTO_INTERNAL void TgatherPingPong2DSlice(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                         TileData &pingTile, TileData &pongTile, int64_t srcBase, int64_t dstBase,
                                         int gShape3, int gShape4, int tileValidRow, int tileValidCol,
                                         const int (&srcPitch)[5], const int (&dstPitch)[5],
                                         const DynStride &srcChunkStride, const DynStride &dstChunkStride, int rank,
                                         TgatherPingPongState &state)
{
    int rowCursor = 0;
    while (rowCursor < gShape3) {
        const int curRows = (gShape3 - rowCursor < tileValidRow) ? (gShape3 - rowCursor) : tileValidRow;
        int colCursor = 0;
        while (colCursor < gShape4) {
            const int curCols = (gShape4 - colCursor < tileValidCol) ? (gShape4 - colCursor) : tileValidCol;
            const int64_t srcOff =
                srcBase + static_cast<int64_t>(rowCursor) * srcPitch[3] + static_cast<int64_t>(colCursor) * srcPitch[4];
            const int64_t dstOff =
                dstBase + static_cast<int64_t>(rowCursor) * dstPitch[3] + static_cast<int64_t>(colCursor) * dstPitch[4];
            TgatherPingPongProcessChunk<ParallelGroupType, GlobalDstData, TileData>(
                parallelGroup, dstGlobalData, pingTile, pongTile, srcOff, dstOff, curRows, curCols, srcChunkStride,
                dstChunkStride, rank, state);
            colCursor += tileValidCol;
        }
        rowCursor += tileValidRow;
    }
}

// 2D sliding chunked gather with ping-pong double buffering
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TgatherChunkedPingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                         TileData &pingTile, TileData &pongTile, int gShape0, int gShape1, int gShape2,
                                         int gShape3, int gShape4, int tileValidRow, int tileValidCol, int nranks,
                                         int perRankRows)
{
    auto &refSrc = parallelGroup[0];
    const int srcPitch[5] = {static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_0)),
                             static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_1)),
                             static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_2)),
                             static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_3)),
                             static_cast<int>(refSrc.GetStride(GlobalTensorDim::DIM_4))};
    const int dstPitch[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                             static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                             static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                             static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                             static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcPitch[0], srcPitch[1], srcPitch[2], srcPitch[3], srcPitch[4]);
    DynStride dstChunkStride(dstPitch[0], dstPitch[1], dstPitch[2], dstPitch[3], dstPitch[4]);
    TgatherPingPongState state;

    for (int r = 0; r < nranks; ++r) {
        const int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstPitch[3];
        for (int i0 = 0; i0 < gShape0; ++i0) {
            for (int i1 = 0; i1 < gShape1; ++i1) {
                for (int i2 = 0; i2 < gShape2; ++i2) {
                    const int64_t srcBase = static_cast<int64_t>(i0) * srcPitch[0] +
                                            static_cast<int64_t>(i1) * srcPitch[1] +
                                            static_cast<int64_t>(i2) * srcPitch[2];
                    const int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstPitch[0] +
                                            static_cast<int64_t>(i1) * dstPitch[1] +
                                            static_cast<int64_t>(i2) * dstPitch[2];
                    TgatherPingPong2DSlice<ParallelGroupType, GlobalDstData, TileData>(
                        parallelGroup, dstGlobalData, pingTile, pongTile, srcBase, dstBase, gShape3, gShape4,
                        tileValidRow, tileValidCol, srcPitch, dstPitch, srcChunkStride, dstChunkStride, r, state);
                }
            }
        }
    }
    TgatherPingPongEpilogue<GlobalDstData, TileData>(dstGlobalData, pingTile, pongTile, state, dstChunkStride);
}

// ============================================================================
// TGATHER_IMPL (ping-pong): Gather with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD of the next
// chunk (MTE2) with TSTORE of the current chunk (MTE3).
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Constraints: same as TGATHER_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &pingTile,
                               TileData &pongTile)
{
    using SrcGlobalT = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename SrcGlobalT::RawDType;

    static_assert(std::is_same_v<ElemT, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<ElemT, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(SrcGlobalT::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    auto &refTensor = parallelGroup[0];
    const int dims[5] = {static_cast<int>(refTensor.GetShape(GlobalTensorDim::DIM_0)),
                         static_cast<int>(refTensor.GetShape(GlobalTensorDim::DIM_1)),
                         static_cast<int>(refTensor.GetShape(GlobalTensorDim::DIM_2)),
                         static_cast<int>(refTensor.GetShape(GlobalTensorDim::DIM_3)),
                         static_cast<int>(refTensor.GetShape(GlobalTensorDim::DIM_4))};

    const int perRankRows = dims[3];
    const int64_t totalRows = static_cast<int64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    const int tileValidRow = pingTile.GetValidRow();
    const int tileValidCol = pingTile.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TGATHER: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TGATHER: tileValidCol must be greater than 0");

    if (totalRows == 0 || dims[4] == 0) {
        return;
    }

    // Simple path: per-rank data fits in UB tile, no ping-pong benefit
    if (totalRows <= tileValidRow && dims[4] <= tileValidCol) {
        TgatherSimple<ParallelGroupType, GlobalDstData, TileData>(
            parallelGroup, dstGlobalData, pingTile, dims[0], dims[1], dims[2], dims[3], dims[4], nranks, perRankRows);
        return;
    }

    // 2D sliding chunked path with ping-pong double buffering
    constexpr bool hasDynRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool hasDynCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!hasDynRow) {
        PTO_ASSERT(dims[3] % tileValidRow == 0,
                   "TGATHER chunked: per-rank DIM_3 must be divisible by tile ValidRow when static.");
    }
    if constexpr (!hasDynCol) {
        PTO_ASSERT(dims[4] % tileValidCol == 0,
                   "TGATHER chunked: DIM_4 must be divisible by tile ValidCol when static.");
    }

    TgatherChunkedPingPong<ParallelGroupType, GlobalDstData, TileData>(parallelGroup, dstGlobalData, pingTile, pongTile,
                                                                       dims[0], dims[1], dims[2], dims[3], dims[4],
                                                                       tileValidRow, tileValidCol, nranks, perRankRows);
}

// CCU engine is only available on A5 hardware.  This stub mirrors the
// `a2a3/async/TPutAsync.hpp` URMA branch: the name must exist as a template
// in `pto::comm` so that `::pto::comm::TGATHER_CCU_IMPL<engine>(...)` in
// pto_comm_inst.hpp parses on A2/A3 builds; the body's static_assert depends
// on the `engine` template parameter and only fires if the overload is
// actually instantiated, which is the misuse it flags.
// `a5/TGather.hpp` defines `PTO_COMM_A5_TGATHER_PROVIDED` before including
// this header so the stub is omitted on A5 builds, where the real
// `TGATHER_CCU_IMPL` overload lives.  Otherwise the generic `Args&&...` pack
// would compete with the real `T&` parameters in `pto::comm`'s overload set
// and could win under partial ordering on some compilers.
#ifndef PTO_COMM_A5_TGATHER_PROVIDED
template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TGATHER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU,
                  "TGATHER<CollEngine::CCU> requires A5 hardware; "
                  "CCU engine is not available on A2/A3.");
}
#endif // PTO_COMM_A5_TGATHER_PROVIDED

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGATHER_HPP
