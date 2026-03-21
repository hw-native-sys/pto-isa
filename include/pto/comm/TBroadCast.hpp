/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TBROADCAST_HPP
#define PTO_COMM_TBROADCAST_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Ping-pong state for chunked TBROADCAST transfers
struct TbroadcastPingPongState {
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;
};

// TLOAD src chunk → sync → TSTORE to all ranks → sync
template <typename ParallelGroupType, typename TileData, typename SrcViewT, typename DstViewT, typename DynShapeT,
          typename DynStrideT>
PTO_INTERNAL void TbroadcastChunkTransfer(ParallelGroupType &parallelGroup, TileData &tile, SrcViewT &srcView,
                                          int64_t dstOffset, const DynShapeT &chunkShape,
                                          const DynStrideT &dstChunkStride, int nranks)
{
    TLOAD(tile, srcView);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    for (int r = 0; r < nranks; ++r) {
        DstViewT dstView(parallelGroup[r].data() + dstOffset, chunkShape, dstChunkStride);
        TSTORE(dstView, tile);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Process one (dim0, dim1, dim2) slice with 2D row/col sliding for chunked broadcast
template <typename ParallelGroupType, typename TileData, typename SrcViewT, typename DstViewT, typename DynShape,
          typename GlobalSrcData, typename DynStride>
PTO_INTERNAL void TbroadcastChunked2DSlice(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                           TileData &stagingTileData, int64_t srcBase, int64_t dstOff, int gShape3,
                                           int gShape4, int tileValidRow, int tileValidCol, const int (&srcPitch)[5],
                                           const int (&dstPitch)[5], const DynStride &srcChunkStride,
                                           const DynStride &dstChunkStride, int nranks)
{
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    for (int rowCursor = 0; rowCursor < gShape3; rowCursor += tileValidRow) {
        int curRows = (rowCursor + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowCursor);
        if constexpr (isDynamicRow)
            stagingTileData.RowMaskInternal = curRows;
        for (int colCursor = 0; colCursor < gShape4; colCursor += tileValidCol) {
            int curCols = (colCursor + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colCursor);
            if constexpr (isDynamicCol)
                stagingTileData.ColMaskInternal = curCols;
            int64_t srcOff =
                srcBase + static_cast<int64_t>(rowCursor) * srcPitch[3] + static_cast<int64_t>(colCursor) * srcPitch[4];
            int64_t curDstOff =
                dstOff + static_cast<int64_t>(rowCursor) * dstPitch[3] + static_cast<int64_t>(colCursor) * dstPitch[4];
            DynShape chunkShape(1, 1, 1, curRows, curCols);
            SrcViewT srcView(srcGlobalData.data() + srcOff, chunkShape, srcChunkStride);
            TbroadcastChunkTransfer<ParallelGroupType, TileData, SrcViewT, DstViewT>(
                parallelGroup, stagingTileData, srcView, curDstOff, chunkShape, dstChunkStride, nranks);
        }
    }
}

// 2D sliding chunked broadcast with single buffer
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TbroadcastChunkedSingle(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                          TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                          int gShape4, int tileValidRow, int tileValidCol, int nranks)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    const int srcPitch[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                             static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                             static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                             static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                             static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    const int dstPitch[5] = {static_cast<int>(parallelGroup[0].GetStride(GlobalTensorDim::DIM_0)),
                             static_cast<int>(parallelGroup[0].GetStride(GlobalTensorDim::DIM_1)),
                             static_cast<int>(parallelGroup[0].GetStride(GlobalTensorDim::DIM_2)),
                             static_cast<int>(parallelGroup[0].GetStride(GlobalTensorDim::DIM_3)),
                             static_cast<int>(parallelGroup[0].GetStride(GlobalTensorDim::DIM_4))};
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcPitch[0], srcPitch[1], srcPitch[2], srcPitch[3], srcPitch[4]);
    DynStride dstChunkStride(dstPitch[0], dstPitch[1], dstPitch[2], dstPitch[3], dstPitch[4]);

    for (int dim0 = 0; dim0 < gShape0; ++dim0) {
        for (int dim1 = 0; dim1 < gShape1; ++dim1) {
            for (int dim2 = 0; dim2 < gShape2; ++dim2) {
                const int64_t srcBase = static_cast<int64_t>(dim0) * srcPitch[0] +
                                        static_cast<int64_t>(dim1) * srcPitch[1] +
                                        static_cast<int64_t>(dim2) * srcPitch[2];
                const int64_t dstBase = static_cast<int64_t>(dim0) * dstPitch[0] +
                                        static_cast<int64_t>(dim1) * dstPitch[1] +
                                        static_cast<int64_t>(dim2) * dstPitch[2];
                TbroadcastChunked2DSlice<ParallelGroupType, TileData, SrcViewT, DstViewT, DynShape>(
                    parallelGroup, srcGlobalData, stagingTileData, srcBase, dstBase, gShape3, gShape4, tileValidRow,
                    tileValidCol, srcPitch, dstPitch, srcChunkStride, dstChunkStride, nranks);
            }
        }
    }
}

// ============================================================================
// TBROADCAST_IMPL: Broadcast data from root NPU to all ranks
//
// The root loads srcGlobalData and stores it to every rank's buffer in the
// ParallelGroup.
//
// When the GlobalTensor exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, shape3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, shape4 must be divisible by ValidCol.
//   - All ranks in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TBROADCAST_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                  TileData &stagingTileData)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TBROADCAST: TileData element type must match GlobalData element type");
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>,
                  "TBROADCAST: ParallelGroup element type must match source element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TBROADCAST: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    const int gShape0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int tileValidRow = stagingTileData.GetValidRow();
    const int tileValidCol = stagingTileData.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TBROADCAST: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TBROADCAST: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    if (nranks == 1) {
        TLOAD(stagingTileData, srcGlobalData);
        PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        TSTORE(parallelGroup[rootIdx], stagingTileData);
        PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        return;
    }

    // Simple path: data fits in UB tile
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        TLOAD(stagingTileData, srcGlobalData);
        PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        for (int r = 0; r < nranks; ++r) {
            TSTORE(parallelGroup[r], stagingTileData);
            PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        }
        return;
    }

    // 2D sliding chunked path
    PTO_ASSERT(tileValidRow > 0, "TBROADCAST: tile ValidRow must be greater than 0 for chunked transfer");
    PTO_ASSERT(tileValidCol > 0, "TBROADCAST: tile ValidCol must be greater than 0 for chunked transfer");
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TBROADCAST chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TBROADCAST chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TbroadcastChunkedSingle<ParallelGroupType, GlobalSrcData, TileData>(parallelGroup, srcGlobalData, stagingTileData,
                                                                        gShape0, gShape1, gShape2, gShape3, gShape4,
                                                                        tileValidRow, tileValidCol, nranks);
}

// Process one chunk iteration with ping-pong for broadcast
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TbroadcastPingPongProcessChunk(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                                 TileData &pingTile, TileData &pongTile, int64_t srcOffset,
                                                 int64_t dstOffset, int currentRows, int currentCols,
                                                 const DynStrideT &srcChunkStride, const DynStrideT &dstChunkStride,
                                                 int nranks, TbroadcastPingPongState &state)
{
    using DstGlobalT = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename GlobalSrcData::RawDType;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, GlobalSrcData::layout>;
    using DstChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, DstGlobalT::layout>;

    const bool currentIsPing = state.usePing;
    TileData &loadTile = currentIsPing ? pingTile : pongTile;
    event_t curEvent = currentIsPing ? EVENT_ID0 : EVENT_ID1;

    constexpr bool hasDynRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool hasDynCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (hasDynRow)
        loadTile.RowMaskInternal = currentRows;
    if constexpr (hasDynCol)
        loadTile.ColMaskInternal = currentCols;

    ChunkShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcChunkView srcView(srcGlobalData.data() + srcOffset, chunkShape, srcChunkStride);

    if (state.hasPending) {
        TileData &storeTile = currentIsPing ? pongTile : pingTile;
        event_t prevEvent = currentIsPing ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

        ChunkShape pendShape(1, 1, 1, state.pendingRows, state.pendingCols);
        for (int r = 0; r < nranks; ++r) {
            DstChunkView dstView(parallelGroup[r].data() + state.pendingDstOffset, pendShape, dstChunkStride);
            TSTORE(dstView, storeTile);
        }
        TLOAD(loadTile, srcView);

        set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
    } else {
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
    }

    state.usePing = !currentIsPing;
    state.hasPending = true;
    state.pendingDstOffset = dstOffset;
    state.pendingRows = currentRows;
    state.pendingCols = currentCols;
}

// Drain last pending broadcast stores
template <typename ParallelGroupType, typename TileData, typename DynStrideT>
PTO_INTERNAL void TbroadcastPingPongEpilogue(ParallelGroupType &parallelGroup, TileData &pingTile, TileData &pongTile,
                                             const TbroadcastPingPongState &state, const DynStrideT &dstChunkStride,
                                             int nranks)
{
    using DstGlobalT = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename DstGlobalT::RawDType;
    if (!state.hasPending)
        return;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DstChunkView = GlobalTensor<ElemT, ChunkShape, DynStrideT, DstGlobalT::layout>;

    const bool finalInPong = state.usePing;
    TileData &lastTile = finalInPong ? pongTile : pingTile;
    event_t lastEvent = finalInPong ? EVENT_ID1 : EVENT_ID0;

    wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);
    ChunkShape lastShape(1, 1, 1, state.pendingRows, state.pendingCols);
    for (int r = 0; r < nranks; ++r) {
        DstChunkView dstView(parallelGroup[r].data() + state.pendingDstOffset, lastShape, dstChunkStride);
        TSTORE(dstView, lastTile);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
    wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
}

// 2D sliding chunked broadcast with ping-pong double buffering
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TbroadcastChunkedPingPong(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                            TileData &pingTile, TileData &pongTile, int gShape0, int gShape1,
                                            int gShape2, int gShape3, int gShape4, int tileValidRow, int tileValidCol,
                                            int nranks)
{
    const int srcStep[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    auto &refDst = parallelGroup[0];
    const int dstStep[5] = {static_cast<int>(refDst.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(refDst.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(refDst.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(refDst.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(refDst.GetStride(GlobalTensorDim::DIM_4))};

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStep[0], srcStep[1], srcStep[2], srcStep[3], srcStep[4]);
    DynStride dstChunkStride(dstStep[0], dstStep[1], dstStep[2], dstStep[3], dstStep[4]);
    TbroadcastPingPongState state;

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                const int64_t srcBase = static_cast<int64_t>(i0) * srcStep[0] + static_cast<int64_t>(i1) * srcStep[1] +
                                        static_cast<int64_t>(i2) * srcStep[2];
                const int64_t dstBase = static_cast<int64_t>(i0) * dstStep[0] + static_cast<int64_t>(i1) * dstStep[1] +
                                        static_cast<int64_t>(i2) * dstStep[2];
                int rowCursor = 0;
                while (rowCursor < gShape3) {
                    const int rowRemain = gShape3 - rowCursor;
                    const int curRows = (rowRemain < tileValidRow) ? rowRemain : tileValidRow;
                    int colCursor = 0;
                    while (colCursor < gShape4) {
                        const int colRemain = gShape4 - colCursor;
                        const int curCols = (colRemain < tileValidCol) ? colRemain : tileValidCol;
                        const int64_t srcOff = srcBase + static_cast<int64_t>(rowCursor) * srcStep[3] +
                                               static_cast<int64_t>(colCursor) * srcStep[4];
                        const int64_t dstOff = dstBase + static_cast<int64_t>(rowCursor) * dstStep[3] +
                                               static_cast<int64_t>(colCursor) * dstStep[4];
                        TbroadcastPingPongProcessChunk<ParallelGroupType, GlobalSrcData, TileData>(
                            parallelGroup, srcGlobalData, pingTile, pongTile, srcOff, dstOff, curRows, curCols,
                            srcChunkStride, dstChunkStride, nranks, state);
                        colCursor += tileValidCol;
                    }
                    rowCursor += tileValidRow;
                }
            }
        }
    }

    TbroadcastPingPongEpilogue<ParallelGroupType, TileData>(parallelGroup, pingTile, pongTile, state, dstChunkStride,
                                                            nranks);
}

// ============================================================================
// TBROADCAST_IMPL (ping-pong): Broadcast with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD of the next
// chunk (MTE2) with TSTORE of the current chunk to all ranks (MTE3).
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [N×TSTORE chunk0] -> [TLOAD chunk1] -> [N×TSTORE chunk1] -> ...
//
// Timeline with ping-pong:
//   [TLOAD chunk0] -> [N×TSTORE chunk0 | TLOAD chunk1] -> [N×TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Constraints: same as TBROADCAST_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TBROADCAST_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData, TileData &pingTile,
                                  TileData &pongTile)
{
    using DstGlobalT = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<ElemT, typename TileData::DType>,
                  "TBROADCAST: TileData element type must match GlobalData element type");
    static_assert(std::is_same_v<ElemT, typename DstGlobalT::RawDType>,
                  "TBROADCAST: ParallelGroup element type must match source element type");
    static_assert(GlobalSrcData::layout == DstGlobalT::layout, "TBROADCAST: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    const int dims[5] = {static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_0)),
                         static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_1)),
                         static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_2)),
                         static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_3)),
                         static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_4))};

    const int64_t totalRows = static_cast<int64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    const int tileValidRow = pingTile.GetValidRow();
    const int tileValidCol = pingTile.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TBROADCAST: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TBROADCAST: tileValidCol must be greater than 0");

    if (totalRows == 0 || dims[4] == 0) {
        return;
    }

    if (nranks == 1) {
        TLOAD(pingTile, srcGlobalData);
        PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        TSTORE(parallelGroup[rootIdx], pingTile);
        PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        return;
    }

    // Simple path: single chunk, no ping-pong benefit
    if (totalRows <= tileValidRow && dims[4] <= tileValidCol) {
        TLOAD(pingTile, srcGlobalData);
        PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        for (int r = 0; r < nranks; ++r) {
            TSTORE(parallelGroup[r], pingTile);
            PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        }
        return;
    }

    // 2D sliding chunked path with ping-pong double buffering
    constexpr bool hasDynRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool hasDynCol = (TileData::ValidCol == DYNAMIC);
    if constexpr (!hasDynRow) {
        PTO_ASSERT(dims[3] % tileValidRow == 0,
                   "TBROADCAST chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!hasDynCol) {
        PTO_ASSERT(dims[4] % tileValidCol == 0,
                   "TBROADCAST chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TbroadcastChunkedPingPong<ParallelGroupType, GlobalSrcData, TileData>(parallelGroup, srcGlobalData, pingTile,
                                                                          pongTile, dims[0], dims[1], dims[2], dims[3],
                                                                          dims[4], tileValidRow, tileValidCol, nranks);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TBROADCAST_HPP
