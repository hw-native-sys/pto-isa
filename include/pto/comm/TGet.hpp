/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_HPP
#define PTO_COMM_TGET_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/constants.hpp"
#include "pto/common/type.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Ping-pong double-buffering state for chunked TGET transfers
struct TgetPingPongState {
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;
};

// Single synchronous transfer: TLOAD from src → sync → TSTORE to dst → sync
template <typename TileData, typename DstGT, typename SrcGT>
PTO_INTERNAL void TgetTransferOnce(DstGT &dst, SrcGT &src, TileData &tile)
{
    TLOAD(tile, src);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked transfer with single buffer
template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TgetChunkedSingle(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                    TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                    int gShape4, int tileValidRow, int tileValidCol)
{
    using T = typename GlobalSrcData::RawDType;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    const int rmtStep[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    const int locStep[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                            static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride remoteChunkStride(rmtStep[0], rmtStep[1], rmtStep[2], rmtStep[3], rmtStep[4]);
    DynStride localChunkStride(locStep[0], locStep[1], locStep[2], locStep[3], locStep[4]);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t rmtBase = static_cast<int64_t>(i0) * rmtStep[0] + static_cast<int64_t>(i1) * rmtStep[1] +
                                  static_cast<int64_t>(i2) * rmtStep[2];
                int64_t locBase = static_cast<int64_t>(i0) * locStep[0] + static_cast<int64_t>(i1) * locStep[1] +
                                  static_cast<int64_t>(i2) * locStep[2];
                for (int rowIdx = 0; rowIdx < gShape3; rowIdx += tileValidRow) {
                    int chunkRows = (rowIdx + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowIdx);
                    if constexpr (isDynamicRow)
                        stagingTileData.RowMaskInternal = chunkRows;
                    for (int colIdx = 0; colIdx < gShape4; colIdx += tileValidCol) {
                        int chunkCols = (colIdx + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colIdx);
                        if constexpr (isDynamicCol)
                            stagingTileData.ColMaskInternal = chunkCols;
                        int64_t remoteOff = rmtBase + static_cast<int64_t>(rowIdx) * rmtStep[3] +
                                            static_cast<int64_t>(colIdx) * rmtStep[4];
                        int64_t localOff = locBase + static_cast<int64_t>(rowIdx) * locStep[3] +
                                           static_cast<int64_t>(colIdx) * locStep[4];
                        DynShape chunkShape(1, 1, 1, chunkRows, chunkCols);
                        SrcViewT srcView(srcGlobalData.data() + remoteOff, chunkShape, remoteChunkStride);
                        DstViewT dstView(dstGlobalData.data() + localOff, chunkShape, localChunkStride);
                        TgetTransferOnce<TileData, DstViewT, SrcViewT>(dstView, srcView, stagingTileData);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TGET_IMPL: Remote read operation implementation
//
// Data flow: srcGlobalData (remote GM) → stagingTileData (UB) → dstGlobalData (local GM)
//
// Chunked transfer follows the same 2D sliding strategy as TPUT_IMPL
// (see TPut.hpp): outer dims iterated explicitly, DIM_3/DIM_4 split into
// tile-sized chunks. Static ValidRow/ValidCol require divisibility;
// use DYNAMIC for partial chunk support.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<typename GlobalDstData::RawDType, T>, "TGET: src/dst element type mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGET: TileData element type must match GlobalData element type");
    constexpr bool layoutMatched = (GlobalSrcData::layout == GlobalDstData::layout);
    static_assert(layoutMatched, "TGET: src/dst layout mismatch");

    const int remoteDims[5] = {static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_0)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_1)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_2)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_3)),
                               static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_4))};

    const int64_t totalRemoteRows = static_cast<int64_t>(remoteDims[0]) * remoteDims[1] * remoteDims[2] * remoteDims[3];
    const int singleTileRows = stagingTileData.GetValidRow();
    const int singleTileCols = stagingTileData.GetValidCol();

    PTO_ASSERT(singleTileRows > 0, "TGET: tileValidRow must be greater than 0");
    PTO_ASSERT(singleTileCols > 0, "TGET: tileValidCol must be greater than 0");

    if (totalRemoteRows == 0 || remoteDims[4] == 0) {
        return;
    }

    // Simple path: data fits in UB tile in both dimensions
    if (totalRemoteRows <= singleTileRows && remoteDims[4] <= singleTileCols) {
        TgetTransferOnce<TileData, GlobalDstData, GlobalSrcData>(dstGlobalData, srcGlobalData, stagingTileData);
        return;
    }

    // 2D sliding chunked path
    PTO_ASSERT(singleTileRows > 0, "TGET: tile ValidRow must be greater than 0 for chunked transfer");
    PTO_ASSERT(singleTileCols > 0, "TGET: tile ValidCol must be greater than 0 for chunked transfer");

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(remoteDims[3] % singleTileRows == 0,
                   "TGET chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(remoteDims[4] % singleTileCols == 0,
                   "TGET chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TgetChunkedSingle<GlobalDstData, GlobalSrcData, TileData>(
        dstGlobalData, srcGlobalData, stagingTileData, remoteDims[0], remoteDims[1], remoteDims[2], remoteDims[3],
        remoteDims[4], singleTileRows, singleTileCols);
}

// Process one chunk in the ping-pong pipeline: overlap TSTORE of previous chunk with TLOAD of current chunk
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename StrideT>
PTO_INTERNAL void TgetPingPongProcessChunk(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                           TileData &pingTile, TileData &pongTile, TgetPingPongState &pp,
                                           int64_t remoteOff, int64_t localOff, int chunkRows, int chunkCols,
                                           const StrideT &remoteChunkStride, const StrideT &localChunkStride)
{
    using DType = typename GlobalSrcData::RawDType;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<DType, DynShape, StrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<DType, DynShape, StrideT, GlobalDstData::layout>;

    TileData &loadTile = pp.usePing ? pingTile : pongTile;
    event_t curEvent = pp.usePing ? EVENT_ID0 : EVENT_ID1;
    if constexpr (isDynamicRow)
        loadTile.RowMaskInternal = chunkRows;
    if constexpr (isDynamicCol)
        loadTile.ColMaskInternal = chunkCols;

    DynShape chunkShape(1, 1, 1, chunkRows, chunkCols);
    SrcViewT srcView(srcGlobalData.data() + remoteOff, chunkShape, remoteChunkStride);

    if (pp.hasPending) {
        TileData &storeTile = pp.usePing ? pongTile : pingTile;
        event_t prevEvent = pp.usePing ? EVENT_ID1 : EVENT_ID0;
        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);
        DynShape pendShape(1, 1, 1, pp.pendingRows, pp.pendingCols);
        DstViewT pendView(dstGlobalData.data() + pp.pendingDstOffset, pendShape, localChunkStride);
        TSTORE(pendView, storeTile);
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
    } else {
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
    }

    pp.pendingDstOffset = localOff;
    pp.pendingRows = chunkRows;
    pp.pendingCols = chunkCols;
    pp.hasPending = true;
    pp.usePing = !pp.usePing;
}

// Flush the last pending TSTORE in the ping-pong pipeline
template <typename GlobalDstData, typename TileData, typename StrideT>
PTO_INTERNAL void TgetPingPongFlush(GlobalDstData &dstGlobalData, TileData &pingTile, TileData &pongTile,
                                    TgetPingPongState &pp, const StrideT &localChunkStride)
{
    if (!pp.hasPending)
        return;

    using DType = typename GlobalDstData::RawDType;
    using ChunkShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DstView = GlobalTensor<DType, ChunkShape, StrideT, GlobalDstData::layout>;

    TileData &finalTile = pp.usePing ? pongTile : pingTile;
    event_t finalEvent = pp.usePing ? EVENT_ID1 : EVENT_ID0;
    wait_flag(PIPE_MTE2, PIPE_MTE3, finalEvent);
    ChunkShape finalShape(1, 1, 1, pp.pendingRows, pp.pendingCols);
    DstView finalView(dstGlobalData.data() + pp.pendingDstOffset, finalShape, localChunkStride);
    TSTORE(finalView, finalTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, finalEvent);
    wait_flag(PIPE_MTE3, PIPE_MTE2, finalEvent);
}

// 2D sliding chunked transfer with ping-pong double buffering
// See TputChunkedPingPong in TPut.hpp for detailed pipeline overlap analysis.
template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TgetChunkedPingPong(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                                      TileData &pongTile, int gShape0, int gShape1, int gShape2, int gShape3,
                                      int gShape4, int tileValidRow, int tileValidCol)
{
    const int remoteStep[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                               static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    const int localStep[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride remoteChunkStride(remoteStep[0], remoteStep[1], remoteStep[2], remoteStep[3], remoteStep[4]);
    DynStride localChunkStride(localStep[0], localStep[1], localStep[2], localStep[3], localStep[4]);

    TgetPingPongState pp;

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t remoteBase = static_cast<int64_t>(i0) * remoteStep[0] +
                                     static_cast<int64_t>(i1) * remoteStep[1] +
                                     static_cast<int64_t>(i2) * remoteStep[2];
                int64_t localBase = static_cast<int64_t>(i0) * localStep[0] + static_cast<int64_t>(i1) * localStep[1] +
                                    static_cast<int64_t>(i2) * localStep[2];
                for (int rowIdx = 0; rowIdx < gShape3; rowIdx += tileValidRow) {
                    int chunkRows = (rowIdx + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowIdx);
                    for (int colIdx = 0; colIdx < gShape4; colIdx += tileValidCol) {
                        int chunkCols = (colIdx + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colIdx);
                        int64_t remoteOff = remoteBase + static_cast<int64_t>(rowIdx) * remoteStep[3] +
                                            static_cast<int64_t>(colIdx) * remoteStep[4];
                        int64_t localOff = localBase + static_cast<int64_t>(rowIdx) * localStep[3] +
                                           static_cast<int64_t>(colIdx) * localStep[4];
                        TgetPingPongProcessChunk<GlobalDstData, GlobalSrcData, TileData>(
                            dstGlobalData, srcGlobalData, pingTile, pongTile, pp, remoteOff, localOff, chunkRows,
                            chunkCols, remoteChunkStride, localChunkStride);
                    }
                }
            }
        }
    }

    TgetPingPongFlush<GlobalDstData, TileData>(dstGlobalData, pingTile, pongTile, pp, localChunkStride);
}

// ============================================================================
// TGET_IMPL (ping-pong): Remote read with double buffering
//
// Same ping-pong pipeline as TPUT_IMPL (see TPut.hpp for detailed timeline
// and synchronization analysis), but reads from remote GM to local GM
// without atomic operation support.
//
// Requirements:
//   - pingTile and pongTile must have the same type and dimensions.
//   - Uses EVENT_ID0 (pingTile) and EVENT_ID1 (pongTile) for synchronization.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                            TileData &pongTile)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGET: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGET: src/dst layout mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGET: TileData element type must match GlobalData element type");

    const int remoteDim0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int remoteDim1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int remoteDim2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int remoteDim3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int remoteDim4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRemoteRows = static_cast<int64_t>(remoteDim0) * remoteDim1 * remoteDim2 * remoteDim3;
    const int pingRows = pingTile.GetValidRow();
    const int pingCols = pingTile.GetValidCol();

    PTO_ASSERT(pingRows > 0, "TGET: tileValidRow must be greater than 0");
    PTO_ASSERT(pingCols > 0, "TGET: tileValidCol must be greater than 0");

    if (totalRemoteRows == 0 || remoteDim4 == 0) {
        return;
    }

    // Simple path: single chunk, no ping-pong benefit
    if (totalRemoteRows <= pingRows && remoteDim4 <= pingCols) {
        TgetTransferOnce<TileData, GlobalDstData, GlobalSrcData>(dstGlobalData, srcGlobalData, pingTile);
        return;
    }

    // 2D sliding chunked path with ping-pong double buffering
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(remoteDim3 % pingRows == 0,
                   "TGET chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(remoteDim4 % pingCols == 0,
                   "TGET chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TgetChunkedPingPong<GlobalDstData, GlobalSrcData, TileData>(dstGlobalData, srcGlobalData, pingTile, pongTile,
                                                                remoteDim0, remoteDim1, remoteDim2, remoteDim3,
                                                                remoteDim4, pingRows, pingCols);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_HPP
