/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_HPP
#define PTO_COMM_TPUT_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// Ping-pong double-buffering state for chunked TPUT transfers
struct TputPingPongState {
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;
};

// Single synchronous transfer: TLOAD from src → sync → TSTORE to dst → sync
template <typename TileData, typename DstGT, typename SrcGT, AtomicType atomicType>
PTO_INTERNAL void TputTransferOnce(DstGT &dst, SrcGT &src, TileData &tile)
{
    TLOAD(tile, src);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_IMPL<TileData, DstGT, atomicType>(dst, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked transfer with single buffer
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TputChunkedSingle(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                    TileData &stagingTileData, int gShape0, int gShape1, int gShape2, int gShape3,
                                    int gShape4, int tileValidRow, int tileValidCol)
{
    using T = typename GlobalSrcData::RawDType;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    const int srcStride[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    const int dstStride[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStride[0], srcStride[1], srcStride[2], srcStride[3], srcStride[4]);
    DynStride dstChunkStride(dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4]);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = static_cast<int64_t>(i0) * srcStride[0] + static_cast<int64_t>(i1) * srcStride[1] +
                                  static_cast<int64_t>(i2) * srcStride[2];
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride[0] + static_cast<int64_t>(i1) * dstStride[1] +
                                  static_cast<int64_t>(i2) * dstStride[2];
                for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                    int curRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);
                    if constexpr (isDynamicRow)
                        stagingTileData.RowMaskInternal = curRows;
                    for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                        int curCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);
                        if constexpr (isDynamicCol)
                            stagingTileData.ColMaskInternal = curCols;
                        int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStride[3] +
                                         static_cast<int64_t>(colOff) * srcStride[4];
                        int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStride[3] +
                                         static_cast<int64_t>(colOff) * dstStride[4];
                        DynShape chunkShape(1, 1, 1, curRows, curCols);
                        SrcViewT srcView(srcGlobalData.data() + srcOff, chunkShape, srcChunkStride);
                        DstViewT dstView(dstGlobalData.data() + dstOff, chunkShape, dstChunkStride);
                        TputTransferOnce<TileData, DstViewT, SrcViewT, atomicType>(dstView, srcView, stagingTileData);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TPUT_IMPL: Remote write operation implementation
//
// Data flow: srcGlobalData (local GM) → stagingTileData (UB) → dstGlobalData (remote GM)
//   - atomicType: Atomic operation type (AtomicNone or AtomicAdd)
//
// When the GlobalTensor exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, shape3 must be divisible by ValidRow.
//     Use DYNAMIC ValidRow for partial row chunk support.
//   - If TileData has static ValidCol, shape4 must be divisible by ValidCol.
//     Use DYNAMIC ValidCol for partial column chunk support.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData,
          AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<typename GlobalDstData::RawDType, T>, "TPUT: src/dst element type mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TPUT: TileData element type must match GlobalData element type");
    constexpr bool sameLayout = (GlobalSrcData::layout == GlobalDstData::layout);
    static_assert(sameLayout, "TPUT: src/dst layout mismatch");

    const int logicalDims[5] = {static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_0)),
                                static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_1)),
                                static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_2)),
                                static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_3)),
                                static_cast<int>(srcGlobalData.GetShape(GlobalTensorDim::DIM_4))};

    const int64_t totalLogicalRows =
        static_cast<int64_t>(logicalDims[0]) * logicalDims[1] * logicalDims[2] * logicalDims[3];
    const int ubChunkRows = stagingTileData.GetValidRow();
    const int ubChunkCols = stagingTileData.GetValidCol();

    PTO_ASSERT(ubChunkRows > 0, "TPUT: tileValidRow must be greater than 0");
    PTO_ASSERT(ubChunkCols > 0, "TPUT: tileValidCol must be greater than 0");

    if (totalLogicalRows == 0 || logicalDims[4] == 0) {
        return;
    }

    // ---- Simple path: data fits in UB tile in both dimensions ----
    if (totalLogicalRows <= ubChunkRows && logicalDims[4] <= ubChunkCols) {
        TLOAD(stagingTileData, srcGlobalData);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE_IMPL<TileData, GlobalDstData, atomicType>(dstGlobalData, stagingTileData);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        return;
    }

    // ---- 2D sliding chunked path ----
    //
    // Strategy (ND layout):
    //   - Iterate over outer dimensions (dim0, dim1, dim2) explicitly.
    //   - Within each (i0, i1, i2) block, slide a (tileValidRow × tileValidCol)
    //     window over the (dim3 × dim4) plane.
    //   - For each chunk, create a view: shape = (1, 1, 1, curRows, curCols),
    //     preserving the original strides for correct GM addressing.
    //   - TLOAD the chunk view into UB, then TSTORE from UB to remote GM.

    PTO_ASSERT(ubChunkRows > 0, "TPUT: tile ValidRow must be greater than 0 for chunked transfer");
    PTO_ASSERT(ubChunkCols > 0, "TPUT: tile ValidCol must be greater than 0 for chunked transfer");

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(logicalDims[3] % ubChunkRows == 0,
                   "TPUT chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(logicalDims[4] % ubChunkCols == 0,
                   "TPUT chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TputChunkedSingle<GlobalDstData, GlobalSrcData, TileData, atomicType>(
        dstGlobalData, srcGlobalData, stagingTileData, logicalDims[0], logicalDims[1], logicalDims[2], logicalDims[3],
        logicalDims[4], ubChunkRows, ubChunkCols);
}

// Process one chunk in the ping-pong pipeline: overlap TSTORE of previous chunk with TLOAD of current chunk
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType, typename StrideT>
PTO_INTERNAL void TputPingPongProcessChunk(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                           TileData &pingTile, TileData &pongTile, TputPingPongState &pp,
                                           int64_t srcOff, int64_t dstOff, int curRows, int curCols,
                                           const StrideT &srcChunkStride, const StrideT &dstChunkStride)
{
    using T = typename GlobalSrcData::RawDType;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, StrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, StrideT, GlobalDstData::layout>;

    TileData &loadTile = pp.usePing ? pingTile : pongTile;
    event_t curEvent = pp.usePing ? EVENT_ID0 : EVENT_ID1;
    if constexpr (isDynamicRow)
        loadTile.RowMaskInternal = curRows;
    if constexpr (isDynamicCol)
        loadTile.ColMaskInternal = curCols;

    DynShape chunkShape(1, 1, 1, curRows, curCols);
    SrcViewT srcView(srcGlobalData.data() + srcOff, chunkShape, srcChunkStride);

    if (pp.hasPending) {
        TileData &storeTile = pp.usePing ? pongTile : pingTile;
        event_t prevEvent = pp.usePing ? EVENT_ID1 : EVENT_ID0;
        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);
        DynShape pendShape(1, 1, 1, pp.pendingRows, pp.pendingCols);
        DstViewT pendView(dstGlobalData.data() + pp.pendingDstOffset, pendShape, dstChunkStride);
        TSTORE_IMPL<TileData, DstViewT, atomicType>(pendView, storeTile);
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
    } else {
        TLOAD(loadTile, srcView);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
    }

    pp.pendingDstOffset = dstOff;
    pp.pendingRows = curRows;
    pp.pendingCols = curCols;
    pp.hasPending = true;
    pp.usePing = !pp.usePing;
}

// Flush the last pending TSTORE in the ping-pong pipeline
template <typename GlobalDstData, typename TileData, AtomicType atomicType, typename StrideT>
PTO_INTERNAL void TputPingPongFlush(GlobalDstData &dstGlobalData, TileData &pingTile, TileData &pongTile,
                                    TputPingPongState &pp, const StrideT &dstChunkStride)
{
    if (!pp.hasPending)
        return;

    using T = typename GlobalDstData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DstViewT = GlobalTensor<T, DynShape, StrideT, GlobalDstData::layout>;

    TileData &lastTile = pp.usePing ? pongTile : pingTile;
    event_t lastEvent = pp.usePing ? EVENT_ID1 : EVENT_ID0;
    wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);
    DynShape lastShape(1, 1, 1, pp.pendingRows, pp.pendingCols);
    DstViewT lastView(dstGlobalData.data() + pp.pendingDstOffset, lastShape, dstChunkStride);
    TSTORE_IMPL<TileData, DstViewT, atomicType>(lastView, lastTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
    wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
}

// 2D sliding chunked transfer with ping-pong double buffering
//
// Pipeline overlap: TSTORE and TLOAD are dispatched to separate HW engines
// (MTE3 and MTE2). Within each iteration they run concurrently. The
// wait_flag at the end ensures storeTile's UB is safe to reuse before
// the NEXT iteration's TLOAD can overwrite it.
//
// MTE2 queue: [..., TLOAD(loadTile), set_flag(curEvent), wait_flag(prevEvent), ...]
// MTE3 queue: [..., wait_flag(prevEvent), TSTORE(storeTile), set_flag(prevEvent), ...]
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TputChunkedPingPong(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                                      TileData &pongTile, int gShape0, int gShape1, int gShape2, int gShape3,
                                      int gShape4, int tileValidRow, int tileValidCol)
{
    const int srcStride[5] = {static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(srcGlobalData.GetStride(GlobalTensorDim::DIM_4))};
    const int dstStride[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStride[0], srcStride[1], srcStride[2], srcStride[3], srcStride[4]);
    DynStride dstChunkStride(dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4]);

    TputPingPongState pp;

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = static_cast<int64_t>(i0) * srcStride[0] + static_cast<int64_t>(i1) * srcStride[1] +
                                  static_cast<int64_t>(i2) * srcStride[2];
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride[0] + static_cast<int64_t>(i1) * dstStride[1] +
                                  static_cast<int64_t>(i2) * dstStride[2];
                for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                    int curRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);
                    for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                        int curCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);
                        int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStride[3] +
                                         static_cast<int64_t>(colOff) * srcStride[4];
                        int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStride[3] +
                                         static_cast<int64_t>(colOff) * dstStride[4];
                        TputPingPongProcessChunk<GlobalDstData, GlobalSrcData, TileData, atomicType>(
                            dstGlobalData, srcGlobalData, pingTile, pongTile, pp, srcOff, dstOff, curRows, curCols,
                            srcChunkStride, dstChunkStride);
                    }
                }
            }
        }
    }

    TputPingPongFlush<GlobalDstData, TileData, atomicType>(dstGlobalData, pingTile, pongTile, pp, dstChunkStride);
}

// ============================================================================
// TPUT_IMPL (ping-pong): Remote write with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD (MTE2) and
// TSTORE (MTE3) for adjacent chunks, effectively hiding one DMA transfer
// behind the other.
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong (overlap TSTORE[i] with TLOAD[i+1]):
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Requirements:
//   - pingTile and pongTile must have the same type and dimensions.
//   - Uses EVENT_ID0 (pingTile) and EVENT_ID1 (pongTile) for synchronization.
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData,
          AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                            TileData &pongTile)
{
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TPUT: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TPUT: src/dst layout mismatch");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TPUT: TileData element type must match GlobalData element type");

    const int gShape0 = srcGlobalData.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = srcGlobalData.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = srcGlobalData.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = srcGlobalData.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcGlobalData.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int tileValidRow = pingTile.GetValidRow();
    const int tileValidCol = pingTile.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TPUT: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TPUT: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    // Simple path: single chunk, no ping-pong benefit
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        TputTransferOnce<TileData, GlobalDstData, GlobalSrcData, atomicType>(dstGlobalData, srcGlobalData, pingTile);
        return;
    }

    // 2D sliding chunked path with ping-pong double buffering
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TPUT chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TPUT chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TputChunkedPingPong<GlobalDstData, GlobalSrcData, TileData, atomicType>(
        dstGlobalData, srcGlobalData, pingTile, pongTile, gShape0, gShape1, gShape2, gShape3, gShape4, tileValidRow,
        tileValidCol);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_HPP
