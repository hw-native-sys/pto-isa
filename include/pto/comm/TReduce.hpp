/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TREDUCE_HPP
#define PTO_COMM_TREDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

namespace detail {

template <typename TileData>
PTO_INTERNAL void ReduceTiles(TileData &acc, TileData &recv, ReduceOp op)
{
    switch (op) {
        case ReduceOp::Sum:
            TADD(acc, acc, recv);
            break;
        case ReduceOp::Max:
            TMAX(acc, acc, recv);
            break;
        case ReduceOp::Min:
            TMIN(acc, acc, recv);
            break;
        default:
            PTO_ASSERT(false, "TREDUCE: unknown ReduceOp");
            break;
    }
}

PTO_INTERNAL int GetRemoteRank(int rootIdx, int remoteOrdinal)
{
    return (remoteOrdinal < rootIdx) ? remoteOrdinal : (remoteOrdinal + 1);
}

} // namespace detail

// Simple reduce path: entire data fits in one tile
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceSimple(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &accTileData,
                                TileData &recvTileData, ReduceOp op, int rootIdx, int nranks)
{
    if (nranks == 1) {
        TLOAD(accTileData, parallelGroup[rootIdx]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobalData, accTileData);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        return;
    }

    TLOAD(accTileData, parallelGroup[rootIdx]);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int r = 0; r < nranks; ++r) {
        if (r == rootIdx)
            continue;
        TLOAD(recvTileData, parallelGroup[r]);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        detail::ReduceTiles(accTileData, recvTileData, op);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobalData, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Process one chunk's full reduce pipeline (single buffer):
// TLOAD root → reduce all remote ranks → TSTORE result
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreduceProcessChunkSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                            TileData &accTileData, TileData &recvTileData, ReduceOp op,
                                            int64_t srcOffset, int64_t dstOffset, int currentRows, int currentCols,
                                            const DynStrideT &srcChunkStride, const DynStrideT &dstChunkStride,
                                            int rootIdx, int nranks)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalDstData::layout>;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (isDynamicRow) {
        accTileData.RowMaskInternal = currentRows;
        recvTileData.RowMaskInternal = currentRows;
    }
    if constexpr (isDynamicCol) {
        accTileData.ColMaskInternal = currentCols;
        recvTileData.ColMaskInternal = currentCols;
    }

    DynShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcViewT rootView(parallelGroup[rootIdx].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(accTileData, rootView);

    if (nranks == 1) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (int r = 0; r < nranks; ++r) {
            if (r == rootIdx)
                continue;
            SrcViewT remoteView(parallelGroup[r].data() + srcOffset, chunkShape, srcChunkStride);
            TLOAD(recvTileData, remoteView);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            detail::ReduceTiles(accTileData, recvTileData, op);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
    TSTORE(dstView, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked reduce with single buffer
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceChunkedSingle(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &recvTileData, ReduceOp op, int gShape0,
                                       int gShape1, int gShape2, int gShape3, int gShape4, int tileValidRow,
                                       int tileValidCol, int rootIdx, int nranks)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    GlobalSrcData &refTensor = parallelGroup[rootIdx];
    const int srcStride0 = refTensor.GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = refTensor.GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = refTensor.GetStride(GlobalTensorDim::DIM_2);
    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = refTensor.GetStride(GlobalTensorDim::DIM_4);
    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = static_cast<int64_t>(i0) * srcStride0 + static_cast<int64_t>(i1) * srcStride1 +
                                  static_cast<int64_t>(i2) * srcStride2;
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride0 + static_cast<int64_t>(i1) * dstStride1 +
                                  static_cast<int64_t>(i2) * dstStride2;
                for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                    int curRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);
                    for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                        int curCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);
                        int64_t srcOff = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                         static_cast<int64_t>(colOff) * srcStride4;
                        int64_t dstOff = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                         static_cast<int64_t>(colOff) * dstStride4;
                        TreduceProcessChunkSingle<ParallelGroupType, GlobalDstData, TileData>(
                            parallelGroup, dstGlobalData, accTileData, recvTileData, op, srcOff, dstOff, curRows,
                            curCols, srcChunkStride, dstChunkStride, rootIdx, nranks);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TREDUCE_IMPL: Reduce operation - root gathers and reduces data from all ranks
//
// The calling NPU is the root and gathers data from all ranks, performing
// element-wise reduction locally.
//
// When the GlobalTensor exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// For each chunk, the full reduce pipeline is executed:
//   1. Load root's chunk into accTileData
//   2. For each remote rank: TLOAD chunk into recvTileData, reduce into acc
//   3. Store reduced chunk to dstGlobalData
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, shape3 must be divisible by ValidRow.
//     Use DYNAMIC ValidRow for partial row chunk support.
//   - If TileData has static ValidCol, shape4 must be divisible by ValidCol.
//     Use DYNAMIC ValidCol for partial column chunk support.
//   - All ranks in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &accTileData,
                               TileData &recvTileData, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TREDUCE: TileData element type must match GlobalData element type");

    const int rootRank = parallelGroup.GetRootIdx();
    const int groupSize = parallelGroup.GetSize();

    PTO_ASSERT(groupSize > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootRank >= 0 && rootRank < groupSize, "rootIdx must be in range [0, nranks)!");

    GlobalSrcData &refTensor = parallelGroup[rootRank];
    const int gShape0 = refTensor.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = refTensor.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = refTensor.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = refTensor.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = refTensor.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int chunkRows = accTileData.GetValidRow();
    const int chunkCols = accTileData.GetValidCol();

    PTO_ASSERT(chunkRows > 0, "TREDUCE: tileValidRow must be greater than 0");
    PTO_ASSERT(chunkCols > 0, "TREDUCE: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    if (totalRows <= chunkRows && gShape4 <= chunkCols) {
        TreduceSimple<ParallelGroupType, GlobalDstData, TileData>(parallelGroup, dstGlobalData, accTileData,
                                                                  recvTileData, op, rootRank, groupSize);
        return;
    }

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % chunkRows == 0,
                   "TREDUCE chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % chunkCols == 0,
                   "TREDUCE chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TreduceChunkedSingle<ParallelGroupType, GlobalDstData, TileData>(
        parallelGroup, dstGlobalData, accTileData, recvTileData, op, gShape0, gShape1, gShape2, gShape3, gShape4,
        chunkRows, chunkCols, rootRank, groupSize);
}

// ============================================================================
// Ping-pong helpers for TREDUCE
// ============================================================================

// Simple reduce path with ping-pong: entire data fits in one tile
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceSimplePingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                        TileData &accTileData, TileData &pingTile, TileData &pongTile, ReduceOp op,
                                        int rootIdx, int nranks, int numRemote)
{
    auto &rootTensor = parallelGroup[rootIdx];
    if (nranks == 1) {
        TLOAD(accTileData, rootTensor);
        PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        TSTORE(dstGlobalData, accTileData);
        PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        return;
    }

    TLOAD(accTileData, rootTensor);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(pingTile, parallelGroup[detail::GetRemoteRank(rootIdx, 0)]);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int i = 0; i < numRemote; ++i) {
        const bool hasNext = (i + 1 < numRemote);
        const bool usePing = ((i & 1) == 0);
        TileData &currentTile = usePing ? pingTile : pongTile;
        TileData &nextTile = usePing ? pongTile : pingTile;
        const auto currentEvent = usePing ? EVENT_ID1 : EVENT_ID2;
        const auto nextEvent = usePing ? EVENT_ID2 : EVENT_ID1;
        if (hasNext) {
            TLOAD(nextTile, parallelGroup[detail::GetRemoteRank(rootIdx, i + 1)]);
            set_flag(PIPE_MTE2, PIPE_V, nextEvent);
        }
        wait_flag(PIPE_MTE2, PIPE_V, currentEvent);
        detail::ReduceTiles(accTileData, currentTile, op);
        if (hasNext) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        } else {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
    }

    TSTORE(dstGlobalData, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Ping-pong reduce loop over remote ranks for one chunk (used in chunked path)
template <typename ParallelGroupType, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreducePingPongLoop(ParallelGroupType &parallelGroup, TileData &accTileData, TileData &pingTile,
                                      TileData &pongTile, ReduceOp op, int64_t srcOffset, int currentRows,
                                      int currentCols, const DynStrideT &srcChunkStride, int rootIdx, int numRemote)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStrideT, GlobalSrcData::layout>;
    DynShape chunkShape(1, 1, 1, currentRows, currentCols);

    const int firstRemoteRank = detail::GetRemoteRank(rootIdx, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    SrcViewT firstView(parallelGroup[firstRemoteRank].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(pingTile, firstView);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int i = 0; i < numRemote; ++i) {
        const bool scheduleNext = (i + 1 < numRemote);
        const bool currentIsPing = ((i & 1) == 0);
        TileData &currentTile = currentIsPing ? pingTile : pongTile;
        TileData &nextTile = currentIsPing ? pongTile : pingTile;
        const event_t currentReady = currentIsPing ? EVENT_ID1 : EVENT_ID2;
        const event_t nextReady = currentIsPing ? EVENT_ID2 : EVENT_ID1;
        if (scheduleNext) {
            const int nextRemoteRank = detail::GetRemoteRank(rootIdx, i + 1);
            SrcViewT nextView(parallelGroup[nextRemoteRank].data() + srcOffset, chunkShape, srcChunkStride);
            TLOAD(nextTile, nextView);
            set_flag(PIPE_MTE2, PIPE_V, nextReady);
        }
        wait_flag(PIPE_MTE2, PIPE_V, currentReady);
        detail::ReduceTiles(accTileData, currentTile, op);
        if (!scheduleNext) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            continue;
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
}

// Process one chunk's full reduce pipeline with ping-pong double buffering
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename DynStrideT>
PTO_INTERNAL void TreduceProcessChunkPingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                              TileData &accTileData, TileData &pingTile, TileData &pongTile,
                                              ReduceOp op, int64_t srcOffset, int64_t dstOffset, int currentRows,
                                              int currentCols, const DynStrideT &srcChunkStride,
                                              const DynStrideT &dstChunkStride, int rootIdx, int nranks, int numRemote)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using ElemT = typename GlobalSrcData::RawDType;
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<ElemT, DynShape, DynStrideT, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<ElemT, DynShape, DynStrideT, GlobalDstData::layout>;
    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (isDynamicRow) {
        accTileData.RowMaskInternal = currentRows;
        pingTile.RowMaskInternal = currentRows;
        pongTile.RowMaskInternal = currentRows;
    }
    if constexpr (isDynamicCol) {
        accTileData.ColMaskInternal = currentCols;
        pingTile.ColMaskInternal = currentCols;
        pongTile.ColMaskInternal = currentCols;
    }

    DynShape chunkShape(1, 1, 1, currentRows, currentCols);
    SrcViewT rootView(parallelGroup[rootIdx].data() + srcOffset, chunkShape, srcChunkStride);
    TLOAD(accTileData, rootView);

    if (nranks == 1) {
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
        TreducePingPongLoop<ParallelGroupType, TileData>(parallelGroup, accTileData, pingTile, pongTile, op, srcOffset,
                                                         currentRows, currentCols, srcChunkStride, rootIdx, numRemote);
    }

    DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
    TSTORE(dstView, accTileData);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// 2D sliding chunked reduce with ping-pong double buffering
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TreduceChunkedPingPong(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                         TileData &accTileData, TileData &pingTile, TileData &pongTile, ReduceOp op,
                                         int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
                                         int tileValidRow, int tileValidCol, int rootIdx, int nranks, int numRemote)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    GlobalSrcData &refTensor = parallelGroup[rootIdx];
    const int srcStride[5] = {static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(refTensor.GetStride(GlobalTensorDim::DIM_4))};
    const int dstStride[5] = {static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_0)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_1)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_2)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3)),
                              static_cast<int>(dstGlobalData.GetStride(GlobalTensorDim::DIM_4))};

    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    DynStride srcChunkStride(srcStride[0], srcStride[1], srcStride[2], srcStride[3], srcStride[4]);
    DynStride dstChunkStride(dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4]);

    for (int dim0 = 0; dim0 < gShape0; ++dim0) {
        for (int dim1 = 0; dim1 < gShape1; ++dim1) {
            for (int dim2 = 0; dim2 < gShape2; ++dim2) {
                const int64_t srcBase = static_cast<int64_t>(dim0) * srcStride[0] +
                                        static_cast<int64_t>(dim1) * srcStride[1] +
                                        static_cast<int64_t>(dim2) * srcStride[2];
                const int64_t dstBase = static_cast<int64_t>(dim0) * dstStride[0] +
                                        static_cast<int64_t>(dim1) * dstStride[1] +
                                        static_cast<int64_t>(dim2) * dstStride[2];
                int rowCursor = 0;
                while (rowCursor < gShape3) {
                    const int rowRemain = gShape3 - rowCursor;
                    const int curRows = (rowRemain < tileValidRow) ? rowRemain : tileValidRow;
                    int colCursor = 0;
                    while (colCursor < gShape4) {
                        const int colRemain = gShape4 - colCursor;
                        const int curCols = (colRemain < tileValidCol) ? colRemain : tileValidCol;
                        const int64_t srcOff = srcBase + static_cast<int64_t>(rowCursor) * srcStride[3] +
                                               static_cast<int64_t>(colCursor) * srcStride[4];
                        const int64_t dstOff = dstBase + static_cast<int64_t>(rowCursor) * dstStride[3] +
                                               static_cast<int64_t>(colCursor) * dstStride[4];
                        TreduceProcessChunkPingPong<ParallelGroupType, GlobalDstData, TileData>(
                            parallelGroup, dstGlobalData, accTileData, pingTile, pongTile, op, srcOff, dstOff, curRows,
                            curCols, srcChunkStride, dstChunkStride, rootIdx, nranks, numRemote);
                        colCursor += tileValidCol;
                    }
                    rowCursor += tileValidRow;
                }
            }
        }
    }
}

// ============================================================================
// TREDUCE_IMPL (ping-pong): Reduce operation with double buffering
//
// The calling NPU is the root and gathers data from all ranks, performing
// element-wise reduction locally. Uses ping-pong double buffering to overlap
// remote data transfer (TLOAD) with reduction computation.
//
// When the GlobalTensor exceeds the UB tile capacity, the operation is
// automatically chunked via 2D sliding (same as TREDUCE_IMPL). Within each
// chunk, ping-pong double buffering is applied to the remote rank reduction:
//
// Timeline without ping-pong:
//   [TLOAD remote0] -> [Reduce] -> [TLOAD remote1] -> [Reduce] -> ...
//
// Timeline with ping-pong (overlap TLOAD[i+1] with Reduce[i]):
//   [TLOAD remote0] -> [Reduce remote0 | TLOAD remote1] -> [Reduce remote1 | TLOAD remote2] -> ...
//
// Constraints: same as TREDUCE_IMPL for chunked mode.
// ============================================================================
template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &accTileData,
                               TileData &pingTile, TileData &pongTile, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TREDUCE: TileData element type must match GlobalData element type");

    const int rootIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    GlobalSrcData &refTensor = parallelGroup[rootIdx];
    const int gShape0 = refTensor.GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = refTensor.GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = refTensor.GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = refTensor.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = refTensor.GetShape(GlobalTensorDim::DIM_4);

    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int tileValidRow = accTileData.GetValidRow();
    const int tileValidCol = accTileData.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TREDUCE: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TREDUCE: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    const int remoteCount = nranks - 1;

    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        TreduceSimplePingPong<ParallelGroupType, GlobalDstData, TileData>(
            parallelGroup, dstGlobalData, accTileData, pingTile, pongTile, op, rootIdx, nranks, remoteCount);
        return;
    }

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TREDUCE chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TREDUCE chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    TreduceChunkedPingPong<ParallelGroupType, GlobalDstData, TileData>(
        parallelGroup, dstGlobalData, accTileData, pingTile, pongTile, op, gShape0, gShape1, gShape2, gShape3, gShape4,
        tileValidRow, tileValidCol, rootIdx, nranks, remoteCount);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TREDUCE_HPP
