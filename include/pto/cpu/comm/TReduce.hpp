/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TREDUCE_HPP
#define PTO_TREDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

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

namespace detail {

// Element-wise reduction helper
template <typename TileData>
PTO_INTERNAL void ReduceTiles(TileData &acc, TileData &recv, pto::comm::ReduceOp op)
{
    // Perform element-wise reduction based on op
    switch (op) {
        case pto::comm::ReduceOp::Sum:
            TADD(acc, acc, recv);
            break;
        case pto::comm::ReduceOp::Max:
            TMAX(acc, acc, recv);
            break;
        case pto::comm::ReduceOp::Min:
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

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &accTileData,
                               TileData &recvTileData, pto::comm::ReduceOp op)
{
    using GlobalSrcData = typename pto::comm::ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    // Type checks
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TREDUCE: TileData element type must match GlobalData element type");

    const int rootIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();

    // Check PG size
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    // Get GlobalTensor dimensions from root's source tensor
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

    // ---- Simple path: data fits in UB tile in both dimensions ----
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        // Single rank case: just copy local data to output
        if (nranks == 1) {
            TLOAD(accTileData, parallelGroup[rootIdx]);
            TSTORE(dstGlobalData, accTileData);
            return;
        }

        // Step 1: Load root data into accumulator
        TLOAD(accTileData, parallelGroup[rootIdx]);

        // Step 2: Reduce data from all other ranks
        for (int r = 0; r < nranks; ++r) {
            if (r == rootIdx) {
                continue; // Skip self, already loaded
            }

            // Load remote data into receive buffer
            TLOAD(recvTileData, parallelGroup[r]);

            // Perform reduction
            detail::ReduceTiles(accTileData, recvTileData, op);
        }
        TSTORE(dstGlobalData, accTileData);
        return;
    }

    // ---- 2D sliding chunked path ----
    //
    // Strategy (ND layout):
    //   - Iterate over outer dimensions (dim0, dim1, dim2) explicitly.
    //   - Within each (i0, i1, i2) block, slide a (tileValidRow x tileValidCol)
    //     window over the (dim3 x dim4) plane.
    //   - For each chunk, execute the full reduce pipeline:
    //     TLOAD root chunk → reduce all remote chunks → TSTORE result.

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    // Row validation: static ValidRow requires shape3 to be exactly divisible
    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TREDUCE chunked: shape3 must be divisible by tile ValidRow when ValidRow is static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    // Column validation: static ValidCol requires shape4 to be exactly divisible
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TREDUCE chunked: shape4 must be divisible by tile ValidCol when ValidCol is static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    // Source strides (from root's tensor, assumed same for all ranks)
    const int srcStride0 = refTensor.GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = refTensor.GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = refTensor.GetStride(GlobalTensorDim::DIM_2);
    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = refTensor.GetStride(GlobalTensorDim::DIM_4);

    // Destination strides
    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    // View types with fully dynamic shape/stride for chunk GlobalTensors
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    // 2D sliding: iterate outer dims, then chunk rows (dim3) and columns (dim4)
    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = static_cast<int64_t>(i0) * srcStride0 + static_cast<int64_t>(i1) * srcStride1 +
                                  static_cast<int64_t>(i2) * srcStride2;
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride0 + static_cast<int64_t>(i1) * dstStride1 +
                                  static_cast<int64_t>(i2) * dstStride2;

                for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                    int currentRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);

                    if constexpr (isDynamicRow) {
                        accTileData.RowMaskInternal = currentRows;
                        recvTileData.RowMaskInternal = currentRows;
                    }

                    for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                        int currentCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);

                        if constexpr (isDynamicCol) {
                            accTileData.ColMaskInternal = currentCols;
                            recvTileData.ColMaskInternal = currentCols;
                        }

                        // Compute element offsets for this chunk
                        int64_t srcOffset = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                            static_cast<int64_t>(colOff) * srcStride4;
                        int64_t dstOffset = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                            static_cast<int64_t>(colOff) * dstStride4;

                        DynShape chunkShape(1, 1, 1, currentRows, currentCols);

                        // Load root's chunk into accumulator
                        SrcViewT rootView(parallelGroup[rootIdx].data() + srcOffset, chunkShape, srcChunkStride);
                        TLOAD(accTileData, rootView);

                        if (nranks != 1) {
                            for (int r = 0; r < nranks; ++r) {
                                if (r == rootIdx)
                                    continue;

                                SrcViewT remoteView(parallelGroup[r].data() + srcOffset, chunkShape, srcChunkStride);
                                TLOAD(recvTileData, remoteView);

                                detail::ReduceTiles(accTileData, recvTileData, op);
                            }
                        }

                        // Store reduced chunk to destination
                        DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
                        TSTORE(dstView, accTileData);
                    }
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
                               TileData &pingTile, TileData &pongTile, pto::comm::ReduceOp op)
{
    using GlobalSrcData = typename pto::comm::ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    // Type checks
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TREDUCE: TileData element type must match GlobalData element type");

    const int rootIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();

    // Check PG size
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    // Get GlobalTensor dimensions from root's source tensor
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

    // Remote ranks are all ranks except root; map by ordinal to concrete rank index.
    const int numRemote = nranks - 1;

    // ---- Simple path: data fits in UB tile in both dimensions ----
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        // Single rank case: just copy local data to output
        if (nranks == 1) {
            TLOAD(accTileData, parallelGroup[rootIdx]);
            TSTORE(dstGlobalData, accTileData);
            return;
        }

        // Step 1: Load root data into accumulator
        TLOAD(accTileData, parallelGroup[rootIdx]);

        // Step 2: Start prefetching first remote data into pingTile
        TLOAD(pingTile, parallelGroup[detail::GetRemoteRank(rootIdx, 0)]);

        // Ping-pong processing: overlap data transfer with computation
        for (int i = 0; i < numRemote; ++i) {
            const bool hasNext = (i + 1 < numRemote);
            const bool usePing = (i % 2 == 0);

            TileData &currentTile = usePing ? pingTile : pongTile;
            TileData &nextTile = usePing ? pongTile : pingTile;

            // Start prefetch of next remote data (overlapped with current reduction)
            if (hasNext) {
                TLOAD(nextTile, parallelGroup[detail::GetRemoteRank(rootIdx, i + 1)]);
            }

            // Perform reduction with current remote data
            detail::ReduceTiles(accTileData, currentTile, op);
        }

        // Step 3: Store final result
        TSTORE(dstGlobalData, accTileData);
        return;
    }

    // ---- 2D sliding chunked path ----
    //
    // Strategy: same 2D sliding as TREDUCE_IMPL, but within each chunk the
    // remote-rank reduction uses ping-pong double buffering.

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

    // Source strides (from root's tensor, assumed same for all ranks)
    const int srcStride0 = refTensor.GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = refTensor.GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = refTensor.GetStride(GlobalTensorDim::DIM_2);
    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = refTensor.GetStride(GlobalTensorDim::DIM_4);

    // Destination strides
    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    // View types with fully dynamic shape/stride for chunk GlobalTensors
    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    // 2D sliding: iterate outer dims, then chunk rows (dim3) and columns (dim4)
    for (int i0 = 0; i0 < gShape0; ++i0) {
        for (int i1 = 0; i1 < gShape1; ++i1) {
            for (int i2 = 0; i2 < gShape2; ++i2) {
                int64_t srcBase = static_cast<int64_t>(i0) * srcStride0 + static_cast<int64_t>(i1) * srcStride1 +
                                  static_cast<int64_t>(i2) * srcStride2;
                int64_t dstBase = static_cast<int64_t>(i0) * dstStride0 + static_cast<int64_t>(i1) * dstStride1 +
                                  static_cast<int64_t>(i2) * dstStride2;

                for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                    int currentRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);

                    if constexpr (isDynamicRow) {
                        accTileData.RowMaskInternal = currentRows;
                        pingTile.RowMaskInternal = currentRows;
                        pongTile.RowMaskInternal = currentRows;
                    }

                    for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                        int currentCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);

                        if constexpr (isDynamicCol) {
                            accTileData.ColMaskInternal = currentCols;
                            pingTile.ColMaskInternal = currentCols;
                            pongTile.ColMaskInternal = currentCols;
                        }

                        // Compute element offsets for this chunk
                        int64_t srcOffset = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                            static_cast<int64_t>(colOff) * srcStride4;
                        int64_t dstOffset = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                            static_cast<int64_t>(colOff) * dstStride4;

                        DynShape chunkShape(1, 1, 1, currentRows, currentCols);

                        // Load root's chunk into accumulator
                        SrcViewT rootView(parallelGroup[rootIdx].data() + srcOffset, chunkShape, srcChunkStride);
                        TLOAD(accTileData, rootView);

                        if (nranks != 1) {
                            // Prefetch first remote chunk into pingTile
                            SrcViewT firstRemoteView(
                                parallelGroup[detail::GetRemoteRank(rootIdx, 0)].data() + srcOffset, chunkShape,
                                srcChunkStride);
                            TLOAD(pingTile, firstRemoteView);

                            // Ping-pong reduce over remote ranks
                            for (int i = 0; i < numRemote; ++i) {
                                const bool hasNext = (i + 1 < numRemote);
                                const bool usePing = (i % 2 == 0);

                                TileData &currentTile = usePing ? pingTile : pongTile;
                                TileData &nextTile = usePing ? pongTile : pingTile;

                                // Start prefetch of next remote chunk (overlapped with reduction)
                                if (hasNext) {
                                    SrcViewT nextRemoteView(
                                        parallelGroup[detail::GetRemoteRank(rootIdx, i + 1)].data() + srcOffset,
                                        chunkShape, srcChunkStride);
                                    TLOAD(nextTile, nextRemoteView);
                                }

                                detail::ReduceTiles(accTileData, currentTile, op);
                            }
                        }

                        // Store reduced chunk to destination
                        DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
                        TSTORE(dstView, accTileData);
                    }
                }
            }
        }
    }
}

} // namespace comm
} // namespace pto

#endif
