/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TSCATTER_HPP
#define PTO_COMM_TSCATTER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TSCATTER_IMPL: Scatter operation - root distributes data to all ranks
//
// The calling NPU (root) splits local source data along DIM_3 (row dimension)
// and distributes the portions to each rank in the parallel group.
// This is the inverse of TGATHER.
//
// The source tensor has shape (D0, D1, D2, N*H, W). Each rank r receives
// rows [r*H, (r+1)*H), i.e., per-rank data of shape (D0, D1, D2, H, W).
//
// When the per-rank data exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, per-rank DIM_3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, DIM_4 must be divisible by ValidCol.
//   - All destination tensors in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename GlobalDataDst, typename GlobalDataSrc>
void Scatter(
    typename GlobalDataDst::DType* dst, typename GlobalDataSrc::DType* src, int64_t dstShape[], int64_t dstStride[],
    int64_t srcStride[], int64_t srcOffset)
{
    auto scatterInner = [&](size_t i, size_t j, size_t k) {
        for (size_t l = 0; l < dstShape[3]; l++) {
            for (size_t m = 0; m < dstShape[4]; m++) {
                int dstIndex =
                    i * dstStride[0] + j * dstStride[1] + k * dstStride[2] + l * dstStride[3] + m * dstStride[4];
                int srcIndex = i * srcStride[0] + j * srcStride[1] + k * srcStride[2] + (l + srcOffset) * srcStride[3] +
                               m * srcStride[4];
                dst[dstIndex] = src[srcIndex];
            }
        }
    };
    for (size_t i = 0; i < dstShape[0]; i++) {
        for (size_t j = 0; j < dstShape[1]; j++) {
            for (size_t k = 0; k < dstShape[2]; k++) {
                scatterInner(i, j, k);
            }
        }
    }
}

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TSCATTER_IMPL(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& stagingTileData)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TSCATTER: GlobalData type mismatch!");
    static_assert(
        std::is_same_v<T, typename TileData::DType>,
        "TSCATTER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TSCATTER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    constexpr size_t numDims = 5;

    int64_t srcStride[numDims] = {
        srcGlobalData.GetStride(0), srcGlobalData.GetStride(1), srcGlobalData.GetStride(2), srcGlobalData.GetStride(3),
        srcGlobalData.GetStride(4)};

    GlobalDstData& dstTensor = parallelGroup[0];
    int64_t dstShape[numDims] = {
        dstTensor.GetShape(0), dstTensor.GetShape(1), dstTensor.GetShape(2), dstTensor.GetShape(3),
        dstTensor.GetShape(4)};
    int64_t dstStride[numDims] = {
        dstTensor.GetStride(0), dstTensor.GetStride(1), dstTensor.GetStride(2), dstTensor.GetStride(3),
        dstTensor.GetStride(4)};
    int64_t DST_DIM_3 = dstTensor.GetShape(3);

    PTO_ASSERT(srcGlobalData.GetShape(0) == dstTensor.GetShape(0), "TSCATTER: src DIM0 must equal dst DIM0!");
    PTO_ASSERT(srcGlobalData.GetShape(1) == dstTensor.GetShape(1), "TSCATTER: src DIM1 must equal dst DIM1!");
    PTO_ASSERT(srcGlobalData.GetShape(2) == dstTensor.GetShape(2), "TSCATTER: src DIM2 must equal dst DIM2!");
    PTO_ASSERT(
        srcGlobalData.GetShape(3) == dstTensor.GetShape(3) * nranks,
        "TSCATTER: src DIM3 must equal dst DIM3 * nranks!");
    PTO_ASSERT(srcGlobalData.GetShape(4) == dstTensor.GetShape(4), "TSCATTER: src DIM4 must equal dst DIM4!");

    for (int r = 0; r < nranks; ++r) {
        GlobalDstData& dstGlobalData = parallelGroup[r];
        int64_t currentSrcOffset = r * DST_DIM_3;
        Scatter<GlobalDstData, GlobalSrcData>(
            dstGlobalData.data(), srcGlobalData.data(), dstShape, dstStride, srcStride, currentSrcOffset);
    }
}

// ============================================================================
// TSCATTER_IMPL (ping-pong): Scatter with double buffering
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
// Constraints: same as TSCATTER_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TSCATTER_IMPL(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile)
{
    pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, pingTile);
}

template <
    CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... WaitEvents>
PTO_INTERNAL void TSCATTER_CCU_IMPL(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& stagingTileData,
    const CcuTriggerContext& ctx, WaitEvents&... events)
{
    pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, stagingTileData);
}

template <
    CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... WaitEvents>
PTO_INTERNAL void TSCATTER_CCU_IMPL(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile,
    const CcuTriggerContext& ctx, WaitEvents&... events)
{
    pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, pingTile);
}

// CCU engine is only available on A5 NPU hardware.  Deferred-fail stub
// mirroring the a2a3 pattern: the template name must exist in `pto::comm`
// so that `::pto::comm::TSCATTER_CCU_IMPL<engine>(...)` in pto_comm_inst.hpp
// parses on CPU simulator builds; the static_assert depends on `engine` and
// fires only if the overload is actually instantiated.
template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TSCATTER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU,
                  "TSCATTER<CollEngine::CCU> is not supported on the CPU simulator; "
                  "CCU engine requires A5 NPU hardware.");
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TSCATTER_HPP
