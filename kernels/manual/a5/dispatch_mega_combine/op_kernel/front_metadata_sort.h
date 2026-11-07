/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_METADATA_SORT_H
#define DISPATCH_MEGA_COMBINE_FRONT_METADATA_SORT_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/const_args.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kSortBlockElems = 32U;
constexpr uint32_t kPackedSortBlockElems = 64U;
constexpr uint32_t kSortMaxElems = 8192U;
constexpr uint32_t kMergeMaxFanIn = 4U;
constexpr float kSortNegInf = -3.4028235e38F;

using SortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, kSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using SortPayloadTile = pto::Tile<pto::TileType::Vec, uint32_t, 1, kSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using PackedSortTile = pto::Tile<pto::TileType::Vec, float, 1, kSortMaxElems * 2U, pto::BLayout::RowMajor, -1, -1>;
using PackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, kSortMaxElems * 2U, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint64_t AlignBytes(uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; }

AICORE inline uint32_t AlignSortBlock(uint32_t elemNum)
{
    return (elemNum + kSortBlockElems - 1U) / kSortBlockElems * kSortBlockElems;
}

AICORE inline uint32_t PackedLen(uint32_t elemCount) { return elemCount * 2U; }

AICORE inline uint32_t PackedOffset(uint32_t elemOffset) { return elemOffset * 2U; }

AICORE inline int32_t FillTailMergePlan(int32_t* mergePlan, int32_t validCols, int32_t blockLen)
{
    int32_t planCount = 0;
    int32_t remainCols = validCols;
    for (int32_t currentBlockLen = blockLen; currentBlockLen >= static_cast<int32_t>(kPackedSortBlockElems);
         currentBlockLen /= 4) {
        int32_t count = 0;
        for (; count < remainCols / currentBlockLen; ++count) {
            mergePlan[planCount++] = currentBlockLen;
        }
        remainCols -= count * currentBlockLen;
    }
    return planCount;
}

AICORE inline void MergeTailPackedRecords(
    PackedSortTile& packedTile, PackedSortTile& tmpTile, uint32_t validCols, uint32_t blockLen)
{
    int32_t mergePlan[15] = {0};
    const int32_t planCount =
        FillTailMergePlan(mergePlan, static_cast<int32_t>(validCols), static_cast<int32_t>(blockLen));
    if (planCount <= 1) {
        return;
    }

    pto::MrgSortExecutedNumList executedNumList{};
    uint16_t mergedCols = 0U;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(tmpTile.data());
    for (int32_t idx = 0; idx < planCount - 1; ++idx) {
        mergedCols += static_cast<uint16_t>(mergePlan[idx]);
        PackedSortTile src0Tile(1, mergedCols);
        PackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[idx + 1]));
        PackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[idx + 1]));
        PackedSortTile mergeTmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[idx + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(mergeTmpTile, tmpAddr);
        pto::TMRGSORT<PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, false>(
            dstTile, executedNumList, mergeTmpTile, src0Tile, src1Tile);
    }
}

AICORE inline void MergePackedRecords(PackedSortTile& packedTile, PackedSortTile& tmpTile, uint32_t validCols)
{
    uint32_t blockLen = kPackedSortBlockElems;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(tmpTile.data());
    for (; blockLen * 4U <= validCols; blockLen *= 4U) {
        const uint16_t cols = static_cast<uint16_t>(validCols / (blockLen * 4U) * (blockLen * 4U));
        PackedSortTile srcTile(1, cols);
        PackedSortTile mergeTmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(mergeTmpTile, tmpAddr);
        pto::TMRGSORT(mergeTmpTile, srcTile, blockLen);
        pto::TMOV(srcTile, mergeTmpTile);
    }
    if (blockLen < validCols) {
        PackedSortTile tailTile(1, validCols);
        PackedSortTile tailTmpTile(1, validCols);
        pto::TASSIGN(tailTile, packedAddr);
        pto::TASSIGN(tailTmpTile, tmpAddr);
        MergeTailPackedRecords(tailTile, tailTmpTile, validCols, blockLen);
    }
}

AICORE inline void SortInt32ToPackedUb(
    uint64_t valueUb, uint64_t payloadUb, uint64_t packedUb, uint64_t mergeTmpUb, uint64_t sortKeyUb, uint32_t elemNum,
    uint32_t alignedElemNum)
{
    PtoCastUb<float, int32_t>(sortKeyUb, valueUb, elemNum, pto::RoundMode::CAST_ROUND);
    PtoMulScalarUb<float>(sortKeyUb, sortKeyUb, elemNum, -1.0F);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (uint32_t idx = elemNum; idx < alignedElemNum; ++idx) {
        PtoSetValue<float>(sortKeyUb, idx, kSortNegInf);
        PtoSetValue<uint32_t>(payloadUb, idx, 0U);
    }
    if (alignedElemNum > elemNum) {
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }

    SortKeyTile keyTile(1, alignedElemNum);
    SortPayloadTile payloadTile(1, alignedElemNum);
    PackedSortTile packedTile(1, alignedElemNum * 2U);
    PackedSortTile tmpTile(1, alignedElemNum * 2U);
    pto::TASSIGN(keyTile, sortKeyUb);
    pto::TASSIGN(payloadTile, payloadUb);
    pto::TASSIGN(packedTile, packedUb);
    pto::TASSIGN(tmpTile, mergeTmpUb);
    pto::TSORT32(packedTile, keyTile, payloadTile);
    MergePackedRecords(packedTile, tmpTile, alignedElemNum * 2U);
}

AICORE inline void ExtractPackedResult(
    uint64_t sortedValueUb, uint64_t sortedPayloadUb, uint64_t valueScratchUb, uint64_t packedUb, uint32_t elemNum)
{
    PackedPayloadTile packedPayloadTile(1, elemNum * 2U);
    SortPayloadTile payloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, packedUb);
    pto::TASSIGN(payloadTile, sortedPayloadUb);
    pto::TGATHER<SortPayloadTile, PackedPayloadTile, pto::MaskPattern::P1010>(payloadTile, packedPayloadTile);

    SortKeyTile keyTile(1, elemNum);
    PackedSortTile packedTile(1, elemNum * 2U);
    pto::TASSIGN(keyTile, valueScratchUb);
    pto::TASSIGN(packedTile, packedUb);
    pto::TGATHER<SortKeyTile, PackedSortTile, pto::MaskPattern::P0101>(keyTile, packedTile);
    PtoMulScalarUb<float>(valueScratchUb, valueScratchUb, elemNum, -1.0F);
    PtoCastUb<int32_t, float>(sortedValueUb, valueScratchUb, elemNum, pto::RoundMode::CAST_ROUND);
}

AICORE inline void MergePackedRecordsWithCounts(
    uint64_t dstUb, uint64_t tmpUb, const uint64_t* srcUb, const uint16_t* elementCount, uint32_t listNum,
    uint32_t* sortedCount)
{
    const uint32_t src0Cols = PackedLen(elementCount[0]);
    const uint32_t src1Cols = listNum >= 2U ? PackedLen(elementCount[1]) : 0U;
    const uint32_t src2Cols = listNum >= 3U ? PackedLen(elementCount[2]) : 0U;
    const uint32_t src3Cols = listNum >= 4U ? PackedLen(elementCount[3]) : 0U;
    const uint32_t dstCols = src0Cols + src1Cols + src2Cols + src3Cols;
    PackedSortTile dstTile(1, dstCols);
    PackedSortTile tmpTile(1, dstCols);
    PackedSortTile src0Tile(1, src0Cols);
    PackedSortTile src1Tile(1, src1Cols);
    PackedSortTile src2Tile(1, src2Cols);
    PackedSortTile src3Tile(1, src3Cols);
    pto::TASSIGN(dstTile, dstUb);
    pto::TASSIGN(tmpTile, tmpUb);
    pto::TASSIGN(src0Tile, srcUb[0]);
    pto::TASSIGN(src1Tile, srcUb[1]);
    if (src2Cols > 0U) {
        pto::TASSIGN(src2Tile, srcUb[2]);
    }
    if (src3Cols > 0U) {
        pto::TASSIGN(src3Tile, srcUb[3]);
    }

    pto::MrgSortExecutedNumList executedNumList{};
    if (listNum == 2U) {
        pto::TMRGSORT<PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (listNum == 3U) {
        pto::TMRGSORT<PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<
            PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, PackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
    }
    sortedCount[0] = executedNumList.mrgSortList0;
    sortedCount[1] = executedNumList.mrgSortList1;
    sortedCount[2] = executedNumList.mrgSortList2;
    sortedCount[3] = executedNumList.mrgSortList3;
}

class Sorter {
public:
    AICORE inline void Init(
        __gm__ int32_t* expertIdPtr, GM_ADDR workspaceGM, const __gm__ MegaMoeFrontReorderTiling* tiling,
        uint32_t coreIdx, uint32_t coreNum)
    {
        expertIdPtr_ = expertIdPtr;
        sortedRouteSlotPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tiling->sortedRouteSlotOffset);
        workspace0Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tiling->sortWorkspace0Offset);
        workspace1Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tiling->sortWorkspace1Offset);
        routeElems_ = tiling->routeElems;
        runElems_ = tiling->sortRunElems;
        runCount_ = tiling->sortRunCount;
        mergeLoopElems_ = kMegaMoeFrontMetadataSortOutLoopElems;
        coreIdx_ = coreIdx;
        coreNum_ = coreNum;
    }

    template <typename GroupBarrier>
    AICORE inline void Run(GroupBarrier& barrier) const
    {
        if (runCount_ == 1U) {
            BuildSingleRunOutput();
            barrier.Sync();
            return;
        }

        BuildVbsRuns();
        barrier.Sync();
        const MergeState finalState = BuildIntermediateMerges(barrier);
        BuildFinalOutput(finalState);
        barrier.Sync();
    }

private:
    struct MergeState {
        uint32_t srcWorkspace = 0U;
        uint32_t listNum = 0U;
        uint32_t perListElems = 0U;
        uint32_t lastListElems = 0U;
    };

    AICORE inline uint64_t PackedBytes(uint32_t elemNum) const
    {
        return AlignBytes(static_cast<uint64_t>(PackedLen(elemNum)) * sizeof(float));
    }

    AICORE inline uint64_t IntBytes(uint32_t elemNum) const
    {
        return AlignBytes(static_cast<uint64_t>(elemNum) * sizeof(int32_t));
    }

    AICORE inline uint64_t MergeInputUb(uint32_t slot, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(slot) * PackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeOutputUb(uint32_t activeLists, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(activeLists) * PackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeTmpUb(uint32_t activeLists, uint32_t perListElems) const
    {
        return MergeOutputUb(activeLists, perListElems) + PackedBytes(activeLists * perListElems);
    }

    AICORE inline uint64_t FinalPayloadUb(uint32_t activeLists, uint32_t perListElems) const
    {
        return IntBytes(activeLists * perListElems);
    }

    AICORE inline uint64_t FinalScratchUb(uint32_t activeLists, uint32_t perListElems) const
    {
        return MergeTmpUb(activeLists, perListElems);
    }

    AICORE inline __gm__ float* Workspace(uint32_t index) const
    {
        return index == 0U ? workspace0Ptr_ : workspace1Ptr_;
    }

    AICORE inline void SortRunToPackedWorkspace(uint32_t runStart, uint32_t runLength) const
    {
        const uint32_t alignedElems = AlignSortBlock(runLength);
        const uint64_t intBytes = AlignBytes(static_cast<uint64_t>(alignedElems) * sizeof(int32_t));
        const uint64_t packedBytes = AlignBytes(static_cast<uint64_t>(alignedElems) * 2U * sizeof(float));
        const uint64_t expertUb = 0U;
        const uint64_t payloadUb = intBytes;
        const uint64_t keyUb = intBytes * 2U;
        const uint64_t packedUb = intBytes * 3U;
        const uint64_t tmpUb = packedUb + packedBytes;

        PtoLoadVector<int32_t>(expertUb, expertIdPtr_ + runStart, runLength);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(payloadUb, static_cast<int32_t>(runStart), 1, runLength);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        SortInt32ToPackedUb(expertUb, payloadUb, packedUb, tmpUb, keyUb, runLength, alignedElems);

        const uint32_t packedOffset = PackedOffset(runStart);
        const uint32_t packedLength = PackedLen(runLength);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<float>(workspace0Ptr_ + packedOffset, packedUb, packedLength);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    AICORE inline void BuildSingleRunOutput() const
    {
        if (coreIdx_ != 0U) {
            return;
        }
        const uint32_t alignedElems = AlignSortBlock(routeElems_);
        const uint64_t intBytes = AlignBytes(static_cast<uint64_t>(alignedElems) * sizeof(int32_t));
        const uint64_t packedBytes = AlignBytes(static_cast<uint64_t>(alignedElems) * 2U * sizeof(float));
        const uint64_t expertUb = 0U;
        const uint64_t payloadUb = intBytes;
        const uint64_t keyUb = intBytes * 2U;
        const uint64_t packedUb = intBytes * 3U;
        const uint64_t tmpUb = packedUb + packedBytes;

        PtoLoadVector<int32_t>(expertUb, expertIdPtr_, routeElems_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(payloadUb, 0, 1, routeElems_);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        SortInt32ToPackedUb(expertUb, payloadUb, packedUb, tmpUb, keyUb, routeElems_, alignedElems);
        ExtractPackedResult(expertUb, payloadUb, keyUb, packedUb, routeElems_);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<int32_t>(sortedRouteSlotPtr_, payloadUb, routeElems_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    AICORE inline void BuildVbsRuns() const
    {
        for (uint32_t run = coreIdx_; run < runCount_; run += coreNum_) {
            const uint32_t runStart = run * runElems_;
            const uint32_t runLength = routeElems_ - runStart > runElems_ ? runElems_ : routeElems_ - runStart;
            SortRunToPackedWorkspace(runStart, runLength);
        }
    }

    AICORE inline void MergeListGroup(
        __gm__ float* srcPtr, __gm__ float* dstPtr, uint32_t inputBaseElem, uint32_t outputBaseElem, uint32_t listNum,
        uint32_t perListElems, uint32_t lastListElems, bool finalOutput) const
    {
        if (listNum == 0U || listNum > kMergeMaxFanIn || perListElems == 0U || lastListElems == 0U) {
            return;
        }

        uint32_t remain[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t offsets[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t totalRemain = 0U;
        for (uint32_t list = 0U; list < listNum; ++list) {
            const uint32_t elems = list == listNum - 1U ? lastListElems : perListElems;
            const uint32_t elemBase = inputBaseElem + list * perListElems;
            remain[list] = elems;
            offsets[list] = PackedOffset(elemBase);
            totalRemain += elems;
        }

        uint32_t outputOffset = finalOutput ? outputBaseElem : PackedOffset(outputBaseElem);
        while (totalRemain > 0U) {
            uint32_t activeLists = 0U;
            for (uint32_t list = 0U; list < listNum; ++list) {
                activeLists += remain[list] != 0U ? 1U : 0U;
            }
            uint16_t elementCount[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t sortedCount[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
            uint64_t inputUb[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t activeToList[kMergeMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t loadedElems = 0U;
            uint32_t active = 0U;
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
            for (uint32_t list = 0U; list < listNum; ++list) {
                if (remain[list] == 0U) {
                    continue;
                }
                const uint32_t elems = remain[list] > mergeLoopElems_ ? mergeLoopElems_ : remain[list];
                inputUb[active] = MergeInputUb(active, mergeLoopElems_);
                PtoLoadVector<float>(inputUb[active], srcPtr + offsets[list], PackedLen(elems));
                elementCount[active] = static_cast<uint16_t>(elems);
                activeToList[active] = list;
                loadedElems += elems;
                ++active;
            }
            const uint64_t mergedUb = MergeOutputUb(activeLists, mergeLoopElems_);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
            if (activeLists == 1U) {
                PtoMoveUb<float>(mergedUb, inputUb[0], PackedLen(elementCount[0]));
                sortedCount[0] = elementCount[0];
            } else {
                MergePackedRecordsWithCounts(
                    mergedUb, MergeTmpUb(activeLists, mergeLoopElems_), inputUb, elementCount, activeLists,
                    sortedCount);
            }

            uint32_t outputElems = 0U;
            for (uint32_t idx = 0U; idx < activeLists; ++idx) {
                uint32_t consumed = sortedCount[idx];
                if (finalOutput && consumed > elementCount[idx]) {
                    consumed = elementCount[idx];
                }
                const uint32_t list = activeToList[idx];
                remain[list] -= consumed;
                offsets[list] += PackedOffset(consumed);
                outputElems += consumed;
            }
            if (outputElems == 0U || outputElems > loadedElems) {
                return;
            }
            totalRemain -= outputElems;

            if (finalOutput) {
                ExtractPackedResult(
                    0U, FinalPayloadUb(activeLists, mergeLoopElems_), FinalScratchUb(activeLists, mergeLoopElems_),
                    mergedUb, outputElems);
                pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
                PtoStoreVector<int32_t>(
                    sortedRouteSlotPtr_ + outputOffset, FinalPayloadUb(activeLists, mergeLoopElems_), outputElems);
                outputOffset += outputElems;
            } else {
                const uint32_t packedLength = PackedLen(outputElems);
                pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
                PtoStoreVector<float>(dstPtr + outputOffset, mergedUb, packedLength);
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
                outputOffset += packedLength;
            }
        }
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    template <typename GroupBarrier>
    AICORE inline MergeState BuildIntermediateMerges(GroupBarrier& barrier) const
    {
        MergeState state{0U, runCount_, runElems_, routeElems_ - (runCount_ - 1U) * runElems_};
        while (state.listNum > kMergeMaxFanIn) {
            const uint32_t groupCount = (state.listNum + kMergeMaxFanIn - 1U) / kMergeMaxFanIn;
            const uint32_t lastGroupLists = state.listNum - (groupCount - 1U) * kMergeMaxFanIn;
            for (uint32_t group = coreIdx_; group < groupCount; group += coreNum_) {
                const uint32_t listNum = group == groupCount - 1U ? lastGroupLists : kMergeMaxFanIn;
                const uint32_t lastListElems = group == groupCount - 1U ? state.lastListElems : state.perListElems;
                const uint32_t baseElem = group * kMergeMaxFanIn * state.perListElems;
                MergeListGroup(
                    Workspace(state.srcWorkspace), Workspace(1U - state.srcWorkspace), baseElem, baseElem, listNum,
                    state.perListElems, lastListElems, false);
            }
            state.lastListElems = state.perListElems * (lastGroupLists - 1U) + state.lastListElems;
            state.perListElems *= kMergeMaxFanIn;
            state.listNum = groupCount;
            state.srcWorkspace = 1U - state.srcWorkspace;
            barrier.Sync();
        }
        return state;
    }

    AICORE inline void BuildFinalOutput(const MergeState& state) const
    {
        if (coreIdx_ != 0U) {
            return;
        }
        MergeListGroup(
            Workspace(state.srcWorkspace), Workspace(state.srcWorkspace), 0U, 0U, state.listNum, state.perListElems,
            state.lastListElems, true);
    }

    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int32_t* sortedRouteSlotPtr_ = nullptr;
    __gm__ float* workspace0Ptr_ = nullptr;
    __gm__ float* workspace1Ptr_ = nullptr;
    uint32_t routeElems_ = 0U;
    uint32_t runElems_ = 0U;
    uint32_t runCount_ = 0U;
    uint32_t mergeLoopElems_ = 0U;
    uint32_t coreIdx_ = 0U;
    uint32_t coreNum_ = 1U;
};

#endif // DISPATCH_MEGA_COMBINE_FRONT_METADATA_SORT_H
