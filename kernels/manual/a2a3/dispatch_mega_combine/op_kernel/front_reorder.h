/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_H

#include "kernel_operator.h"

#include <type_traits>

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kFrontSortBlockElems = 32;
constexpr uint32_t kFrontPackedSortBlockElems = 64;
constexpr uint32_t kFrontSortMaxElems = 8192;
constexpr float kFrontSortNegInf = -3.4028235e38F;

using FrontSortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, kFrontSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using FrontSortPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, kFrontSortMaxElems, pto::BLayout::RowMajor, -1, -1>;
using FrontPackedSortTile =
    pto::Tile<pto::TileType::Vec, float, 1, kFrontSortMaxElems * 2, pto::BLayout::RowMajor, -1, -1>;
using FrontPackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, kFrontSortMaxElems * 2, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint32_t FrontAlignSortBlock(uint32_t elemNum)
{
    return ((elemNum + kFrontSortBlockElems - 1U) / kFrontSortBlockElems) * kFrontSortBlockElems;
}

AICORE inline int32_t FrontFillTailMergeArray(int32_t *mergePlan, int32_t validCols, int32_t blockLen)
{
    int32_t planCount = 0;
    int32_t remainCols = validCols;
    for (int32_t curBlockLen = blockLen; curBlockLen >= static_cast<int32_t>(kFrontPackedSortBlockElems);
         curBlockLen /= 4) {
        int32_t count = 0;
        for (; count < remainCols / curBlockLen; ++count) {
            mergePlan[planCount++] = curBlockLen;
        }
        remainCols -= count * curBlockLen;
    }
    return planCount;
}

AICORE inline void FrontMergeTailPackedSortRecords(FrontPackedSortTile &packedSortTile,
                                                   FrontPackedSortTile &mergeTmpTile, uint32_t validCols,
                                                   uint32_t blockLen)
{
    int32_t mergePlan[15] = {0};
    const int32_t mergePlanCount =
        FrontFillTailMergeArray(mergePlan, static_cast<int32_t>(validCols), static_cast<int32_t>(blockLen));
    if (mergePlanCount <= 1) {
        return;
    }

    pto::MrgSortExecutedNumList executedNumList{};
    uint16_t mergedCols = 0;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (int32_t i = 0; i < mergePlanCount - 1; ++i) {
        mergedCols += static_cast<uint16_t>(mergePlan[i]);
        FrontPackedSortTile src0Tile(1, mergedCols);
        FrontPackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[i + 1]));
        FrontPackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        FrontPackedSortTile tmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, false>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
    }
}

AICORE inline void FrontMergePackedSortRecords(FrontPackedSortTile &packedSortTile, FrontPackedSortTile &mergeTmpTile,
                                               uint32_t validCols)
{
    uint32_t blockLen = kFrontPackedSortBlockElems; // 32个排序单元，每个单元是[key,payload] 2个元素，因此是64
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (; blockLen * 4U <= validCols; blockLen *= 4U) { /* 按照4批递归，128->512->.... */
        const uint16_t cols = static_cast<uint16_t>(validCols / (blockLen * 4U) * (blockLen * 4U));
        FrontPackedSortTile srcTile(1, cols);
        FrontPackedSortTile tmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT(tmpTile, srcTile, blockLen); // 4路归并排序
        pipe_barrier(PIPE_V);
        pto::TMOV(srcTile, tmpTile);
        pipe_barrier(PIPE_V);
    }

    if (blockLen < validCols) {
        FrontPackedSortTile tailTile(1, validCols);
        FrontPackedSortTile tailTmpTile(1, validCols);
        pto::TASSIGN(tailTile, packedAddr);
        pto::TASSIGN(tailTmpTile, tmpAddr);
        FrontMergeTailPackedSortRecords(tailTile, tailTmpTile, validCols, blockLen); // 最后不足4路的归并排序
    }
}

AICORE inline void FrontBuildAscendingSortKey(uint64_t sortKeyUb, uint64_t inputValueUb, uint32_t elemNum)
{
    PtoCastUb<float, int32_t>(sortKeyUb, inputValueUb, elemNum, pto::RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
    PtoMulScalarUb<float>(sortKeyUb, sortKeyUb, elemNum, -1.0F);
    pipe_barrier(PIPE_V);
}

AICORE inline void FrontRestoreAscendingSortValue(uint64_t sortedValueUb, uint64_t sortedValueScratchUb,
                                                  uint32_t elemNum)
{
    PtoMulScalarUb<float>(sortedValueScratchUb, sortedValueScratchUb, elemNum, -1.0F);
    pipe_barrier(PIPE_V);
    PtoCastUb<int32_t, float>(sortedValueUb, sortedValueScratchUb, elemNum, pto::RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
}

/* 把一组 int32 expertId 和对应 payload routeIdx 在 UB 里排成 packed record，并按 expert 升序排好
inputValueUb    原始 expertId 数组，int32[elemNum]
inputPayloadUb  原始 routeIdx 数组，uint32[alignedElemNum]，通常是 0,1,2... 或 loopStart...
packedSortUb    输出 packed records 的 UB 地址
mergeTmpUb      TMRGSORT 临时 UB
sortKeyUb       float sort key 临时 UB, float[alignedElemNum]
elemNum         有效元素数
alignedElemNum  按 32 对齐后的排序元素数
 */
AICORE inline void FrontSortInt32ToPackedUb(uint64_t inputValueUb, uint64_t inputPayloadUb, uint64_t packedSortUb,
                                            uint64_t mergeTmpUb, uint64_t sortKeyUb, uint32_t elemNum,
                                            uint32_t alignedElemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (alignedElemNum > kFrontSortMaxElems) {
        return;
    }
    if (alignedElemNum < FrontAlignSortBlock(elemNum)) {
        return;
    }

    // 把expertid 从int转float,再乘以-1，便于TSORT32最终按 expertId 升序排序
    FrontBuildAscendingSortKey(sortKeyUb, inputValueUb, elemNum);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (uint32_t i = elemNum; i < alignedElemNum; ++i) { // padding
        PtoSetValue<float>(sortKeyUb, i, kFrontSortNegInf);
        PtoSetValue<uint32_t>(inputPayloadUb, i, 0U);
    }
    if (alignedElemNum > elemNum) {
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }

    // 准备排序用的UB片上数据
    FrontSortKeyTile srcTile(1, alignedElemNum);
    FrontSortPayloadTile payloadTile(1, alignedElemNum);
    FrontPackedSortTile packedTile(1, alignedElemNum * 2U);
    FrontPackedSortTile mergeTmpTile(1, alignedElemNum * 2U);
    pto::TASSIGN(srcTile, sortKeyUb);
    pto::TASSIGN(payloadTile, inputPayloadUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TASSIGN(mergeTmpTile, mergeTmpUb);

    pto::TSORT32(packedTile, srcTile, payloadTile); // 完成按照32切分的排序，排序完毕后是每32个元素内部顺序排列
    pipe_barrier(PIPE_V);
    FrontMergePackedSortRecords(packedTile, mergeTmpTile, alignedElemNum * 2U); // 将所有的32元素block做归并排序
}

AICORE inline void FrontExtractPackedSortResult(uint64_t sortedValueUb, uint64_t sortedPayloadUb,
                                                uint64_t sortedValueScratchUb, uint64_t packedSortUb, uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }

    FrontPackedPayloadTile packedPayloadTile(1, elemNum * 2U);
    FrontSortPayloadTile sortedPayloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, packedSortUb);
    pto::TASSIGN(sortedPayloadTile, sortedPayloadUb);
    pto::TGATHER<FrontSortPayloadTile, FrontPackedPayloadTile, pto::MaskPattern::P1010>(sortedPayloadTile,
                                                                                        packedPayloadTile);
    pipe_barrier(PIPE_V);

    FrontSortKeyTile sortedKeyTile(1, elemNum);
    FrontPackedSortTile packedTile(1, elemNum * 2U);
    pto::TASSIGN(sortedKeyTile, sortedValueScratchUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TGATHER<FrontSortKeyTile, FrontPackedSortTile, pto::MaskPattern::P0101>(sortedKeyTile, packedTile);
    pipe_barrier(PIPE_V);

    FrontRestoreAscendingSortValue(sortedValueUb, sortedValueScratchUb, elemNum);
}

constexpr uint32_t kLargeFullRowMaxK = 8192;

template <typename T>
AICORE inline uint64_t AlignBytes(uint64_t value)
{
    return (value + 31U) / 32U * 32U;
}

constexpr uint32_t kFrontMergeOutMaxFanIn = 4U;
constexpr uint32_t kFrontMetadataChunkElems = 4096U;
constexpr uint32_t kFrontCountMarkerStrideElems = 128U;
constexpr int32_t kFrontCountMarkerBias = 0x800000;
constexpr uint32_t kFrontSrcToDstAssistElems = 256U;
constexpr uint32_t kFrontSrcToDstAssistIndexElems = 32U;
constexpr uint32_t kFrontInt32OneBlockElems = 8U;
constexpr uint32_t kFrontSortAlignElement = 32U;
constexpr uint32_t kFrontMaxExpertNum = 5120U;
constexpr uint32_t kFrontDynamicQuantColsBuffer = 21U;
constexpr uint32_t kFrontDynamicQuantScaleBytes = 64U;
constexpr uint32_t kFrontFullLoadHLimit = 7168U;

struct FrontRowSplitCoreTiling {
    uint32_t needCoreNum = 0;
    uint32_t perCoreRows = 0;
    uint32_t lastCoreRows = 0;
};

struct FrontRowSplitLoopTiling : FrontRowSplitCoreTiling {
    uint32_t perCorePerLoopRows = 0;
    uint32_t perCoreLastLoopRows = 0;
    uint32_t lastCorePerLoopRows = 0;
    uint32_t lastCoreLastLoopRows = 0;
};

using FrontSrcToDstTiling = FrontRowSplitLoopTiling;

struct FrontGatherQuantTiling : FrontRowSplitLoopTiling {
    uint32_t activateRows = 0;
    uint32_t perCoreLoops = 0;
    uint32_t lastCoreLoops = 0;
    uint32_t perLoopCols = 0;
    uint32_t lastLoopCols = 0;
    uint32_t colLoops = 0;
};

AICORE inline uint32_t FrontGetPerOrLastValue(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return value <= divisor ? value : value % divisor;
}

AICORE inline FrontRowSplitCoreTiling FrontBuildRouteCoreTiling(uint32_t routeElems, uint32_t coreNum)
{
    FrontRowSplitCoreTiling tiling;
    if (routeElems == 0U || coreNum == 0U) {
        return tiling;
    }
    tiling.perCoreRows = static_cast<uint32_t>(ceilDiv(routeElems, coreNum));
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }
    tiling.needCoreNum = static_cast<uint32_t>(ceilDiv(routeElems, tiling.perCoreRows));
    tiling.lastCoreRows = routeElems - tiling.perCoreRows * (tiling.needCoreNum - 1U);
    return tiling;
}

AICORE inline void FrontApplyPerCoreRowLoopSplit(FrontRowSplitLoopTiling &tiling, uint32_t perLoopMaxRows)
{
    if (tiling.perCoreRows == 0U || perLoopMaxRows == 0U) {
        tiling = FrontRowSplitLoopTiling{};
        return;
    }
    if (perLoopMaxRows >= tiling.perCoreRows) {
        tiling.perCorePerLoopRows = tiling.perCoreRows;
        tiling.perCoreLastLoopRows = tiling.perCoreRows;
    } else {
        tiling.perCorePerLoopRows = perLoopMaxRows;
        const uint32_t loops = static_cast<uint32_t>(ceilDiv(tiling.perCoreRows, perLoopMaxRows));
        tiling.perCoreLastLoopRows = tiling.perCoreRows - (loops - 1U) * perLoopMaxRows;
    }
    if (perLoopMaxRows >= tiling.lastCoreRows) {
        tiling.lastCorePerLoopRows = tiling.lastCoreRows;
        tiling.lastCoreLastLoopRows = tiling.lastCoreRows;
    } else {
        tiling.lastCorePerLoopRows = perLoopMaxRows;
        const uint32_t loops = static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, perLoopMaxRows));
        tiling.lastCoreLastLoopRows = tiling.lastCoreRows - (loops - 1U) * perLoopMaxRows;
    }
}

struct FrontCoreRowLoopView {
    bool active = false;
    uint32_t coreRows = 0;
    uint32_t perLoopRows = 0;
    uint32_t lastLoopRows = 0;
    uint32_t rowLoops = 0;
    uint32_t coreBase = 0;
};

struct FrontUbStack {
    uint64_t cursor = 0;

    AICORE inline uint64_t Push(uint64_t bytes)
    {
        const uint64_t base = cursor;
        cursor += bytes;
        return base;
    }

    AICORE inline uint64_t End() const
    {
        return cursor;
    }
};

AICORE inline uint32_t FrontExpertTokenOutExpertNumUbAlign(uint32_t expertNum)
{
    const uint32_t alignedExpertNum = static_cast<uint32_t>(alignUp(expertNum, kFrontInt32OneBlockElems));
    return alignedExpertNum > kFrontMaxExpertNum ? kFrontMaxExpertNum : alignedExpertNum;
}

struct FrontExpertTokenOutUbPlan {
    uint64_t countUb = 0;
    uint64_t chunkUb = 0;
    uint64_t scratchUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline explicit FrontExpertTokenOutUbPlan(uint32_t expertNum, uint32_t perLoopRows = 0U)
    {
        const uint32_t expertAlign = FrontExpertTokenOutExpertNumUbAlign(expertNum);
        const uint64_t countBytes = AlignBytes<int32_t>(static_cast<uint64_t>(expertAlign) * sizeof(int32_t));
        const uint64_t chunkBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(kFrontMetadataChunkElems) * sizeof(int32_t));
        countUb = 0U;
        chunkUb = countBytes;
        requiredBytes = chunkUb + chunkBytes;
        if (perLoopRows == 0U) {
            return;
        }
        const uint64_t inputBytes = AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t));
        const uint64_t scratchBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(expertAlign + kFrontInt32OneBlockElems) * sizeof(int32_t));
        const uint64_t inputEnd = chunkUb + inputBytes;
        scratchUb = inputEnd;
        if (inputEnd > requiredBytes) {
            requiredBytes = inputEnd;
        }
        const uint64_t scratchEnd = inputEnd + scratchBytes;
        if (scratchEnd > requiredBytes) {
            requiredBytes = scratchEnd;
        }
    }
};

struct FrontSrcToDstUbPlan {
    uint64_t inputUb = 0;
    uint64_t outputUb = 0;
    uint64_t assistUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline explicit FrontSrcToDstUbPlan(uint32_t perLoopRows)
    {
        const uint64_t inputBytes = AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t));
        const uint64_t assistGroups =
            (static_cast<uint64_t>(perLoopRows) + kFrontSrcToDstAssistElems - 1U) / kFrontSrcToDstAssistElems;
        const uint64_t outputBytes =
            AlignBytes<int32_t>(assistGroups * kFrontSrcToDstAssistElems * kFrontInt32OneBlockElems * sizeof(int32_t));
        const uint64_t assistBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(kFrontSrcToDstAssistElems) * sizeof(int32_t));
        FrontUbStack stack;
        inputUb = stack.Push(inputBytes);
        outputUb = stack.Push(outputBytes);
        assistUb = stack.Push(assistBytes);
        requiredBytes = stack.End();
    }
};

template <typename InputElement>
struct FrontGatherQuantUbPlan {
    uint64_t rawUb = 0;
    uint64_t fp32Ub = 0;
    uint64_t tmpUb = 0;
    uint64_t outUb = 0;
    uint64_t scaleUb = 0;
    uint64_t indexUb = 0;
    uint64_t requiredBytes = 0;

    AICORE inline FrontGatherQuantUbPlan(uint32_t problemK, uint32_t perLoopRows)
    {
        FrontUbStack stack;
        rawUb = stack.Push(AlignBytes<InputElement>(static_cast<uint64_t>(problemK) * sizeof(InputElement)));
        fp32Ub = stack.Push(AlignBytes<float>(static_cast<uint64_t>(problemK) * sizeof(float)));
        tmpUb = stack.Push(AlignBytes<float>(static_cast<uint64_t>(problemK) * sizeof(float)));
        outUb = stack.Push(AlignBytes<int8_t>(static_cast<uint64_t>(problemK) * sizeof(int8_t)));
        scaleUb = stack.Push(AlignBytes<float>(8U * sizeof(float)));
        indexUb = stack.Push(AlignBytes<int32_t>(static_cast<uint64_t>(perLoopRows) * sizeof(int32_t)));
        requiredBytes = stack.End();
    }
};

template <typename T>
AICORE inline uint32_t FrontPtoGetSortLen(uint32_t elemCount)
{
    static_assert(std::is_same_v<T, float>, "front sort packed records use float slots");
    return elemCount * 2U;
}

template <typename T>
AICORE inline uint32_t FrontPtoGetSortOffset(uint32_t elemOffset)
{
    static_assert(std::is_same_v<T, float>, "front sort packed records use float slots");
    return elemOffset * 2U;
}

AICORE inline uint32_t FrontPackedRowStride(uint32_t k)
{
    return k + static_cast<uint32_t>(UB_ALIGN);
}

AICORE inline uint64_t FrontPackedRowOffset(uint32_t row, uint32_t k)
{
    return static_cast<uint64_t>(row) * FrontPackedRowStride(k);
}

AICORE inline void FrontMergePackedSortRecordsChecked(uint64_t dstUb, uint64_t tmpUb, uint64_t src0Ub, uint64_t src1Ub,
                                                      uint64_t src2Ub, uint64_t src3Ub,
                                                      const uint16_t *elementCountList, uint32_t remainListNum,
                                                      uint32_t *listSortedNums)
{
    const uint32_t src0Cols = FrontPtoGetSortLen<float>(elementCountList[0]);
    const uint32_t src1Cols = (remainListNum >= 2U) ? FrontPtoGetSortLen<float>(elementCountList[1]) : 0U;
    const uint32_t src2Cols = (remainListNum >= 3U) ? FrontPtoGetSortLen<float>(elementCountList[2]) : 0U;
    const uint32_t src3Cols = (remainListNum >= 4U) ? FrontPtoGetSortLen<float>(elementCountList[3]) : 0U;
    const uint32_t dstCols = src0Cols + src1Cols + src2Cols + src3Cols;
    if (dstCols > kFrontSortMaxElems * 2U) {
        return;
    }

    FrontPackedSortTile dstTile(1, dstCols);
    FrontPackedSortTile tmpTile(1, dstCols);
    FrontPackedSortTile src0Tile(1, src0Cols);
    FrontPackedSortTile src1Tile(1, src1Cols);
    FrontPackedSortTile src2Tile(1, src2Cols);
    FrontPackedSortTile src3Tile(1, src3Cols);
    pto::TASSIGN(dstTile, dstUb);
    pto::TASSIGN(tmpTile, tmpUb);
    pto::TASSIGN(src0Tile, src0Ub);
    pto::TASSIGN(src1Tile, src1Ub);
    if (src2Cols > 0U) {
        pto::TASSIGN(src2Tile, src2Ub);
    }
    if (src3Cols > 0U) {
        pto::TASSIGN(src3Tile, src3Ub);
    }

    pto::MrgSortExecutedNumList executedNumList{};
    if (remainListNum == 2U) {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (remainListNum == 3U) {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile,
                      FrontPackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile, FrontPackedSortTile,
                      FrontPackedSortTile, FrontPackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile,
                                                                      src1Tile, src2Tile, src3Tile);
    }
    pipe_barrier(PIPE_V);

    listSortedNums[0] = executedNumList.mrgSortList0;
    listSortedNums[1] = executedNumList.mrgSortList1;
    listSortedNums[2] = executedNumList.mrgSortList2;
    listSortedNums[3] = executedNumList.mrgSortList3;
}

struct FrontReorderCommonState {
    AICORE inline void InitCommonInputs(GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                                        const __gm__ MegaMoeTilingData *tilingData)
    {
        xPtr_ = xGM;
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t *>(expertIdGM);
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        const auto &info = tilingData_->megaMoeInfo;
        problemM_ = info.M;
        problemK_ = info.K;
        topK_ = info.topK;
        expertPerRank_ = info.expertPerRank;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
    }

    AICORE inline void InitCommonCoreIdx()
    {
        coreIdx_ = get_block_idx();
        coreNum_ = get_block_num();
        if ASCEND_IS_AIV {
            coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
            coreNum_ = get_block_num() * get_subblockdim();
        }
    }

    AICORE inline void InitCommonPeerWindow()
    {
        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(remoteWindow_);
        offsetAPtr_ = reinterpret_cast<__gm__ int8_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetA);
        tokenPerExpertPtr_ =
            reinterpret_cast<__gm__ int32_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    }

    AICORE inline void InitCommonWorkspacePtrs(const __gm__ MegaMoeFrontReorderTiling &front, bool extended)
    {
        expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.expandedRowIdxOffset);
        localTokenPerExpertPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.localTokenPerExpertOffset);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.cumsumMMOffset);
        preSumBeforeRankPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.preSumBeforeRankOffset);
        if (extended) {
            frontExpandedExpertPtr_ =
                reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandedExpertOffset);
            frontExpandDstToSrcPtr_ =
                reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandDstToSrcOffset);
        }
    }

    AICORE inline void InitCommonSortTiling(const __gm__ MegaMoeFrontReorderTiling &front)
    {
        expertNum_ = front.expertNum;
        expertNumAligned_ = front.expertNumAligned;
        routeElems_ = front.routeElems;
        alignedRouteElems_ = front.alignedRouteElems;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
        sortNeedCoreNum_ = front.sortNeedCoreNum;
        sortPerCoreElems_ = front.sortPerCoreElems;
        sortLastCoreElems_ = front.sortLastCoreElems;
        sortPerCoreLoops_ = front.sortPerCoreLoops;
        sortPerCorePerLoopElems_ = front.sortPerCorePerLoopElems;
        sortPerCoreLastLoopElems_ = front.sortPerCoreLastLoopElems;
        sortLastCoreLoops_ = front.sortLastCoreLoops;
        sortLastCorePerLoopElems_ = front.sortLastCorePerLoopElems;
        sortLastCoreLastLoopElems_ = front.sortLastCoreLastLoopElems;
        sortOutLoopMaxElems_ = front.sortOutLoopMaxElems;
    }

    AICORE inline void InitMinimalSortTiling(const __gm__ MegaMoeFrontReorderTiling &front)
    {
        expertNum_ = front.expertNum;
        expertNumAligned_ = front.expertNumAligned;
        routeElems_ = front.routeElems;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
        sortLastCorePerLoopElems_ = front.sortLastCorePerLoopElems;
    }

    GM_ADDR xPtr_ = nullptr;
    __gm__ int32_t *expertIdPtr_ = nullptr;
    __gm__ int32_t *expertTokenNumsPtr_ = nullptr;
    __gm__ int32_t *expandedRowIdxPtr_ = nullptr;
    __gm__ int32_t *frontExpandedExpertPtr_ = nullptr;
    __gm__ int32_t *frontExpandDstToSrcPtr_ = nullptr;
    __gm__ int32_t *localTokenPerExpertPtr_ = nullptr;
    __gm__ int32_t *tokenPerExpertPtr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ int8_t *offsetAPtr_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t routeElems_ = 0;
    uint32_t alignedRouteElems_ = 0;
    uint32_t sortLoopMaxElement_ = 0;
    uint32_t sortNeedCoreNum_ = 0;
    uint32_t sortPerCoreElems_ = 0;
    uint32_t sortLastCoreElems_ = 0;
    uint32_t sortPerCoreLoops_ = 0;
    uint32_t sortPerCorePerLoopElems_ = 0;
    uint32_t sortPerCoreLastLoopElems_ = 0;
    uint32_t sortLastCoreLoops_ = 0;
    uint32_t sortLastCorePerLoopElems_ = 0;
    uint32_t sortLastCoreLastLoopElems_ = 0;
    uint32_t sortOutLoopMaxElems_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 0;
};

class FrontReorderPathBase {
public:
    AICORE inline explicit FrontReorderPathBase(FrontReorderCommonState &op) : op_(op)
    {}

    AICORE inline FrontReorderCommonState &common()
    {
        return op_;
    }
    AICORE inline const FrontReorderCommonState &common() const
    {
        return op_;
    }

protected:
    FrontReorderCommonState &op_;
};

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontRowSplitCoreTiling &tiling, uint32_t perLoopRows,
                                                           uint32_t lastLoopRows, uint32_t explicitRowLoops = 0U)
{
    FrontCoreRowLoopView view;
    if (tiling.needCoreNum == 0U || op.coreIdx_ >= tiling.needCoreNum || tiling.perCoreRows == 0U) {
        return view;
    }
    view.active = true;
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    view.coreRows = isLastCore ? tiling.lastCoreRows : tiling.perCoreRows;
    view.perLoopRows = perLoopRows;
    view.lastLoopRows = lastLoopRows;
    view.rowLoops = explicitRowLoops != 0U ?
                        explicitRowLoops :
                        (view.perLoopRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(view.coreRows, view.perLoopRows)));
    view.coreBase = op.coreIdx_ * tiling.perCoreRows;
    return view;
}

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontRowSplitLoopTiling &tiling)
{
    if (op.coreIdx_ >= tiling.needCoreNum) {
        return FrontCoreRowLoopView{};
    }
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    return FrontGetCoreRowLoopView(op, tiling, isLastCore ? tiling.lastCorePerLoopRows : tiling.perCorePerLoopRows,
                                   isLastCore ? tiling.lastCoreLastLoopRows : tiling.perCoreLastLoopRows, 0U);
}

AICORE inline FrontCoreRowLoopView FrontGetCoreRowLoopView(const FrontReorderCommonState &op,
                                                           const FrontGatherQuantTiling &tiling)
{
    if (op.coreIdx_ >= tiling.needCoreNum) {
        return FrontCoreRowLoopView{};
    }
    const bool isLastCore = op.coreIdx_ == tiling.needCoreNum - 1U;
    return FrontGetCoreRowLoopView(op, tiling, isLastCore ? tiling.lastCorePerLoopRows : tiling.perCorePerLoopRows,
                                   isLastCore ? tiling.lastCoreLastLoopRows : tiling.perCoreLastLoopRows,
                                   isLastCore ? tiling.lastCoreLoops : tiling.perCoreLoops);
}

AICORE inline uint32_t FrontSrcToDstPerLoopMaxRows(uint32_t coreNum)
{
    const uint64_t reservedBytes = static_cast<uint64_t>(kFrontSrcToDstAssistElems) * sizeof(float) +
                                   static_cast<uint64_t>(coreNum) * kFrontSortAlignElement;
    if (AtlasA2::UB_SIZE <= reservedBytes) {
        return 0U;
    }
    return static_cast<uint32_t>((AtlasA2::UB_SIZE - reservedBytes) / (kFrontSortAlignElement * 2U) / 2U);
}

AICORE inline FrontSrcToDstTiling FrontBuildSrcToDstTiling(uint32_t routeElems, uint32_t coreNum)
{
    FrontSrcToDstTiling tiling;
    static_cast<FrontRowSplitCoreTiling &>(tiling) = FrontBuildRouteCoreTiling(routeElems, coreNum);
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }
    const uint32_t perLoopMaxRows = FrontSrcToDstPerLoopMaxRows(coreNum);
    FrontApplyPerCoreRowLoopSplit(tiling, perLoopMaxRows);
    return tiling;
}

AICORE inline bool FrontGatherQuantPreferSingleColOneLoop(uint32_t problemK, uint64_t colSize, uint64_t scaleSize)
{
    return AtlasA2::UB_SIZE > colSize + scaleSize + 32U * 4U * 4U && problemK == kFrontFullLoadHLimit;
}

AICORE inline void FrontApplyGatherQuantOneLoop(FrontGatherQuantTiling &tiling, uint32_t perCoreRows,
                                                uint32_t onceRowSize, uint32_t problemK, bool forceSplit)
{
    const uint32_t perCoreOnceRows = forceSplit ? (onceRowSize < perCoreRows ? onceRowSize : perCoreRows) : perCoreRows;
    const uint32_t lastCoreOnceRows =
        forceSplit ? (onceRowSize < tiling.lastCoreRows ? onceRowSize : tiling.lastCoreRows) : tiling.lastCoreRows;
    tiling.perCorePerLoopRows = perCoreOnceRows;
    tiling.perCoreLastLoopRows = forceSplit ? FrontGetPerOrLastValue(perCoreRows, perCoreOnceRows) : perCoreRows;
    tiling.lastCorePerLoopRows = lastCoreOnceRows;
    tiling.lastCoreLastLoopRows =
        forceSplit ? FrontGetPerOrLastValue(tiling.lastCoreRows, lastCoreOnceRows) : tiling.lastCoreRows;
    tiling.perCoreLoops = forceSplit ? static_cast<uint32_t>(ceilDiv(perCoreRows, perCoreOnceRows)) : 1U;
    tiling.lastCoreLoops = forceSplit ? static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, lastCoreOnceRows)) : 1U;
    tiling.perLoopCols = problemK;
    tiling.lastLoopCols = problemK;
    tiling.colLoops = 1U;
}

AICORE inline void FrontApplyGatherQuantColumnSplit(FrontGatherQuantTiling &tiling, uint32_t perCoreRows,
                                                    uint32_t problemK, uint64_t rowSize, uint64_t colSize,
                                                    uint64_t scaleSize)
{
    uint32_t baseMaxCols = 6144U;
    const uint64_t totalColSize =
        AlignBytes<int8_t>(static_cast<uint64_t>(baseMaxCols) * sizeof(int8_t)) * kFrontDynamicQuantColsBuffer;
    uint32_t basePerLoopMaxRows =
        static_cast<uint32_t>(AlignBytes<int32_t>((AtlasA2::UB_SIZE - totalColSize - scaleSize) / sizeof(int32_t))) /
        4U;
    if (problemK < 6144U) {
        basePerLoopMaxRows =
            static_cast<uint32_t>(AlignBytes<int32_t>((AtlasA2::UB_SIZE - colSize - scaleSize) / sizeof(int32_t))) / 4U;
    } else if (perCoreRows < basePerLoopMaxRows) {
        baseMaxCols = static_cast<uint32_t>(AlignBytes<int32_t>(AtlasA2::UB_SIZE - rowSize - scaleSize)) /
                      kFrontDynamicQuantColsBuffer;
    }
    tiling.perLoopCols = baseMaxCols < problemK ? baseMaxCols : problemK;
    tiling.lastLoopCols = FrontGetPerOrLastValue(problemK, baseMaxCols);
    tiling.colLoops = baseMaxCols == 0U ? 0U : static_cast<uint32_t>(ceilDiv(problemK, baseMaxCols));
    tiling.perCorePerLoopRows = perCoreRows < basePerLoopMaxRows ? perCoreRows : basePerLoopMaxRows;
    tiling.perCoreLastLoopRows = FrontGetPerOrLastValue(perCoreRows, basePerLoopMaxRows);
    tiling.perCoreLoops =
        basePerLoopMaxRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(perCoreRows, basePerLoopMaxRows));
    tiling.lastCorePerLoopRows = tiling.lastCoreRows < basePerLoopMaxRows ? tiling.lastCoreRows : basePerLoopMaxRows;
    tiling.lastCoreLastLoopRows = FrontGetPerOrLastValue(tiling.lastCoreRows, basePerLoopMaxRows);
    tiling.lastCoreLoops =
        basePerLoopMaxRows == 0U ? 0U : static_cast<uint32_t>(ceilDiv(tiling.lastCoreRows, basePerLoopMaxRows));
}

AICORE inline FrontGatherQuantTiling FrontBuildGatherQuantTiling(uint32_t routeElems, uint32_t coreNum,
                                                                 uint32_t problemK)
{
    FrontGatherQuantTiling tiling;
    tiling.activateRows = routeElems;
    static_cast<FrontRowSplitCoreTiling &>(tiling) = FrontBuildRouteCoreTiling(routeElems, coreNum);
    if (tiling.perCoreRows == 0U) {
        return tiling;
    }

    const uint32_t perCoreRows = tiling.perCoreRows;
    const uint64_t rowSize = AlignBytes<int32_t>(static_cast<uint64_t>(perCoreRows) * sizeof(int32_t)) * 4U;
    const uint64_t colSize =
        AlignBytes<int8_t>(static_cast<uint64_t>(problemK) * sizeof(int8_t)) * kFrontDynamicQuantColsBuffer;
    const uint64_t scaleSize = kFrontDynamicQuantScaleBytes;
    uint32_t onceRowSize = 0U;
    if (AtlasA2::UB_SIZE > colSize + scaleSize + 32U * 4U * 3U) {
        onceRowSize =
            static_cast<uint32_t>((AtlasA2::UB_SIZE - colSize - scaleSize - 32U * 4U * 3U) / (sizeof(int32_t) * 4U));
        onceRowSize = onceRowSize / kFrontInt32OneBlockElems * kFrontInt32OneBlockElems;
    }
    const bool ifOneLoop = FrontGatherQuantPreferSingleColOneLoop(problemK, colSize, scaleSize);
    if (rowSize + colSize + scaleSize < AtlasA2::UB_SIZE || ifOneLoop) {
        FrontApplyGatherQuantOneLoop(tiling, perCoreRows, onceRowSize, problemK, ifOneLoop);
        return tiling;
    }

    FrontApplyGatherQuantColumnSplit(tiling, perCoreRows, problemK, rowSize, colSize, scaleSize);
    return tiling;
}

AICORE inline bool FrontRowLoopUbFits(const FrontCoreRowLoopView &coreLoop, uint64_t requiredUbBytes)
{
    return coreLoop.coreRows != 0U && coreLoop.perLoopRows != 0U && requiredUbBytes <= AtlasA2::UB_SIZE;
}

AICORE inline bool FrontGatherQuantCoreLoopReady(const FrontCoreRowLoopView &coreLoop, uint64_t requiredUbBytes)
{
    return FrontRowLoopUbFits(coreLoop, requiredUbBytes) && coreLoop.rowLoops != 0U;
}

AICORE inline bool FrontPostSortGatherQuantEnabled(const FrontGatherQuantTiling &tiling, uint32_t problemK)
{
    // Post-sort gather only implements single full-K quant tiles (colLoops == 1).
    return problemK <= kLargeFullRowMaxK && tiling.needCoreNum != 0U && tiling.colLoops == 1U;
}

AICORE inline bool FrontSrcToDstInputsValid(uint32_t routeElems, const FrontSrcToDstTiling &tiling)
{
    return routeElems != 0U && tiling.needCoreNum != 0U && tiling.perCoreRows != 0U;
}

AICORE inline bool FrontExpertTokenOutInputsValid(uint32_t routeElems, uint32_t expertNumUbAlign)
{
    return routeElems != 0U && expertNumUbAlign != 0U;
}

template <typename InputElement>
AICORE inline float FrontDynamicQuantRowToUb(__gm__ InputElement *xRow, uint32_t k, uint64_t rawUb, uint64_t fp32Ub,
                                             uint64_t tmpUb, uint64_t outUb, uint64_t scaleUb)
{
    PtoLoadVector<InputElement>(rawUb, xRow, k);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (std::is_same_v<InputElement, float>) {
        PtoMoveUb<float>(fp32Ub, rawUb, k);
    } else {
        CastDynamicQuantInputToFp32<InputElement>(fp32Ub, rawUb, k);
    }
    pipe_barrier(PIPE_V);

    BuildDynamicQuantAbs(tmpUb, fp32Ub, k);
    pipe_barrier(PIPE_V);
    ReduceDynamicQuantAbsMax(scaleUb, tmpUb, k);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float scaleValue = PtoGetValue<float>(scaleUb, 0U) / 127.0f;
    PtoFillUb<float>(scaleUb, scaleValue, 8U);
    PtoFillUb<float>(tmpUb, scaleValue, k);
    pipe_barrier(PIPE_V);

    DivideDynamicQuantInputByScale(tmpUb, fp32Ub, tmpUb, k);
    pipe_barrier(PIPE_V);
    PtoCastUb<half, float>(tmpUb, tmpUb, k, pto::RoundMode::CAST_TRUNC);
    pipe_barrier(PIPE_V);
    PtoCastUb<int8_t, half>(outUb, tmpUb, k, pto::RoundMode::CAST_ROUND);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    return scaleValue;
}

template <typename InputElement>
AICORE inline void FrontQuantAndScatterPackedRows(const FrontReorderCommonState &op, uint32_t routeRowStart,
                                                  uint32_t routeRowCount, uint64_t dstRowIdxUb, uint32_t maxDstRow,
                                                  bool relativeDstIdx, uint64_t rawUb, uint64_t fp32Ub, uint64_t tmpUb,
                                                  uint64_t outUb, uint64_t scaleUb)
{
    if (routeRowCount == 0U || op.topK_ == 0U) {
        return;
    }

    uint32_t curRouteRow = routeRowStart;
    const uint32_t routeRowEnd = routeRowStart + routeRowCount - 1U;
    const uint32_t startTokenRow = routeRowStart / op.topK_;
    const uint32_t endTokenRow = routeRowEnd / op.topK_;

    for (uint32_t tokenRow = startTokenRow; tokenRow <= endTokenRow && tokenRow < op.problemM_; ++tokenRow) {
        (void)FrontDynamicQuantRowToUb<InputElement>(
            reinterpret_cast<__gm__ InputElement *>(op.xPtr_) + static_cast<uint64_t>(tokenRow) * op.problemK_,
            op.problemK_, rawUb, fp32Ub, tmpUb, outUb, scaleUb);

        bool rowStored = false;
        while (curRouteRow <= routeRowEnd && curRouteRow / op.topK_ == tokenRow) {
            const uint32_t ubIdx = relativeDstIdx ? (curRouteRow - routeRowStart) : curRouteRow;
            const int32_t dstRow = PtoGetValue<int32_t>(dstRowIdxUb, ubIdx);
            ++curRouteRow;
            if (dstRow < 0 || static_cast<uint32_t>(dstRow) >= maxDstRow) {
                continue;
            }
            PtoStoreVector<int8_t>(op.offsetAPtr_ + FrontPackedRowOffset(static_cast<uint32_t>(dstRow), op.problemK_),
                                   outUb, FrontPackedRowStride(op.problemK_));
            rowStored = true;
        }
        if (rowStored) {
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
        }
    }
}

constexpr uint32_t kFrontEndMaxExpertNumAligned = 384U;

using FrontEndShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using FrontEndStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

AICORE inline uint64_t FrontEndCountRowBytes(const FrontReorderCommonState &op)
{
    return AlignBytes<int32_t>(static_cast<uint64_t>(op.expertNumAligned_) * sizeof(int32_t));
}

AICORE inline uint64_t FrontEndCountRowUb()
{
    return 0U;
}

AICORE inline uint64_t FrontEndPrefixUb(const FrontReorderCommonState &op)
{
    return FrontEndCountRowUb() + FrontEndCountRowBytes(op);
}

AICORE inline uint64_t FrontEndTputUb(const FrontReorderCommonState &op)
{
    return FrontEndPrefixUb(op) + FrontEndCountRowBytes(op);
}

AICORE inline uint64_t FrontEndCumsumUb(const FrontReorderCommonState &op)
{
    return FrontEndTputUb(op) + FrontEndCountRowBytes(op);
}

template <uint32_t ExpertPerRank>
AICORE inline uint64_t FrontEndCountExchangeRequiredUbBytes(const FrontReorderCommonState &op)
{
    return FrontEndCumsumUb(op) + static_cast<uint64_t>(op.rankSize_) * ExpertPerRank * sizeof(int32_t);
}

AICORE inline __gm__ int32_t *FrontEndTokenPerExpertRow(const FrontReorderCommonState &op, uint32_t srcRank)
{
    return op.tokenPerExpertPtr_ + tokenPerExpertOffset(static_cast<int32_t>(srcRank), 0, 0,
                                                        static_cast<int32_t>(op.expertNumAligned_),
                                                        static_cast<int32_t>(op.expertPerRank_));
}

template <uint32_t ExpertPerRank>
AICORE inline bool FrontEndPostprocessEnabled(const FrontReorderCommonState &op)
{
    return op.rankSize_ != 0U && op.expertNumAligned_ != 0U && op.expertPerRank_ != 0U &&
           op.expertPerRank_ == ExpertPerRank && op.expertNumAligned_ <= kFrontEndMaxExpertNumAligned &&
           FrontEndCountExchangeRequiredUbBytes<ExpertPerRank>(op) <= AtlasA2::UB_SIZE;
}

/* 本 rank 把自己的整行 localTokenPerExpert[expertNumAligned]
写到每个 peer 的 tokenPerExpert[rank, :]
 */
AICORE inline void FrontEndPublishCountRowsToPeers(const FrontReorderCommonState &op)
{
    for (uint32_t dstRank = op.coreIdx_; dstRank < op.rankSize_; dstRank += op.coreNum_) {
        if (dstRank == op.rank_) {
            continue;
        }
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), op.localTokenPerExpertPtr_, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        PtoAddScalarUb<int32_t>(FrontEndCountRowUb(), FrontEndCountRowUb(), op.expertNumAligned_,
                                kFrontCountMarkerBias);
        pipe_barrier(PIPE_V);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        __gm__ int32_t *remoteRow =
            op.remoteWindow_.RemotePtr(FrontEndTokenPerExpertRow(op, op.rank_), static_cast<int32_t>(dstRank));
        PtoStoreVector<int32_t>(remoteRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
}

AICORE inline void FrontEndWaitCountRowMarker(const FrontReorderCommonState &op, uint32_t srcRank)
{
    if (srcRank == op.rank_) {
        return;
    }

    using MarkerGlobal = pto::GlobalTensor<int32_t, FrontEndShapeDyn, FrontEndStrideDyn, pto::Layout::ND>;
    const uint32_t markerNum = static_cast<uint32_t>(ceilDiv(op.expertNumAligned_, kFrontCountMarkerStrideElems));
    FrontEndShapeDyn markerShape(1, 1, 1, 1, markerNum);
    FrontEndStrideDyn markerStride(kFrontCountMarkerStrideElems, kFrontCountMarkerStrideElems,
                                   kFrontCountMarkerStrideElems, kFrontCountMarkerStrideElems,
                                   kFrontCountMarkerStrideElems);
    MarkerGlobal marker(FrontEndTokenPerExpertRow(op, srcRank), markerShape, markerStride);
    pto::comm::TWAIT(marker, 0, pto::comm::WaitCmp::NE);
    V5DcciGmRange(static_cast<__gm__ void *>(FrontEndTokenPerExpertRow(op, srcRank)), FrontEndCountRowBytes(op));
}

AICORE inline void FrontEndRestoreCountRowAndBuildPreSum(const FrontReorderCommonState &op, uint32_t srcRank)
{
    __gm__ int32_t *srcRow = FrontEndTokenPerExpertRow(op, srcRank);
    FrontEndWaitCountRowMarker(op, srcRank);

    if (srcRank != op.rank_) {
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), srcRow, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        PtoAddScalarUb<int32_t>(FrontEndCountRowUb(), FrontEndCountRowUb(), op.expertNumAligned_,
                                -kFrontCountMarkerBias);
        pipe_barrier(PIPE_V);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<int32_t>(srcRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    } else {
        PtoLoadVector<int32_t>(FrontEndCountRowUb(), op.localTokenPerExpertPtr_, op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        PtoStoreVector<int32_t>(srcRow, FrontEndCountRowUb(), op.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
    PtoFillUb<int32_t>(FrontEndPrefixUb(op), 0, op.expertPerRank_);
    pipe_barrier(PIPE_V);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();

    int32_t prevSum = 0;
    uint32_t localExpert = 0U;
    const uint32_t localBegin = op.rank_ * op.expertPerRank_;
    const uint32_t prefixEnd = localBegin + op.expertPerRank_;
    for (uint32_t expert = 0U; expert < prefixEnd && expert < op.expertNumAligned_; ++expert) {
        if (expert >= localBegin && localExpert < op.expertPerRank_) {
            PtoSetValue<int32_t>(FrontEndPrefixUb(op), localExpert, prevSum);
            ++localExpert;
        }
        if (expert < op.expertNum_) {
            prevSum += PtoGetValue<int32_t>(FrontEndCountRowUb(), expert);
        }
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreVector<int32_t>(op.preSumBeforeRankPtr_ + static_cast<uint64_t>(srcRank) * op.expertPerRank_,
                            FrontEndPrefixUb(op), op.expertPerRank_);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

/* 每个 rank 把自己的全局 expert 计数行发给所有 peer；
每个 rank 收到每个 srcRank 的计数行后，
对这行从 globalExpert=0 起求前缀，
取出“本 rank 的 local expert 在 srcRank.offsetA 里的起始行”，
写成 preSumBeforeRank[srcRank, localExpert]。
 */
AICORE inline void FrontEndBuildCountExchangeAndPreSum(const FrontReorderCommonState &op)
{
    FrontEndPublishCountRowsToPeers(op);

    for (uint32_t srcRank = op.coreIdx_; srcRank < op.rankSize_; srcRank += op.coreNum_) {
        FrontEndRestoreCountRowAndBuildPreSum(op, srcRank);
    }

    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

template <uint32_t Pitch>
AICORE inline void FrontEndBuildCumsumForPitch(const FrontReorderCommonState &op)
{
    static_assert(Pitch == 8U || Pitch == 16U, "front cumsum supports expertPerRank 8 or 16");
    using CumsumTile = pto::Tile<pto::TileType::Vec, int32_t, 16, Pitch, pto::BLayout::RowMajor, -1, -1>;
    using CumsumGlobal = pto::GlobalTensor<int32_t, FrontEndShapeDyn, FrontEndStrideDyn, pto::Layout::ND>;
    const uint32_t localBegin = op.rank_ * Pitch;
    CumsumTile cumsumTile(op.rankSize_, Pitch);
    pto::TASSIGN(cumsumTile, FrontEndCumsumUb(op));

    FrontEndShapeDyn loadShape(1, 1, 1, op.rankSize_, Pitch);
    FrontEndStrideDyn loadStride(op.rankSize_ * op.expertNumAligned_, op.rankSize_ * op.expertNumAligned_,
                                 op.rankSize_ * op.expertNumAligned_, op.expertNumAligned_, 1);
    CumsumGlobal srcGlobal(op.tokenPerExpertPtr_ + localBegin, loadShape, loadStride);
    pto::TLOAD(cumsumTile, srcGlobal);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();

    for (uint32_t srcRank = 1U; srcRank < op.rankSize_; ++srcRank) {
        PtoAddUb<int32_t>(FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank) * Pitch * sizeof(int32_t),
                          FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank) * Pitch * sizeof(int32_t),
                          FrontEndCumsumUb(op) + static_cast<uint64_t>(srcRank - 1U) * Pitch * sizeof(int32_t), Pitch);
        pipe_barrier(PIPE_V);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    FrontEndShapeDyn storeShape(1, 1, 1, op.rankSize_, Pitch);
    FrontEndStrideDyn storeStride(op.rankSize_ * Pitch, op.rankSize_ * Pitch, op.rankSize_ * Pitch, Pitch, 1);
    CumsumGlobal dstGlobal(op.cumsumMMPtr_, storeShape, storeStride);
    pto::TSTORE(dstGlobal, cumsumTile);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

/* 输入：tokenPerExpert[srcRank, globalExpert]
输出：
  cumsumMM[srcRank, localExpert]
  expertTokenNums[localExpert]
   */
template <uint32_t ExpertPerRank>
AICORE inline void FrontEndBuildCumsumAndExpertTokenNums(const FrontReorderCommonState &op)
{
    static_assert(ExpertPerRank == 8U || ExpertPerRank == 16U, "front cumsum supports expertPerRank 8 or 16");
    if (op.coreIdx_ == 0U) {
        FrontEndBuildCumsumForPitch<ExpertPerRank>(op);
    }

    if (op.coreIdx_ == 0U) {
        PtoLoadVector<int32_t>(FrontEndPrefixUb(op),
                               op.cumsumMMPtr_ + static_cast<uint64_t>(op.rankSize_ - 1U) * ExpertPerRank,
                               ExpertPerRank);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_MTE3>();
        PtoStoreVector<int32_t>(op.expertTokenNumsPtr_, FrontEndPrefixUb(op), ExpertPerRank);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }
}

template <uint32_t ExpertPerRank>
AICORE inline void FrontFinalizeRankMetadata(const FrontReorderCommonState &op)
{
    if (!FrontEndPostprocessEnabled<ExpertPerRank>(op)) {
        return;
    }
    FrontEndBuildCountExchangeAndPreSum(op);

    FrontEndBuildCumsumAndExpertTokenNums<ExpertPerRank>(op);

    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

AICORE inline void FrontStoreExpertTokenOutCountSlice(const FrontReorderCommonState &op, int32_t firstExpertId,
                                                      uint32_t copyLength, uint64_t countUb, uint64_t storeScratchUb)
{
    if (copyLength == 0U) {
        return;
    }
    const uint32_t first = static_cast<uint32_t>(firstExpertId);
    const uint32_t alignedFirst = first / kFrontInt32OneBlockElems * kFrontInt32OneBlockElems;
    const uint32_t headElems = first - alignedFirst;
    const uint32_t alignedLength = static_cast<uint32_t>(alignUp(headElems + copyLength, kFrontInt32OneBlockElems));

    PtoFillUb<int32_t>(storeScratchUb, 0, alignedLength);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (uint32_t idx = 0U; idx < copyLength; ++idx) {
        PtoSetValue<int32_t>(storeScratchUb, headElems + idx, PtoGetValue<int32_t>(countUb, idx));
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreAtomicAddVector<int32_t>(op.localTokenPerExpertPtr_ + alignedFirst, storeScratchUb, alignedLength);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

AICORE inline void FrontUpdateExpertTokenOutCount(uint64_t countUb, int32_t curExpertId, int32_t &lastExpertId,
                                                  int32_t &tokenCount)
{
    tokenCount++;
    if (lastExpertId < curExpertId) {
        PtoSetValue<int32_t>(countUb, static_cast<uint32_t>(lastExpertId), tokenCount - 1);
        tokenCount = 1;
        lastExpertId = curExpertId;
    }
}

AICORE inline void FrontBuildExpertTokenOut(const FrontReorderCommonState &op)
{
    const FrontSrcToDstTiling tiling = FrontBuildSrcToDstTiling(op.routeElems_, op.coreNum_);
    const uint32_t expertNumUbAlign = FrontExpertTokenOutExpertNumUbAlign(op.expertNum_);
    const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
    if (!FrontExpertTokenOutInputsValid(op.routeElems_, expertNumUbAlign) || !coreLoop.active) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        return;
    }
    const FrontExpertTokenOutUbPlan ubPlan(op.expertNum_, coreLoop.perLoopRows);
    if (!FrontRowLoopUbFits(coreLoop, ubPlan.requiredBytes)) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        return;
    }

    PtoFillUb<int32_t>(ubPlan.countUb, 0, expertNumUbAlign);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    int32_t tokenCount = 0;
    int32_t lastExpertId = -1;
    for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
        const uint32_t currentLoopRows = loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
        PtoLoadVector<int32_t>(ubPlan.chunkUb,
                               op.frontExpandedExpertPtr_ + coreLoop.coreBase + loop * coreLoop.perLoopRows,
                               currentLoopRows);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        for (uint32_t idx = 0U; idx < currentLoopRows; ++idx) {
            const int32_t expertId = PtoGetValue<int32_t>(ubPlan.chunkUb, idx);
            if (lastExpertId == -1) {
                lastExpertId = expertId;
            }
            FrontUpdateExpertTokenOutCount(ubPlan.countUb, expertId, lastExpertId, tokenCount);
        }
        if (lastExpertId >= 0) {
            PtoSetValue<int32_t>(ubPlan.countUb, static_cast<uint32_t>(lastExpertId), tokenCount);
        }
    }
    if (lastExpertId >= 0) {
        FrontStoreExpertTokenOutCountSlice(op, 0, expertNumUbAlign, ubPlan.countUb, ubPlan.scratchUb);
    }
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

AICORE inline void FrontPrepareSrcToDstAssist(const FrontReorderCommonState &op, uint64_t assistUb,
                                              const FrontSrcToDstTiling &tiling)
{
    using AssistTile = PtoVecTile<int32_t, kFrontSrcToDstAssistElems>;
    AssistTile assistTile(1, kFrontSrcToDstAssistElems);
    pto::TASSIGN(assistTile, assistUb);
    const int32_t baseRow = static_cast<int32_t>(op.coreIdx_ * tiling.perCoreRows);
    for (uint32_t idx = 0U; idx < kFrontSrcToDstAssistElems; ++idx) {
        int32_t rowOffset = 0;
        if (idx % kFrontInt32OneBlockElems == 0U) {
            rowOffset = baseRow + static_cast<int32_t>(idx / kFrontInt32OneBlockElems);
        }
        assistTile.SetValue(idx, rowOffset);
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

AICORE inline void FrontComputeSrcToDstRows(uint32_t progress, uint32_t perLoopRows, uint32_t currentLoopRows,
                                            uint64_t outputUb, uint64_t assistUb)
{
    const uint32_t loops = (currentLoopRows + kFrontSrcToDstAssistIndexElems - 1U) / kFrontSrcToDstAssistIndexElems;
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        PtoAddScalarUb<int32_t, kFrontSrcToDstAssistElems>(
            outputUb + static_cast<uint64_t>(loop) * kFrontSrcToDstAssistElems * sizeof(int32_t), assistUb,
            kFrontSrcToDstAssistElems,
            static_cast<int32_t>(progress * perLoopRows + loop * kFrontSrcToDstAssistIndexElems));
    }
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}

AICORE inline void FrontCopyOutSrcToDstRows(const FrontReorderCommonState &op, uint32_t currentLoopRows,
                                            uint64_t inputUb, uint64_t outputUb, bool hasNextLoop)
{
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    for (uint32_t idx = 0U; idx < currentLoopRows; ++idx) {
        const uint32_t srcRoute = static_cast<uint32_t>(PtoGetValue<int32_t>(inputUb, idx));
        PtoStoreVector<int32_t>(op.expandedRowIdxPtr_ + srcRoute,
                                outputUb + static_cast<uint64_t>(idx) * kFrontInt32OneBlockElems * sizeof(int32_t), 1);
    }
    if (hasNextLoop) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
}
/*
inputUb  : 存 srcRoute 列表，大小约 perLoopRows * 4B
outputUb : 存 dstRow 列表，但每个 dstRow 占 8-int block，大小约 ceil(perLoopRows/256) * 256 * 8 *  4B
assistUb : 32 个 8-int block 模板，用来批量生成 dstRow，固定 1024B
8-int对齐是为了写GM的时候，由于是按照srcRoute乱序写，这么做能做到32B对齐
assistUb 是为了expandedRowIdx[srcRoute] = dstRow的时候， dstRow的生成32个元素依次加偏移性能差，可以用aiv批量加偏移
 */
AICORE inline void FrontBuildSrcToDst(const FrontReorderCommonState &op)
{
    const FrontSrcToDstTiling tiling = FrontBuildSrcToDstTiling(op.routeElems_, op.coreNum_);
    if (FrontSrcToDstInputsValid(op.routeElems_, tiling)) {
        const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
        if (coreLoop.active) {
            const FrontSrcToDstUbPlan ubPlan(coreLoop.perLoopRows);
            if (FrontRowLoopUbFits(coreLoop, ubPlan.requiredBytes)) {
                FrontPrepareSrcToDstAssist(op, ubPlan.assistUb, tiling);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
                    const uint32_t currentLoopRows =
                        loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
                    const bool hasNextLoop = loop + 1U < coreLoop.rowLoops;
                    PtoLoadVector<int32_t>(ubPlan.inputUb,
                                           op.frontExpandDstToSrcPtr_ + coreLoop.coreBase + loop * coreLoop.perLoopRows,
                                           currentLoopRows);
                    FrontComputeSrcToDstRows(loop, coreLoop.perLoopRows, currentLoopRows, ubPlan.outputUb,
                                             ubPlan.assistUb);
                    FrontCopyOutSrcToDstRows(op, currentLoopRows, ubPlan.inputUb, ubPlan.outputUb, hasNextLoop);
                }
            }
        }
    }
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

template <typename InputElement>
AICORE inline void FrontCopyInGatherExpandedRowIdx(const FrontReorderCommonState &op, uint32_t coreBase,
                                                   uint32_t progress, uint32_t perLoopRows, uint32_t currentLoopRows,
                                                   uint64_t indexUb)
{
    PtoLoadVector<int32_t>(indexUb, op.expandedRowIdxPtr_ + coreBase + progress * perLoopRows, currentLoopRows);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
}

template <typename InputElement>
AICORE inline void FrontRunGatherQuantLoop(const FrontReorderCommonState &op)
{
    const FrontGatherQuantTiling tiling = FrontBuildGatherQuantTiling(op.routeElems_, op.coreNum_, op.problemK_);
    const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
    if (FrontPostSortGatherQuantEnabled(tiling, op.problemK_) && coreLoop.active) {
        const FrontGatherQuantUbPlan<InputElement> ubPlan(op.problemK_, coreLoop.perLoopRows);
        if (FrontGatherQuantCoreLoopReady(coreLoop, ubPlan.requiredBytes)) {
            for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
                uint32_t currentLoopRows =
                    loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
                if (currentLoopRows == 0U && loop != coreLoop.rowLoops - 1U) {
                    currentLoopRows = coreLoop.perLoopRows;
                }
                FrontCopyInGatherExpandedRowIdx<InputElement>(op, coreLoop.coreBase, loop, coreLoop.perLoopRows,
                                                              currentLoopRows, ubPlan.indexUb);
                const uint32_t routeRowStart = coreLoop.coreBase + coreLoop.perLoopRows * loop;
                FrontQuantAndScatterPackedRows<InputElement>(op, routeRowStart, currentLoopRows, ubPlan.indexUb,
                                                             op.routeElems_, true, ubPlan.rawUb, ubPlan.fp32Ub,
                                                             ubPlan.tmpUb, ubPlan.outUb, ubPlan.scaleUb);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

template <typename InputElement, uint32_t ExpertPerRank>
AICORE inline void FrontRunPostSortPipeline(const FrontReorderCommonState &op)
{
    FrontBuildExpertTokenOut(op); // 生成 localTokenPerExpert[global_expert] = count

    FrontBuildSrcToDst(op); // 生成 expandedRowIdx[srcRoute] = dstRow

    FrontRunGatherQuantLoop<InputElement>(op);

    FrontFinalizeRankMetadata<ExpertPerRank>(op);
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
