/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_VMS_SORT_H
#define DISPATCH_MEGA_COMBINE_FRONT_VMS_SORT_H

constexpr uint32_t kFrontSingleCoreSortOwnerCore = 0U;
constexpr uint32_t kFrontSingleCoreSortAlignElems = 32U;

template <typename InputElement>
class FrontReorderVmsSort : public FrontReorderPathBase {
public:
    AICORE inline explicit FrontReorderVmsSort(FrontReorderCommonState& op) : FrontReorderPathBase(op) {}

    AICORE inline void Init(
        GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData)
    {
        op_.InitCommonInputs(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        op_.InitCommonSortTiling(op_.tilingData_->frontReorderTiling);
        op_.InitCommonCoreIdx();
        op_.InitCommonWorkspacePtrs(op_.tilingData_->frontReorderTiling, true);
        op_.InitCommonPeerWindow();
    }

    AICORE inline void RunSort() const
    {
        if (IsSingleCorePath()) {
            BuildSingleCoreSort();
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
            return;
        }
        BuildVbsSortRuns();
        BuildOneCoreVmsProcess();
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        BuildVmsProcess();
        BuildSortOutProcess();
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    }

    struct SortMergeState {
        uint32_t srcWsIndex = 0;
        uint32_t listNum = 0;
        uint32_t perListElements = 0;
        uint32_t lastListElements = 0;
    };

private:
    struct MergeListState {
        uint32_t remain[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t offsets[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t allRemain = 0U;
        uint32_t outOffset = 0U;
    };

    struct MergeLoopState {
        uint16_t elementCountList[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t listSortedNums[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint64_t tmpUbInputs[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t activeToList[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t loadedElems = 0U;
        uint32_t activeListNum = 0U;
    };

    AICORE inline bool IsSingleCorePath() const
    {
        return op_.tilingData_->frontReorderTiling.frontCase == kFrontCaseOneCoreDynamic;
    }

    AICORE inline uint32_t SingleSortTileLength() const
    {
        const uint32_t vbsLastCoreElems =
            op_.sortLastCorePerLoopElems_ == 0U ? op_.routeElems_ : op_.sortLastCorePerLoopElems_;
        return static_cast<uint32_t>(alignUp(vbsLastCoreElems, static_cast<uint32_t>(sizeof(int32_t))));
    }

    AICORE inline uint32_t SingleSortNum() const
    {
        return static_cast<uint32_t>(alignUp(SingleSortTileLength(), kFrontSingleCoreSortAlignElems));
    }

    AICORE inline bool SingleCoreSortEnabled(uint32_t sortNum, uint64_t actualUbBytes, uint64_t requiredUbBytes) const
    {
        return op_.coreIdx_ == kFrontSingleCoreSortOwnerCore && op_.routeElems_ != 0U &&
               sortNum <= kFrontSortMaxElems && actualUbBytes <= AtlasA2::UB_SIZE &&
               requiredUbBytes <= AtlasA2::UB_SIZE;
    }

    AICORE inline void BuildSingleCoreSort() const
    {
        const uint32_t sortNum = SingleSortNum();
        const uint64_t sortIntBytes = AlignBytes<int32_t>(static_cast<uint64_t>(sortNum) * sizeof(int32_t));
        const uint64_t sortPackedBytes = AlignBytes<float>(static_cast<uint64_t>(sortNum) * 2U * sizeof(float));
        const uint64_t sortKeyBytes = AlignBytes<float>(static_cast<uint64_t>(sortNum) * sizeof(float));
        const uint64_t expertUb = 0U;
        const uint64_t payloadUb = sortIntBytes;
        const uint64_t packedUb = payloadUb + sortIntBytes;
        const uint64_t mergeTmpUb = packedUb + sortPackedBytes;
        const uint64_t sortKeyUb = mergeTmpUb + sortPackedBytes;
        const uint64_t actualUbBytes = sortKeyUb + sortKeyBytes;
        constexpr uint32_t kFrontSortBufferCount = 4U;
        constexpr uint32_t kFrontSortPayloadAndKey = 2U;
        const uint64_t requiredUbBytes =
            static_cast<uint64_t>(sortNum) * sizeof(int32_t) * kFrontSortPayloadAndKey * kFrontSortBufferCount;
        if (!SingleCoreSortEnabled(sortNum, actualUbBytes, requiredUbBytes)) {
            return;
        }

        const uint32_t totalLength = op_.routeElems_;
        PtoLoadVector<int32_t>(expertUb, op_.expertIdPtr_, totalLength);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(payloadUb, 0, 1, sortNum);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        FrontSortInt32ToPackedUb(expertUb, payloadUb, packedUb, mergeTmpUb, sortKeyUb, totalLength, sortNum);
        pipe_barrier(PIPE_V);
        FrontExtractPackedSortResult(expertUb, payloadUb, sortKeyUb, packedUb, totalLength);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<int32_t>(op_.frontExpandedExpertPtr_, expertUb, totalLength);
        PtoStoreVector<int32_t>(op_.frontExpandDstToSrcPtr_, payloadUb, totalLength);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    AICORE inline uint64_t MergeOutPackedBytes(uint32_t elemNum) const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FrontPtoGetSortLen<float>(elemNum)) * sizeof(float));
    }

    AICORE inline uint64_t MergeOutIntBytes(uint32_t elemNum) const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(elemNum) * sizeof(int32_t));
    }

    AICORE inline uint64_t MergeOutInputUb(uint32_t slot, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(slot) * MergeOutPackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeOutMergedUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(activeListNum) * MergeOutPackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeOutTmpUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutMergedUb(activeListNum, perListElems) + MergeOutPackedBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t MergeOnlyRequiredUbBytes(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutTmpUb(activeListNum, perListElems) + MergeOutPackedBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t SortOutPayloadUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutIntBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t SortOutScratchUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutTmpUb(activeListNum, perListElems);
    }

    AICORE inline uint64_t SortOutRequiredUbBytes(uint32_t activeListNum, uint32_t perListElems) const
    {
        const uint64_t totalElems = static_cast<uint64_t>(activeListNum) * perListElems;
        const uint64_t mergeBytes = MergeOnlyRequiredUbBytes(activeListNum, perListElems);
        const uint64_t payloadBytes = SortOutPayloadUb(activeListNum, perListElems) + MergeOutIntBytes(totalElems);
        const uint64_t scratchBytes = SortOutScratchUb(activeListNum, perListElems) + MergeOutIntBytes(totalElems);
        uint64_t requiredBytes = mergeBytes > payloadBytes ? mergeBytes : payloadBytes;
        requiredBytes = requiredBytes > scratchBytes ? requiredBytes : scratchBytes;
        return requiredBytes;
    }

    AICORE inline uint64_t FrontSortWorkspaceBytes() const
    {
        const uint64_t sortedIntBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(op_.alignedRouteElems_) * 2U * sizeof(int32_t));
        const uint64_t packedRunBytes =
            AlignBytes<float>(static_cast<uint64_t>(op_.alignedRouteElems_) * 2U * sizeof(float));
        return sortedIntBytes > packedRunBytes ? sortedIntBytes : packedRunBytes;
    }

    AICORE inline __gm__ float* FrontSortWsPtr(uint32_t srcWsIndex) const
    {
        const auto& front = op_.tilingData_->frontReorderTiling;
        const uint64_t offset = srcWsIndex == 0U ? front.frontSortWs0Offset : front.frontSortWs1Offset;
        return reinterpret_cast<__gm__ float*>(op_.workspaceGM_ + offset);
    }

    AICORE inline uint32_t BuildOneCoreVmsSrcWsIndex(uint32_t listNum) const
    {
        uint32_t srcWsIndex = 0U;
        while (listNum > 1U) {
            listNum = (listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            srcWsIndex = 1U - srcWsIndex;
        }
        return srcWsIndex;
    }

    AICORE inline SortMergeState BuildLocalSortState() const
    {
        SortMergeState state;
        state.srcWsIndex = op_.sortPerCoreLoops_ > 1U ? BuildOneCoreVmsSrcWsIndex(op_.sortPerCoreLoops_) : 0U;
        state.listNum = op_.sortNeedCoreNum_;
        state.perListElements = op_.sortPerCoreElems_;
        state.lastListElements = op_.sortLastCoreElems_;
        return state;
    }

    AICORE inline SortMergeState BuildFinalSortState() const
    {
        SortMergeState state = BuildLocalSortState();
        while (state.listNum > kFrontMergeOutMaxFanIn) {
            const uint32_t currentStageNeedCoreNum =
                (state.listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            const uint32_t remainListNum = state.listNum - (currentStageNeedCoreNum - 1U) * kFrontMergeOutMaxFanIn;
            state.lastListElements = state.perListElements * (remainListNum - 1U) + state.lastListElements;
            state.perListElements *= kFrontMergeOutMaxFanIn;
            state.listNum = currentStageNeedCoreNum;
            state.srcWsIndex = 1U - state.srcWsIndex;
        }
        return state;
    }

    AICORE inline bool InitMergeListState(
        MergeListState& state, uint32_t inputBaseElem, uint32_t outputBaseElem, uint32_t listNum,
        uint32_t perListElements, uint32_t lastListElements, uint64_t srcWorkspaceBytes, bool extractToFinal) const
    {
        for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
            const uint32_t elems = listIdx == listNum - 1U ? lastListElements : perListElements;
            const uint32_t elemBase = inputBaseElem + listIdx * perListElements;
            const uint32_t packedOffset = FrontPtoGetSortOffset<float>(elemBase);
            const uint32_t packedLen = FrontPtoGetSortLen<float>(elems);
            if (static_cast<uint64_t>(packedOffset + packedLen) * sizeof(float) > srcWorkspaceBytes) {
                return false;
            }
            state.remain[listIdx] = elems;
            state.offsets[listIdx] = packedOffset;
            state.allRemain += elems;
        }
        state.outOffset = extractToFinal ? outputBaseElem : FrontPtoGetSortOffset<float>(outputBaseElem);
        return !extractToFinal || state.allRemain == op_.routeElems_;
    }

    AICORE inline uint32_t CountActiveMergeLists(const MergeListState& state, uint32_t listNum) const
    {
        uint32_t activeListNum = 0U;
        for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
            activeListNum += state.remain[listIdx] > 0U ? 1U : 0U;
        }
        return activeListNum;
    }

    AICORE inline bool LoadMergeLoopInputs(
        __gm__ float* srcPtr, MergeListState& listState, MergeLoopState& loopState, uint32_t listNum,
        uint32_t perListElems) const
    {
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
        uint32_t activeIdx = 0U;
        for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
            if (listState.remain[listIdx] == 0U) {
                continue;
            }
            const uint32_t curElems =
                listState.remain[listIdx] > perListElems ? perListElems : listState.remain[listIdx];
            const uint32_t packedLen = FrontPtoGetSortLen<float>(curElems);
            const uint64_t inputUb = MergeOutInputUb(activeIdx, perListElems);
            PtoLoadVector<float>(inputUb, srcPtr + listState.offsets[listIdx], packedLen);
            loopState.tmpUbInputs[activeIdx] = inputUb;
            loopState.elementCountList[activeIdx] = static_cast<uint16_t>(curElems);
            loopState.activeToList[activeIdx] = listIdx;
            loopState.loadedElems += curElems;
            ++activeIdx;
        }
        return activeIdx == loopState.activeListNum && loopState.loadedElems != 0U;
    }

    AICORE inline void MergeLoopInputs(
        MergeLoopState& loopState, uint32_t perListElems, uint64_t mergedUb, bool extractToFinal) const
    {
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        if (loopState.activeListNum == 1U) {
            PtoMoveUb<float>(
                mergedUb, loopState.tmpUbInputs[0], FrontPtoGetSortLen<float>(loopState.elementCountList[0]));
            loopState.listSortedNums[0] = loopState.elementCountList[0];
            if (extractToFinal) {
                pipe_barrier(PIPE_V);
            }
            return;
        }
        FrontMergePackedSortRecordsChecked(
            mergedUb, MergeOutTmpUb(loopState.activeListNum, perListElems), loopState.tmpUbInputs[0],
            loopState.tmpUbInputs[1], loopState.activeListNum >= 3U ? loopState.tmpUbInputs[2] : 0U,
            loopState.activeListNum >= 4U ? loopState.tmpUbInputs[3] : 0U, loopState.elementCountList,
            loopState.activeListNum, loopState.listSortedNums);
    }

    AICORE inline bool UpdateMergeListState(
        MergeListState& listState, const MergeLoopState& loopState, uint32_t perListElems, bool extractToFinal,
        uint32_t& curLoopSortedNum) const
    {
        curLoopSortedNum = 0U;
        for (uint32_t idx = 0U; idx < loopState.activeListNum; ++idx) {
            uint32_t sortedNum = loopState.listSortedNums[idx];
            if (extractToFinal && sortedNum > loopState.elementCountList[idx]) {
                sortedNum = loopState.elementCountList[idx];
            }
            const uint32_t listIdx = loopState.activeToList[idx];
            listState.remain[listIdx] -= sortedNum;
            listState.offsets[listIdx] += FrontPtoGetSortOffset<float>(sortedNum);
            curLoopSortedNum += sortedNum;
        }
        if (curLoopSortedNum == 0U || curLoopSortedNum > loopState.loadedElems ||
            curLoopSortedNum > loopState.activeListNum * perListElems) {
            return false;
        }
        listState.allRemain -= curLoopSortedNum;
        return true;
    }

    AICORE inline bool StoreMergeLoopOutput(
        __gm__ float* dstPtr, MergeListState& listState, const MergeLoopState& loopState, uint32_t perListElems,
        uint64_t mergedUb, uint32_t curLoopSortedNum, uint64_t dstWorkspaceBytes, bool extractToFinal) const
    {
        if (extractToFinal) {
            FrontExtractPackedSortResult(
                0U, SortOutPayloadUb(loopState.activeListNum, perListElems),
                SortOutScratchUb(loopState.activeListNum, perListElems), mergedUb, curLoopSortedNum);
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            PtoStoreVector<int32_t>(op_.frontExpandedExpertPtr_ + listState.outOffset, 0U, curLoopSortedNum);
            PtoStoreVector<int32_t>(
                op_.frontExpandDstToSrcPtr_ + listState.outOffset,
                SortOutPayloadUb(loopState.activeListNum, perListElems), curLoopSortedNum);
            listState.outOffset += curLoopSortedNum;
            return true;
        }
        const uint32_t outLen = FrontPtoGetSortLen<float>(curLoopSortedNum);
        if (static_cast<uint64_t>(listState.outOffset + outLen) * sizeof(float) > dstWorkspaceBytes) {
            return false;
        }
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<float>(dstPtr + listState.outOffset, mergedUb, outLen);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        listState.outOffset += outLen;
        return true;
    }

    /*
        srcPtr             输入 packed workspace
        dstPtr             输出 packed workspace
        inputBaseElem      输入 run 组在全局 record 空间里的起始 record 下标
        outputBaseElem     输出位置的起始 record 下标
        listNum            要归并几路，1~4
        perListElements    普通每一路有多少 record
        lastListElements   最后一路有多少 record
        srcWorkspaceBytes  输入 workspace 字节数
        dstWorkspaceBytes  输出 workspace 字节数
        extractToFinal     是否最终抽取为 int32 结果 */
    AICORE inline bool MergePackedListGroupImpl(
        __gm__ float* srcPtr, __gm__ float* dstPtr, uint32_t inputBaseElem, uint32_t outputBaseElem, uint32_t listNum,
        uint32_t perListElements, uint32_t lastListElements, uint64_t srcWorkspaceBytes, uint64_t dstWorkspaceBytes,
        bool extractToFinal) const
    {
        if (listNum == 0U || listNum > kFrontMergeOutMaxFanIn || perListElements == 0U || lastListElements == 0U) {
            return false;
        }

        MergeListState listState;
        if (!InitMergeListState(
                listState, inputBaseElem, outputBaseElem, listNum, perListElements, lastListElements, srcWorkspaceBytes,
                extractToFinal)) {
            return false;
        }
        while (listState.allRemain > 0U) {
            MergeLoopState loopState;
            loopState.activeListNum = CountActiveMergeLists(listState, listNum);
            if (loopState.activeListNum == 0U) {
                return false;
            }
            const uint32_t perListElems = op_.sortOutLoopMaxElems_;
            const uint64_t requiredUbBytes = extractToFinal ?
                                                 SortOutRequiredUbBytes(loopState.activeListNum, perListElems) :
                                                 MergeOnlyRequiredUbBytes(loopState.activeListNum, perListElems);
            if (perListElems == 0U || requiredUbBytes > AtlasA2::UB_SIZE) {
                return false;
            }

            if (!LoadMergeLoopInputs(srcPtr, listState, loopState, listNum, perListElems)) {
                return false;
            }
            const uint64_t mergedUb = MergeOutMergedUb(loopState.activeListNum, perListElems);
            MergeLoopInputs(loopState, perListElems, mergedUb, extractToFinal);
            uint32_t curLoopSortedNum = 0U;
            if (!UpdateMergeListState(listState, loopState, perListElems, extractToFinal, curLoopSortedNum)) {
                return false;
            }
            if (!StoreMergeLoopOutput(
                    dstPtr, listState, loopState, perListElems, mergedUb, curLoopSortedNum, dstWorkspaceBytes,
                    extractToFinal)) {
                return false;
            }
        }
        return true;
    }

    AICORE inline void BuildOneCoreVmsProcess() const
    {
        uint32_t runStart = 0U;
        uint32_t runElems = 0U;
        GetVbsSortRunRange(runStart, runElems);
        if (runElems == 0U) {
            return;
        }

        const bool isLastSortCore = op_.coreIdx_ == op_.sortNeedCoreNum_ - 1U;
        uint32_t listNum = isLastSortCore ? op_.sortLastCoreLoops_ : op_.sortPerCoreLoops_;
        uint32_t perListElements = isLastSortCore ? op_.sortLastCorePerLoopElems_ : op_.sortPerCorePerLoopElems_;
        uint32_t lastListElements = isLastSortCore ? op_.sortLastCoreLastLoopElems_ : op_.sortPerCoreLastLoopElems_;
        if (listNum <= 1U || perListElements == 0U || lastListElements == 0U) {
            return;
        }

        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        uint32_t srcWsIndex = 0U;
        while (listNum > 1U) {
            const uint32_t loops =
                (listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn; // kFrontMergeOutMaxFanIn=4路归并
            const uint32_t remainListNum = listNum - (loops - 1U) * kFrontMergeOutMaxFanIn;
            __gm__ float* srcPtr = FrontSortWsPtr(srcWsIndex);
            __gm__ float* dstPtr = FrontSortWsPtr(1U - srcWsIndex);
            for (uint32_t loop = 0U; loop < loops; ++loop) {
                const uint32_t groupListNum = loop == loops - 1U ? remainListNum : kFrontMergeOutMaxFanIn;
                const uint32_t groupLastListElements = loop == loops - 1U ? lastListElements : perListElements;
                const uint32_t baseElem = runStart + loop * kFrontMergeOutMaxFanIn * perListElements;
                if (!MergePackedListGroupImpl(
                        srcPtr, dstPtr, baseElem, baseElem, groupListNum, perListElements, groupLastListElements,
                        workspaceBytes, workspaceBytes, false)) {
                    return;
                }
            }
            lastListElements = perListElements * (remainListNum - 1U) + lastListElements;
            perListElements *= kFrontMergeOutMaxFanIn;
            listNum = loops; // 继续下一轮的4路归并
            srcWsIndex = 1U - srcWsIndex;
        }
    }

    AICORE inline void BuildVmsProcess() const
    {
        if (op_.sortNeedCoreNum_ <= kFrontMergeOutMaxFanIn || op_.sortPerCoreElems_ == 0U ||
            op_.sortLastCoreElems_ == 0U || op_.routeElems_ == 0U) {
            return;
        }

        SortMergeState state = BuildLocalSortState();
        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        while (state.listNum > kFrontMergeOutMaxFanIn) {
            const uint32_t currentStageNeedCoreNum =
                (state.listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            const uint32_t remainListNum =
                state.listNum - (currentStageNeedCoreNum - 1U) * kFrontMergeOutMaxFanIn; // 4路归并
            __gm__ float* srcPtr = FrontSortWsPtr(state.srcWsIndex);
            __gm__ float* dstPtr = FrontSortWsPtr(1U - state.srcWsIndex);
            if (op_.coreIdx_ < currentStageNeedCoreNum) {
                const uint32_t groupListNum =
                    op_.coreIdx_ == currentStageNeedCoreNum - 1U ? remainListNum : kFrontMergeOutMaxFanIn;
                const uint32_t groupLastListElements =
                    op_.coreIdx_ == currentStageNeedCoreNum - 1U ? state.lastListElements : state.perListElements;
                const uint32_t baseElem = op_.coreIdx_ * kFrontMergeOutMaxFanIn * state.perListElements;
                (void)MergePackedListGroupImpl(
                    srcPtr, dstPtr, baseElem, baseElem, groupListNum, state.perListElements, groupLastListElements,
                    workspaceBytes, workspaceBytes, false);
            }
            state.lastListElements = state.perListElements * (remainListNum - 1U) + state.lastListElements;
            state.perListElements *= kFrontMergeOutMaxFanIn;
            state.listNum = currentStageNeedCoreNum; // 继续下一轮
            state.srcWsIndex = 1U - state.srcWsIndex;
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        }
    }

    AICORE inline void GetVbsSortRunRange(uint32_t& runStart, uint32_t& runElems) const
    {
        runStart = op_.coreIdx_ * op_.sortPerCoreElems_;
        runElems = 0U;
        if (op_.coreIdx_ >= op_.sortNeedCoreNum_ || runStart >= op_.routeElems_) {
            return;
        }
        runElems = (op_.coreIdx_ == op_.sortNeedCoreNum_ - 1U) ? op_.sortLastCoreElems_ : op_.sortPerCoreElems_;
        if (runStart + runElems > op_.routeElems_) {
            runElems = op_.routeElems_ - runStart;
        }
    }

    AICORE inline bool VbsSortRunFitsUb(uint32_t runAlignedElems) const
    {
        const uint64_t runSortBytes = AlignBytes<int32_t>(static_cast<uint64_t>(runAlignedElems) * sizeof(int32_t));
        const uint64_t runSortPackedBytes =
            AlignBytes<float>(static_cast<uint64_t>(runAlignedElems) * 2U * sizeof(float));
        const uint64_t requiredBytes = runSortBytes * 3U + runSortPackedBytes * 2U;
        return runAlignedElems <= kFrontSortMaxElems && runAlignedElems <= op_.sortLoopMaxElement_ &&
               requiredBytes <= AtlasA2::UB_SIZE;
    }

    AICORE inline bool ProcessOneVbsSortRun(__gm__ float* packedPtr, uint32_t loopStart, uint32_t loopElems) const
    {
        const uint32_t loopAlignedElems = FrontAlignSortBlock(loopElems);
        if (!VbsSortRunFitsUb(loopAlignedElems)) {
            return false;
        }
        const uint64_t runSortBytes = AlignBytes<int32_t>(static_cast<uint64_t>(loopAlignedElems) * sizeof(int32_t));
        const uint64_t runPackedSortBytes =
            AlignBytes<float>(static_cast<uint64_t>(loopAlignedElems) * 2U * sizeof(float));
        const uint64_t runExpertUb = 0U;
        const uint64_t runPayloadUb = runSortBytes;
        const uint64_t runSortKeyUb = runSortBytes * 2U;
        const uint64_t runPackedSortUb = runSortBytes * 3U;
        const uint64_t runMergeTmpUb = runPackedSortUb + runPackedSortBytes;

        PtoLoadVector<int32_t>(runExpertUb, op_.expertIdPtr_ + loopStart, loopElems);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(runPayloadUb, static_cast<int32_t>(loopStart), 1, loopElems);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        FrontSortInt32ToPackedUb(
            runExpertUb, runPayloadUb, runPackedSortUb, runMergeTmpUb, runSortKeyUb, loopElems, loopAlignedElems);
        pipe_barrier(PIPE_V);

        const uint32_t packedOffset = FrontPtoGetSortOffset<float>(loopStart);
        const uint32_t packedLen = FrontPtoGetSortLen<float>(loopElems);
        if (static_cast<uint64_t>(packedOffset + packedLen) * sizeof(float) > FrontSortWorkspaceBytes()) {
            return false;
        }
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<float>(packedPtr + packedOffset, runPackedSortUb, packedLen);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        return true;
    }

    AICORE inline void BuildVbsSortRuns() const
    {
        uint32_t runStart = 0U;
        uint32_t runElems = 0U;
        GetVbsSortRunRange(runStart, runElems); // 多核切分，获取本核负责的总行数
        if (runElems == 0U) {
            return;
        }

        const bool isLastSortCore = op_.coreIdx_ == op_.sortNeedCoreNum_ - 1U;
        const uint32_t coreLoops = isLastSortCore ?
                                       op_.sortLastCoreLoops_ :
                                       op_.sortPerCoreLoops_; // 本核分到的总行数，需要分成几轮来处理，每轮处理6144条
        const uint32_t corePerLoopElems = isLastSortCore ? op_.sortLastCorePerLoopElems_ : op_.sortPerCorePerLoopElems_;
        const uint32_t coreLastLoopElems =
            isLastSortCore ? op_.sortLastCoreLastLoopElems_ : op_.sortPerCoreLastLoopElems_;
        if (coreLoops == 0U || corePerLoopElems == 0U || coreLastLoopElems == 0U) {
            return;
        }

        const auto& front = op_.tilingData_->frontReorderTiling;
        __gm__ float* packedPtr = reinterpret_cast<__gm__ float*>(op_.workspaceGM_ + front.frontSortWs0Offset);
        for (uint32_t loop = 0U; loop < coreLoops; ++loop) { // 按照多轮6144遍历
            const uint32_t loopStart = runStart + loop * corePerLoopElems;
            if (loopStart >= runStart + runElems || loopStart >= op_.routeElems_) {
                return;
            }
            uint32_t loopElems = (loop == coreLoops - 1U) ? coreLastLoopElems : corePerLoopElems;
            if (loopStart + loopElems > runStart + runElems) {
                loopElems = runStart + runElems - loopStart;
            }
            if (loopStart + loopElems > op_.routeElems_) {
                loopElems = op_.routeElems_ - loopStart;
            }
            if (loopElems == 0U) {
                return;
            }

            if (!ProcessOneVbsSortRun(packedPtr, loopStart, loopElems)) {
                return;
            }
        }
    }

    AICORE inline void BuildSortOutProcess() const
    {
        if (op_.coreIdx_ != 0U || op_.sortNeedCoreNum_ == 0U || op_.sortPerCoreElems_ == 0U || op_.routeElems_ == 0U) {
            return;
        }

        const SortMergeState finalState = BuildFinalSortState();
        const uint32_t finalListNum = finalState.listNum;
        const uint32_t finalPerListElements = finalState.perListElements;
        const uint32_t finalLastListElements = finalState.lastListElements;
        if (finalListNum == 0U || finalListNum > kFrontMergeOutMaxFanIn || finalPerListElements == 0U ||
            finalLastListElements == 0U) {
            return;
        }
        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        __gm__ float* srcPackedPtr = FrontSortWsPtr(finalState.srcWsIndex);
        (void)MergePackedListGroupImpl(
            srcPackedPtr, srcPackedPtr, 0U, 0U, finalListNum, finalPerListElements, finalLastListElements,
            workspaceBytes, 0U, true);
    }
};

#endif // DISPATCH_MEGA_COMBINE_FRONT_VMS_SORT_H
