/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H
#define DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H

template <typename InputElement>
class FrontReorderFullLoad : public FrontReorderPathBase {
public:
    AICORE inline explicit FrontReorderFullLoad(FrontReorderCommonState &op) : FrontReorderPathBase(op)
    {}

    AICORE inline void Init(GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ MegaMoeTilingData *tilingData)
    {
        op_.InitCommonInputs(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        op_.InitMinimalSortTiling(op_.tilingData_->frontReorderTiling);
        op_.InitCommonCoreIdx();
        op_.InitCommonWorkspacePtrs(op_.tilingData_->frontReorderTiling, false);
        op_.InitCommonPeerWindow();
    }

    struct FullLoadRouteTiling {
        uint32_t needCoreNum = 0;
        uint32_t perCoreRows = 0;
        uint32_t lastCoreRows = 0;
        uint32_t activateRows = 0;
        uint32_t coreRows = 0;
    };

    AICORE inline FullLoadRouteTiling BuildFullLoadRouteTiling() const
    {
        FullLoadRouteTiling tiling;
        tiling.activateRows = op_.routeElems_;
        if (op_.routeElems_ == 0U || op_.coreNum_ == 0U) {
            return tiling;
        }
        tiling.perCoreRows = static_cast<uint32_t>(ceilDiv(op_.routeElems_, op_.coreNum_));
        if (tiling.perCoreRows == 0U) {
            return tiling;
        }
        tiling.needCoreNum = static_cast<uint32_t>(ceilDiv(op_.routeElems_, tiling.perCoreRows));
        tiling.lastCoreRows = op_.routeElems_ - tiling.perCoreRows * (tiling.needCoreNum - 1U);
        if (op_.coreIdx_ < tiling.needCoreNum) {
            tiling.coreRows = op_.coreIdx_ == tiling.needCoreNum - 1U ? tiling.lastCoreRows : tiling.perCoreRows;
        }
        return tiling;
    }

    AICORE inline uint32_t FullLoadTileLength() const
    {
        const uint32_t lastCorePerLoop =
            op_.sortLastCorePerLoopElems_ == 0U ? op_.routeElems_ : op_.sortLastCorePerLoopElems_;
        return static_cast<uint32_t>(alignUp(lastCorePerLoop, sizeof(int32_t)));
    }

    AICORE inline uint32_t FullLoadSortNum() const
    {
        return static_cast<uint32_t>(alignUp(FullLoadTileLength(), kFrontSortAlignElement));
    }

    AICORE inline uint64_t FullLoadRequiredUbBytes() const
    {
        constexpr uint64_t kOneCoreSortBuffer = 6U;
        constexpr uint64_t kOtherRouteBuffer = 3U;
        constexpr uint64_t kDynamicQuantFullLoadColsBuffer = 13U;
        constexpr uint64_t kScaleOutBytes = 64U;
        const uint64_t alignedRouteElems = alignUp(op_.routeElems_, UB_ALIGN);
        const uint64_t sortSpace = alignedRouteElems * sizeof(int32_t) * kOneCoreSortBuffer;
        const uint64_t otherSpace = alignedRouteElems * sizeof(int32_t) * kOtherRouteBuffer;
        const uint64_t expertSpace = alignUp(static_cast<uint64_t>(op_.expertNum_) * sizeof(int32_t), UB_ALIGN);
        const uint64_t quantSpace =
            alignUp(static_cast<uint64_t>(op_.problemK_), UB_ALIGN) * kDynamicQuantFullLoadColsBuffer;
        return sortSpace + otherSpace + expertSpace + quantSpace + kScaleOutBytes;
    }

    AICORE inline bool FullLoadDynamicCapable() const
    {
        return op_.routeElems_ != 0U && op_.routeElems_ <= op_.sortLoopMaxElement_ &&
               op_.problemK_ <= kLargeFullRowMaxK && op_.problemK_ % UB_ALIGN == 0U &&
               FullLoadRequiredUbBytes() <= AtlasA2::UB_SIZE;
    }

    AICORE inline uint64_t FullLoadRouteBytes() const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(FullLoadSortNum()) * sizeof(int32_t));
    }

    AICORE inline uint64_t FullLoadPackedSortBytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FullLoadSortNum()) * 2U * sizeof(float));
    }

    AICORE inline uint64_t FullLoadSortKeyBytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FullLoadSortNum()) * sizeof(float));
    }

    AICORE inline uint64_t FullLoadCountBytes() const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(op_.expertNumAligned_) * sizeof(int32_t));
    }

    AICORE inline uint64_t QuantRawBytes() const
    {
        return AlignBytes<InputElement>(static_cast<uint64_t>(op_.problemK_) * sizeof(InputElement));
    }

    AICORE inline uint64_t QuantFp32Bytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(op_.problemK_) * sizeof(float));
    }

    AICORE inline uint64_t QuantOutBytes() const
    {
        return AlignBytes<int8_t>(static_cast<uint64_t>(op_.problemK_) * sizeof(int8_t));
    }

    AICORE inline uint64_t QuantScaleBytes() const
    {
        return AlignBytes<float>(8U * sizeof(float));
    }

    AICORE inline uint64_t FullLoadPayloadUb() const
    {
        return FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadPackedSortUb() const
    {
        return FullLoadPayloadUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadMergeTmpUb() const
    {
        return FullLoadPackedSortUb() + FullLoadPackedSortBytes();
    }

    AICORE inline uint64_t FullLoadSortKeyUb() const
    {
        return FullLoadMergeTmpUb() + FullLoadPackedSortBytes();
    }

    AICORE inline uint64_t FullLoadExpandedExpertUb() const
    {
        return FullLoadSortKeyUb() + FullLoadSortKeyBytes();
    }

    AICORE inline uint64_t FullLoadExpandDstToSrcUb() const
    {
        return FullLoadExpandedExpertUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadExpandedRowIdxUb() const
    {
        return FullLoadExpandDstToSrcUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadSortScratchUb() const
    {
        return FullLoadExpandedRowIdxUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadCountUb() const
    {
        return FullLoadSortScratchUb() + FullLoadSortKeyBytes();
    }

    AICORE inline uint64_t QuantRawUb() const
    {
        return FullLoadSortScratchUb();
    }

    AICORE inline uint64_t QuantFp32Ub() const
    {
        return QuantRawUb() + QuantRawBytes();
    }

    AICORE inline uint64_t QuantTmpUb() const
    {
        return QuantFp32Ub() + QuantFp32Bytes();
    }

    AICORE inline uint64_t QuantOutUb() const
    {
        return QuantTmpUb() + QuantFp32Bytes();
    }

    AICORE inline uint64_t QuantScaleUb() const
    {
        return QuantOutUb() + QuantOutBytes();
    }

    AICORE inline uint64_t QuantActualUbBytes() const
    {
        return QuantScaleUb() + QuantScaleBytes();
    }

    AICORE inline uint64_t FullLoadSortActualUbBytes() const
    {
        return FullLoadCountUb() + FullLoadCountBytes();
    }

    AICORE inline bool FullLoadSortEnabled() const
    {
        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        return op_.coreIdx_ < tiling.needCoreNum && FullLoadDynamicCapable() &&
               FullLoadSortNum() <= kFrontSortMaxElems && FullLoadSortActualUbBytes() <= AtlasA2::UB_SIZE;
    }

    /* 输入 FullLoadExpandDstToSrcUb()[dstRow] = srcRoute，
       输出 FullLoadExpandedRowIdxUb()[srcRoute] = dstRow。 */
    AICORE inline void BuildInverseExpandedRowIdx(uint32_t totalLength, uint32_t sortNum) const
    {
        PtoCastUb<float, int32_t>(FullLoadSortKeyUb(), FullLoadExpandDstToSrcUb(), totalLength,
                                  pto::RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_V);
        PtoMulScalarUb<float>(FullLoadSortKeyUb(), FullLoadSortKeyUb(), totalLength, -1.0F);
        pipe_barrier(PIPE_V);

        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        for (uint32_t idx = totalLength; idx < sortNum; ++idx) {
            PtoSetValue<float>(FullLoadSortKeyUb(), idx, kFrontSortNegInf);
            PtoSetValue<uint32_t>(FullLoadPayloadUb(), idx, 0U);
        }
        if (sortNum > totalLength) {
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        }
        PtoFillArithProgressionInt32(FullLoadPayloadUb(), 0, 1, totalLength);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        FrontPackedSortTile packedTile(1, sortNum * 2U);
        FrontPackedSortTile mergeTmpTile(1, sortNum * 2U);
        FrontSortKeyTile srcTile(1, sortNum);
        FrontSortPayloadTile payloadTile(1, sortNum);
        pto::TASSIGN(srcTile, FullLoadSortKeyUb());
        pto::TASSIGN(payloadTile, FullLoadPayloadUb());
        pto::TASSIGN(packedTile, FullLoadPackedSortUb());
        pto::TASSIGN(mergeTmpTile, FullLoadMergeTmpUb());
        pto::TSORT32(packedTile, srcTile, payloadTile);
        pipe_barrier(PIPE_V);
        FrontMergePackedSortRecords(packedTile, mergeTmpTile, sortNum * 2U);

        FrontPackedPayloadTile packedPayloadTile(1, totalLength * 2U);
        FrontSortPayloadTile expandedRowIdxTile(1, totalLength);
        pto::TASSIGN(packedPayloadTile, FullLoadPackedSortUb());
        pto::TASSIGN(expandedRowIdxTile, FullLoadExpandedRowIdxUb());
        pto::TGATHER<FrontSortPayloadTile, FrontPackedPayloadTile, pto::MaskPattern::P1010>(expandedRowIdxTile,
                                                                                            packedPayloadTile);
        pipe_barrier(PIPE_V);
    }

    /*  FullLoad means the whole route sort/count/quant workspace fits in one UB.
         Active AIVs repeat this full sort, then split quant/scatter by route range. */
    AICORE inline void RunFullLoadSort() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }

        const uint32_t totalLength = op_.routeElems_;

        const uint32_t sortNum = FullLoadSortNum();                // TSORT32 需要32元素对齐.
        PtoLoadVector<int32_t>(0U, op_.expertIdPtr_, totalLength); // 加载expertId[M, topK]=exprtidvalue 到UB

        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(FullLoadPayloadUb(), 0, 1,
                                     totalLength); // payload[srcRoute] = srcRoute, 辅助编号加快排序速度
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        // First sort output is packed as (sortedExpert, srcRoute) records ordered by expert.
        FrontSortInt32ToPackedUb(0U, FullLoadPayloadUb(), FullLoadPackedSortUb(), FullLoadMergeTmpUb(),
                                 FullLoadSortKeyUb(), totalLength, sortNum);

        pipe_barrier(PIPE_V);
        // expandedExpert[dstRow] = sortedExpert, expandDstToSrc[dstRow] = srcRoute.
        FrontExtractPackedSortResult(FullLoadExpandedExpertUb(), FullLoadExpandDstToSrcUb(), FullLoadSortScratchUb(),
                                     FullLoadPackedSortUb(), totalLength);
        pipe_barrier(PIPE_V);

        // Second sort: use srcRoute as key and dstRow as payload.
        // Final UB result: FullLoadExpandedRowIdxUb()[srcRoute] = dstRow in expert-sorted order.
        BuildInverseExpandedRowIdx(totalLength, sortNum);
    }

    AICORE inline void StoreExpandedRowIdxToGm() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }
        if (op_.coreIdx_ == 0U) {
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            PtoStoreVector<int32_t>(op_.expandedRowIdxPtr_, FullLoadExpandedRowIdxUb(), op_.routeElems_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
    }

    AICORE inline void BuildLocalTokenPerExpertFromSort() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }

        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        const uint32_t ownerCore = tiling.needCoreNum == 0U ? 0U : tiling.needCoreNum - 1U;
        if (op_.coreIdx_ != ownerCore) {
            return;
        }

        PtoFillUb<int32_t>(FullLoadCountUb(), 0, op_.expertNumAligned_);
        pipe_barrier(PIPE_V);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();

        int32_t lastExpertId = PtoGetValue<int32_t>(FullLoadExpandedExpertUb(), 0U);
        int32_t tokenCount = 0;
        for (uint32_t idx = 0; idx < op_.routeElems_; ++idx) {
            const int32_t curExpertId = PtoGetValue<int32_t>(FullLoadExpandedExpertUb(), idx);
            ++tokenCount;
            while (lastExpertId < curExpertId) {
                if (lastExpertId >= 0 && static_cast<uint32_t>(lastExpertId) < op_.expertNumAligned_) {
                    PtoSetValue<int32_t>(FullLoadCountUb(), static_cast<uint32_t>(lastExpertId), tokenCount - 1);
                }
                tokenCount = 1;
                ++lastExpertId;
            }
        }
        if (lastExpertId >= 0 && static_cast<uint32_t>(lastExpertId) < op_.expertNumAligned_) {
            PtoSetValue<int32_t>(FullLoadCountUb(), static_cast<uint32_t>(lastExpertId), tokenCount);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        PtoStoreVector<int32_t>(op_.localTokenPerExpertPtr_, FullLoadCountUb(), op_.expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    AICORE inline bool FullLoadQuantUbReady(const FullLoadRouteTiling &tiling) const
    {
        return tiling.coreRows != 0U && QuantActualUbBytes() <= AtlasA2::UB_SIZE;
    }

    AICORE inline bool FullLoadQuantEnabled() const
    {
        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        return FullLoadSortEnabled() && FullLoadQuantUbReady(tiling);
    }

    AICORE inline void QuantAndScatterPackedRows() const
    {
        if (!FullLoadQuantEnabled()) {
            return;
        }

        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        uint32_t curRowsStart = op_.coreIdx_ * tiling.perCoreRows;
        const uint32_t curRowsEnd = curRowsStart + tiling.coreRows - 1U;
        const uint32_t startXRow = curRowsStart / op_.topK_;
        const uint32_t endXRow = curRowsEnd / op_.topK_;

        for (uint32_t row = startXRow; row <= endXRow && row < op_.problemM_; ++row) {
            (void)FrontDynamicQuantRowToUb<InputElement>(
                reinterpret_cast<__gm__ InputElement *>(op_.xPtr_) + static_cast<uint64_t>(row) * op_.problemK_,
                op_.problemK_, QuantRawUb(), QuantFp32Ub(), QuantTmpUb(), QuantOutUb(), QuantScaleUb());

            bool rowStored = false;
            while (curRowsStart <= curRowsEnd && curRowsStart / op_.topK_ == row) {
                const int32_t outIndex = PtoGetValue<int32_t>(FullLoadExpandedRowIdxUb(), curRowsStart);
                ++curRowsStart;
                if (outIndex < 0 || static_cast<uint32_t>(outIndex) >= tiling.activateRows) {
                    continue;
                }
                PtoStoreVector<int8_t>(
                    op_.offsetAPtr_ + FrontPackedRowOffset(static_cast<uint32_t>(outIndex), op_.problemK_),
                    QuantOutUb(), FrontPackedRowStride(op_.problemK_));
                rowStored = true;
            }
            if (rowStored) {
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
            }
        }
    }

private:
};

#endif // DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H
