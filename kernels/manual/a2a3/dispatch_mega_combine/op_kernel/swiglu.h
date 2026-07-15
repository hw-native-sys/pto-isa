/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_SWIGLU_H
#define DISPATCH_MEGA_COMBINE_SWIGLU_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_vector.hpp"
#include "utils/pto_sync_substrate.hpp"

constexpr uint32_t kSwigluWaitSourceC2VOnly = 1U;
constexpr uint32_t kSwigluPipelineModeInputOutputSplit = 1U;
constexpr uint32_t kSwigluMetadataModeSharedSegmentMeta = 1U;
constexpr uint32_t kSwigluVecTileElems = 1024U;
constexpr uint32_t kSwigluFullRowIoBlockChunks = 4U;
constexpr uint32_t kSwigluUbStageNum = 2U;
constexpr uint32_t kSwigluScaleTileElems = 128U;
constexpr uint32_t kSwigluScaleChunkBuffers = 2U;
constexpr uint32_t kSwigluScalarScratchBytes = 32U;
constexpr float kSwigluDynamicQuantEps = 1.0e-6f;

template <typename InputElement>
class Swiglu {
public:
    AICORE inline void Init(GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void Process();

private:
    AICORE inline __gm__ MegaMoeSwigluSegmentRuntimeMeta* SegmentMetaPtr() const
    {
        return reinterpret_cast<__gm__ MegaMoeSwigluSegmentRuntimeMeta*>(
            workspaceGM_ + tilingData_->swigluTiling.swigluSegmentMetaOffset);
    }
    AICORE inline void WriteSharedSegmentMetadata(uint32_t segmentIdx) const;
    AICORE inline void ReadSharedSegmentMetadata(
        uint32_t segmentIdx, uint32_t& segmentStartExpert, uint32_t& segmentEndExpert, uint32_t& segmentRowBase,
        uint32_t& segmentRows, uint32_t& cumsumRows, uint32_t& expertTokenRows, uint32_t& rowSplitBase,
        uint32_t& rowSplitRem) const;
    AICORE inline uint64_t AlignUbBytes(uint64_t value) const { return (value + 31U) / 32U * 32U; }
    AICORE inline uint64_t SwigluMaxScratchBytes() const
    {
        const uint64_t bytes = static_cast<uint64_t>(outputN_ / 2U) * sizeof(float);
        return bytes < kSwigluScalarScratchBytes ? kSwigluScalarScratchBytes : AlignUbBytes(bytes);
    }
    AICORE inline uint64_t SwigluStageBytes() const;
    AICORE inline uint64_t SwigluCOffset(uint32_t bufferId) const
    {
        if (bufferId == 0U) {
            return 0;
        }
        return static_cast<uint64_t>(bufferId) * SwigluStageBytes();
    }
    AICORE inline uint64_t SwigluDOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCOffset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(half));
    }
    AICORE inline uint64_t SwigluCFp32Offset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluDOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(int8_t));
    }
    AICORE inline uint64_t SwigluWorkOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCFp32Offset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluAbsOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluWorkOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluMaxOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluAbsOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluMaxOffset(bufferId) + SwigluMaxScratchBytes());
    }
    AICORE inline uint64_t SwigluScaleOutputBytes() const
    {
        return AlignUbBytes(static_cast<uint64_t>(kSwigluScaleTileElems) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOutputOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluStageBytes() * kSwigluUbStageNum) +
               static_cast<uint64_t>(bufferId) * SwigluScaleOutputBytes();
    }
    AICORE inline bool FullRowUbFits() const
    {
        return SwigluScaleOutputOffset(kSwigluScaleChunkBuffers - 1U) + SwigluScaleOutputBytes() <= AtlasA2::UB_SIZE;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreDoneEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t ScaleStoreEvent(uint32_t bufferId) const { return bufferId == 0U ? EVENT_ID2 : EVENT_ID3; }
    AICORE inline void InitFullRowPipeline() const;
    AICORE inline void FinalizeFullRowPipeline() const;
    AICORE inline void RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const;
    AICORE inline void IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline void ApplyFullRowPerTokenScale(uint32_t bufferId, float perTokenScale) const;
    AICORE inline void ComputeFullRowSwiglu(uint32_t bufferId) const;
    AICORE inline float ReduceFullRowMaxAbs(uint32_t bufferId) const;
    AICORE inline void QuantizeFullRowOutput(uint32_t bufferId, float quantScale) const;
    AICORE inline void StoreFullRowOutput(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float ComputeAndStorePreparedFullRow(uint32_t rowIdx, uint32_t bufferId, float perTokenScale) const;
    AICORE inline void IssueStoreScale2Chunk(uint32_t rowStart, uint32_t rowCount, uint32_t scaleBufferId) const;

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ half* gmCPtr_ = nullptr;
    __gm__ float* perTokenScalePtr_ = nullptr;
    __gm__ int8_t* gmPermutedTokenPtr_ = nullptr;
    __gm__ float* perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;
    uint32_t problemN_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t stageNum_ = 0;
};

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::Init(
    GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    (void)sizeof(InputElement);
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    problemN_ = tilingData_->megaMoeInfo.N;
    outputN_ = problemN_ / 2U;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    stageNum_ = tilingData_->frontReorderTiling.stageNum;

    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    gmCPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM_ + tilingData_->gmm1Tiling.gmCOffset);
    perTokenScalePtr_ = reinterpret_cast<__gm__ float*>(workspaceGM_ + tilingData_->dispatchTiling.perTokenScaleOffset);
    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t*>(workspaceGM_ + tilingData_->swigluTiling.gmPermutedTokenOffset);
    perTokenScale2Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM_ + tilingData_->swigluTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);
}
template <typename InputElement>
AICORE inline void Swiglu<InputElement>::WriteSharedSegmentMetadata(uint32_t segmentIdx) const
{
    if (coreIdx_ != 0U) {
        return;
    }

    uint32_t segmentStartExpert = 0;
    uint32_t segmentEndExpert = 0;
    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    uint32_t cumsumRows = 0;
    uint32_t expertTokenRows = 0;
    MoeBuildSegmentMetadata(
        segmentIdx, expertPerRank_, maxOutputSize_, cumsumMMPtr_, expertTokenNumsPtr_, rankSize_, segmentStartExpert,
        segmentEndExpert, segmentRowBase, segmentRows, cumsumRows, expertTokenRows);
    const uint32_t rowSplitBase = segmentRows / coreNum_;
    const uint32_t rowSplitRem = segmentRows - rowSplitBase * coreNum_;

    volatile __gm__ MegaMoeSwigluSegmentRuntimeMeta* entry = SegmentMetaPtr() + segmentIdx;
    entry->valid = 0U;
    entry->segmentIdx = segmentIdx;
    entry->segmentStartExpert = segmentStartExpert;
    entry->segmentEndExpert = segmentEndExpert;
    entry->segmentRowBase = segmentRowBase;
    entry->segmentRows = segmentRows;
    entry->cumsumRows = cumsumRows;
    entry->expertTokenRows = expertTokenRows;
    entry->rowSplitBase = rowSplitBase;
    entry->rowSplitRem = rowSplitRem;
    entry->generation = stageNum_;
    entry->producerCoreIdx = coreIdx_;
    entry->metadataMode = kSwigluMetadataModeSharedSegmentMeta;
    entry->segmentNum = MoeSwigluSegmentNum(expertPerRank_);
    entry->epilogueGranularity = MoeSwigluEpilogueGranularity(expertPerRank_);
    entry->marker = 1U;
    pipe_barrier(PIPE_ALL);
    entry->valid = 1U;
    pipe_barrier(PIPE_ALL);
    V5DcciGmRange(
        reinterpret_cast<__gm__ void*>(SegmentMetaPtr() + segmentIdx), sizeof(MegaMoeSwigluSegmentRuntimeMeta));
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ReadSharedSegmentMetadata(
    uint32_t segmentIdx, uint32_t& segmentStartExpert, uint32_t& segmentEndExpert, uint32_t& segmentRowBase,
    uint32_t& segmentRows, uint32_t& cumsumRows, uint32_t& expertTokenRows, uint32_t& rowSplitBase,
    uint32_t& rowSplitRem) const
{
    volatile __gm__ MegaMoeSwigluSegmentRuntimeMeta* entry = SegmentMetaPtr() + segmentIdx;
    segmentStartExpert = entry->segmentStartExpert;
    segmentEndExpert = entry->segmentEndExpert;
    segmentRowBase = entry->segmentRowBase;
    segmentRows = entry->segmentRows;
    cumsumRows = entry->cumsumRows;
    expertTokenRows = entry->expertTokenRows;
    rowSplitBase = entry->rowSplitBase;
    rowSplitRem = entry->rowSplitRem;
}
template <typename InputElement>
AICORE inline uint64_t Swiglu<InputElement>::SwigluStageBytes() const
{
    return AlignUbBytes(SwigluScaleOffset(0) + kSwigluScalarScratchBytes);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::InitFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbStageNum; ++bufferId) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::FinalizeFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbStageNum; ++bufferId) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        wait_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const
{
    if (localRows == 0U || outputN_ == 0U || !FullRowUbFits()) {
        return;
    }
    InitFullRowPipeline();
    uint32_t scaleChunkStart = 0;
    uint32_t scaleChunkCount = 0;
    uint32_t scaleBufferId = 0;
    IssueFullRowLoad(localRowStart, 0U);
    for (uint32_t localRow = 0; localRow < localRows; ++localRow) {
        const uint32_t bufferId = localRow % kSwigluUbStageNum;
        const uint32_t rowIdx = localRowStart + localRow;
        const float perTokenScale =
            PrepareFullRowCompute(rowIdx, bufferId); //  用来反量化 GMM1 结果，进入 SwiGLU fp32 计算
        if (localRow + 1U < localRows) {
            IssueFullRowLoad(localRowStart + localRow + 1U, (localRow + 1U) % kSwigluUbStageNum);
        }
        // SwiGLU 输出重新量化后的 per-token scale，供 combine 阶段反量化 GMM2 输出使用
        const float scale2 = ComputeAndStorePreparedFullRow(rowIdx, bufferId, perTokenScale);
        if (scaleChunkCount == 0U) {
            wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
        }
        PtoSetValue<float, kSwigluScaleTileElems>(SwigluScaleOutputOffset(scaleBufferId), scaleChunkCount, scale2);
        ++scaleChunkCount;
        if (scaleChunkCount == kSwigluScaleTileElems) { // 满128行，一起写回GM
            IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
            scaleChunkStart += scaleChunkCount;
            scaleChunkCount = 0;
            scaleBufferId = scaleBufferId + 1U == kSwigluScaleChunkBuffers ? 0U : scaleBufferId + 1U;
        }
    }
    if (scaleChunkCount > 0U) {
        IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
    }
    FinalizeFullRowPipeline();
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::IssueStoreScale2Chunk(
    uint32_t rowStart, uint32_t rowCount, uint32_t scaleBufferId) const
{
    if (rowCount == 0U) {
        return;
    }
    set_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    wait_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    PtoStoreVector<float, kSwigluScaleTileElems>(
        perTokenScale2Ptr_ + rowStart, SwigluScaleOutputOffset(scaleBufferId), rowCount);
    set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CGlobal = pto::GlobalTensor<half, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileC = pto::Tile<
        pto::TileType::Vec, half, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    __gm__ half* gmCRow = gmCPtr_ + static_cast<uint64_t>(rowIdx) * problemN_;

    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    uint32_t offset = 0;
    while (problemN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (problemN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileC cTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        BlockShape cShape(chunkRows, kSwigluVecTileElems);
        BlockStride cStride(
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        CBlockGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < problemN_) {
        const uint32_t cur = problemN_ - offset;
        TileC cTile(1, cur);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        VectorShape cShape(cur);
        VectorStride cStride(cur, cur, cur, cur);
        CGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);

    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 fp32Tile(1, cur);
        TileC cTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(fp32Tile, ubCFp32Offset + elemOffset * sizeof(float));
        pto::TASSIGN(cTile, ubCOffset + elemOffset * sizeof(half));
        pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    return perTokenScalePtr_[rowIdx];
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ApplyFullRowPerTokenScale(uint32_t bufferId, float perTokenScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(srcTile, ubCFp32Offset + elemOffset);
        pto::TMULS(dstTile, srcTile, perTokenScale);
    }
    pipe_barrier(PIPE_V);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::ComputeFullRowSwiglu(uint32_t bufferId) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        pto::TASSIGN(dstTile, ubWorkOffset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(srcTile, ubCFp32Offset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TMULS(dstTile, srcTile, -1.0f);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TEXP(dstTile, srcTile);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TADDS(dstTile, srcTile, 1.0f);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 xTile(1, cur);
        TileFp32 denomTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(xTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(denomTile, ubWorkOffset + elemOffset);
        pto::TDIV(dstTile, xTile, denomTile);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 siluTile(1, cur);
        TileFp32 gateTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(siluTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(gateTile, ubCFp32Offset + static_cast<uint64_t>(outputN_ + offset) * sizeof(float));
        pto::TMUL(dstTile, siluTile, gateTile);
    }
    pipe_barrier(PIPE_V);
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::ReduceFullRowMaxAbs(uint32_t bufferId) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);
    const uint64_t ubMaxOffset = SwigluMaxOffset(bufferId);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TABS(absTile, srcTile);
    }
    pipe_barrier(PIPE_V);

    bool firstReduceChunk = true;
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(tmpTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(rowMaxTile, firstReduceChunk ? ubMaxOffset : ubAbsOffset);
        pto::TROWMAX(rowMaxTile, absTile, tmpTile);
        pto::TSYNC<pto::Op::TROWMAX>();
        if (!firstReduceChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, ubMaxOffset);
            pto::TASSIGN(newTile, ubAbsOffset);
            pto::TASSIGN(dstTile, ubMaxOffset);
            pto::TMAX(dstTile, accTile, newTile);
            pto::TSYNC<pto::Op::TMAX>();
        }
        firstReduceChunk = false;
    }
    pipe_barrier(PIPE_V);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    TileFp32 maxScalarTile(1, 1);
    pto::TASSIGN(maxScalarTile, ubMaxOffset);
    return maxScalarTile.GetValue(0);
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::QuantizeFullRowOutput(uint32_t bufferId, float quantScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    using TileI32 = PtoVecTile<int32_t, kSwigluVecTileElems>;
    using TileHalf = PtoVecTile<half, kSwigluVecTileElems>;
    const uint64_t ubDOffset = SwigluDOffset(bufferId);
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TMULS(dstTile, srcTile, quantScale);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileI32 i32Tile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(i32Tile, ubAbsOffset + elemOffset * sizeof(int32_t));
        pto::TASSIGN(srcTile, ubAbsOffset + elemOffset * sizeof(float));
        pto::TCVT(i32Tile, srcTile, pto::RoundMode::CAST_RINT);
    }
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileHalf halfTile(1, cur);
        TileI32 i32Tile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(halfTile, ubAbsOffset + elemOffset * sizeof(half));
        pto::TASSIGN(i32Tile, ubAbsOffset + elemOffset * sizeof(int32_t));
        pto::TCVT(halfTile, i32Tile, pto::RoundMode::CAST_RINT);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        PtoVecTile<int8_t, kSwigluVecTileElems> dTile(1, cur);
        TileHalf halfTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(dTile, ubDOffset + elemOffset * sizeof(int8_t));
        pto::TASSIGN(halfTile, ubAbsOffset + elemOffset * sizeof(half));
        pto::TCVT(dTile, halfTile, pto::RoundMode::CAST_RINT);
    }
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::StoreFullRowOutput(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileD = PtoVecTile<int8_t, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<int8_t, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileD = pto::Tile<
        pto::TileType::Vec, int8_t, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DBlockGlobal = pto::GlobalTensor<int8_t, BlockShape, BlockStride, pto::Layout::ND>;
    const uint64_t ubDOffset = SwigluDOffset(bufferId);
    __gm__ int8_t* gmDRow = gmPermutedTokenPtr_ + static_cast<uint64_t>(rowIdx) * outputN_;
    uint32_t offset = 0;
    while (outputN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (outputN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileD dTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        BlockShape dShape(chunkRows, kSwigluVecTileElems);
        BlockStride dStride(
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        DBlockGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < outputN_) {
        const uint32_t cur = outputN_ - offset;
        TileD dTile(1, cur);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        VectorShape dShape(cur);
        VectorStride dStride(cur, cur, cur, cur);
        DGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
    }
}

template <typename InputElement>
AICORE inline float Swiglu<InputElement>::ComputeAndStorePreparedFullRow(
    uint32_t rowIdx, uint32_t bufferId, float perTokenScale) const
{
    ApplyFullRowPerTokenScale(bufferId, perTokenScale);
    ComputeFullRowSwiglu(bufferId);
    const float maxAbs = ReduceFullRowMaxAbs(bufferId);
    const float scale2 = maxAbs > 0.0f ? maxAbs / 127.0f : kSwigluDynamicQuantEps / 127.0f;
    const float quantScale = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;

    wait_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    QuantizeFullRowOutput(bufferId, quantScale);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    StoreFullRowOutput(rowIdx, bufferId);
    set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    return scale2;
}

template <typename InputElement>
AICORE inline void Swiglu<InputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }

    const uint32_t segmentNum = MoeSwigluSegmentNum(expertPerRank_);
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        CrossCoreWaitFlag<0x2>(MegaMoeC2VHardFlagId(segmentIdx));
        WriteSharedSegmentMetadata(segmentIdx); // core 0负责分配任务给多个aiv
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        uint32_t segmentStartExpert = 0;
        uint32_t segmentEndExpert = 0;
        uint32_t segmentRowBase = 0;
        uint32_t segmentRows = 0;
        uint32_t cumsumRows = 0;
        uint32_t expertTokenRows = 0;
        uint32_t localRowStart = 0;
        uint32_t localRows = 0;
        uint32_t rowSplitBase = 0;
        uint32_t rowSplitRem = 0;
        ReadSharedSegmentMetadata(
            segmentIdx, segmentStartExpert, segmentEndExpert, segmentRowBase, segmentRows, cumsumRows, expertTokenRows,
            rowSplitBase, rowSplitRem);
        localRows = rowSplitBase + (coreIdx_ < rowSplitRem ? 1U : 0U);
        const uint32_t prefixRows = coreIdx_ * rowSplitBase + (coreIdx_ < rowSplitRem ? coreIdx_ : rowSplitRem);
        localRowStart = segmentRowBase + prefixRows;
        RunFullRowEpilogue(localRowStart, localRows);

        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(MegaMoeV2CHardFlagId(segmentIdx));
    }
}

#endif // DISPATCH_MEGA_COMBINE_SWIGLU_H
