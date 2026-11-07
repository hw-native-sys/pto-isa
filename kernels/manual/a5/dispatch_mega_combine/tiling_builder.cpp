/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tiling_builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "op_kernel/utils/const_args.hpp"
#include "op_kernel/utils/mega_wave_schedule.hpp"

namespace {
constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchMinBufferCount = 2U;
constexpr uint32_t kDispatchMaxBufferCount = 6U;
constexpr uint32_t kDispatchBaseRouteItemsPerBatch = 12288U;
constexpr uint32_t kDispatchRouteItemAlignment = 256U;
constexpr uint32_t kDispatchMetaSlotBytes = 32U;
constexpr uint32_t kDispatchRouteCountBytes = 32U;
constexpr uint32_t kDispatchMaskLoadGuardBytes = 256U;
constexpr uint32_t kMaxUnpermuteVecTileElems = 8192U;
constexpr uint32_t kHalfDataBlockElems = 16U;
constexpr uint32_t kFrontMetadataSortRunMaxElems = 6144U;
constexpr uint32_t kFrontMetadataSortAlignElems = 32U;
constexpr uint32_t kUnpermuteMetadataTokenBatchTarget = 64U;
constexpr uint32_t kMxDataAlignmentBytes = 256U;
constexpr uint32_t kMxScaleAlignmentBytes = 32U;

uint64_t AlignUp(uint64_t value, uint64_t align);

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char* name);

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char* name);

uint32_t Pow4Ceil(uint32_t value)
{
    uint32_t result = 1U;
    while (result < value) {
        if (result > UINT32_MAX / 4U) {
            throw std::runtime_error("front metadata sort run count overflows uint32");
        }
        result *= 4U;
    }
    return result;
}

uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return value / divisor + static_cast<uint32_t>(value % divisor != 0U);
}

// Bootstrap GMM roles; actual groups are selected by the runtime publication prefix.
constexpr A5CoreTopology kA5CoreTopologies[] = {
    {.physicalAicNum = 28U, .gmm1GroupSize = 20U, .gmm2GroupSize = 8U},
    {.physicalAicNum = 32U, .gmm1GroupSize = 21U, .gmm2GroupSize = 11U},
    {.physicalAicNum = 36U, .gmm1GroupSize = 24U, .gmm2GroupSize = 12U},
};
constexpr uint32_t kUnpermuteTwoPhaseMinM = 512U;

void RequireMxShape(uint32_t k, uint32_t n)
{
    if (n == 0U || (n & 1U) != 0U) {
        throw std::runtime_error("N must be positive and even for SwiGLU");
    }
    if (k % 128U != 0U || (n / 2U) % 128U != 0U) {
        throw std::runtime_error("MXFP8 GMM reduction dimensions must be multiples of 128");
    }
}

uint64_t MxQuantDataStorageBytes(uint32_t cols) { return AlignUp(cols, kMxDataAlignmentBytes); }

uint64_t MxQuantScaleCols(uint32_t cols) { return cols / kMegaMoeMxGroupSize; }

uint64_t MxQuantScaleStorageBytes(uint32_t cols) { return AlignUp(MxQuantScaleCols(cols), kMxScaleAlignmentBytes); }

uint64_t MxPackedRowStride(uint32_t cols) { return MxQuantDataStorageBytes(cols) + MxQuantScaleStorageBytes(cols); }

void RequireDispatchPackedRowCapacity(uint64_t packedRowStride)
{
    if (packedRowStride > kDispatchGatherPackedTileCols) {
        throw std::runtime_error("dispatch packed row exceeds the A5 vector tile width");
    }
}

void RequireFrontQuantUbCapacity(uint32_t k)
{
    const uint64_t scaleCols = MxQuantScaleCols(k);
    const uint64_t oneBufferBytes = AlignUp(static_cast<uint64_t>(k) * sizeof(uint16_t), UB_ALIGN) +
                                    AlignUp(static_cast<uint64_t>(k), UB_ALIGN) + AlignUp(scaleCols, UB_ALIGN) +
                                    AlignUp(scaleCols * sizeof(uint16_t), UB_ALIGN) * 2U;
    if (oneBufferBytes * 2U > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("front quant ping-pong buffers exceed A5 main UB budget");
    }
}

uint64_t UnpermuteUbBytes(
    uint32_t metadataTokenRows, uint32_t topk, uint32_t tileCols, uint32_t metadataBufferCount,
    uint32_t inputBufferCount)
{
    uint64_t metadataBytes = 0U;
    for (uint32_t i = 0; i < metadataBufferCount; ++i) {
        metadataBytes += AlignUp(static_cast<uint64_t>(metadataTokenRows) * topk * sizeof(int32_t), UB_ALIGN);
        metadataBytes += AlignUp(static_cast<uint64_t>(metadataTokenRows) * topk * sizeof(float), UB_ALIGN);
    }
    const uint64_t bf16SlotBytes = AlignUp(static_cast<uint64_t>(tileCols) * sizeof(uint16_t), UB_ALIGN);
    const uint64_t fp32SlotBytes = AlignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    const uint64_t dataSlotBytes = CheckedAdd(bf16SlotBytes, fp32SlotBytes, "Unpermute data slot");
    return CheckedAdd(
        metadataBytes, CheckedMul(inputBufferCount + 1U, dataSlotBytes, "Unpermute data ring"), "Unpermute UB");
}

void RequireUnpermuteUbCapacity(
    uint32_t metadataTokenRows, uint32_t topk, uint32_t tileCols, uint32_t metadataBufferCount,
    uint32_t inputBufferCount)
{
    if (inputBufferCount < kMegaMoeUnpermuteMinInputBufferCount ||
        inputBufferCount > kMegaMoeUnpermuteMaxInputBufferCount ||
        UnpermuteUbBytes(metadataTokenRows, topk, tileCols, metadataBufferCount, inputBufferCount) > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("unpermute UB buffers exceed A5 main UB budget");
    }
}

uint32_t ChooseUnpermuteTileCols(uint32_t k, uint32_t metadataTokenRows, uint32_t topk, uint32_t metadataBufferCount)
{
    uint32_t tileCols = std::min(k, kMaxUnpermuteVecTileElems);
    tileCols = tileCols / kHalfDataBlockElems * kHalfDataBlockElems;
    while (tileCols > kHalfDataBlockElems && UnpermuteUbBytes(
                                                 metadataTokenRows, topk, tileCols, metadataBufferCount,
                                                 kMegaMoeUnpermuteMinInputBufferCount) > A5_MAIN_UB_SIZE) {
        tileCols -= kHalfDataBlockElems;
    }
    if (tileCols == 0U ||
        UnpermuteUbBytes(metadataTokenRows, topk, tileCols, metadataBufferCount, kMegaMoeUnpermuteMinInputBufferCount) >
            A5_MAIN_UB_SIZE) {
        throw std::runtime_error("unable to choose unpermute tile cols within A5 main UB budget");
    }
    return tileCols;
}

uint32_t ChooseUnpermuteInputBufferCount(
    uint32_t metadataTokenRows, uint32_t topk, uint32_t tileCols, uint32_t metadataBufferCount)
{
    const uint32_t accumulationItemCount = metadataTokenRows * topk;
    for (uint32_t count = kMegaMoeUnpermuteMaxInputBufferCount; count >= kMegaMoeUnpermuteMinInputBufferCount;
         --count) {
        if ((count <= accumulationItemCount || count == kMegaMoeUnpermuteMinInputBufferCount) &&
            UnpermuteUbBytes(metadataTokenRows, topk, tileCols, metadataBufferCount, count) <= A5_MAIN_UB_SIZE) {
            return count;
        }
    }
    throw std::runtime_error("unable to choose unpermute input ring within A5 main UB budget");
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        throw std::runtime_error("AlignUp requires nonzero align");
    }
    if (value > UINT64_MAX - (align - 1U)) {
        throw std::runtime_error("AlignUp overflows uint64");
    }
    return (value + align - 1) / align * align;
}

void PopulateDispatchBufferTiling(MegaMoeDispatchTiling& dispatch, const MegaMoeFrontReorderTiling& front)
{
    if (front.routeElems == 0U || front.packedRowStride == 0U) {
        throw std::runtime_error("dispatch requires nonzero route items and packed row stride");
    }

    const uint64_t alignedTotalRouteItems = AlignUp(front.routeElems, kDispatchRouteItemAlignment);
    uint64_t routeItemsPerBatch = std::min<uint64_t>(alignedTotalRouteItems, kDispatchBaseRouteItemsPerBatch);
    const uint64_t copyBufferBytes = front.packedRowStride;
    const uint64_t dispatchSlotBytes = copyBufferBytes;
    const uint64_t countTableBytes = AlignUp(static_cast<uint64_t>(front.expertNum) * sizeof(int32_t), UB_ALIGN);

    auto routeBufferBytes = [](uint64_t routeItems) {
        const uint64_t routeIndexBytes = AlignUp(routeItems * sizeof(uint32_t), UB_ALIGN);
        const uint64_t maskBytes = AlignUp((routeItems + 7U) / 8U, UB_ALIGN);
        return routeIndexBytes + maskBytes + kDispatchRouteCountBytes + kDispatchMaskLoadGuardBytes;
    };

    const uint64_t nonSlotBytes = routeBufferBytes(routeItemsPerBatch) + 2U * countTableBytes;
    const uint64_t slotBudget = nonSlotBytes < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - nonSlotBytes : 0U;
    uint64_t bufferCount = std::min<uint64_t>(slotBudget / dispatchSlotBytes, kDispatchMaxBufferCount);
    if (bufferCount < kDispatchMinBufferCount) {
        throw std::runtime_error("dispatch cannot fit the minimum two token ring slots in A5 main UB");
    }

    if (routeItemsPerBatch < alignedTotalRouteItems) {
        const uint64_t fixedBytes = bufferCount * dispatchSlotBytes + kDispatchRouteCountBytes +
                                    kDispatchMaskLoadGuardBytes + 2U * countTableBytes;
        const uint64_t routeBudget = fixedBytes < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - fixedBytes : 0U;
        uint64_t expandedRouteItems = routeBudget * 8U / 33U;
        expandedRouteItems = expandedRouteItems / kDispatchRouteItemAlignment * kDispatchRouteItemAlignment;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        routeItemsPerBatch = std::max(routeItemsPerBatch, expandedRouteItems);
    }

    dispatch.routeItemsPerBatch = static_cast<uint32_t>(routeItemsPerBatch);
    dispatch.bufferCount = static_cast<uint32_t>(bufferCount);
    dispatch.copyBufferBytes = static_cast<uint32_t>(copyBufferBytes);

    uint64_t ubOffset = bufferCount * copyBufferBytes;
    dispatch.routeIndexUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset = AlignUp(ubOffset + routeItemsPerBatch * sizeof(uint32_t), UB_ALIGN);
    dispatch.routeCountUbOffset = static_cast<uint32_t>(ubOffset);
    ubOffset += kDispatchRouteCountBytes;
    dispatch.maskBufferUbOffset = static_cast<uint32_t>(ubOffset);
    const uint64_t maskBufferBytes = AlignUp((routeItemsPerBatch + 7U) / 8U, UB_ALIGN);
    ubOffset += maskBufferBytes + kDispatchMaskLoadGuardBytes;
    dispatch.cumsumUbOffset = static_cast<uint32_t>(AlignUp(ubOffset, UB_ALIGN));
    dispatch.expertCountUbOffset = dispatch.cumsumUbOffset + static_cast<uint32_t>(countTableBytes);
    ubOffset = dispatch.expertCountUbOffset + countTableBytes;
    if (ubOffset > A5_MAIN_UB_SIZE) {
        throw std::runtime_error("dispatch token ring and route buffers exceed A5 main UB budget");
    }
}

uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    if (divisor == 0U) {
        throw std::runtime_error("CeilDivU64 requires a nonzero divisor");
    }
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

void PopulateWaveSchedule(MegaMoeTilingData& tiling, const A5CoreTopology& schedule, const CaseConfig& cfg)
{
    MegaMoeFixedGroupTiling& fixed = tiling.fixedGroupTiling;
    MegaMoeWavePlannerInput plannerInput = {
        .inputRows = cfg.m,
        .topK = cfg.topk,
        .expertCount = cfg.expert_per_rank,
        .activeAicNum = schedule.physicalAicNum,
        .gmm1TileM = kMegaMoeGmmTileM,
        .gmm1TileN = kMegaMoeGmmTileN,
        .gmm1OutputN = cfg.n,
        .gmm2TileM = kMegaMoeGmmTileM,
        .gmm2TileN = kMegaMoeGmmTileN,
        .gmm2OutputN = cfg.k,
    };
    fixed.fullAicExpertsPerWave = CalcExpertsPerWave(plannerInput);
    plannerInput.activeAicNum = fixed.gmm1GroupSize;
    fixed.expertsPerWave = CalcExpertsPerWave(plannerInput);
}

void ValidateFixedSchedule(const MegaMoeFixedGroupTiling& fixed, const CaseConfig& cfg)
{
    if (fixed.gmm1GroupSize + fixed.gmm2GroupSize != fixed.physicalAicNum) {
        throw std::runtime_error("fixed AIC group sizes must cover all physical AICs");
    }
    if (fixed.dispatchGroupSize == 0U || fixed.dispatchGroupSize > kMegaMoeFixedDispatchGroupSize ||
        fixed.dispatchGroupSize + 1U >= fixed.physicalAicNum || fixed.gmm1GroupSize == 0U ||
        fixed.gmm1GroupSize > kMegaMoeFixedGmm1GroupSize || fixed.gmm2GroupSize == 0U ||
        fixed.gmm2GroupSize > kMegaMoeFixedGmm2GroupSize) {
        throw std::runtime_error("fixed Dispatch/GMM/SwiGLU groups exceed the A5 mixed-core capacities");
    }
    const uint32_t totalWaveCount = GetTotalWaveCount(
        cfg.expert_per_rank, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
    if (fixed.fullAicExpertsPerWave == 0U || fixed.fullAicExpertsPerWave > cfg.expert_per_rank ||
        fixed.expertsPerWave == 0U || fixed.expertsPerWave > cfg.expert_per_rank || totalWaveCount == 0U ||
        fixed.fullAicGmm1WaveCount != 1U || fixed.fullAicGmm1WaveCount > totalWaveCount) {
        throw std::runtime_error("invalid expert wave partition");
    }
}

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char* name)
{
    if (lhs != 0U && rhs > UINT64_MAX / lhs) {
        throw std::runtime_error(std::string(name) + " byte size overflows uint64");
    }
    return lhs * rhs;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char* name)
{
    if (rhs > UINT64_MAX - lhs) {
        throw std::runtime_error(std::string(name) + " end offset overflows uint64");
    }
    return lhs + rhs;
}

void RequireAlignedRange(const char* name, uint64_t offset, uint64_t bytes)
{
    if (offset % 512U != 0U || bytes % 512U != 0U) {
        throw std::runtime_error(std::string(name) + " must be 512-byte aligned");
    }
}

uint64_t AllocateAlignedSection(uint64_t& offset, uint64_t rawBytes, const char* name)
{
    const uint64_t begin = AlignUp(offset, 512U);
    offset = AlignUp(CheckedAdd(begin, rawBytes, name), 512U);
    return begin;
}

uint64_t PopulateFrontTiling(
    MegaMoeFrontReorderTiling& front, const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    constexpr uint64_t kPeerAlignBytes = 512U;
    constexpr uint64_t kPeerSignalBytes = MB_SIZE;

    const uint64_t expertNum = CheckedMul(cfg.world_size, cfg.expert_per_rank, "expert count");
    const uint64_t routeElems = CheckedMul(cfg.m, cfg.topk, "route count");
    if (expertNum > INT32_MAX || routeElems > INT32_MAX) {
        throw std::runtime_error("expert or route count exceeds int32 metadata capacity");
    }

    const uint64_t quantDataStorageBytes = MxQuantDataStorageBytes(cfg.k);
    const uint64_t quantScaleCols = MxQuantScaleCols(cfg.k);
    const uint64_t quantScaleStorageBytes = MxQuantScaleStorageBytes(cfg.k);
    const uint64_t packedRowStride = CheckedAdd(quantDataStorageBytes, quantScaleStorageBytes, "MX token record");
    if (quantDataStorageBytes > UINT32_MAX || quantScaleCols > UINT32_MAX || quantScaleStorageBytes > UINT32_MAX ||
        packedRowStride > UINT32_MAX) {
        throw std::runtime_error("MX token record fields exceed uint32");
    }

    front.expertNum = static_cast<uint32_t>(expertNum);
    front.routeElems = static_cast<uint32_t>(routeElems);
    front.quantDataStorageBytes = static_cast<uint32_t>(quantDataStorageBytes);
    front.quantScaleCols = static_cast<uint32_t>(quantScaleCols);
    front.packedRowStride = static_cast<uint32_t>(packedRowStride);
    front.maskBytes = static_cast<uint32_t>(AlignUp((routeElems + 7U) / 8U, 32U));
    front.maskSlotBytes = front.maskBytes + kMegaMoeFrontMaskCountRecordBytes;
    constexpr uint64_t kMaskMinBufferCount = 2U;
    constexpr uint64_t kMaskMaxBufferCount = 6U;
    constexpr uint64_t kMaskRouteItemAlignment = 256U;
    constexpr uint64_t kMaskBaseRouteItemsPerBatch = 12288U;
    constexpr uint64_t kMaskMaxRouteItemsPerBatch = 24576U;
    const uint64_t alignedMaskRoutes = AlignUp(routeElems, kMaskRouteItemAlignment);
    uint64_t maskRouteItems = std::min(alignedMaskRoutes, kMaskBaseRouteItemsPerBatch);
    const uint64_t routeBatchCount = CeilDivU64(routeElems, maskRouteItems);
    const uint64_t ownedExpertMax = CeilDivU64(expertNum, cfg.aiv_num);
    const uint64_t maskPushCount = routeBatchCount * ownedExpertMax;
    const uint64_t quantBytesPerBuffer = AlignUp(cfg.k * sizeof(uint16_t), UB_ALIGN) + AlignUp(cfg.k, UB_ALIGN) +
                                         AlignUp(quantScaleCols, UB_ALIGN) +
                                         2U * AlignUp(quantScaleCols * sizeof(uint16_t), UB_ALIGN);
    const uint64_t fixedBufferBytes = 2U * quantBytesPerBuffer + AlignUp(ownedExpertMax * 4U, UB_ALIGN) + UB_ALIGN;
    const uint64_t routeIndexBytes = maskRouteItems * sizeof(int32_t);
    const uint64_t maskBufferBytes = AlignUp((maskRouteItems + 7U) / 8U, UB_ALIGN);
    const uint64_t bytesWithoutMask = fixedBufferBytes + 2U * routeIndexBytes;
    const uint64_t maskBudget = bytesWithoutMask < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - bytesWithoutMask : 0U;
    uint64_t maskBufferCount = std::min(maskBudget / maskBufferBytes, kMaskMaxBufferCount);
    if (maskPushCount != 0U)
        maskBufferCount = std::min(maskBufferCount, maskPushCount);
    maskBufferCount = std::max(maskBufferCount, kMaskMinBufferCount);
    if (maskRouteItems < routeElems) {
        const uint64_t fixedWithMasks = fixedBufferBytes + maskBufferCount * UB_ALIGN;
        const uint64_t budget = fixedWithMasks < A5_MAIN_UB_SIZE ? A5_MAIN_UB_SIZE - fixedWithMasks : 0U;
        uint64_t expanded = budget * 8U / (64U + maskBufferCount);
        expanded = expanded / kMaskRouteItemAlignment * kMaskRouteItemAlignment;
        expanded = std::min(std::min(expanded, alignedMaskRoutes), kMaskMaxRouteItemsPerBatch);
        maskRouteItems = std::max(maskRouteItems, expanded);
    }
    front.maskRouteItemsPerBatch = static_cast<uint32_t>(maskRouteItems);
    front.maskBufferCount = static_cast<uint32_t>(maskBufferCount);

    const uint64_t sourceTokenRecordBytes = CheckedMul(cfg.m, front.packedRowStride, "source token records");
    front.routeMaskOffset = AlignUp(sourceTokenRecordBytes, kPeerAlignBytes);
    const uint64_t maskSlotCount = CheckedMul(cfg.expert_per_rank, cfg.world_size, "route mask slots");
    const uint64_t routeMaskBytes = CheckedMul(maskSlotCount, front.maskSlotBytes, "route mask slots");
    front.expertCountPeerOffset =
        AlignUp(CheckedAdd(front.routeMaskOffset, routeMaskBytes, "route mask slots"), kPeerAlignBytes);
    const uint64_t expertCountBytes =
        AlignUp(CheckedMul(maskSlotCount, sizeof(int32_t), "raw expert counts"), kPeerAlignBytes);
    front.combineOutputOffset =
        AlignUp(CheckedAdd(front.expertCountPeerOffset, expertCountBytes, "raw expert counts"), kPeerAlignBytes);
    const uint64_t combineRowsAndCols = CheckedMul(routeElems, cfg.k, "combine route output");
    const uint64_t combineOutputBytes = CheckedMul(combineRowsAndCols, sizeof(uint16_t), "combine route output");
    front.preSumBeforeRankPeerOffset =
        AlignUp(CheckedAdd(front.combineOutputOffset, combineOutputBytes, "combine route output"), kPeerAlignBytes);
    const uint64_t preSumBeforeRankPeerBytes =
        AlignUp(CheckedMul(expertNum, sizeof(int32_t), "preSumBeforeRank peer rows"), kPeerAlignBytes);
    const uint64_t peerDataBytes = AlignUp(
        CheckedAdd(front.preSumBeforeRankPeerOffset, preSumBeforeRankPeerBytes, "preSumBeforeRank peer rows"),
        kPeerAlignBytes);

    const uint64_t windowBytes = runtime.hccl.WindowBytes();
    if (windowBytes < kPeerSignalBytes) {
        throw std::runtime_error("HCCL window is smaller than the reserved signal section");
    }
    const uint64_t peerSignalOffset = windowBytes - kPeerSignalBytes;
    if (peerDataBytes > peerSignalOffset) {
        throw std::runtime_error(
            "HCCL window is too small for Mask Pull peer layout: windowBytes=" + std::to_string(windowBytes) +
            " dataBytes=" + std::to_string(peerDataBytes) + " signalOffset=" + std::to_string(peerSignalOffset));
    }
    RequireAlignedRange("Mask Pull peer data", 0U, peerDataBytes);

    uint64_t frontWorkspaceOffset = 0U;
    front.cumsumMMOffset = frontWorkspaceOffset;
    const uint64_t cumsumBytes = CheckedMul(maskSlotCount, sizeof(int32_t), "cumsumMM");
    frontWorkspaceOffset = AlignUp(CheckedAdd(frontWorkspaceOffset, cumsumBytes, "cumsumMM"), kPeerAlignBytes);

    front.expandedRowIdxOffset = frontWorkspaceOffset;
    const uint64_t expandedRowIdxBytes =
        AlignUp(CheckedMul(routeElems, sizeof(int32_t), "expandedRowIdx"), kPeerAlignBytes);
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, expandedRowIdxBytes, "expandedRowIdx"), kPeerAlignBytes);

    const uint64_t alignedRouteElems = AlignUp(routeElems, 128U);
    const uint32_t minimumRunCount = CeilDivU32(front.routeElems, kFrontMetadataSortRunMaxElems);
    front.sortRunCount = Pow4Ceil(minimumRunCount);
    front.sortRunElems =
        static_cast<uint32_t>(AlignUp(CeilDivU32(front.routeElems, front.sortRunCount), kFrontMetadataSortAlignElems));
    if (front.sortRunElems == 0U || front.sortRunElems > kFrontMetadataSortRunMaxElems ||
        static_cast<uint64_t>(front.sortRunCount - 1U) * front.sortRunElems >= front.routeElems) {
        throw std::runtime_error("invalid front metadata VBS run split");
    }

    const uint64_t sortedArrayBytes =
        AlignUp(CheckedMul(alignedRouteElems, sizeof(int32_t), "front sorted metadata"), kPeerAlignBytes);
    front.sortedRouteSlotOffset = frontWorkspaceOffset;
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, sortedArrayBytes, "front sorted route slots"), kPeerAlignBytes);

    const uint64_t sortWorkspaceBytes =
        AlignUp(CheckedMul(alignedRouteElems, 2U * sizeof(float), "front packed sort workspace"), kPeerAlignBytes);
    front.sortWorkspace0Offset = frontWorkspaceOffset;
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, sortWorkspaceBytes, "front packed sort workspace 0"), kPeerAlignBytes);
    front.sortWorkspace1Offset = frontWorkspaceOffset;
    frontWorkspaceOffset =
        AlignUp(CheckedAdd(frontWorkspaceOffset, sortWorkspaceBytes, "front packed sort workspace 1"), kPeerAlignBytes);
    RequireAlignedRange("expandedRowIdx", front.expandedRowIdxOffset, expandedRowIdxBytes);
    RequireAlignedRange("sortedRouteSlot", front.sortedRouteSlotOffset, sortedArrayBytes);
    RequireAlignedRange("sortWorkspace0", front.sortWorkspace0Offset, sortWorkspaceBytes);
    RequireAlignedRange("sortWorkspace1", front.sortWorkspace1Offset, sortWorkspaceBytes);
    return frontWorkspaceOffset;
}

uint64_t AllocatePipelineWorkspace(MegaMoeTilingData& tiling, const CaseConfig& cfg, uint64_t frontWorkspaceBytes)
{
    auto& dispatch = tiling.dispatchTiling;
    auto& swiglu = tiling.swigluTiling;

    const uint64_t rowCount = cfg.max_output_size;
    const uint64_t gmAElems = CheckedMul(rowCount, cfg.k, "gmA elements");
    const uint64_t gmAScaleElems = CheckedMul(rowCount, cfg.k / kMegaMoeMxGroupSize, "gmAScale elements");
    const uint64_t swigluCols = cfg.n / 2U;
    const uint64_t gmSwigluAElems = CheckedMul(rowCount, swigluCols, "gmSwigluA elements");
    const uint64_t gmSwigluScaleElems =
        CheckedMul(rowCount, swigluCols / kMegaMoeMxGroupSize, "gmSwigluScale elements");

    uint64_t workspaceOffset = frontWorkspaceBytes;
    dispatch.gmAOffset = AllocateAlignedSection(workspaceOffset, gmAElems, "gmA");
    dispatch.gmAScaleOffset = AllocateAlignedSection(workspaceOffset, gmAScaleElems, "gmAScale");
    swiglu.gmSwigluAOffset = AllocateAlignedSection(workspaceOffset, gmSwigluAElems, "gmSwigluA");
    swiglu.gmSwigluScaleOffset = AllocateAlignedSection(workspaceOffset, gmSwigluScaleElems, "gmSwigluScale");

    dispatch.gmm1ExpertCountStrideBytes = static_cast<uint32_t>(AlignUp(cfg.expert_per_rank * 4ULL, 64U));
    dispatch.gmm1ExpertCountOffset = AllocateAlignedSection(
        workspaceOffset, CheckedMul(cfg.aic_num, dispatch.gmm1ExpertCountStrideBytes, "paired expert counts"),
        "paired expert counts");

    const uint64_t maxTilesPerExpert = CeilDivU64(rowCount, kMegaMoeGmmTileM);
    if (maxTilesPerExpert == 0U || maxTilesPerExpert > UINT32_MAX) {
        throw std::runtime_error("Dispatch ready-count tile capacity exceeds uint32");
    }
    dispatch.readyCountMaxTilesPerExpert = static_cast<uint32_t>(maxTilesPerExpert);
    const uint64_t readyCountExpertStrideBytes =
        CheckedMul(maxTilesPerExpert, kMegaMoeReadyCountSlotBytes, "Dispatch ready-count expert stride");
    const uint64_t readyCountRawBytes =
        CheckedMul(cfg.expert_per_rank, readyCountExpertStrideBytes, "Dispatch ready-count workspace");
    dispatch.readyCountOffset =
        AllocateAlignedSection(workspaceOffset, readyCountRawBytes, "Dispatch ready-count workspace");
    const uint64_t readyCountBytes = workspaceOffset - dispatch.readyCountOffset;
    RequireAlignedRange("Dispatch ready-count workspace", dispatch.readyCountOffset, readyCountBytes);
    return workspaceOffset;
}

void AllocateGmmQueueWorkspace(
    MegaMoeGmmQueueTiling& queue, uint64_t normalTaskCapacity, uint64_t dependencySlotCount,
    uint32_t completionSlotCount, const char* stageName, uint64_t& workspaceOffset)
{
    if (normalTaskCapacity == 0U || normalTaskCapacity > UINT32_MAX || dependencySlotCount > UINT32_MAX) {
        throw std::runtime_error(std::string(stageName) + " task capacity exceeds the queue protocol");
    }

    queue.taskOffset = AllocateAlignedSection(
        workspaceOffset, CheckedMul(normalTaskCapacity, sizeof(MegaMoeGmmTaskDescriptor), "GMM task descriptors"),
        "GMM task descriptors");
    const uint64_t taskBytes = workspaceOffset - queue.taskOffset;
    uint64_t dependencyBytes = 0U;
    if (dependencySlotCount != 0U) {
        queue.dependencyOffset = AllocateAlignedSection(
            workspaceOffset, CheckedMul(dependencySlotCount, kMegaMoeFixedSyncSlotBytes, "GMM dependency counters"),
            "GMM dependency counters");
        dependencyBytes = workspaceOffset - queue.dependencyOffset;
    }
    uint64_t completionBytes = 0U;
    if (completionSlotCount != 0U) {
        queue.completionOffset = AllocateAlignedSection(
            workspaceOffset,
            CheckedMul(completionSlotCount, kMegaMoeFixedSyncSlotBytes, "GMM expert completion counters"),
            "GMM expert completion counters");
        completionBytes = workspaceOffset - queue.completionOffset;
    }
    RequireAlignedRange("GMM task descriptors", queue.taskOffset, taskBytes);
    if (dependencySlotCount != 0U) {
        RequireAlignedRange("GMM dependency counters", queue.dependencyOffset, dependencyBytes);
    }
    if (completionSlotCount != 0U) {
        RequireAlignedRange("GMM expert completion counters", queue.completionOffset, completionBytes);
    }
}

void AllocateGmmSchedulerWorkspace(MegaMoeTilingData& tiling, const CaseConfig& cfg, uint64_t& workspaceOffset)
{
    const MegaMoeFixedGroupTiling& fixed = tiling.fixedGroupTiling;
    MegaMoeGmmSchedulerTiling& scheduler = tiling.gmmSchedulerTiling;
    const uint64_t maxMTiles = CeilDivU64(cfg.max_output_size, kMegaMoeGmmTileM);
    const uint64_t mTileCapacity = CheckedAdd(maxMTiles, cfg.expert_per_rank - 1U, "GMM M-tile capacity");
    const uint64_t gmm1NTiles = CeilDivU64(cfg.n / 2U, kMegaMoeGmmTileN);
    const uint64_t gmm2NTiles = CeilDivU64(cfg.k, kMegaMoeGmmTileN);
    if (cfg.expert_per_rank > kGmmTaskExpertMask + 1U || maxMTiles > kGmmTaskBlockMMask + 1U ||
        gmm1NTiles > kGmmTaskBlockNMask + 1U || gmm2NTiles > kGmmTaskBlockNMask + 1U) {
        throw std::runtime_error("GMM task coordinates exceed the packed descriptor bit capacity");
    }
    const uint64_t gmm1NormalCapacity = CheckedMul(mTileCapacity, gmm1NTiles, "GMM1 normal task capacity");
    const uint64_t gmm2NormalCapacity = CheckedMul(mTileCapacity, gmm2NTiles, "GMM2 normal task capacity");
    const uint64_t gmm2DependencySlots = cfg.expert_per_rank;

    AllocateGmmQueueWorkspace(scheduler.gmm1, gmm1NormalCapacity, 0U, 0U, "GMM1", workspaceOffset);
    AllocateGmmQueueWorkspace(
        scheduler.gmm2, gmm2NormalCapacity, gmm2DependencySlots, cfg.expert_per_rank, "GMM2", workspaceOffset);

    const uint64_t gmm2TicketBase = CheckedAdd(1U, gmm1NormalCapacity, "GMM2 mailbox ticket base");
    const uint64_t gmm2TicketEnd = CheckedAdd(gmm2TicketBase, gmm2NormalCapacity, "GMM2 mailbox ticket span");
    if (gmm2TicketEnd >= kGmmMailboxTerminalTicket) {
        throw std::runtime_error("GMM mailbox ticket namespace exceeds the control-token range");
    }

    MegaMoeGmmMailboxTiling& mailbox = scheduler.mailbox;
    mailbox.gmm2TicketBase = static_cast<uint32_t>(gmm2TicketBase);
    mailbox.p2cOffset = AllocateAlignedSection(
        workspaceOffset, GmmMailboxP2cStorageBytes(kMegaMoeFixedPhysicalAicNum), "GMM P2C mailbox");
    const uint64_t p2cBytes = workspaceOffset - mailbox.p2cOffset;
    mailbox.c2pOffset = AllocateAlignedSection(
        workspaceOffset, CheckedMul(fixed.physicalAicNum, sizeof(MegaMoeGmmC2pSlot), "GMM C2P mailbox"),
        "GMM C2P mailbox");
    const uint64_t c2pBytes = workspaceOffset - mailbox.c2pOffset;
    RequireAlignedRange("GMM P2C mailbox", mailbox.p2cOffset, p2cBytes);
    RequireAlignedRange("GMM C2P mailbox", mailbox.c2pOffset, c2pBytes);
}

void PopulateRuntimeGmmSplitTable(MegaMoeFixedGroupTiling& fixed, const CaseConfig& cfg)
{
    const uint32_t maxLag = cfg.expert_per_rank - 1U;
    const uint32_t minLag = std::min(cfg.m <= 128U ? 4U : 1U, maxLag);
    const uint32_t maxLagWindow = std::min(cfg.m <= 128U ? 6U : 2U, maxLag);
    const uint32_t preferredLagNumerator = std::min(cfg.m <= 128U ? 10U : 3U, 2U * maxLagWindow);
    constexpr uint32_t preferredLagDenominator = 2U;
    for (uint32_t prefix = 0U; prefix <= kMegaMoeFixedMaxExperts; ++prefix) {
        fixed.runtimeGmm1GroupSizeByCalcStartExpert[prefix] = 0U;
        if (prefix >= cfg.expert_per_rank)
            continue;
        const MegaMoeGmmGroupSplit split = SelectGmmGroupSplitForLagWindow(
            fixed.physicalAicNum, cfg.expert_per_rank, prefix, 2U, 1U, minLag, maxLagWindow, preferredLagNumerator,
            preferredLagDenominator);
        if (split.gmm1GroupSize == 0U || split.gmm2GroupSize == 0U) {
            throw std::runtime_error("runtime GMM balancing found no dependency-safe AIC split");
        }
        fixed.runtimeGmm1GroupSizeByCalcStartExpert[prefix] = split.gmm1GroupSize;
    }
}

void AllocateFixedGroupWorkspace(
    MegaMoeTilingData& tiling, const A5CoreTopology& schedule, const CaseConfig& cfg, uint64_t& workspaceOffset)
{
    MegaMoeFixedGroupTiling& fixed = tiling.fixedGroupTiling;
    fixed.physicalAicNum = schedule.physicalAicNum;
    // Keep enough Dispatch workers for all ranks, reserving metadata and producer roles.
    fixed.dispatchGroupSize = std::max(schedule.DefaultDispatchAiv0Count(), cfg.world_size);
    fixed.gmm1GroupSize = schedule.gmm1GroupSize;
    fixed.gmm2GroupSize = schedule.gmm2GroupSize;
    fixed.fullAicGmm1WaveCount = kMegaMoeFullAicGmm1WaveCount;
    PopulateWaveSchedule(tiling, schedule, cfg);
    PopulateRuntimeGmmSplitTable(fixed, cfg);
    ValidateFixedSchedule(fixed, cfg);

    AllocateGmmSchedulerWorkspace(tiling, cfg, workspaceOffset);

    fixed.syncOffset = AlignUp(workspaceOffset, 512U);
    workspaceOffset = fixed.syncOffset + kMegaMoeFixedSyncBytes;
    fixed.completionOffset = AlignUp(workspaceOffset, 512U);
    const uint64_t completionBytes = MegaMoeFixedCompletionBytes(schedule.physicalAicNum);
    workspaceOffset = CheckedAdd(fixed.completionOffset, completionBytes, "completion doorbell workspace");
    RequireAlignedRange("fixedGroupSync", fixed.syncOffset, kMegaMoeFixedSyncBytes);
    RequireAlignedRange("fixedGroupCompletion", fixed.completionOffset, completionBytes);
}

bool CanUseRankStreaming(const CaseConfig& cfg, uint32_t workerCount, uint32_t initialWorkerCount)
{
    if (workerCount == 0U) {
        return false;
    }
    const uint64_t tokensPerWorker = CeilDivU64(cfg.m, workerCount);
    const uint64_t phase1TokensPerWorker = initialWorkerCount == 0U ? 0U : CeilDivU64(cfg.m, initialWorkerCount);
    return cfg.m != 0U && cfg.topk != 0U && cfg.topk <= 32U && cfg.expert_per_rank != 0U && cfg.world_size != 0U &&
           cfg.world_size <= kMegaMoeExpertProgressMaxRanks &&
           tokensPerWorker <= kMegaMoeRankStreamingMaxTokensPerWorker &&
           (initialWorkerCount == 0U || phase1TokensPerWorker <= kMegaMoeRankStreamingMaxTokensPerWorker);
}

void PopulateUnpermuteTiling(MegaMoeTilingData& tiling, const CaseConfig& cfg)
{
    MegaMoeUnpermuteTiling& unpermute = tiling.unpermuteTiling;
    const uint32_t physicalAicCount = tiling.fixedGroupTiling.physicalAicNum;
    const uint32_t physicalAivCount = physicalAicCount * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
    const uint32_t workerCount = physicalAivCount;
    const uint32_t availableAiv0Workers = physicalAivCount - physicalAicCount;
    const bool twoPhase = cfg.m >= kUnpermuteTwoPhaseMinM;
    const uint32_t initialWorkerCount =
        twoPhase ? std::min(availableAiv0Workers, tiling.fixedGroupTiling.dispatchGroupSize) : 0U;
    if (!CanUseRankStreaming(cfg, workerCount, initialWorkerCount)) {
        throw std::runtime_error("shape exceeds the rank-streaming Unpermute capability");
    }
    unpermute.rankStreamingInitialWorkerCount = initialWorkerCount;
    const uint32_t phase2TokenBatch = static_cast<uint32_t>(CeilDivU64(cfg.m, workerCount));
    const uint32_t phase1TokenBatch =
        initialWorkerCount == 0U ? phase2TokenBatch : static_cast<uint32_t>(CeilDivU64(cfg.m, initialWorkerCount));
    const uint32_t boundedPhase1TokenBatch = std::min(phase1TokenBatch, kUnpermuteMetadataTokenBatchTarget);
    unpermute.unpermuteTokenBatch = std::max(phase2TokenBatch, boundedPhase1TokenBatch);
    if (unpermute.unpermuteTokenBatch == 0U ||
        unpermute.unpermuteTokenBatch > kMegaMoeRankStreamingMaxTokensPerWorker) {
        throw std::runtime_error("invalid final-phase unpermute worker handoff");
    }
    constexpr uint32_t metadataBufferCount = 1U;
    unpermute.unpermuteTileCols =
        ChooseUnpermuteTileCols(cfg.k, unpermute.unpermuteTokenBatch, cfg.topk, metadataBufferCount);
    unpermute.unpermuteInputBufferCount = ChooseUnpermuteInputBufferCount(
        unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols, metadataBufferCount);
    RequireUnpermuteUbCapacity(
        unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols, metadataBufferCount,
        unpermute.unpermuteInputBufferCount);
}

void PopulateMultiServerTiling(MegaMoeBuildResult& result, const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    if (!runtime.urma_enabled)
        return;
    auto& multi = result.tiling.multiServerTiling;
    const auto& front = result.tiling.frontReorderTiling;
    const uint32_t ranks = cfg.world_size;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t perServer = runtime.rank_num_per_server;
    if (ranks != static_cast<uint32_t>(runtime.hccl.world_size) || rank >= ranks || perServer == 0U ||
        perServer > ranks || ranks % perServer != 0U || runtime.urma_workspace == nullptr || cfg.aic_num == 0U ||
        runtime.urma_jetty_count == 0U || runtime.urma_jetty_count % cfg.aic_num != 0U) {
        throw std::runtime_error("URMA topology/workspace is incomplete");
    }
    multi.enabled = 1U;
    multi.rankNumPerServer = perServer;
    multi.serverNum = ranks / perServer;
    multi.serverId = rank / perServer;
    multi.urmaWorkspace = reinterpret_cast<uint64_t>(runtime.urma_workspace);
    multi.urmaJettyCount = runtime.urma_jetty_count;
    multi.readySlotBytes = 64U;
    const uint64_t masks = CheckedMul(cfg.expert_per_rank, front.maskSlotBytes, "URMA mask block");
    const uint64_t counts = AlignUp(masks, 32U);
    const uint64_t preSum = AlignUp(CheckedAdd(counts, cfg.expert_per_rank * 4ULL, "URMA counts"), 32U);
    const uint64_t block = AlignUp(CheckedAdd(preSum, cfg.expert_per_rank * 4ULL, "URMA preSum"), 512U);
    if (block > UINT32_MAX)
        throw std::runtime_error("URMA metadata block exceeds WQE size field");
    multi.metadataCountOffsetBytes = static_cast<uint32_t>(counts);
    multi.metadataPreSumOffsetBytes = static_cast<uint32_t>(preSum);
    multi.metadataBlockBytes = block;
    uint64_t offset = AlignUp(
        CheckedAdd(
            front.preSumBeforeRankPeerOffset,
            AlignUp(CheckedMul(ranks, cfg.expert_per_rank * 4ULL, "preSum end"), 512U), "peer end"),
        512U);
    multi.metadataSendOffset = AllocateAlignedSection(offset, CheckedMul(ranks, block, "URMA send"), "URMA send");
    multi.metadataRecvOffset = AllocateAlignedSection(offset, CheckedMul(ranks, block, "URMA recv"), "URMA recv");
    multi.metadataReadyOffset = AllocateAlignedSection(offset, ranks * 64ULL, "URMA front epoch");
    multi.startReadyOffset = AllocateAlignedSection(offset, ranks * 64ULL, "URMA start epoch");
    multi.relayChunkTokens = 256U;
    multi.relayChunkCount = CeilDivU32(cfg.m, multi.relayChunkTokens);
    multi.relaySourceStrideBytes = AlignUp(CheckedMul(cfg.m, front.packedRowStride, "URMA token rows"), 512U);
    multi.relayDataOffset = AllocateAlignedSection(
        offset, CheckedMul(multi.serverNum, multi.relaySourceStrideBytes, "URMA relay"), "URMA relay");
    multi.combineTxChunkRows = 16U;
    multi.combineTxSlotCount = 2U;
    multi.combineTxOwnerStrideBytes = AlignUp(CheckedMul(cfg.k, 2ULL * 16U * 2U, "URMA TX slots"), 512U);
    multi.combineTxStagingOffset = AllocateAlignedSection(
        offset, CheckedMul(ranks, multi.combineTxOwnerStrideBytes, "URMA TX owners"), "URMA TX owners");
    multi.transportReadyOffset = AllocateAlignedSection(offset, ranks * 64ULL, "URMA transport epoch");
    multi.finalizeReadyOffset = AllocateAlignedSection(offset, 64U, "URMA final epoch");
    multi.peerDataBytes = offset;
    if (runtime.hccl.WindowBytes() < MB_SIZE || offset > runtime.hccl.WindowBytes() - MB_SIZE) {
        throw std::runtime_error(
            "HCCL window too small for URMA: dataBytes=" + std::to_string(offset) +
            " windowBytes=" + std::to_string(runtime.hccl.WindowBytes()));
    }
    const uint64_t maxReceiveRows = std::min<uint64_t>(
        cfg.max_output_size,
        CheckedMul(CheckedMul(ranks, cfg.m, "URMA global tokens"), cfg.topk, "URMA global routes"));
    const uint64_t combineStagingBytes =
        AlignUp(CheckedMul(maxReceiveRows, cfg.k * 2ULL, "URMA Combine workspace"), 512U);
    multi.combineStagingOffset =
        AllocateAlignedSection(result.workspace_bytes, combineStagingBytes, "URMA Combine workspace");
}

} // namespace

const A5CoreTopology* FindA5CoreTopology(uint32_t effectiveAicNum)
{
    for (const A5CoreTopology& schedule : kA5CoreTopologies) {
        if (schedule.physicalAicNum == effectiveAicNum) {
            return &schedule;
        }
    }
    return nullptr;
}

MegaMoeBuildResult BuildMegaMoeTiling(const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    const A5CoreTopology* schedule = FindA5CoreTopology(cfg.aic_num);
    if (schedule == nullptr) {
        throw std::runtime_error(
            "unsupported A5 AICore count " + std::to_string(cfg.aic_num) + "; supported counts are 28, 32, and 36");
    }
    if (cfg.aiv_num != schedule->PhysicalAivNum()) {
        throw std::runtime_error("AIV count does not match the selected A5 AICore topology");
    }
    const uint32_t maxDispatchRanks = std::min(kMegaMoeFixedDispatchGroupSize, schedule->physicalAicNum - 2U);
    if (runtime.hccl.world_size <= 0 || static_cast<uint32_t>(runtime.hccl.world_size) > maxDispatchRanks) {
        throw std::runtime_error(
            "fixed-group dispatch requires a positive rank size no larger than " + std::to_string(maxDispatchRanks) +
            " for the selected topology");
    }
    if (cfg.expert_per_rank < 2U || cfg.expert_per_rank > kMegaMoeFixedMaxExperts) {
        throw std::runtime_error("expert count exceeds runtime GMM policy capacity");
    }
    RequireMxShape(cfg.k, cfg.n);
    RequireDispatchPackedRowCapacity(MxPackedRowStride(cfg.k));
    RequireFrontQuantUbCapacity(cfg.k);

    MegaMoeBuildResult result;
    result.block_dim = cfg.aic_num;
    result.tiling.megaMoeInfo = {
        .M = cfg.m,
        .K = cfg.k,
        .N = cfg.n,
        .expertPerRank = cfg.expert_per_rank,
        .topK = cfg.topk,
        .maxOutputSize = cfg.max_output_size,
    };
    result.tiling.runtimeInfo = {
        .remoteWindowContext = reinterpret_cast<uint64_t>(runtime.hccl.RemoteWindowContextPtr()),
        .rank = static_cast<uint32_t>(runtime.hccl.rank_id),
        .rankSize = static_cast<uint32_t>(runtime.hccl.world_size),
    };
    const uint64_t frontWorkspaceBytes = PopulateFrontTiling(result.tiling.frontReorderTiling, cfg, runtime);
    PopulateDispatchBufferTiling(result.tiling.dispatchTiling, result.tiling.frontReorderTiling);
    result.workspace_bytes = AllocatePipelineWorkspace(result.tiling, cfg, frontWorkspaceBytes);
    AllocateFixedGroupWorkspace(result.tiling, *schedule, cfg, result.workspace_bytes);
    PopulateUnpermuteTiling(result.tiling, cfg);
    PopulateMultiServerTiling(result, cfg, runtime);
    return result;
}
