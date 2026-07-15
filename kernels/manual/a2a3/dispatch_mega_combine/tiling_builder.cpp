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

namespace {
void RequirePositive(const char* name, uint32_t value)
{
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void RequireInt8RowAligned(uint32_t k)
{
    constexpr uint32_t kDataBlockBytes = 32;
    if (k % kDataBlockBytes != 0) {
        throw std::runtime_error("K must be 32-byte aligned for int8 offsetA rows");
    }
}

void RequirePackedOffsetACapacity(const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    constexpr uint64_t kPackedScalePadBytes = 32;
    const uint64_t rawOffsetABytes = runtime.hccl.WindowBytes() / 3U;
    const uint64_t offsetABytes = (rawOffsetABytes + 511U) / 512U * 512U;
    const uint64_t requiredBytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.k + kPackedScalePadBytes);
    if (requiredBytes > offsetABytes) {
        throw std::runtime_error(
            "HCCL window offsetA section is too small for packed rows: windowBytes=" +
            std::to_string(runtime.hccl.WindowBytes()) + " offsetABytes=" + std::to_string(offsetABytes) +
            " requiredBytes=" + std::to_string(requiredBytes) +
            " maxOutputSize=" + std::to_string(cfg.max_output_size) + " K=" + std::to_string(cfg.k));
    }
}

void RequireDispatchTileCapacity(uint64_t dispatchTileBytes)
{
    constexpr uint64_t kMaxDispatchPingPongBytes = 128U * 1024U;
    if (dispatchTileBytes * 2U > kMaxDispatchPingPongBytes) {
        throw std::runtime_error("dispatch tile ping-pong buffers exceed UB budget");
    }
}

void RequireUnpermuteUbCapacity(uint32_t tokenBatch, uint32_t topk, uint32_t tileCols)
{
    constexpr uint64_t kMaxUnpermuteMainUbBytes = 128U * 1024U;
    auto alignUpBytes = [](uint64_t value) { return (value + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN; };
    uint64_t ubBytes = 0;
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tokenBatch) * topk * sizeof(int32_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(tokenBatch) * topk * sizeof(float));
    }
    ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    for (uint32_t i = 0; i < 2U; ++i) {
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
        ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(float));
    }
    ubBytes += alignUpBytes(static_cast<uint64_t>(tileCols) * sizeof(uint16_t));
    if (ubBytes > kMaxUnpermuteMainUbBytes) {
        throw std::runtime_error("unpermute UB buffers exceed 128 KiB main budget");
    }
}

uint32_t CalcMixAic1To2BlockDim(uint32_t aic_num, uint32_t aiv_num)
{
    if (aiv_num > aic_num * 2U) {
        throw std::runtime_error("direct mixed launch expects aiv_num <= aic_num * 2");
    }
    return aic_num;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        throw std::runtime_error("AlignUp requires nonzero align");
    }
    return (value + align - 1) / align * align;
}

uint32_t DivCeil(uint32_t value, uint32_t divisor) { return divisor == 0U ? 0U : (value + divisor - 1U) / divisor; }

uint32_t Pow4Ceil(uint32_t value)
{
    if (value <= 1U) {
        return 1U;
    }
    uint32_t out = 1U;
    while (out < value && out <= UINT32_MAX / 4U) {
        out *= 4U;
    }
    return out;
}

constexpr uint32_t kFrontMaxColsOneLoopQuant = 8192U;
constexpr uint32_t kFrontSortAlignElems = 32U;
constexpr uint32_t kFrontSortOutLoopMaxElems = 2040U;

void PopulateFrontSortLoopFields(MegaMoeFrontReorderTiling& front)
{
    if (front.sortNeedCoreNum == 0U || front.sortPerCoreElems == 0U || front.sortLastCoreElems == 0U ||
        front.sortLoopMaxElement == 0U) {
        return;
    }
    front.sortPerCoreLoops = DivCeil(front.sortPerCoreElems, front.sortLoopMaxElement);
    front.sortPerCorePerLoopElems = std::min(front.sortPerCoreElems, front.sortLoopMaxElement);
    front.sortPerCoreLastLoopElems =
        front.sortPerCoreElems - (front.sortPerCoreLoops - 1U) * front.sortPerCorePerLoopElems;
    front.sortLastCoreLoops = front.sortPerCoreLoops;
    const uint32_t lastCoreAvgElems = DivCeil(front.sortLastCoreElems, front.sortLastCoreLoops);
    front.sortLastCorePerLoopElems = DivCeil(lastCoreAvgElems, kFrontSortAlignElems) * kFrontSortAlignElems;
    const uint64_t lastCoreLoopConsumed =
        static_cast<uint64_t>(front.sortLastCoreLoops - 1U) * front.sortLastCorePerLoopElems;
    front.sortLastCoreLastLoopElems = lastCoreLoopConsumed >= front.sortLastCoreElems ?
                                          0U :
                                          static_cast<uint32_t>(front.sortLastCoreElems - lastCoreLoopConsumed);
}

uint32_t FrontSortLoopMaxElement()
{
    constexpr uint32_t kSort32AlignElement = 32U;
    constexpr uint32_t kFrontSortBytesPerRouteElem = sizeof(int32_t) * 2U * 4U;
    return static_cast<uint32_t>(
        AtlasA2::UB_SIZE / kFrontSortBytesPerRouteElem / kSort32AlignElement * kSort32AlignElement);
}

uint32_t FrontAlignedRouteElems(uint32_t routeElems)
{
    constexpr uint64_t kFrontRouteAlign = 128U;
    return static_cast<uint32_t>(AlignUp(routeElems, kFrontRouteAlign));
}

uint64_t FullLoadDynamicUbBudgetBytes(uint32_t routeElems, uint32_t k, uint32_t expertNum)
{
    constexpr uint64_t kOneCoreSortBuffer = 6U;
    constexpr uint64_t kOtherRouteBuffer = 3U;
    constexpr uint64_t kDynamicQuantFullLoadColsBuffer = 13U;
    constexpr uint64_t kScaleOutBytes = 64U;
    const uint64_t alignedRouteElems = AlignUp(routeElems, UB_ALIGN);
    const uint64_t sortSpace = alignedRouteElems * sizeof(int32_t) * kOneCoreSortBuffer;
    const uint64_t otherSpace = alignedRouteElems * sizeof(int32_t) * kOtherRouteBuffer;
    const uint64_t expertSpace = AlignUp(static_cast<uint64_t>(expertNum) * sizeof(int32_t), UB_ALIGN);
    const uint64_t quantSpace = AlignUp(static_cast<uint64_t>(k), UB_ALIGN) * kDynamicQuantFullLoadColsBuffer;
    return sortSpace + otherSpace + expertSpace + quantSpace + kScaleOutBytes;
}

uint64_t FrontRouteWorkspaceBytes(uint32_t alignedRouteElems)
{
    const uint64_t sortedIntBytes = AlignUp(static_cast<uint64_t>(alignedRouteElems) * 2U * sizeof(int32_t), 512U);
    const uint64_t packedRunBytes = AlignUp(static_cast<uint64_t>(alignedRouteElems) * 2U * sizeof(float), 512U);
    return std::max(sortedIntBytes, packedRunBytes);
}

bool IsFrontFullLoadDynamic(const CaseConfig& cfg, uint32_t routeElems, uint32_t sortLoopMaxElement)
{
    if (routeElems == 0U || routeElems > sortLoopMaxElement || cfg.k > kFrontMaxColsOneLoopQuant ||
        cfg.k % UB_ALIGN != 0U) {
        return false;
    }
    const uint32_t expertNum = cfg.world_size * cfg.expert_per_rank;
    return FullLoadDynamicUbBudgetBytes(routeElems, cfg.k, expertNum) <= AtlasA2::UB_SIZE;
}

uint32_t SelectFrontCase(const CaseConfig& cfg, uint32_t routeElems, uint32_t sortLoopMaxElement)
{
    if (routeElems == 0U) {
        return 0U;
    }
    if (IsFrontFullLoadDynamic(cfg, routeElems, sortLoopMaxElement)) {
        return kFrontCaseFullLoadDynamic;
    }
    if (routeElems <= sortLoopMaxElement) {
        return kFrontCaseOneCoreDynamic;
    }
    return kFrontCaseMultiCoreDynamic;
}

void PopulateFrontSortSplit(MegaMoeFrontReorderTiling& front, uint32_t aivNum)
{
    front.sortOutLoopMaxElems = static_cast<uint16_t>(kFrontSortOutLoopMaxElems);
    if (front.routeElems == 0U) {
        return;
    }
    if (front.frontCase != kFrontCaseMultiCoreDynamic || aivNum == 0U) {
        front.sortNeedCoreNum = static_cast<uint16_t>(1U);
        front.sortPerCoreElems = front.routeElems;
        front.sortLastCoreElems = front.routeElems;
        PopulateFrontSortLoopFields(front);
        return;
    }

    uint32_t needCoreNum = DivCeil(front.routeElems, front.sortLoopMaxElement);
    needCoreNum = std::min<uint32_t>(Pow4Ceil(needCoreNum), aivNum);
    if (needCoreNum == 0U) {
        return;
    }

    uint32_t perCoreElems = front.routeElems / needCoreNum;
    uint32_t floorPerCoreElems = perCoreElems - perCoreElems % kFrontSortAlignElems;
    if (floorPerCoreElems == 0U) {
        floorPerCoreElems = kFrontSortAlignElems;
    }
    const uint32_t lastCoreElemsWithFloor = front.routeElems - (needCoreNum - 1U) * floorPerCoreElems;
    uint32_t ceilPerCoreElems = perCoreElems + kFrontSortAlignElems - perCoreElems % kFrontSortAlignElems;
    if (perCoreElems % kFrontSortAlignElems == 0U) {
        // reference uses the next aligned block for the ceil-side capacity probe.
        ceilPerCoreElems = perCoreElems + kFrontSortAlignElems;
    }
    if (lastCoreElemsWithFloor > ceilPerCoreElems) {
        perCoreElems = ceilPerCoreElems;
        needCoreNum = DivCeil(front.routeElems, perCoreElems);
    } else {
        perCoreElems = floorPerCoreElems;
    }

    while (perCoreElems >= kFrontSortAlignElems) {
        front.sortNeedCoreNum = static_cast<uint16_t>(needCoreNum);
        front.sortPerCoreElems = perCoreElems;
        front.sortLastCoreElems = front.routeElems - (front.sortNeedCoreNum - 1U) * front.sortPerCoreElems;
        PopulateFrontSortLoopFields(front);
        if (front.sortLastCoreLastLoopElems != 0U || perCoreElems <= kFrontSortAlignElems) {
            break;
        }
        perCoreElems -= kFrontSortAlignElems;
    }
}

void RequireAlignedRange(const char* name, uint64_t offset, uint64_t bytes)
{
    if (offset % 512U != 0U || bytes % 512U != 0U) {
        throw std::runtime_error(std::string(name) + " must be 512-byte aligned");
    }
}

void PopulateMegaMoeInfo(MegaMoeInfo& info, const CaseConfig& cfg)
{
    info.M = cfg.m;
    info.K = cfg.k;
    info.N = cfg.n;
    info.topK = cfg.topk;
    info.expertPerRank = cfg.expert_per_rank;
    info.worldSize = cfg.world_size;
    info.maxOutputSize = cfg.max_output_size;
    info.aivNum = cfg.aiv_num;
}

void PopulateRuntimeInfo(MegaMoeRuntimeInfo& runtimeInfo, const StandaloneRankRuntime& runtime)
{
    runtimeInfo.remoteWindowContext = reinterpret_cast<uint64_t>(runtime.hccl.RemoteWindowContextPtr());
    runtimeInfo.rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    runtimeInfo.rankSize = static_cast<uint32_t>(runtime.hccl.world_size);
}

void PopulateFrontTiling(MegaMoeFrontReorderTiling& front, const CaseConfig& cfg)
{
    const uint64_t expertNum = static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank;
    front.expertNum = static_cast<uint32_t>(expertNum);
    front.expertNumAligned = static_cast<uint32_t>(AlignUp(expertNum + 1U, 128U));
    front.stageNum = static_cast<uint16_t>(14U);
    front.routeElems = static_cast<uint32_t>(static_cast<uint64_t>(cfg.m) * cfg.topk);
    front.alignedRouteElems = FrontAlignedRouteElems(front.routeElems);
    front.sortLoopMaxElement = FrontSortLoopMaxElement();
    front.frontCase = static_cast<uint16_t>(SelectFrontCase(cfg, front.routeElems, front.sortLoopMaxElement));
    PopulateFrontSortSplit(front, cfg.aiv_num);
    front.expandedRowIdxOffset = 0;
}

uint64_t AllocateFrontRouteWorkspace(MegaMoeFrontReorderTiling& front, uint64_t frontWorkspaceOffset)
{
    frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
    front.frontExpandedExpertOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U);

    frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
    front.frontExpandDstToSrcOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U);

    frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
    front.frontSortWs0Offset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += FrontRouteWorkspaceBytes(front.alignedRouteElems);

    frontWorkspaceOffset = AlignUp(frontWorkspaceOffset, 512U);
    front.frontSortWs1Offset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += FrontRouteWorkspaceBytes(front.alignedRouteElems);

    RequireAlignedRange(
        "frontExpandedExpert", front.frontExpandedExpertOffset,
        AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U));
    RequireAlignedRange(
        "frontExpandDstToSrc", front.frontExpandDstToSrcOffset,
        AlignUp(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U));
    RequireAlignedRange("frontSortWs0", front.frontSortWs0Offset, FrontRouteWorkspaceBytes(front.alignedRouteElems));
    RequireAlignedRange("frontSortWs1", front.frontSortWs1Offset, FrontRouteWorkspaceBytes(front.alignedRouteElems));
    return frontWorkspaceOffset;
}

void AllocateFrontWorkspace(MegaMoeFrontReorderTiling& front, const CaseConfig& cfg)
{
    const uint64_t expandedRowIdxBytes = ((cfg.m + 255) / 256) * 256 * cfg.topk * sizeof(int32_t);
    uint64_t frontWorkspaceOffset = expandedRowIdxBytes;
    front.localTokenPerExpertOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(front.expertNumAligned) * sizeof(int32_t);
    front.cumsumMMOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(int32_t);
    front.preSumBeforeRankOffset = static_cast<uint32_t>(frontWorkspaceOffset);
    frontWorkspaceOffset += static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(int32_t);
    if (front.routeElems != 0U) {
        frontWorkspaceOffset = AllocateFrontRouteWorkspace(front, frontWorkspaceOffset);
    }
    front.frontWorkspaceBytes = static_cast<uint32_t>(AlignUp(frontWorkspaceOffset, 512U));
}

void PopulateDispatchScratch(MegaMoeDispatchTiling& dispatch, const CaseConfig& cfg, uint64_t& workspaceOffset)
{
    dispatch.dispatchTileBytes = std::max<uint64_t>(32U * 1024U, AlignUp(static_cast<uint64_t>(cfg.k) + 32U, 32U));
    RequireDispatchTileCapacity(dispatch.dispatchTileBytes);
    dispatch.dispatchGatherTileBytes = dispatch.dispatchTileBytes;
    dispatch.dispatchGatherScratchBytesPerAiv = AlignUp(dispatch.dispatchTileBytes * 2U, 512U);
    dispatch.dispatchGatherScratchOffset = workspaceOffset;
    dispatch.dispatchGatherScratchBytes =
        static_cast<uint64_t>(cfg.aiv_num) * dispatch.dispatchGatherScratchBytesPerAiv;
    RequireAlignedRange(
        "dispatchGatherScratch", dispatch.dispatchGatherScratchOffset, dispatch.dispatchGatherScratchBytes);
    workspaceOffset = AlignUp(workspaceOffset + dispatch.dispatchGatherScratchBytes, 512U);
}

void PopulateSwigluMetadata(MegaMoeSwigluTiling& swiglu, const CaseConfig& cfg, uint64_t& workspaceOffset)
{
    const uint32_t swigluSegmentNum = MoeSwigluSegmentNum(cfg.expert_per_rank);
    swiglu.swigluSegmentMetaOffset = workspaceOffset;
    swiglu.swigluSegmentMetaBytes =
        AlignUp(static_cast<uint64_t>(swigluSegmentNum) * sizeof(MegaMoeSwigluSegmentRuntimeMeta), 512U);
    workspaceOffset = AlignUp(workspaceOffset + swiglu.swigluSegmentMetaBytes, 512U);
}

uint64_t AllocatePipelineWorkspace(MegaMoeTilingData& tiling, const CaseConfig& cfg)
{
    auto& dispatch = tiling.dispatchTiling;
    auto& swiglu = tiling.swigluTiling;
    auto& gmm1 = tiling.gmm1Tiling;
    auto& gmm2 = tiling.gmm2Tiling;
    auto& combine = tiling.combineTiling;

    uint64_t workspaceOffset = tiling.frontReorderTiling.frontWorkspaceBytes;
    dispatch.perTokenScaleOffset = workspaceOffset;
    workspaceOffset = AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * sizeof(float), 512U);
    swiglu.perTokenScale2Offset = workspaceOffset;
    workspaceOffset = AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * sizeof(float), 512U);
    dispatch.gmAOffset = workspaceOffset;
    workspaceOffset =
        AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t), 512U);
    swiglu.gmPermutedTokenOffset = workspaceOffset;
    workspaceOffset =
        AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * (cfg.n / 2) * sizeof(int8_t), 512U);
    gmm1.gmCOffset = workspaceOffset;
    workspaceOffset =
        AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(int16_t), 512U);
    gmm2.gmm2OutputOffset = workspaceOffset;
    workspaceOffset =
        AlignUp(workspaceOffset + static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int16_t), 512U);

    combine.gmm2OutputOffset = gmm2.gmm2OutputOffset;
    combine.perTokenScale2Offset = swiglu.perTokenScale2Offset;
    PopulateDispatchScratch(dispatch, cfg, workspaceOffset);
    PopulateSwigluMetadata(swiglu, cfg, workspaceOffset);
    return workspaceOffset;
}

void PopulateUnpermuteTiling(MegaMoeUnpermuteTiling& unpermute, const CaseConfig& cfg)
{
    unpermute.unpermuteTileCols = 2048U;
    unpermute.unpermuteTokenBatch = 256U;
    RequireUnpermuteUbCapacity(unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols);
}

} // namespace

MegaMoeBuildResult BuildMegaMoeTiling(const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    RequirePositive("aic_num", cfg.aic_num);
    RequirePositive("aiv_num", cfg.aiv_num);
    RequireInt8RowAligned(cfg.k);
    RequirePackedOffsetACapacity(cfg, runtime);

    MegaMoeBuildResult result;
    result.block_dim = CalcMixAic1To2BlockDim(cfg.aic_num, cfg.aiv_num);
    PopulateMegaMoeInfo(result.tiling.megaMoeInfo, cfg);
    PopulateRuntimeInfo(result.tiling.runtimeInfo, runtime);
    PopulateFrontTiling(result.tiling.frontReorderTiling, cfg);
    AllocateFrontWorkspace(result.tiling.frontReorderTiling, cfg);
    result.workspace_bytes = AllocatePipelineWorkspace(result.tiling, cfg);
    PopulateUnpermuteTiling(result.tiling.unpermuteTiling, cfg);
    return result;
}
