/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_LAYOUT_H_
#define MOE_COMBINE_LAYOUT_H_

#include "common.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace moe_combine {

constexpr uint64_t kSyncSoftSlotInt32 = 8;

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        throw std::invalid_argument("alignment must be nonzero");
    }
    uint64_t rem = value % alignment;
    if (rem == 0) {
        return value;
    }
    uint64_t add = alignment - rem;
    if (value > std::numeric_limits<uint64_t>::max() - add) {
        throw std::overflow_error("AlignUp overflow");
    }
    return value + add;
}

inline uint64_t CheckedMul(uint64_t a, uint64_t b, const char* label)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return a * b;
}

inline uint64_t EffectiveAivBlocks(const MoeCombineShape& shape) { return shape.aivBlocks == 0 ? 1 : shape.aivBlocks; }

inline uint64_t ExpertNumPadded(const MoeCombineShape& shape)
{
    return AlignUp(shape.expertNum, kMoeCombineMetadataPad);
}

inline uint64_t AppendField(uint64_t* offset, uint64_t bytes)
{
    *offset = AlignUp(*offset, 64);
    uint64_t fieldOffset = *offset;
    if (*offset > std::numeric_limits<uint64_t>::max() - bytes) {
        throw std::overflow_error("layout byte overflow");
    }
    *offset += bytes;
    return fieldOffset;
}

inline WorkspaceLayout ComputeWorkspaceLayout(const MoeCombineShape& shape)
{
    constexpr uint64_t kI32 = 4;
    uint64_t aivBlocks = EffectiveAivBlocks(shape);
    uint64_t expertNumPadded = ExpertNumPadded(shape);
    uint64_t offset = 0;
    WorkspaceLayout layout{};
    uint64_t syncSlots = CheckedMul(aivBlocks, kSyncSoftSlotInt32 + expertNumPadded, "localSync slots");
    if (syncSlots < 64) {
        syncSlots = 64;
    }
    layout.localSync = AppendField(&offset, CheckedMul(syncSlots, kI32, "localSync"));
    layout.totalBytes = AlignUp(offset, 64);
    return layout;
}

inline CombineRouteMetaLayout ComputeCombineRouteMetaLayout(const MoeCombineShape& shape)
{
    constexpr uint64_t kI32 = 4;
    uint64_t expertNumPadded = ExpertNumPadded(shape);
    uint64_t expandedRows = CheckedMul(shape.m, shape.topK, "expanded rows");
    uint64_t offset = 0;
    CombineRouteMetaLayout layout{};
    layout.peerTokenPerExpert = AppendField(
        &offset,
        CheckedMul(
            CheckedMul(shape.ep, expertNumPadded, "peerTokenPerExpert elems"), kI32, "peerTokenPerExpert bytes"));
    layout.expandedRowIdx = AppendField(&offset, CheckedMul(expandedRows, kI32, "expandedRowIdx"));
    layout.cumsumPerExpert = AppendField(
        &offset,
        CheckedMul(CheckedMul(shape.ep, expertNumPadded, "cumsumPerExpert elems"), kI32, "cumsumPerExpert bytes"));
    layout.dispatchOffset = AppendField(&offset, CheckedMul(shape.expertPerRank, kI32, "dispatchOffset"));
    layout.prevSumBeforeRank = AppendField(
        &offset,
        CheckedMul(
            CheckedMul(shape.ep, shape.expertPerRank, "prevSumBeforeRank elems"), kI32, "prevSumBeforeRank bytes"));
    layout.totalBytes = AlignUp(offset, 64);
    return layout;
}

inline PeerWindowLayout ComputePeerWindowLayout(const MoeCombineShape& shape)
{
    constexpr uint64_t kI32 = 4;
    constexpr uint64_t kHalf = 2;
    uint64_t expandedRows = CheckedMul(shape.m, shape.topK, "expanded rows");
    uint64_t offset = 0;
    PeerWindowLayout layout{};
    layout.ptrD =
        AppendField(&offset, CheckedMul(CheckedMul(expandedRows, shape.k, "ptrD elems"), kHalf, "ptrD bytes"));
    layout.countReadySignal = AppendField(&offset, CheckedMul(shape.ep, kI32, "countReadySignal"));
    layout.combineDoneSignal = AppendField(&offset, CheckedMul(shape.ep, kI32, "combineDoneSignal"));
    layout.totalBytes = AlignUp(offset, 64);
    return layout;
}

inline uint64_t EstimateHcclBuffSizeMb(const MoeCombineShape&, const PeerWindowLayout& peerWindowLayout)
{
    constexpr uint64_t kMiB = 1024ULL * 1024ULL;
    constexpr uint64_t kSafetyMargin = 64ULL * kMiB;
    uint64_t bytes = peerWindowLayout.totalBytes + kSafetyMargin;
    return AlignUp(bytes, kMiB) / kMiB;
}

} // namespace moe_combine

#endif // MOE_COMBINE_LAYOUT_H_
