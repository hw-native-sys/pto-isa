/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DISPATCH_WORKER_PARTITION_HPP
#define DISPATCH_MEGA_COMBINE_DISPATCH_WORKER_PARTITION_HPP

#include <cstdint>

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE __aicore__ inline
#else
#define MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE inline
#endif

struct DispatchRankWorkerRange {
    uint32_t begin = 0U;
    uint32_t count = 0U;
};

struct DispatchWorkerAssignment {
    uint32_t rank = 0U;
    uint32_t shardIndex = 0U;
    uint32_t shardCount = 0U;
};

// The first workerCount % rankSize ranks own one extra worker. Keep ownership
// contiguous and identical on every consumer; every worker belongs to one rank.
MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE DispatchRankWorkerRange
GetDispatchRankWorkerRange(uint32_t workerCount, uint32_t rankSize, uint32_t rank)
{
    DispatchRankWorkerRange range;
    if (rankSize == 0U || workerCount < rankSize || rank >= rankSize) {
        return range;
    }
    const uint32_t base = workerCount / rankSize;
    const uint32_t remainder = workerCount % rankSize;
    range.begin = rank * base + (rank < remainder ? rank : remainder);
    range.count = base + (rank < remainder ? 1U : 0U);
    return range;
}

// Invert the quotient/remainder partition once per Dispatch worker, not once
// per token. Unlike uniform division, this does not discard remainder lanes.
MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE DispatchWorkerAssignment
GetDispatchWorkerAssignment(uint32_t workerIndex, uint32_t workerCount, uint32_t rankSize)
{
    DispatchWorkerAssignment assignment;
    if (rankSize == 0U || workerCount < rankSize || workerIndex >= workerCount) {
        return assignment;
    }
    const uint32_t base = workerCount / rankSize;
    const uint32_t remainder = workerCount % rankSize;
    const uint32_t wideWorkerCount = (base + 1U) * remainder;
    if (workerIndex < wideWorkerCount) {
        assignment.rank = workerIndex / (base + 1U);
        assignment.shardIndex = workerIndex % (base + 1U);
        assignment.shardCount = base + 1U;
    } else {
        const uint32_t narrowWorkerIndex = workerIndex - wideWorkerCount;
        assignment.rank = remainder + narrowWorkerIndex / base;
        assignment.shardIndex = narrowWorkerIndex % base;
        assignment.shardCount = base;
    }
    return assignment;
}

// Row shards may be empty or cross source-rank boundaries.
MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE uint32_t
DispatchRowShardBegin(uint32_t rows, uint32_t shardIndex, uint32_t shardCount)
{
    return shardCount == 0U ? 0U : static_cast<uint32_t>((static_cast<uint64_t>(rows) * shardIndex) / shardCount);
}

#undef MEGA_MOE_DISPATCH_HOST_DEVICE_INLINE

#endif // DISPATCH_MEGA_COMBINE_DISPATCH_WORKER_PARTITION_HPP
