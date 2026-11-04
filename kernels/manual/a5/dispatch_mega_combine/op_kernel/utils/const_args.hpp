/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP

#include <cstdint>

#include "pto/common/buffer_limits.hpp"

constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t UB_ALIGN = 32;

struct AtlasA5 {
    static constexpr uint32_t UB_SIZE = PTO_UBUF_SIZE_BYTES;
    static constexpr uint32_t L1_SIZE = PTO_CBUF_SIZE_BYTES;
    static constexpr uint32_t L0A_SIZE = PTO_L0A_SIZE_BYTES;
    static constexpr uint32_t L0B_SIZE = PTO_L0B_SIZE_BYTES;
    static constexpr uint32_t L0C_SIZE = PTO_L0C_SIZE_BYTES;
};

// Compile-time capacities for synchronization and remote-window layouts. The
// selected 28/32/36-core split is carried in MegaMoeFixedGroupTiling at runtime.
constexpr uint32_t kMegaMoeFixedPhysicalAicNum = 36U;
constexpr uint32_t kMegaMoeFixedAivSubblocksPerPhysicalBlock = 2U;
constexpr uint32_t kMegaMoeFixedPhysicalAivNum =
    kMegaMoeFixedPhysicalAicNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
constexpr uint32_t kMegaMoeFixedGmm1GroupSize = 24U;
constexpr uint32_t kMegaMoeFixedGmm2GroupSize = 16U;
constexpr uint32_t kMegaMoeFixedDispatchGroupSize = 32U;
// Every physical AIC can produce a CV tile during the full-AIC GMM1 waves;
// reserve the SwiGLU synchronization lanes for the complete AIC set.
constexpr uint32_t kMegaMoeFixedSwigluGroupSize = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedUnpermuteGroupSize = kMegaMoeFixedPhysicalAivNum;
// Peer-signal slots have fixed offsets; the host selects phase-1 workers.
constexpr uint32_t kMegaMoeFixedInitialUnpermuteAiv0WorkerCapacity = 32U;
constexpr uint32_t kMegaMoeExpertProgressMaxRanks = 32U;
constexpr uint32_t kMegaMoeFrontMaskCountRecordBytes = 32U;
constexpr uint32_t kMegaMoeRouteMetaFields = 8U;
constexpr uint32_t kMegaMoeMxGroupSize = 32U;
constexpr uint32_t kMegaMoeMxScalePrefetchK = 4096U;

constexpr uint16_t kMegaMoeFixedSecondAivSubblockFlagOffset = 16U;

constexpr uint32_t A5_UB_SYNC_RESERVE_BYTES = 40U * 1024U;
constexpr uint32_t A5_MAIN_UB_SIZE = AtlasA5::UB_SIZE - A5_UB_SYNC_RESERVE_BYTES;

static_assert(kMegaMoeFixedAivSubblocksPerPhysicalBlock == 2U);
static_assert(kMegaMoeFixedGmm1GroupSize < kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedGmm2GroupSize < kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedDispatchGroupSize >= kMegaMoeFixedGmm1GroupSize);
static_assert(kMegaMoeFixedDispatchGroupSize < kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedSwigluGroupSize == kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedUnpermuteGroupSize == kMegaMoeFixedPhysicalAivNum);
static_assert(kMegaMoeFixedInitialUnpermuteAiv0WorkerCapacity <= kMegaMoeFixedPhysicalAicNum);
static_assert(AtlasA5::UB_SIZE == 256U * 1024U);
static_assert(A5_MAIN_UB_SIZE == 216U * 1024U);

#endif // CONST_ARGS_HPP
