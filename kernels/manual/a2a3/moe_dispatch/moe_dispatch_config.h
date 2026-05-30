/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

// ============================================================================
// MoE Dispatch Operator Configuration
//
// Standalone Dispatch communication operator for MegaMoE PTO-ISA validation.
// Pulls quantized tokens from remote ranks' shmem into local workspace,
// separating token data and per-token scales.
// ============================================================================

// Default shape parameters (overridable via cmake -D)
#ifndef CONFIG_EP
#define CONFIG_EP 2
#endif

#ifndef CONFIG_EXPERT_PER_RANK
#define CONFIG_EXPERT_PER_RANK 1
#endif

#ifndef CONFIG_HIDDEN_SIZE
#define CONFIG_HIDDEN_SIZE 128
#endif

#ifndef CONFIG_MAX_TOKENS_PER_RANK
#define CONFIG_MAX_TOKENS_PER_RANK 64
#endif

#ifndef CONFIG_MAX_OUTPUT_SIZE
#define CONFIG_MAX_OUTPUT_SIZE 512
#endif

#ifndef CONFIG_FIRST_DEVICE_ID
#define CONFIG_FIRST_DEVICE_ID 0
#endif

// Hardware constants (matching MegaMoE reference implementation)
static constexpr int32_t UB_ALIGN = 32;
static constexpr int32_t UB_HALF_SIZE = 96 * 1024;
static constexpr int32_t UB_MOVE_NUM_MAX = 16;

// Backward compat: existing code referencing UB_MOVE_NUM still compiles
static constexpr int32_t UB_MOVE_NUM = UB_MOVE_NUM_MAX;

// Compile-time helper: compute actual MOVE_NUM based on UB capacity
template <int TILE_COLS>
struct DispatchTraits {
    static constexpr int32_t MAX_ROWS = UB_HALF_SIZE / TILE_COLS;
    static constexpr int32_t MOVE_NUM =
        (MAX_ROWS >= UB_MOVE_NUM_MAX) ? UB_MOVE_NUM_MAX : (MAX_ROWS >= 1 ? MAX_ROWS : 1);
};

// Per-row byte stride in remote shmem: hiddenSize bytes of int8 data + UB_ALIGN padding (containing float scale)
inline constexpr int32_t ShmemRowStride(int32_t hiddenSize)
{
    return hiddenSize + UB_ALIGN;
}

// ============================================================================
// CrossRankSync — shmem layout and DataAsFlag constants
// ============================================================================

static constexpr int32_t DATA_AS_FLAG_OFFSET = 0x800000;

// Alignment for tokenPerExpert exchange area (int32 elements, must be multiple of 8 for DMA)
inline constexpr int32_t PaddedExpertNum(int32_t EP, int32_t expertPerRank)
{
    int32_t raw = EP * expertPerRank;
    return (raw + 7) & ~7;
}

// Total bytes for the tokenPerExpert exchange area in shmem
// Layout: [EP rows] x [paddedExpertNum columns] of int32_t
// Each srcRank writes its localTPE into row[srcRank] of every remote rank's area
inline constexpr int64_t TPEAreaBytes(int32_t EP, int32_t expertPerRank)
{
    return static_cast<int64_t>(EP) * PaddedExpertNum(EP, expertPerRank) * sizeof(int32_t);
}

// Workspace layout for CrossRankSync computed routing tables:
//   [0 .. EP*paddedExpNum)                               : cumsumMM (padded rows for DMA alignment)
//   [EP*paddedExpNum .. EP*paddedExpNum + EP*expertPerRank) : preSumBeforeRank (int32)
//   [EP*paddedExpNum + EP*expertPerRank .. end)           : tokenPerExpert (padded, int32)
inline constexpr int64_t SyncWorkspaceBytes(int32_t EP, int32_t expertPerRank)
{
    int32_t paddedExpNum = ((EP * expertPerRank) + 7) & ~7;
    int32_t cumsumSize = EP * paddedExpNum;
    int32_t psbrSize = EP * expertPerRank;
    int32_t tpeSize = EP * paddedExpNum;
    return static_cast<int64_t>(cumsumSize + psbrSize + tpeSize) * sizeof(int32_t);
}

// SYNCALL soft barrier workspace: each core needs 8 int32 slots
static constexpr int32_t SYNCALL_SOFT_SLOT_INT32 = 8;

// ============================================================================
// Dispatch kernel launch parameters
// ============================================================================

struct MoeDispatchParams {
    int32_t EP;
    int32_t expertPerRank;
    int32_t hiddenSize;
    int32_t maxOutputSize;
    int32_t maxTokensPerRank;
};
