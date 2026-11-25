/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef FLASH_MLA_BF16_COMMON_H_
#define FLASH_MLA_BF16_COMMON_H_

#include <cstdint>

#ifndef FLASH_MLA_SIM_RUNTIME
#define FLASH_MLA_SIM_RUNTIME 1
#endif
#ifndef FLASH_MLA_MARK_STAMP_COMPUTE_PIPE
#define FLASH_MLA_MARK_STAMP_COMPUTE_PIPE 0
#endif
#ifndef FLASH_MLA_MARK_STAMP_DATA_PIPE
#define FLASH_MLA_MARK_STAMP_DATA_PIPE 0
#endif
static_assert(FLASH_MLA_SIM_RUNTIME == 0 || FLASH_MLA_SIM_RUNTIME == 1, "FLASH_MLA_SIM_RUNTIME must be 0 or 1");
static_assert(
    FLASH_MLA_MARK_STAMP_COMPUTE_PIPE == 0 || FLASH_MLA_MARK_STAMP_COMPUTE_PIPE == 1,
    "FLASH_MLA_MARK_STAMP_COMPUTE_PIPE must be 0 or 1");
static_assert(
    FLASH_MLA_MARK_STAMP_DATA_PIPE == 0 || FLASH_MLA_MARK_STAMP_DATA_PIPE == 1,
    "FLASH_MLA_MARK_STAMP_DATA_PIPE must be 0 or 1");
// The runtime/stamp contract applies to both host and device compilation.
static_assert(
    !FLASH_MLA_SIM_RUNTIME || (!FLASH_MLA_MARK_STAMP_COMPUTE_PIPE && !FLASH_MLA_MARK_STAMP_DATA_PIPE),
    "Profiling stamps require RUN_MODE=npu; simulator builds require "
    "FLASH_MLA_MARK_STAMP_COMPUTE_PIPE=0 and FLASH_MLA_MARK_STAMP_DATA_PIPE=0");

#if defined(__CCE__) || defined(__CPU_SIM)
#include <pto/pto-inst.hpp>
#endif

namespace flash_mla_bf16 {
#if defined(__CCE__) || defined(__CPU_SIM)
using namespace pto;
#endif
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

constexpr uint32_t M = 64;
constexpr uint32_t D = 512;
#ifndef FLASH_MLA_TILE_N
#define FLASH_MLA_TILE_N 128
#endif
static_assert(FLASH_MLA_TILE_N == 64 || FLASH_MLA_TILE_N == 128, "FLASH_MLA_TILE_N must be 64 or 128");
constexpr uint32_t N = FLASH_MLA_TILE_N;
constexpr uint32_t AIV_M = M / 2;
constexpr uint32_t AIV_N = N / 2;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t Q_BUFFER_NUM = 1;
constexpr uint32_t KV_BUFFER_NUM = 3;
constexpr uint32_t INDEX_BUFFER_NUM = 3;
constexpr uint32_t QK_K = 128;
constexpr uint32_t KV_REGION_TOKENS = 64;
constexpr uint32_t PV_N = 256;
constexpr uint32_t PV_K = KV_REGION_TOKENS;
constexpr uint32_t KV_REGIONS = N / KV_REGION_TOKENS;
constexpr uint32_t CACHE_GROUP = 16;
constexpr uint32_t CACHE_SCALE_D = D / CACHE_GROUP;
constexpr uint32_t CACHE_RECORD_BYTES = D / 2 + CACHE_SCALE_D;
constexpr uint32_t K_CHUNK = 32;
static_assert(AIV_N % K_CHUNK == 0, "Each AIV must gather a whole number of 32-token slices");
// Each AIV contributes one 32-token slice to every K64 region.
static_assert(PV_K == 2 * K_CHUNK, "A KV region must contain one gather slice from each AIV");
static_assert(
    N % KV_REGION_TOKENS == 0 && KV_REGIONS >= 1 && KV_REGIONS <= 2,
    "QK and PV require one or two complete K64 regions per tile");
constexpr uint32_t JOB_STATE_NUM = 2;
constexpr uint32_t SCORE_BUFFER_NUM = 1;
constexpr uint32_t SCORE_CREDIT_NUM = SCORE_BUFFER_NUM;
constexpr uint32_t ALPHA_BUFFER_NUM = BUFFER_NUM;
constexpr uint32_t ALPHA_SLOT_STRIDE = AIV_M * sizeof(float);
// A padded row changes the UB bank of successive 32-byte NZ panels.
constexpr uint32_t KV_NZ_PAD = 1;
constexpr uint32_t P_NZ_PAD = 1;
constexpr uint32_t KV_NZ_ROWS = K_CHUNK + KV_NZ_PAD;
constexpr uint32_t KV_NZ_BYTES = KV_NZ_ROWS * D * sizeof(uint16_t);
constexpr uint32_t P_NZ_ROWS = AIV_M + P_NZ_PAD;
constexpr uint32_t P_NZ_BYTES = P_NZ_ROWS * N * sizeof(uint16_t);
// Regional KV readiness travels V->C: region0 uses IDs0..2 and region1
// uses IDs3..5. Release travels C->V with the two ranges exchanged.
enum Event : uint32_t {
    KV_READY = 0,
    L1_FREE = 3,
    SCORE_READY = 6,
    SCORE_FREE = 6,
    RESCALE_READY = 7,
    P_READY = 8,
    PV_READY = 10,
    PV_FREE = 12,
    P_FREE = 14
};
// Event numbers belong to a direction and an AIV endpoint. Opposite directions
// may reuse a number; each endpoint has 16 IDs, with Vec1 addressed at ID + 16.
static_assert(KV_READY + KV_BUFFER_NUM <= L1_FREE, "Regional KV events require disjoint ranges");
static_assert(
    L1_FREE + KV_BUFFER_NUM <= SCORE_FREE && SCORE_FREE + SCORE_CREDIT_NUM <= RESCALE_READY &&
        RESCALE_READY + 1 <= P_READY && P_READY + BUFFER_NUM <= PV_FREE && PV_FREE + BUFFER_NUM <= 16,
    "V-to-C KV, score-release, decision, P-ready and PV-release events must fit disjoint 16-ID ranges");
static_assert(
    L1_FREE + KV_BUFFER_NUM <= SCORE_READY && SCORE_READY + SCORE_CREDIT_NUM <= RESCALE_READY &&
        RESCALE_READY + 1 <= PV_READY && PV_READY + BUFFER_NUM <= P_FREE && P_FREE + BUFFER_NUM <= 16,
    "C-to-V KV, score-ready, acknowledgement, PV-ready and P-release events must fit disjoint 16-ID ranges");
static_assert(SCORE_CREDIT_NUM == 1, "The single physical UB score tile requires exactly one reuse credit");
// Factor-256 slack expressed in natural-log scaled-score units.
constexpr float RESCALE_THRESHOLD = 5.545177444479562f;
constexpr uint32_t RESCALE_SSBUF_BASE = 0x100;
constexpr uint32_t RESCALE_SSBUF_SLOT_BYTES = 32;
constexpr uint32_t RESCALE_SSBUF_REPLY_OFFSET = 16;
#if defined(__CCE__) || defined(__CPU_SIM)
AICORE inline uint32_t kvReadyEvent(uint32_t slot, uint32_t region)
{
    return (region == 0 ? KV_READY : L1_FREE) + slot;
}

AICORE inline uint32_t kvReleaseEvent(uint32_t slot, uint32_t region)
{
    return (region == 0 ? L1_FREE : KV_READY) + slot;
}

AICORE inline uint32_t rescaleSlotAddress(uint32_t slot)
{
    return RESCALE_SSBUF_BASE + (slot % BUFFER_NUM) * RESCALE_SSBUF_SLOT_BYTES;
}
#endif
constexpr uint32_t L1_Q = 0;
constexpr uint32_t L1_K = Q_BUFFER_NUM * M * D * sizeof(uint16_t);
constexpr uint32_t L1_V = L1_K;
constexpr uint32_t KV_SLOT_STRIDE = N * D * sizeof(uint16_t);
constexpr uint32_t P_SLOT_STRIDE = M * N * sizeof(uint16_t);
constexpr uint32_t L1_P = L1_K + KV_BUFFER_NUM * KV_SLOT_STRIDE;
constexpr uint32_t L1_END = L1_P + BUFFER_NUM * P_SLOT_STRIDE;
static_assert(Q_BUFFER_NUM == 1, "Query-half reuse operates on one shared L1 query allocation");
static_assert(L1_Q + M * D * sizeof(uint16_t) == L1_K, "The L1 query allocation must end at the first KV bank");
static_assert(L1_END <= 512 * 1024, "Q, KV and P allocations must fit the 512-KiB L1 capacity");

constexpr uint32_t UB_SCORE = 0;
constexpr uint32_t UB_SCORE_SLOT_STRIDE = AIV_M * N * sizeof(float);
constexpr uint32_t UB_PV = UB_SCORE + SCORE_BUFFER_NUM * UB_SCORE_SLOT_STRIDE;
constexpr uint32_t UB_PV_SLOT_STRIDE = AIV_M * PV_N * sizeof(float);
constexpr uint32_t UB_RUNNING_O = UB_PV + BUFFER_NUM * UB_PV_SLOT_STRIDE;
constexpr uint32_t UB_SCRATCH = UB_RUNNING_O + AIV_M * D * sizeof(float);

constexpr uint32_t L0_AB_SLOT_STRIDE = 32768;
constexpr uint32_t L0_RIGHT_SLOT_NUM = 4;
constexpr uint32_t L0_RIGHT_SLOT_STRIDE = 16 * 1024;
constexpr uint32_t PV_MMAD_N = 128;
static_assert(
    BUFFER_NUM * L0_AB_SLOT_STRIDE <= 64 * 1024 && L0_RIGHT_SLOT_NUM * L0_RIGHT_SLOT_STRIDE <= 64 * 1024,
    "Operand slot arenas must fit the 64-KiB capacity of L0A and L0B");
static_assert(
    QK_K * KV_REGION_TOKENS * sizeof(uint16_t) <= L0_RIGHT_SLOT_STRIDE, "A QK K operand must fit one L0B slot");
static_assert(PV_K * PV_MMAD_N * sizeof(uint16_t) <= L0_RIGHT_SLOT_STRIDE, "A PV V operand must fit one L0B slot");
constexpr uint32_t L0_QK = 0;
constexpr uint32_t L0_QK_SLOT_STRIDE = M * N * sizeof(float);
constexpr uint32_t L0_PV = BUFFER_NUM * L0_QK_SLOT_STRIDE;
constexpr uint32_t L0_PV_SLOT_STRIDE = M * PV_N * sizeof(float);
static_assert(M * QK_K * sizeof(uint16_t) <= L0_AB_SLOT_STRIDE, "A QK Q operand must fit one L0A slot");
static_assert(M * PV_K * sizeof(uint16_t) <= L0_AB_SLOT_STRIDE, "A PV P operand must fit one L0A slot");
static_assert(
    L0_PV + BUFFER_NUM * L0_PV_SLOT_STRIDE <= 256 * 1024, "QK and PV accumulators must fit the 256-KiB L0C capacity");

#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
using MatQ = Tile<TileType::Mat, bfloat16_t, M, D, BLayout::ColMajor, M, D, SLayout::RowMajor, 512>;
using MatKInsert = Tile<TileType::Mat, bfloat16_t, N, D, BLayout::ColMajor, N, D, SLayout::RowMajor, 512>;
using MatK = Tile<TileType::Mat, bfloat16_t, D, N, BLayout::RowMajor, D, N, SLayout::ColMajor, 512>;
using MatV = MatKInsert;
using MatP = Tile<TileType::Mat, bfloat16_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
using VecScore = Tile<TileType::Vec, float, AIV_M, N, BLayout::RowMajor, AIV_M, N>;
using VecPv = Tile<TileType::Vec, float, AIV_M, D, BLayout::RowMajor, AIV_M, D>;
#endif

#if defined(__CCE_AICORE__)
using RescaleCtrl = Tile<TileType::Ctrl, uint32_t, 1, 1>;
using RescaleRequestConfig = CtrlPipeConfig<CtrlMergeMode::OR, VecCubeRatio::V2C1_VECS, true, false>;
using RescaleReplyConfig = CtrlPipeConfig<CtrlMergeMode::NONE, VecCubeRatio::V2C1_VECS, true, false>;
// Each AIV consumes its previous reply before its next request. AIC waits for
// both requests before replying, so this exchange owns reuse of the SSBUF slot.
// Depth-one pipes use the caller's slot and retain event 7 in both directions.
// Backward credits are disabled: event 8 already belongs to P_READY.
using RescaleRequest = TPipe<RESCALE_READY, Direction::DIR_V2C_CTRL, 4, 1, 1, false, false, RescaleRequestConfig>;
using RescaleReply = TPipe<RESCALE_READY, Direction::DIR_C2V_CTRL, 4, 1, 1, false, false, RescaleReplyConfig>;
#endif

} // namespace flash_mla_bf16
#endif // FLASH_MLA_BF16_COMMON_H_
