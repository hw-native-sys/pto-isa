/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef FLASH_MLA_BF16_AIC_H_
#define FLASH_MLA_BF16_AIC_H_

#include "flash_mla_common.h"
#include "custom/TPipe.hpp"
#include "flash_mla_profile.h"

namespace flash_mla_bf16 {

// MTE1<->M events protect L0A and L0B independently. FIX<->M
// accumulator credits use a different pipe pair and may reuse these IDs.
constexpr uint32_t L0_LEFT_EVENT = 2;
constexpr uint32_t L0_RIGHT_EVENT = 4;
constexpr uint32_t Q_JOB_EVENT_BASE = 6;
constexpr uint32_t Q_PART_D = 256;
constexpr uint32_t Q_PART_NUM = D / Q_PART_D;
static_assert(Q_PART_D == 2 * QK_K && Q_PART_NUM == 2);
static_assert(Q_JOB_EVENT_BASE + Q_PART_NUM <= 8);
constexpr uint32_t QK_ACC_EVENT = 2;
constexpr uint32_t PV_ACC_EVENT = 4;
constexpr uint32_t PV_FINAL_EVENT = 4;
constexpr uint32_t PV_SUBVIEWS = PV_N / PV_MMAD_N;
static_assert(PV_N == 2 * PV_MMAD_N);
// MTE1<->M operand events, FIX->M accumulator leases, and M->FIX
// final relays occupy independent event domains.
static_assert(L0_LEFT_EVENT + BUFFER_NUM <= L0_RIGHT_EVENT);
static_assert(L0_RIGHT_EVENT + L0_RIGHT_SLOT_NUM <= 8);
static_assert(QK_ACC_EVENT + BUFFER_NUM <= PV_ACC_EVENT);
static_assert(PV_ACC_EVENT + (D / PV_N) * PV_SUBVIEWS <= 8);
static_assert(QK_ACC_EVENT + BUFFER_NUM <= PV_FINAL_EVENT);
static_assert(PV_FINAL_EVENT + D / PV_N <= 8);

// Pair a producer event with its consumer wait on the selected pipe domain.
template <pipe_t Src, pipe_t Dst>
AICORE inline void localFence()
{
    set_flag(Src, Dst, EVENT_ID0);
    wait_flag(Src, Dst, EVENT_ID0);
}

template <pipe_t Pipe>
AICORE inline void waitVectors(uint32_t event)
{
    wait_intra_block(Pipe, event);
    wait_intra_block(Pipe, event + 16);
}

template <pipe_t Pipe>
AICORE inline void signalVectors(uint32_t event)
{
    set_intra_block(Pipe, event);
    set_intra_block(Pipe, event + 16);
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void setLocalCredit(uint32_t id)
{
    set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void waitLocalCredit(uint32_t id)
{
    wait_flag(Src, Dst, static_cast<event_t>(id));
}

// Seed credits once per physical block, including across empty logical jobs.
// Local ID0 remains reserved for the paired localFence helper.
AICORE inline void initAicPipeline()
{
    if constexpr (DAV_CUBE) {
#pragma unroll
        for (uint32_t part = 0; part < Q_PART_NUM; ++part) {
            setLocalCredit<PIPE_MTE1, PIPE_MTE2>(Q_JOB_EVENT_BASE + part);
        }
        for (uint32_t slot = 0; slot < KV_BUFFER_NUM; ++slot) {
            for (uint32_t region = 0; region < KV_REGIONS; ++region) {
                signalVectors<PIPE_MTE1>(kvReleaseEvent(slot, region));
            }
        }
        for (uint32_t slot = 0; slot < BUFFER_NUM; ++slot) {
            signalVectors<PIPE_MTE1>(P_FREE + slot);
            setLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + slot);
            setLocalCredit<PIPE_FIX, PIPE_M>(QK_ACC_EVENT + slot);
#pragma unroll
            for (uint32_t subview = 0; subview < PV_SUBVIEWS; ++subview) {
                setLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + slot * PV_SUBVIEWS + subview);
            }
        }
        for (uint32_t slot = 0; slot < L0_RIGHT_SLOT_NUM; ++slot) {
            setLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + slot);
        }
    }
}

// Collect the physical credits once after all assigned logical jobs finish.
AICORE inline void drainAicPipeline()
{
    if constexpr (DAV_CUBE) {
#pragma unroll
        for (uint32_t part = 0; part < Q_PART_NUM; ++part) {
            waitLocalCredit<PIPE_MTE1, PIPE_MTE2>(Q_JOB_EVENT_BASE + part);
        }
        waitVectors<PIPE_FIX>(SCORE_FREE);
        for (uint32_t slot = 0; slot < BUFFER_NUM; ++slot) {
            waitVectors<PIPE_FIX>(PV_FREE + slot);
            waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + slot);
            waitLocalCredit<PIPE_FIX, PIPE_M>(QK_ACC_EVENT + slot);
#pragma unroll
            for (uint32_t subview = 0; subview < PV_SUBVIEWS; ++subview) {
                waitLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + slot * PV_SUBVIEWS + subview);
            }
        }
        for (uint32_t slot = 0; slot < L0_RIGHT_SLOT_NUM; ++slot) {
            waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + slot);
        }
    }
}

// Two independently ready halves share one L1 query allocation.
// Both transfers use the full GM row stride and the 64-row NZ pitch.
AICORE inline void loadQ(__gm__ bfloat16_t* q, uint64_t row, uint64_t job)
{
    if constexpr (DAV_CUBE) {
        using GlobalQ = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, Q_PART_D>, pto::Stride<M * D, M * D, M * D, D, 1>>;
        using QueryPart =
            Tile<TileType::Mat, bfloat16_t, M, Q_PART_D, BLayout::ColMajor, M, Q_PART_D, SLayout::RowMajor, 512>;
#pragma unroll
        for (uint32_t part = 0; part < Q_PART_NUM; ++part) {
            QueryPart qTile;
            GlobalQ qGlobal(q + row * D + part * Q_PART_D);

            TASSIGN(qTile, L1_Q + part * M * Q_PART_D * sizeof(bfloat16_t));
            waitLocalCredit<PIPE_MTE1, PIPE_MTE2>(Q_JOB_EVENT_BASE + part);
            TLOAD(qTile, qGlobal);
            setLocalCredit<PIPE_MTE2, PIPE_MTE1>(Q_JOB_EVENT_BASE + part);
        }
        stampData<PIPE_MTE2, StampStage::Q_LOAD>(job);
    }
}

// Finish each K64 token region across all feature slices before acquiring the
// next region. Cube can consume region0 while MTE3 is still filling region1.
AICORE inline void qk(uint32_t scoreSlot, uint32_t kvSlot, uint32_t tile, bool last, bool hasPv)
{
    if constexpr (DAV_CUBE) {
        using QLeft = TileLeft<bfloat16_t, M, QK_K>;
        using KRight = TileRight<bfloat16_t, QK_K, KV_REGION_TOKENS>;
        using RegionAcc = TileAcc<float, M, KV_REGION_TOKENS>;
        MatQ qTile;
        MatK kTile;

        TASSIGN(qTile, L1_Q);
        TASSIGN(kTile, L1_K + kvSlot * KV_SLOT_STRIDE);
        waitLocalCredit<PIPE_FIX, PIPE_M>(QK_ACC_EVENT + scoreSlot);
        // Relay the lease generation before arithmetic. Unit flags supply
        // per-unit readiness; FIX returns the lease after both AIV copies.
        setLocalCredit<PIPE_M, PIPE_FIX>(QK_ACC_EVENT + scoreSlot);
        const bool tailPrefill = (N == 2 * PV_K) && hasPv;
        static_assert(M * D * sizeof(bfloat16_t) == BUFFER_NUM * L0_AB_SLOT_STRIDE);
        static_assert(L0_AB_SLOT_STRIDE == 2 * M * QK_K * sizeof(bfloat16_t));
        static_assert(M * PV_K * sizeof(bfloat16_t) <= L0_AB_SLOT_STRIDE / 2);
#pragma unroll
        for (uint32_t region = 0; region < KV_REGIONS; ++region) {
            // The NZ score layout has a 64-row pitch for both views, so these
            // disjoint regions form the full-width score tile.
            RegionAcc score;
            TASSIGN(score, L0_QK + scoreSlot * L0_QK_SLOT_STRIDE + region * M * KV_REGION_TOKENS * sizeof(float));
#pragma unroll
            for (uint32_t k = 0; k < D; k += QK_K) {
                const uint32_t rightSlot = (k / QK_K) % L0_RIGHT_SLOT_NUM;
                QLeft left;
                KRight right;

                // First tiles initialize each Q slice before its K-side
                // waits, overlapping Q loading with incoming KV.
                TASSIGN(right, rightSlot * L0_RIGHT_SLOT_STRIDE);
                TASSIGN(left, k * M * sizeof(bfloat16_t));
                if (tile == 0 && region == 0) {
                    if (tailPrefill && k % Q_PART_D == 0) {
                        // Preceding PV acquired both halves after the preceding
                        // QK readers completed, and occupies their prefixes.
                        waitLocalCredit<PIPE_MTE2, PIPE_MTE1>(Q_JOB_EVENT_BASE + k / Q_PART_D);
                        TileLeft<bfloat16_t, M, QK_K - PV_K> suffix;
                        TASSIGN(suffix, (k + PV_K) * M * sizeof(bfloat16_t));
                        TEXTRACT(suffix, qTile, 0, k + PV_K);
                        waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + k / Q_PART_D);
                        TileLeft<bfloat16_t, M, PV_K> prefix;
                        TASSIGN(prefix, k * M * sizeof(bfloat16_t));
                        TEXTRACT(prefix, qTile, 0, k);
                    } else {
                        if (k % Q_PART_D == 0) {
                            waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + k / Q_PART_D);
                            waitLocalCredit<PIPE_MTE2, PIPE_MTE1>(Q_JOB_EVENT_BASE + k / Q_PART_D);
                        }
                        TEXTRACT(left, qTile, 0, k);
                    }
                    if (last && k % Q_PART_D + QK_K == Q_PART_D) {
                        setLocalCredit<PIPE_MTE1, PIPE_MTE2>(Q_JOB_EVENT_BASE + k / Q_PART_D);
                    }
                }
                // K owns an independent L0B slot. On subsequent tiles,
                // start its copy while PV may still be reading P in L0A.
                waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + rightSlot);
                if (k == 0) {
                    custom::V2C<MatK> kvPipe(L1_K + kvSlot * KV_SLOT_STRIDE);
                    kvPipe.waitReady(kvReadyEvent(kvSlot, region));
                    TPOP<decltype(kvPipe), MatK, TILE_UP_DOWN>(kvPipe, kTile, 0);
                }
                TEXTRACT(right, kTile, k, region * KV_REGION_TOKENS);
                if (tile != 0 && region == 0 && k % Q_PART_D == 0) {
                    waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + k / Q_PART_D);
                    // PV overwrites only the first 8 KiB of each half.
                    // Restore that prefix; retain all other Q features.
                    TileLeft<bfloat16_t, M, PV_K> restore;
                    TASSIGN(restore, k * M * sizeof(bfloat16_t));
                    TEXTRACT(restore, qTile, 0, k);
                    if (last) {
                        setLocalCredit<PIPE_MTE1, PIPE_MTE2>(Q_JOB_EVENT_BASE + k / Q_PART_D);
                    }
                }
                setLocalCredit<PIPE_MTE1, PIPE_M>(L0_RIGHT_EVENT + rightSlot);
                stampData<PIPE_MTE1, StampStage::QK>(tile, region * (D / QK_K) + k / QK_K);
                // Operand readiness follows both K and Q extraction in MTE1,
                // so it also orders each retained Q slice before its MMAD.
                waitLocalCredit<PIPE_MTE1, PIPE_M>(L0_RIGHT_EVENT + rightSlot);
                if (k == 0) {
                    TMATMUL<AccPhase::Partial>(score, left, right);
                } else if (k + QK_K == D) {
                    TMATMUL_ACC<AccPhase::Final>(score, score, left, right);
                } else {
                    TMATMUL_ACC<AccPhase::Partial>(score, score, left, right);
                }
                setLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + rightSlot);
                if (region + 1 == KV_REGIONS && ((k + QK_K) * M * sizeof(bfloat16_t)) % L0_AB_SLOT_STRIDE == 0) {
                    // Return each half after its last regional read, so
                    // following PV may load P while QK uses the other half.
                    setLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + k * M * sizeof(bfloat16_t) / L0_AB_SLOT_STRIDE);
                }
            }
        }
        stampCompute<PIPE_M, StampStage::QK>(tile, KV_REGIONS * (D / QK_K) - 1);
    }
}

// A required PV flush is submitted before score publication.
// QK computation can overlap that segment's FIXP transfer.
AICORE inline void exportQk(uint32_t scoreSlot, uint32_t tile)
{
    if constexpr (DAV_CUBE) {
        using ScoreAcc = TileAcc<float, M, N, M, N>;
        ScoreAcc score;
        TASSIGN(score, L0_QK + scoreSlot * L0_QK_SLOT_STRIDE);
        waitLocalCredit<PIPE_M, PIPE_FIX>(QK_ACC_EVENT + scoreSlot);
        waitVectors<PIPE_FIX>(SCORE_FREE);
        // Rotation explicitly acquires the single physical score tile before
        // FIXP writes it, independently of the probability-ready dependency.
        custom::C2V<VecScore, STPhase::Final> scorePipe(UB_SCORE);
        TPUSH<decltype(scorePipe), ScoreAcc, TILE_UP_DOWN>(scorePipe, score);
        signalVectors<PIPE_FIX>(SCORE_READY);
        setLocalCredit<PIPE_FIX, PIPE_M>(QK_ACC_EVENT + scoreSlot);
        stampData<PIPE_FIX, StampStage::QK>(tile, KV_REGIONS * (D / QK_K) - 1);
    }
}

// One L0C segment spans any number of tiles sharing their row scaling bases.
// A threshold crossing exports the accumulated segment before the current PV starts.
// Mid-job exports use explicit M/FIX ownership. Unit flags span the whole job:
// first MMAD acquires them, last MMAD publishes them, final FIXP releases them.

AICORE inline bool receiveRescaleDecision(uint32_t slot)
{
    RescaleRequest request(nullptr, 0, rescaleSlotAddress(slot));
    RescaleReply reply(nullptr, rescaleSlotAddress(slot) + RESCALE_SSBUF_REPLY_OFFSET, 0);
    RescaleCtrl decision;
    TPOP<RescaleRequest, RescaleCtrl, TileSplitAxis::TILE_NO_SPLIT>(request, decision);
    TPUSH<RescaleReply, RescaleCtrl, TileSplitAxis::TILE_NO_SPLIT>(reply, decision);
    return decision.data() != 0;
}

// Return each conditional-flush L0C view after both AIV copies. The valid
// destination width is one view, but its row stride remains the full GU tile.
template <uint32_t Subview>
AICORE inline void exportPvSubview(uint32_t fragment)
{
    static_assert(Subview < PV_SUBVIEWS);
    using PvAcc = TileAcc<float, M, PV_MMAD_N, M, PV_MMAD_N>;
    using VecPvChunk = Tile<TileType::Vec, float, AIV_M, PV_N, BLayout::RowMajor, AIV_M, PV_MMAD_N>;
    PvAcc output;
    TASSIGN(output, L0_PV + fragment * L0_PV_SLOT_STRIDE + Subview * M * PV_MMAD_N * sizeof(float));
    custom::C2V<VecPvChunk> pvPipe(UB_PV + fragment * UB_PV_SLOT_STRIDE + Subview * PV_MMAD_N * sizeof(float));
    TPUSH<decltype(pvPipe), PvAcc, TILE_UP_DOWN>(pvPipe, output);
    setLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + fragment * PV_SUBVIEWS + Subview);
}

template <bool Final>
AICORE inline void exportPvFragment(uint32_t fragment, uint32_t tile)
{
    waitVectors<PIPE_FIX>(PV_FREE + fragment);
    if constexpr (!Final) {
        exportPvSubview<0>(fragment);
        exportPvSubview<1>(fragment);
    } else {
        // Final exports use the full-width unit-flag transfer. Return the L0C
        // leases after both AIV destination copies; PV_FREE protects later GU reads.
        using PvAcc = TileAcc<float, M, PV_N, M, PV_N>;
        using VecPvChunk = Tile<TileType::Vec, float, AIV_M, PV_N, BLayout::RowMajor, AIV_M, PV_N>;
        PvAcc output;
        TASSIGN(output, L0_PV + fragment * L0_PV_SLOT_STRIDE);
        custom::C2V<VecPvChunk, STPhase::Final> pvPipe(UB_PV + fragment * UB_PV_SLOT_STRIDE);
        TPUSH<decltype(pvPipe), PvAcc, TILE_UP_DOWN>(pvPipe, output);
    }
    signalVectors<PIPE_FIX>(PV_READY + fragment);
    if constexpr (Final) {
#pragma unroll
        for (uint32_t subview = 0; subview < PV_SUBVIEWS; ++subview) {
            setLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + fragment * PV_SUBVIEWS + subview);
        }
    }
    stampData<PIPE_FIX, StampStage::PV>(tile, fragment);
}

// Template steps resolve each PV operand slot and event ID at compile time.
template <uint32_t Col, uint32_t K, typename Output, typename PMatrix, typename VMatrix>
AICORE inline void pvMmad(
    Output& output, PMatrix& pTile, VMatrix& vTile, uint32_t pSlot, uint32_t kvSlot, uint32_t tile, bool last,
    bool hasNextQk, uint32_t nextScoreSlot, uint32_t nextQkTile, bool& rescale)
{
    static_assert(Col < D && Col % PV_MMAD_N == 0);
    static_assert(K < N && K % PV_K == 0);
    constexpr uint32_t fragment = Col / PV_N;
    using PLeft = TileLeft<bfloat16_t, M, PV_K>;
    using VRight = TileRight<bfloat16_t, PV_K, PV_MMAD_N>;
    constexpr uint32_t rightSlot = ((Col / PV_MMAD_N) * (N / PV_K) + K / PV_K) % L0_RIGHT_SLOT_NUM;
    // P is independent of the output-column fragment. Keep each
    // K region in one L0A bank while L0B continues to rotate.
    constexpr uint32_t leftSlot = K / PV_K;
    PLeft left;
    VRight right;

    TASSIGN(left, leftSlot * L0_AB_SLOT_STRIDE);
    TASSIGN(right, rightSlot * L0_RIGHT_SLOT_STRIDE);
    // Submit both first PV operands before the S wait.
    // The operand pipelines continue while scalar reads the VF flag.
    waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + rightSlot);
    TEXTRACT(right, vTile, K, Col);
    if constexpr (Col + PV_MMAD_N == D) {
        // Each PV_K-token region can be replaced after its final
        // V read; Cube subsequently consumes the L0B copy.
        signalVectors<PIPE_MTE1>(kvReleaseEvent(kvSlot, K / PV_K));
    }
    setLocalCredit<PIPE_MTE1, PIPE_M>(L0_RIGHT_EVENT + rightSlot);
    if constexpr (Col == 0 && K == 0) {
        custom::V2C<MatP> probabilityPipe(L1_P + pSlot * P_SLOT_STRIDE);
        probabilityPipe.waitReady(P_READY + pSlot);
        TPOP<decltype(probabilityPipe), MatP, TILE_UP_DOWN>(probabilityPipe, pTile, 0);
    }
    if constexpr (Col == 0) {
        waitLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + leftSlot);
        TEXTRACT(left, pTile, 0, K);
    }
    if constexpr (Col == 0 && K + PV_K == N) {
        // Remaining P reads use the retained L0A copy, so
        // the L1 slot can already be filled again.
        signalVectors<PIPE_MTE1>(P_FREE + pSlot);
    }
    if constexpr (Col == 0) {
        setLocalCredit<PIPE_MTE1, PIPE_M>(L0_LEFT_EVENT + leftSlot);
    }
    stampData<PIPE_MTE1, StampStage::PV>(tile, (Col / PV_MMAD_N) * (N / PV_K) + K / PV_K);
    if constexpr (Col == 0 && K == 0) {
        // Predicate handoffs start at tile 1; tile 0 initializes the segment.
        if (tile != 0) {
            rescale = receiveRescaleDecision(pSlot);
        }
        if (rescale) {
            // The preceding iteration fenced this accumulated segment before
            // submitting its trailing QK.
            exportPvFragment<false>(0, tile);
            exportPvFragment<false>(1, tile);
        }
        if (hasNextQk) {
            exportQk(nextScoreSlot, nextQkTile);
        }
    }
    if constexpr (Col % PV_N == 0 && K == 0) {
        if (tile == 0 || rescale) {
            // Acquire the fragment before its forward relay; this also orders
            // reuse after the preceding job's final consumer.
            waitLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + Col / PV_MMAD_N);
        }
        if (last) {
            // Relay before arithmetic; final unit readiness protects
            // each L0C unit while FIXP streams the last segment.
            setLocalCredit<PIPE_M, PIPE_FIX>(PV_FINAL_EVENT + fragment);
        }
    }
    waitLocalCredit<PIPE_MTE1, PIPE_M>(L0_RIGHT_EVENT + rightSlot);
    if constexpr (Col == 0) {
        waitLocalCredit<PIPE_MTE1, PIPE_M>(L0_LEFT_EVENT + leftSlot);
    }
    const bool initialize = K == 0 && (tile == 0 || rescale);
    if (last && K + PV_K == N) {
        if (initialize) {
            if constexpr (Col % PV_N != 0) {
                waitLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + Col / PV_MMAD_N);
            }
            TMATMUL<AccPhase::Final>(output, left, right);
        } else {
            TMATMUL_ACC<AccPhase::Final>(output, output, left, right);
        }
    } else if (tile == 0 && K == 0) {
        if constexpr (Col % PV_N != 0) {
            waitLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + Col / PV_MMAD_N);
        }
        TMATMUL<AccPhase::Partial>(output, left, right);
    } else if (initialize) {
        if constexpr (Col % PV_N != 0) {
            waitLocalCredit<PIPE_FIX, PIPE_M>(PV_ACC_EVENT + Col / PV_MMAD_N);
        }
        TMATMUL(output, left, right);
    } else {
        TMATMUL_ACC(output, output, left, right);
    }
    if constexpr (Col + PV_MMAD_N == D) {
        setLocalCredit<PIPE_M, PIPE_MTE1>(L0_LEFT_EVENT + leftSlot);
    }
    setLocalCredit<PIPE_M, PIPE_MTE1>(L0_RIGHT_EVENT + rightSlot);
}

template <uint32_t Col, typename PMatrix, typename VMatrix>
AICORE inline void pvFragment(
    PMatrix& pTile, VMatrix& vTile, uint32_t pSlot, uint32_t kvSlot, uint32_t tile, bool last, bool hasNextQk,
    uint32_t nextScoreSlot, uint32_t nextQkTile, bool& rescale)
{
    constexpr uint32_t fragment = Col / PV_N;
    TileAcc<float, M, PV_MMAD_N, M, PV_MMAD_N> output;
    TASSIGN(output, L0_PV + fragment * L0_PV_SLOT_STRIDE);
    pvMmad<Col, 0>(output, pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
    if constexpr (N > PV_K) {
        pvMmad<Col, PV_K>(
            output, pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
    }
    // The second view uses the full 64-row NZ pitch and its own accumulator
    // lease. GU publication covers the complete 256-column fragment.
    TileAcc<float, M, PV_MMAD_N, M, PV_MMAD_N> output1;
    TASSIGN(output1, L0_PV + fragment * L0_PV_SLOT_STRIDE + M * PV_MMAD_N * sizeof(float));
    pvMmad<Col + PV_MMAD_N, 0>(
        output1, pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
    if constexpr (N > PV_K) {
        pvMmad<Col + PV_MMAD_N, PV_K>(
            output1, pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
    }
    stampCompute<PIPE_M, StampStage::PV>(tile, fragment);
    if (last) {
        waitLocalCredit<PIPE_M, PIPE_FIX>(PV_FINAL_EVENT + fragment);
        exportPvFragment<true>(fragment, tile);
    }
}

AICORE inline void pv(
    uint32_t pSlot, uint32_t kvSlot, uint32_t tile, bool last, bool hasNextQk, uint32_t nextScoreSlot,
    uint32_t nextQkTile)
{
    if constexpr (DAV_CUBE) {
        static_assert(D == 2 * PV_N);
        MatP pTile;
        MatV vTile;

        TASSIGN(pTile, L1_P + pSlot * P_SLOT_STRIDE);
        TASSIGN(vTile, L1_V + kvSlot * KV_SLOT_STRIDE);
        bool rescale = false;
        pvFragment<0>(pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
        pvFragment<PV_N>(pTile, vTile, pSlot, kvSlot, tile, last, hasNextQk, nextScoreSlot, nextQkTile, rescale);
    }
}
} // namespace flash_mla_bf16
#endif // FLASH_MLA_BF16_AIC_H_
