/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FLASH_MLA_BF16_AIV_H_
#define FLASH_MLA_BF16_AIV_H_

#include "flash_mla_aic.h"
#include "flash_mla_api.h"
#include "custom/MGather.hpp"
#include "custom/TStore.hpp"
#include "custom/TGatherMetaData.hpp"
#include "custom/TDequantKey.hpp"
#include "custom/TSoftMaxRow.hpp"
#include "custom/TSoftMaxProb.hpp"
#include "custom/TGlobalUpdate.hpp"
#include "custom/TOutputNormalize.hpp"
#include "custom/TExpandS.hpp"

namespace flash_mla_bf16 {
// Raw, decoded KV and probability sources have independent ownership.
// Index masks/offsets/pair controls share a bank released by its last softmax.
constexpr uint32_t RAW_SLOT_STRIDE = K_CHUNK * RAW_STRIDE;
constexpr uint32_t UB_RAW = UB_SCRATCH;
constexpr uint32_t UB_MAX = UB_RAW + BUFFER_NUM * RAW_SLOT_STRIDE;
constexpr uint32_t UB_SUM = UB_MAX + JOB_STATE_NUM * 128;
constexpr uint32_t UB_ALPHA = UB_SUM + JOB_STATE_NUM * 128;
constexpr uint32_t UB_SINK = UB_ALPHA + ALPHA_BUFFER_NUM * ALPHA_SLOT_STRIDE;
constexpr uint32_t UB_LSE = UB_SINK + JOB_STATE_NUM * 128;
// Reserve 32 bytes per row around the packed 128-byte LSE.
constexpr uint32_t UB_FLOAT = (UB_LSE + AIV_M * 32 + 511) / 512 * 512;
constexpr uint32_t KV_NZ_SLOT_STRIDE = KV_NZ_BYTES;
constexpr uint32_t UB_DEQUANT_SCALES = UB_FLOAT + BUFFER_NUM * KV_NZ_BYTES;
constexpr uint32_t UB_P_NZ = (UB_DEQUANT_SCALES + K_CHUNK * CACHE_SCALE_D * 2 + 511) / 512 * 512;
constexpr uint32_t UB_SOFTMAX_ROWS = (UB_P_NZ + P_NZ_BYTES + 255) / 256 * 256;
constexpr uint32_t UB_INDEX = UB_SOFTMAX_ROWS + 512;
// Held row bases and two scalar decisions follow the metadata arena.
constexpr uint32_t UB_ANCHOR = UB_INDEX + INDEX_BUFFER_NUM * INDEX_SLOT_STRIDE;
constexpr uint32_t UB_RESCALE = UB_ANCHOR + JOB_STATE_NUM * 128;
// The second anchor's NORM load reads a full register; upper-lane padding
// extends beyond the two 32-byte decision slots for its 32 valid lanes.
constexpr uint32_t UB_END = UB_RESCALE + 128;
constexpr uint32_t RESCALE_LOCAL_EVENT = 4;
constexpr uint32_t RAW_EVENT_BASE = 4;
constexpr uint32_t INDEX_EVENT_BASE = 1;
constexpr uint32_t KV_NZ_EVENT_BASE = 2;
constexpr uint32_t P_NZ_EVENT = 1;
constexpr uint32_t OUTPUT_READY_EVENT_BASE = 4;
constexpr uint32_t OUTPUT_REUSE_EVENT = 4;
constexpr uint32_t JOB_ROW_EVENT_BASE = 6;
// V<->MTE2 uses index1..3, raw4..5 and row-state6..7; ID0 is for local fences.
// The same IDs on MTE3<->V protect independent resources.
static_assert(INDEX_BUFFER_NUM == 3 && INDEX_TOKENS % N == 0);
// V<->MTE2: metadata and raw slots have independent reuse/readiness credits.
static_assert(INDEX_EVENT_BASE + INDEX_BUFFER_NUM <= RAW_EVENT_BASE);
// V->S: metadata dispatch and rescale decisions use distinct publications.
static_assert(INDEX_EVENT_BASE + INDEX_BUFFER_NUM <= RESCALE_LOCAL_EVENT);
static_assert(RAW_EVENT_BASE + BUFFER_NUM <= JOB_ROW_EVENT_BASE);
static_assert(JOB_ROW_EVENT_BASE + JOB_STATE_NUM <= 8);
// Forward output-ready4/5 are distinct so both may be pending on MTE3.
// Backward output-reuse4 covers both halves and LSE in the opposite pipe pair.
static_assert(KV_NZ_EVENT_BASE + BUFFER_NUM <= OUTPUT_READY_EVENT_BASE);
static_assert(OUTPUT_READY_EVENT_BASE + D / PV_N <= 8);
static_assert(UB_END <= 256 * 1024, "BF16 UB layout must fit the 256-KiB physical capacity");
static_assert(KV_NZ_SLOT_STRIDE % 32 == 0 && RAW_SLOT_STRIDE % 256 == 0);
static_assert(
    AIV_M == 32 && D == 512 && PV_N == 256, "Output VFs require 32 rows, 512 features and 256-column fragments");
// Valid signed-int32 token indices fit the DMA's 40-bit source-stride field.
static_assert(uint64_t{INT32_MAX} * CACHE_RECORD_BYTES <= ((uint64_t{1} << 40) - 1));
static_assert(RAW_STRIDE < (uint32_t{1} << 21));

struct WideValidity {
    uint32_t countBits = 0;
};
AICORE inline uint32_t indexMaskAddress(uint32_t metadataSlot, uint32_t metadataPart)
{
    return UB_INDEX + metadataSlot * INDEX_SLOT_STRIDE + metadataPart * N * sizeof(uint32_t);
}
AICORE inline uint32_t pairMetadataAddress(uint32_t metadataSlot, uint32_t metadataPart, uint32_t slice)
{
    return UB_INDEX + metadataSlot * INDEX_SLOT_STRIDE + INDEX_PAIR_OFFSET +
           (metadataPart * N + slice * K_CHUNK) / 2 * sizeof(uint64_t);
}
AICORE inline uint32_t rawSlotAddress(uint32_t slot) { return UB_RAW + slot * RAW_SLOT_STRIDE; }
AICORE inline event_t rawSlotEvent(uint32_t slot) { return static_cast<event_t>(RAW_EVENT_BASE + slot); }

// Every K64 region contains one slice from each AIV. AIV0 gathers
// slices 0/2 and AIV1 gathers slices 1/3 at N128.
AICORE inline uint32_t kvSlice(uint32_t ordinal, uint32_t subblock) { return ordinal * 2 + subblock; }

AICORE inline void clearRawSlot(uint32_t slot)
{
    using Raw = Tile<TileType::Vec, uint32_t, 1, RAW_SLOT_STRIDE / 4>;
    Raw raw;
    TASSIGN(raw, rawSlotAddress(slot));
    TEXPANDS(raw, 0u);
}

AICORE inline void initVectorPipeline()
{
    if constexpr (DAV_VEC) {
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(OUTPUT_REUSE_EVENT));
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(P_NZ_EVENT));
        for (uint32_t slot = 0; slot < BUFFER_NUM; ++slot) {
            set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(JOB_ROW_EVENT_BASE + slot));
            if (slot < SCORE_CREDIT_NUM)
                set_intra_block(PIPE_V, SCORE_FREE + slot);
            set_intra_block(PIPE_V, PV_FREE + slot);
            clearRawSlot(slot);
            set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(KV_NZ_EVENT_BASE + slot));
            set_flag(PIPE_V, PIPE_MTE2, rawSlotEvent(slot));
        }
        for (uint32_t slot = 0; slot < INDEX_BUFFER_NUM; ++slot) {
            set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(INDEX_EVENT_BASE + slot));
        }
    }
}

AICORE inline void drainVectorPipeline()
{
    if constexpr (DAV_VEC) {
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(OUTPUT_REUSE_EVENT));
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(P_NZ_EVENT));
        for (uint32_t slot = 0; slot < KV_BUFFER_NUM; ++slot) {
            for (uint32_t region = 0; region < KV_REGIONS; ++region) {
                wait_intra_block(PIPE_MTE3, kvReleaseEvent(slot, region));
            }
        }
        for (uint32_t slot = 0; slot < BUFFER_NUM; ++slot) {
            wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(JOB_ROW_EVENT_BASE + slot));
            wait_intra_block(PIPE_MTE3, P_FREE + slot);
            wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(KV_NZ_EVENT_BASE + slot));
            wait_flag(PIPE_V, PIPE_MTE2, rawSlotEvent(slot));
        }
        for (uint32_t slot = 0; slot < INDEX_BUFFER_NUM; ++slot) {
            wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(INDEX_EVENT_BASE + slot));
        }
    }
}

AICORE inline void initJobRows(__gm__ float* sink, uint32_t head, uint32_t jobSlot)
{
    const auto event = static_cast<event_t>(JOB_ROW_EVENT_BASE + jobSlot);
    wait_flag(PIPE_V, PIPE_MTE2, event);
    using State = Tile<TileType::Vec, float, 1, AIV_M, BLayout::RowMajor, 1, AIV_M>;
    // FirstTile SM1/SM2 initialize maximum, anchor and sum for nonempty jobs.
    State sinkTile;
    TASSIGN(sinkTile, UB_SINK + jobSlot * 128);
    if (sink != nullptr) {
        using GlobalSink = GlobalTensor<float, Shape<1, 1, 1, 1, AIV_M>, pto::Stride<1, 1, 1, AIV_M, 1>>;
        GlobalSink source(sink + head + get_subblockid() * AIV_M);
        TLOAD(sinkTile, source);
    }
    set_flag(PIPE_MTE2, PIPE_V, event);
    wait_flag(PIPE_MTE2, PIPE_V, event);
    if (sink == nullptr) {
        using Bits = Tile<TileType::Vec, uint32_t, 1, AIV_M, BLayout::RowMajor, 1, AIV_M>;
        Bits bits;
        TASSIGN(bits, UB_SINK + jobSlot * 128);
        TEXPANDS(bits, 0xff800000u);
    }
}

AICORE inline void initJobOutput(bool empty)
{
    // Running O shares storage with the preceding job's output store.
    // Metadata, dequant and softmax are submitted before this reuse acquisition.
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(OUTPUT_REUSE_EVENT));
    // Nonempty jobs initialize running O in their first consumed PV update.
    if (empty) {
        VecPv running;
        TASSIGN(running, UB_RUNNING_O);
        TEXPANDS(running, 0.0f);
    }
}

AICORE inline void processIndices(uint32_t count, uint32_t capacity, uint32_t slot)
{
    custom::GatherMetadataTile metadata;
    TASSIGN(metadata, UB_INDEX + slot * INDEX_SLOT_STRIDE);
    custom::TGatherMetaData(metadata, count, capacity);
}

AICORE inline void prepareIndices(
    __gm__ int32_t* indices, uint32_t start, uint32_t length, uint64_t capacity, uint32_t slot)
{
    const auto event = static_cast<event_t>(INDEX_EVENT_BASE + slot);
    // Called only for a nonempty tile in the selected cache scope, hence
    // start < length and its selected indices pointer is nonnull.
    const uint32_t count = length - start < INDEX_TOKENS ? length - start : INDEX_TOKENS;
    // Softmax is the last mask reader. Acquire its credit before replacing
    // this metadata slot, including when a physical AIV starts its next job.
    wait_flag(PIPE_V, PIPE_MTE2, event);
    using IndexTile = Tile<TileType::Vec, int32_t, 1, INDEX_TOKENS, BLayout::RowMajor, 1, -1>;
    using IndexGlobal = GlobalTensor<int32_t, Shape<1, 1, 1, 1, -1>, pto::Stride<1, 1, 1, -1, 1>>;
    IndexTile indexTile(count);
    IndexGlobal source(indices + start, Shape<1, 1, 1, 1, -1>(count), pto::Stride<1, 1, 1, -1, 1>(count));
    TASSIGN(indexTile, UB_INDEX + slot * INDEX_SLOT_STRIDE);
    TLOAD(indexTile, source);
    set_flag(PIPE_MTE2, PIPE_V, event);
    wait_flag(PIPE_MTE2, PIPE_V, event);
    processIndices(count, static_cast<uint32_t>(capacity), slot);
    // This handoff is required: scalar dispatch reads VF-produced UB offsets.
    set_flag(PIPE_V, PIPE_S, event);
}

AICORE inline void gatherPreparedSlice(
    __gm__ uint8_t* cache, uint32_t rawSlot, uint32_t tile, uint32_t slice, uint32_t metadataSlot,
    uint32_t metadataPart)
{
    using RawTile = Tile<TileType::Vec, uint8_t, 1, RAW_SLOT_STRIDE>;
    using DescriptorTile = Tile<TileType::Vec, uint64_t, 1, K_CHUNK / 2>;
    using CacheTensor =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, CACHE_RECORD_BYTES>, pto::Stride<1, 1, 1, CACHE_RECORD_BYTES, 1>>;
    RawTile raw;
    DescriptorTile offsets, controls;
    CacheTensor source(cache);
    TASSIGN(raw, rawSlotAddress(rawSlot));
    TASSIGN(
        offsets, UB_INDEX + metadataSlot * INDEX_SLOT_STRIDE + INDEX_OFFSET +
                     (metadataPart * N + slice * K_CHUNK) / 2 * sizeof(uint64_t));
    TASSIGN(controls, pairMetadataAddress(metadataSlot, metadataPart, slice));
    const auto event = rawSlotEvent(rawSlot);
    wait_flag(PIPE_V, PIPE_MTE2, event);
    custom::MGATHER(raw, source, offsets, controls);
    set_flag(PIPE_MTE2, PIPE_V, event);
    stampData<PIPE_MTE2, StampStage::DEQUANT>(tile, dequantStampPart(slice));
}

// Decode four inline scale rows per gather. The exact E2M1 * E4M3 product
// fits BF16. Paired DMA may sort its sources; restore both payload and scale
// together before writing the selected token order to padded BF16 NZ.
AICORE inline void promoteKey(uint32_t rawSlot, uint32_t pairAddress, uint32_t dstAddress)
{
    custom::RawKeyTile raw;
    custom::PairControlTile controls;
    custom::DequantScaleTile scales;
    custom::DequantKeyTile decoded;

    TASSIGN(raw, rawSlotAddress(rawSlot));
    TASSIGN(controls, pairAddress);
    TASSIGN(scales, UB_DEQUANT_SCALES);
    TASSIGN(decoded, dstAddress);
    custom::TDequantKey(decoded, scales, raw, controls);
}

// Capture each decoded NZ handoff immediately after its producer VF. Slice 0
// is decoded and queued to L1 by beginKvGather before the second raw gather.
template <uint32_t Ordinal>
AICORE inline void dequantSlice(uint32_t tile, uint32_t metadataSlot, uint32_t metadataPart, uint32_t subblock)
{
    constexpr uint32_t rawSlot = Ordinal % BUFFER_NUM;
    const uint32_t slice = kvSlice(Ordinal, subblock);
    const auto rawEvent = rawSlotEvent(rawSlot);
    wait_flag(PIPE_MTE2, PIPE_V, rawEvent);
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(KV_NZ_EVENT_BASE + rawSlot));
    promoteKey(rawSlot, pairMetadataAddress(metadataSlot, metadataPart, slice), UB_FLOAT + rawSlot * KV_NZ_SLOT_STRIDE);
    set_flag(PIPE_V, PIPE_MTE2, rawEvent);
    stampCompute<PIPE_V, StampStage::DEQUANT>(tile, dequantStampPart(slice));
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(KV_NZ_EVENT_BASE + rawSlot));
}

// Queue each decoded NZ slice into its regional L1 destination. These waits
// stay on MTE3; scalar does not explicitly wait for the copy to finish.
template <uint32_t Ordinal>
AICORE inline void insertKvSlice(uint32_t kvSlot, uint32_t tile, uint32_t subblock)
{
    constexpr uint32_t rawSlot = Ordinal % BUFFER_NUM;
    const uint32_t slice = kvSlice(Ordinal, subblock);
    using Nz = Tile<TileType::Vec, bfloat16_t, KV_NZ_ROWS, D, BLayout::ColMajor, K_CHUNK, D, SLayout::RowMajor, 512>;
    // The transfer uses the padded NZ source row pitch.
    static_assert(Nz::Compact == CompactMode::Null);
    Nz decoded;
    TASSIGN(decoded, UB_FLOAT + rawSlot * KV_NZ_SLOT_STRIDE);
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(KV_NZ_EVENT_BASE + rawSlot));
    // PV has finished every feature read of this K64 region.
    // The other region may still be live while this copy proceeds.
    wait_intra_block(PIPE_MTE3, kvReleaseEvent(kvSlot, Ordinal));
    custom::V2C<MatKInsert> kvPipe(L1_K + kvSlot * KV_SLOT_STRIDE, slice * K_CHUNK);
    TPUSH<decltype(kvPipe), Nz, TILE_UP_DOWN>(kvPipe, decoded);
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(KV_NZ_EVENT_BASE + rawSlot));
    // QK may consume this K64 region while the next one fills.
    set_intra_block(PIPE_MTE3, kvReadyEvent(kvSlot, Ordinal));
    stampData<PIPE_MTE3, StampStage::DEQUANT>(tile, dequantStampPart(slice));
}

AICORE inline WideValidity beginKvGather(
    __gm__ uint8_t* cache, uint32_t kvSlot, uint32_t tile, uint32_t metadataSlot, uint32_t metadataPart)
{
    WideValidity validity{};
    if constexpr (DAV_VEC) {
        if (metadataPart == 0) {
            wait_flag(PIPE_V, PIPE_S, static_cast<event_t>(INDEX_EVENT_BASE + metadataSlot));
        }
        validity.countBits = *reinterpret_cast<__ubuf__ uint32_t*>(
            UB_INDEX + metadataSlot * INDEX_SLOT_STRIDE + INDEX_ANY_OFFSET + metadataPart * sizeof(float));
        const uint32_t subblock = get_subblockid();
        // Queue the first decoded slice to L1 before the second raw gather.
        // MTE3 waits for its own regional destination credit independently.
        gatherPreparedSlice(cache, 0, tile, kvSlice(0, subblock), metadataSlot, metadataPart);
        dequantSlice<0>(tile, metadataSlot, metadataPart, subblock);
        insertKvSlice<0>(kvSlot, tile, subblock);
        if constexpr (N == 128) {
            gatherPreparedSlice(cache, 1, tile, kvSlice(1, subblock), metadataSlot, metadataPart);
        }
    }
    return validity;
}

// Decode and insert the remaining slice for N128.
template <uint32_t Ordinal>
AICORE inline void prepareKvWideSlices(
    __gm__ uint8_t* cache, uint32_t kvSlot, uint32_t tile, uint32_t metadataSlot, uint32_t metadataPart,
    uint32_t subblock)
{
    dequantSlice<Ordinal>(tile, metadataSlot, metadataPart, subblock);
    insertKvSlice<Ordinal>(kvSlot, tile, subblock);
}

AICORE inline void prepareKvWide(
    __gm__ uint8_t* cache, uint32_t kvSlot, uint32_t tile, uint32_t metadataSlot, uint32_t metadataPart)
{
    if constexpr (DAV_VEC) {
        const uint32_t subblock = get_subblockid();
        if constexpr (N == 128) {
            prepareKvWideSlices<1>(cache, kvSlot, tile, metadataSlot, metadataPart, subblock);
        }
    }
}

// The first VF publishes row-state and the predicate before the second VF
// emits probabilities and their denominator sum. BF16 P stores go directly
// to padded NZ panels; only row summaries cross the vector-function boundary.
// P and the denominator share a held row basis. Only a threshold crossing
// changes that basis; the Cube then exports the preceding resident PV segment.
template <bool AllValid, bool FirstTile, typename ScoreTile>
AICORE inline void onlineSoftmaxRows(
    ScoreTile& score, float scale, uint32_t slot, uint32_t metadataSlot, uint32_t metadataPart, uint32_t jobSlot)
{
    custom::RowStateTile maxima;
    custom::RowStateTile anchors;
    custom::DecisionTile decision;
    custom::RowStateTile sums;
    custom::RowStateTile alphas;
    custom::IndexMaskTile mask;
    custom::SoftMaxScratchTile scratch;

    TASSIGN(maxima, UB_MAX + jobSlot * 128);
    TASSIGN(anchors, UB_ANCHOR + jobSlot * 128);
    TASSIGN(decision, UB_RESCALE + slot * 32);
    TASSIGN(sums, UB_SUM + jobSlot * 128);
    TASSIGN(alphas, UB_ALPHA + slot * ALPHA_SLOT_STRIDE);
    TASSIGN(mask, indexMaskAddress(metadataSlot, metadataPart));
    TASSIGN(scratch, UB_SOFTMAX_ROWS);
    custom::TSoftMaxRow<AllValid, FirstTile>(maxima, anchors, decision, alphas, scratch, score, sums, mask, scale);
    // Publish only row summaries; probability packing keeps its own VF boundary.
    if constexpr (!FirstTile) {
        set_flag(PIPE_V, PIPE_S, static_cast<event_t>(RESCALE_LOCAL_EVENT));
    }
}

template <bool AllValid, bool FirstTile>
AICORE inline void onlineSoftmaxProbabilities(
    float scale, uint32_t slot, uint32_t metadataSlot, uint32_t metadataPart, uint32_t jobSlot)
{
    VecScore score;
    custom::ProbabilityTile probabilities;
    custom::RowStateTile anchors;
    custom::RowStateTile sums;
    custom::RowStateTile alphas;
    custom::IndexMaskTile mask;
    custom::SoftMaxScratchTile scratch;

    TASSIGN(score, UB_SCORE);
    TASSIGN(probabilities, UB_P_NZ);
    TASSIGN(anchors, UB_ANCHOR + jobSlot * 128);
    TASSIGN(sums, UB_SUM + jobSlot * 128);
    TASSIGN(alphas, UB_ALPHA + slot * ALPHA_SLOT_STRIDE);
    TASSIGN(mask, indexMaskAddress(metadataSlot, metadataPart));
    TASSIGN(scratch, UB_SOFTMAX_ROWS);
    custom::TSoftMaxProb<AllValid, FirstTile>(probabilities, sums, scratch, score, anchors, alphas, mask, scale);
}

AICORE inline void beginSoftmaxRows(
    const WideValidity& valid, float scale, uint32_t slot, uint32_t jobSlot, uint32_t metadataSlot,
    uint32_t metadataPart, bool firstTile)
{
    if constexpr (DAV_VEC) {
        VecScore score;
        custom::C2V<VecScore> scorePipe(UB_SCORE);
        scorePipe.waitReady(SCORE_READY);
        TPOP<decltype(scorePipe), VecScore, TILE_UP_DOWN>(scorePipe, score, 0);
        // Acquire P-NZ reuse early; the probability VF writes it after the row stage.
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(P_NZ_EVENT));
        constexpr uint32_t FULL_COUNT_BITS = N == 128 ? 0x43000000u : 0x42800000u;
        if (firstTile) {
            if (valid.countBits == FULL_COUNT_BITS)
                onlineSoftmaxRows<true, true>(score, scale, slot, metadataSlot, metadataPart, jobSlot);
            else
                onlineSoftmaxRows<false, true>(score, scale, slot, metadataSlot, metadataPart, jobSlot);
        } else {
            if (valid.countBits == FULL_COUNT_BITS)
                onlineSoftmaxRows<true, false>(score, scale, slot, metadataSlot, metadataPart, jobSlot);
            else
                onlineSoftmaxRows<false, false>(score, scale, slot, metadataSlot, metadataPart, jobSlot);
        }
    }
}

// Keep the score and metadata leases until this VF has reread both inputs.
// A preceding output update may run between the row and probability stages.
AICORE inline void finishSoftmaxProbabilities(
    const WideValidity& valid, float scale, uint32_t slot, uint32_t tile, uint32_t jobSlot, uint32_t metadataSlot,
    uint32_t metadataPart, bool releaseMetadata, bool firstTile)
{
    if constexpr (DAV_VEC) {
        constexpr uint32_t FULL_COUNT_BITS = N == 128 ? 0x43000000u : 0x42800000u;
        if (firstTile) {
            if (valid.countBits == FULL_COUNT_BITS)
                onlineSoftmaxProbabilities<true, true>(scale, slot, metadataSlot, metadataPart, jobSlot);
            else
                onlineSoftmaxProbabilities<false, true>(scale, slot, metadataSlot, metadataPart, jobSlot);
        } else {
            if (valid.countBits == FULL_COUNT_BITS)
                onlineSoftmaxProbabilities<true, false>(scale, slot, metadataSlot, metadataPart, jobSlot);
            else
                onlineSoftmaxProbabilities<false, false>(scale, slot, metadataSlot, metadataPart, jobSlot);
        }
        if (releaseMetadata) {
            set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(INDEX_EVENT_BASE + metadataSlot));
        }
        set_intra_block(PIPE_V, SCORE_FREE + slot % SCORE_CREDIT_NUM);
        stampCompute<PIPE_V, StampStage::SOFTMAX>(tile);
        using Nz = Tile<TileType::Vec, bfloat16_t, P_NZ_ROWS, N, BLayout::ColMajor, AIV_M, N, SLayout::RowMajor, 512>;
        Nz probabilities;
        TASSIGN(probabilities, UB_P_NZ);
        wait_intra_block(PIPE_MTE3, P_FREE + slot);
        set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(P_NZ_EVENT));
        wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(P_NZ_EVENT));
        custom::V2C<MatP> probabilityPipe(L1_P + slot * P_SLOT_STRIDE, get_subblockid() * AIV_M);
        TPUSH<decltype(probabilityPipe), Nz, TILE_UP_DOWN>(probabilityPipe, probabilities);
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(P_NZ_EVENT));
        set_intra_block(PIPE_MTE3, P_READY + slot);
        stampData<PIPE_MTE3, StampStage::SOFTMAX>(tile);
    }
}

// Each nonfirst nonempty stream tile has one publication and acknowledgement.
// Tile 0 initializes its held basis; publications start at tile 1.
// Each publication follows consumption of the preceding acknowledgement.
AICORE inline void publishRescaleDecision(uint32_t slot)
{
    if constexpr (DAV_VEC) {
        wait_flag(PIPE_V, PIPE_S, static_cast<event_t>(RESCALE_LOCAL_EVENT));
        custom::DecisionTile decision;
        TASSIGN(decision, UB_RESCALE + slot * 32);
        RescaleRequest request(nullptr, 0, rescaleSlotAddress(slot));
        TPUSH<RescaleRequest, custom::DecisionTile, TILE_NO_SPLIT>(request, decision);
    }
}

AICORE inline bool consumeRescaleDecision(uint32_t slot)
{
    if constexpr (DAV_VEC) {
        RescaleReply reply(nullptr, rescaleSlotAddress(slot) + RESCALE_SSBUF_REPLY_OFFSET, 0);
        RescaleCtrl decision;
        TPOP<RescaleReply, RescaleCtrl, TileSplitAxis::TILE_NO_SPLIT>(reply, decision);
        return decision.data() != 0;
    }
    return false;
}

template <typename PartialTile>
AICORE inline void updateOutput(PartialTile& partial, uint32_t fragment, uint32_t slot, bool initialize)
{
    custom::OutputFragmentTile running;
    custom::RowStateTile alphas;

    TASSIGN(running, UB_RUNNING_O + fragment * PV_N * sizeof(float));
    TASSIGN(alphas, UB_ALPHA + (slot % ALPHA_BUFFER_NUM) * ALPHA_SLOT_STRIDE);
    custom::TGlobalUpdate<custom::GlobalUpdateMode::STEADY>(running, partial, alphas, initialize);
}

template <uint32_t Fragment>
AICORE inline void consumeOutputFragments(uint32_t slot, uint32_t tile, bool initialize)
{
    using PartialTile = Tile<TileType::Vec, float, AIV_M, PV_N, BLayout::RowMajor, AIV_M, PV_N>;
    PartialTile partial;
    custom::C2V<PartialTile> pvPipe(UB_PV + Fragment * UB_PV_SLOT_STRIDE);
    pvPipe.waitReady(PV_READY + Fragment);
    TPOP<decltype(pvPipe), PartialTile, TILE_UP_DOWN>(pvPipe, partial, 0);
    updateOutput(partial, Fragment, slot, initialize);
    set_intra_block(PIPE_V, PV_FREE + Fragment);
    stampCompute<PIPE_V, StampStage::GU>(tile, Fragment);
    if constexpr (Fragment + 1 < D / PV_N) {
        consumeOutputFragments<Fragment + 1>(slot, tile, initialize);
    }
}

AICORE inline void consumeOutput(uint32_t slot, uint32_t tile, bool& hasRunningO)
{
    if constexpr (DAV_VEC) {
        consumeOutputFragments<0>(slot, tile, !hasRunningO);
        hasRunningO = true;
    }
}

// Only the final update consumes these summaries. PIPE_V order protects the
// shared scratch until both fragments have read their row multipliers.
AICORE inline void prepareOutputNormalization(uint32_t jobSlot)
{
    custom::RowStateTile maximum;
    custom::RowStateTile sum;
    custom::RowStateTile sink;
    custom::RowStateTile lse;
    custom::RowStateTile multipliers;

    TASSIGN(maximum, UB_ANCHOR + jobSlot * 128);
    TASSIGN(sum, UB_SUM + jobSlot * 128);
    TASSIGN(sink, UB_SINK + jobSlot * 128);
    TASSIGN(lse, UB_LSE);
    TASSIGN(multipliers, UB_DEQUANT_SCALES);
    custom::TOutputNormalize(lse, multipliers, maximum, sum, sink);
}

template <typename PartialTile>
AICORE inline void updateFinalOutput(PartialTile& partial, uint32_t fragment, bool initialize)
{
    custom::OutputFragmentTile running;
    custom::PackedOutputFragmentTile output;
    custom::RowStateTile multipliers;

    TASSIGN(running, UB_RUNNING_O + fragment * PV_N * sizeof(float));
    TASSIGN(output, UB_RUNNING_O + fragment * PV_N * sizeof(bfloat16_t));
    TASSIGN(multipliers, UB_DEQUANT_SCALES);
    custom::TGlobalUpdate<custom::GlobalUpdateMode::FINAL>(output, running, partial, multipliers, initialize);
}

template <uint32_t Fragment>
AICORE inline void consumeFinalOutputFragments(
    __gm__ bfloat16_t* output, uint64_t row, uint32_t tile, uint64_t job, bool initialize)
{
    using PartialTile = Tile<TileType::Vec, float, AIV_M, PV_N, BLayout::RowMajor, AIV_M, PV_N>;
    PartialTile partial;
    custom::C2V<PartialTile> pvPipe(UB_PV + Fragment * UB_PV_SLOT_STRIDE);
    pvPipe.waitReady(PV_READY + Fragment);
    TPOP<decltype(pvPipe), PartialTile, TILE_UP_DOWN>(pvPipe, partial, 0);
    updateFinalOutput(partial, Fragment, initialize);
    set_intra_block(PIPE_V, PV_FREE + Fragment);
    stampCompute<PIPE_V, StampStage::GU>(tile, Fragment);
    // Each packed half has disjoint storage from the other final VF.
    // Its GM transfer can overlap that VF using the FP32-sized UB row pitch.
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(OUTPUT_READY_EVENT_BASE + Fragment));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(OUTPUT_READY_EVENT_BASE + Fragment));
    using OutputTile = Tile<TileType::Vec, bfloat16_t, AIV_M, 2 * D, BLayout::RowMajor, AIV_M, PV_N>;
    using OutputGlobal = GlobalTensor<bfloat16_t, Shape<1, 1, 1, AIV_M, PV_N>, pto::Stride<1, 1, 1, D, 1>>;
    OutputTile packed;
    OutputGlobal destination(output + (row + get_subblockid() * AIV_M) * D + Fragment * PV_N);
    TASSIGN(packed, UB_RUNNING_O + Fragment * PV_N * sizeof(bfloat16_t));
    custom::TSTORE(destination, packed);
    stampData<PIPE_MTE3, StampStage::OUTPUT>(job, Fragment);
    if constexpr (Fragment + 1 < D / PV_N) {
        consumeFinalOutputFragments<Fragment + 1>(output, row, tile, job, initialize);
    }
}

AICORE inline void consumeFinalOutput(
    __gm__ bfloat16_t* output, uint64_t row, uint32_t tile, uint64_t job, uint32_t jobSlot, bool hasRunningO)
{
    if constexpr (DAV_VEC) {
        prepareOutputNormalization(jobSlot);
        consumeFinalOutputFragments<0>(output, row, tile, job, !hasRunningO);
    }
}

// initJobOutput already cleared every byte consumed as packed BF16. Empty
// jobs need only their exact +infinity LSE, independent of the sink value.
AICORE inline void finishEmptyOutput()
{
    constexpr uint32_t INFINITY_BITS = 0x7f800000u;
    custom::RowStateTile lse;
    TASSIGN(lse, UB_LSE);
    custom::TEXPANDS<INFINITY_BITS>(lse);
}

// Finish O and packed LSE stores before releasing their shared UB allocation
// to the next job. Row-state reuse has a separate MTE2 credit.
// Nonempty callers must first run consumeFinalOutput: its final-fragment
// V->MTE3 ID5 orders both packed O and the preceding LSE writes. MTE3 then
// queues the LSE copy after both O copies under that producer fence.
AICORE inline void finishJobVector(
    __gm__ bfloat16_t* output, __gm__ float* lse, uint64_t row, uint32_t batch, uint32_t query, uint32_t head,
    const FlashMLABf16Config& cfg, uint64_t job, uint32_t jobSlot, bool empty)
{
    if constexpr (DAV_VEC) {
        if (empty) {
            finishEmptyOutput();
        }
        stampCompute<PIPE_V, StampStage::OUTPUT>(job);
        set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(JOB_ROW_EVENT_BASE + jobSlot));
        const uint32_t sub = get_subblockid();
        if (empty) {
            // Empty jobs have no final-fragment handoff; fence their zero O and LSE.
            localFence<PIPE_V, PIPE_MTE3>();
            using OutputTile = Tile<TileType::Vec, bfloat16_t, AIV_M, 2 * D, BLayout::RowMajor, AIV_M, D>;
            using OutputGlobal = GlobalTensor<bfloat16_t, Shape<1, 1, 1, AIV_M, D>, pto::Stride<1, 1, 1, D, 1>>;
            OutputTile packed;
            OutputGlobal destination(output + (row + sub * AIV_M) * D);
            TASSIGN(packed, UB_RUNNING_O);
            custom::TSTORE(destination, packed);
            stampData<PIPE_MTE3, StampStage::OUTPUT>(job, D / PV_N - 1);
        }
        const uint64_t lseBase =
            (static_cast<uint64_t>(batch) * cfg.numHeads + head + sub * AIV_M) * cfg.queryLength + query;
        const bool contiguousLse = cfg.queryLength == 1;
        const uint32_t lseBursts = contiguousLse ? 1 : AIV_M;
        const uint32_t lseBurstBytes = contiguousLse ? AIV_M * sizeof(float) : sizeof(float);
        using LseTile = Tile<TileType::Vec, float, 1, AIV_M>;
        using LseGlobal = GlobalTensor<float, Shape<1, 1, 1, -1, -1>, pto::Stride<1, 1, 1, -1, 1>>;
        LseTile packedLse;
        LseGlobal destination(
            lse + lseBase, Shape<1, 1, 1, -1, -1>(lseBursts, lseBurstBytes / sizeof(float)),
            pto::Stride<1, 1, 1, -1, 1>(cfg.queryLength));
        TASSIGN(packedLse, UB_LSE);
        custom::TSTORE<true>(destination, packedLse);
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(OUTPUT_REUSE_EVENT));
        stampData<PIPE_MTE3, StampStage::OUTPUT>(job, D / PV_N);
    }
}

} // namespace flash_mla_bf16
#endif // FLASH_MLA_BF16_AIV_H_
