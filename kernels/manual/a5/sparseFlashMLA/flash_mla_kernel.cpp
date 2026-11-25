/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "flash_mla_aic.h"
#include "flash_mla_aiv.h"
#include "flash_mla_api.h"

namespace {
using namespace flash_mla_bf16;

AICORE inline uint32_t selectionLength(__gm__ int32_t* lengths, uint32_t batch, uint32_t maximum)
{
    if (lengths == nullptr) {
        return maximum;
    }
    // Device metadata is validated by the caller: every entry is in [0,maximum].
    return static_cast<uint32_t>(lengths[batch]);
}

// Queue up to 256 indices from one cache scope into a metadata bank.
AICORE inline void prepareJobIndices(
    uint32_t tile, uint32_t mainTiles, uint32_t mainLength, uint32_t extraLength, uint32_t queryRow,
    __gm__ int32_t* indices, __gm__ int32_t* extraIndices, const FlashMLABf16Config& cfg, uint32_t metadataSlot)
{
    if constexpr (DAV_VEC) {
        if (tile < mainTiles) {
            prepareIndices(
                indices + static_cast<uint64_t>(queryRow) * cfg.topK, tile * N, mainLength,
                static_cast<uint64_t>(cfg.pageSize) * cfg.numPages, metadataSlot);
        } else {
            prepareIndices(
                extraIndices + static_cast<uint64_t>(queryRow) * cfg.extraTopK, (tile - mainTiles) * N, extraLength,
                static_cast<uint64_t>(cfg.extraPageSize) * cfg.extraNumPages, metadataSlot);
        }
    }
}

struct PipelineWork {
    uint64_t job = 0;
    uint32_t jobSlot = 0;
    uint32_t mainLength = 0, extraLength = 0, mainTiles = 0, totalTiles = 0;
    uint32_t tile = 0, streamTile = 0, metadataSlot = 0, metadataPart = 0;
    bool valid = false;

    // Validated launches have one or two M-head tiles per query. The job
    // order is [query row, head tile], so its output row is exactly job*M.
    AICORE uint64_t row() const { return job * M; }
    AICORE uint32_t queryRow(const FlashMLABf16Config& cfg) const
    {
        return static_cast<uint32_t>(job) >> (cfg.numHeads / M - 1);
    }
    AICORE uint32_t head(const FlashMLABf16Config& cfg) const
    {
        return (static_cast<uint32_t>(job) & (cfg.numHeads / M - 1)) * M;
    }
    AICORE bool last() const { return tile + 1 >= totalTiles; }
    AICORE bool metadataLast() const
    {
        return metadataPart + 1 == INDEX_BATCH_TILES || tile + 1 == mainTiles || last();
    }
    AICORE uint32_t slot() const { return streamTile % BUFFER_NUM; }
    AICORE uint32_t kvSlot() const { return streamTile % KV_BUFFER_NUM; }
};

struct PipelineCursor {
    PipelineWork work;
    uint32_t logicalJobs, launchCount;
    __gm__ int32_t* lengths;
    __gm__ int32_t* extraLengths;
    const FlashMLABf16Config& cfg;

    AICORE void loadJob()
    {
        work.valid = work.job < logicalJobs;
        if (!work.valid) {
            return;
        }
        const uint32_t batch = work.queryRow(cfg) / cfg.queryLength;
        work.mainLength = selectionLength(lengths, batch, cfg.topK);
        work.extraLength = selectionLength(extraLengths, batch, cfg.extraTopK);
        work.mainTiles = (work.mainLength + N - 1) / N;
        work.totalTiles = work.mainTiles + (work.extraLength + N - 1) / N;
    }

    AICORE PipelineWork take()
    {
        const PipelineWork result = work;
        if (work.valid) {
            // The modulo-2 score/P ring and modulo-3 KV ring share this phase.
            // Wrap at their common period of six stream tiles.
            if (++work.streamTile == BUFFER_NUM * KV_BUFFER_NUM) {
                work.streamTile = 0;
            }
            // A batch ends at 256 tokens, a cache boundary or a job boundary.
            // Empty jobs consume stream positions, but no metadata bank.
            if (work.totalTiles != 0) {
                if (work.metadataLast()) {
                    if (++work.metadataSlot == INDEX_BUFFER_NUM) {
                        work.metadataSlot = 0;
                    }
                    work.metadataPart = 0;
                } else {
                    ++work.metadataPart;
                }
            }
            if (work.last()) {
                work.tile = 0;
                work.job += launchCount;
                work.jobSlot ^= 1;
                loadJob();
            } else {
                ++work.tile;
            }
        }
        return result;
    }
};

AICORE inline void prefetchWorkIndices(
    const PipelineWork& work, __gm__ int32_t* indices, __gm__ int32_t* extraIndices, const FlashMLABf16Config& cfg)
{
    if constexpr (DAV_VEC) {
        if (work.valid && work.totalTiles != 0 && work.metadataPart == 0) {
            prepareJobIndices(
                work.tile, work.mainTiles, work.mainLength, work.extraLength, work.queryRow(cfg), indices, extraIndices,
                cfg, work.metadataSlot);
        }
    }
}

AICORE inline void runCrossJobPipeline(
    __gm__ bfloat16_t* output, __gm__ float* lse, __gm__ bfloat16_t* q, __gm__ uint8_t* cache, __gm__ float* sink,
    __gm__ int32_t* indices, __gm__ int32_t* topkLength, __gm__ uint8_t* extraCache, __gm__ int32_t* extraIndices,
    __gm__ int32_t* extraTopkLength, const FlashMLABf16Config& cfg, uint32_t logicalJobs, uint32_t launchCount)
{
    PipelineCursor cursor{{}, logicalJobs, launchCount, topkLength, extraTopkLength, cfg};
    cursor.work.job = block_idx;
    cursor.loadJob();
    PipelineWork current = cursor.take();
    PipelineWork next = cursor.take();
    PipelineWork scoreWork{}, softmaxWork{}, outputWork{};
    WideValidity valid[KV_BUFFER_NUM]{};
    bool hasRunningO = false;
    prefetchWorkIndices(current, indices, extraIndices, cfg);
    if constexpr (DAV_VEC) {
        if (current.valid && current.totalTiles != 0) {
            valid[current.kvSlot()] = beginKvGather(
                current.tile < current.mainTiles ? cache : extraCache, current.kvSlot(), current.tile,
                current.metadataSlot, current.metadataPart);
        }
        // Prime indices for the next two items after beginKvGather queues
        // the first KV insertion and any second raw gather.
        // Short scopes can use all three distinct metadata banks; a nonzero
        // metadataPart already belongs to a prepared batch and is a no-op.
        prefetchWorkIndices(next, indices, extraIndices, cfg);
        prefetchWorkIndices(cursor.work, indices, extraIndices, cfg);
    }
    if constexpr (DAV_CUBE) {
        if (current.valid && current.totalTiles != 0) {
            loadQ(q, current.row(), current.job);
        }
    }
    // Three KV slots keep DQ two tiles ahead of SM/PV. Consume the oldest
    // V before submitting current QK so MTE3 can refill its L1 regions sooner.
    // The preceding QK waits in L0C until the single UB score tile accepts it.
    // Three invalid items drain score publication, SM/PV and GU across jobs.
    while (current.valid || scoreWork.valid || softmaxWork.valid || outputWork.valid) {
        if (current.valid && current.tile == 0) {
            stampJob(current.job, 0);
        }
        if constexpr (DAV_CUBE) {
            const bool hasQk = current.valid && current.totalTiles != 0;
            const bool hasScore = scoreWork.valid && scoreWork.totalTiles != 0;
            const bool hasPv = softmaxWork.valid && softmaxWork.totalTiles != 0;
            // Consume the oldest PV while gather/DQ retain their lookahead.
            // PV exports the preceding score after any resident segment flush;
            // the score's L0C parity is free before its next QK reuse.
            if (hasPv) {
                pv(softmaxWork.slot(), softmaxWork.kvSlot(), softmaxWork.tile, softmaxWork.last(), hasScore,
                   scoreWork.slot(), scoreWork.tile);
            } else if (hasScore) {
                exportQk(scoreWork.slot(), scoreWork.tile);
            }
            if (hasScore && scoreWork.tile != 0) {
                // The next iteration may flush this PV segment. Capture its
                // producer before unrelated QK enters M; FIX consumes the
                // token even when that next rescale predicate skips.
                localFence<PIPE_M, PIPE_FIX>();
            }
            if (hasQk) {
                qk(current.slot(), current.kvSlot(), current.tile, current.last(), hasPv);
            }
            if (next.valid && next.tile == 0 && next.totalTiles != 0) {
                // Current QK already queued each half's last read and release,
                // including a singleton, before the next load can wait on it.
                loadQ(q, next.row(), next.job);
            }
        }
        if constexpr (DAV_VEC) {
            if (current.valid && current.totalTiles != 0) {
                const bool main = current.tile < current.mainTiles;
                prepareKvWide(
                    main ? cache : extraCache, current.kvSlot(), current.tile, current.metadataSlot,
                    current.metadataPart);
            }
            // Normally submit the next batch immediately after dequant. A
            // singleton batch can put next and pending softmax in the same
            // bank; submit that reuse only after softmax releases its credit.
            const bool deferIndices = next.valid && next.totalTiles != 0 && next.metadataPart == 0 &&
                                      softmaxWork.valid && softmaxWork.totalTiles != 0 &&
                                      next.metadataSlot == softmaxWork.metadataSlot;
            // The first two next items were primed above. Softmax becomes
            // valid on iteration two; that stage bit marks the end of warmup.
            if (!deferIndices && softmaxWork.valid) {
                prefetchWorkIndices(next, indices, extraIndices, cfg);
            }
            if (softmaxWork.valid) {
                const bool firstTile = softmaxWork.tile == 0;
                if (firstTile) {
                    initJobRows(sink, softmaxWork.head(cfg), softmaxWork.jobSlot);
                }
                if (softmaxWork.totalTiles != 0) {
                    beginSoftmaxRows(
                        valid[softmaxWork.kvSlot()], cfg.softmaxScale, softmaxWork.slot(), softmaxWork.jobSlot,
                        softmaxWork.metadataSlot, softmaxWork.metadataPart, firstTile);
                    // Return the preceding PV destination before probability
                    // exponentiation. Its alpha lives in the other parity.
                    // Final/init/store work runs after probability publication.
                    // A nonfinal output item is followed by this same job's
                    // next softmax tile. Nonfirst items are always nonempty.
                    if (outputWork.valid && outputWork.tile != 0 && !outputWork.last()) {
                        if (consumeRescaleDecision(outputWork.slot())) {
                            consumeOutput(outputWork.slot(), outputWork.tile, hasRunningO);
                        }
                    }
                    finishSoftmaxProbabilities(
                        valid[softmaxWork.kvSlot()], cfg.softmaxScale, softmaxWork.slot(), softmaxWork.tile,
                        softmaxWork.jobSlot, softmaxWork.metadataSlot, softmaxWork.metadataPart,
                        softmaxWork.metadataLast(), firstTile);
                }
            }
            if (deferIndices) {
                prefetchWorkIndices(next, indices, extraIndices, cfg);
            }
            if (outputWork.valid) {
                if (outputWork.tile == 0) {
                    initJobOutput(outputWork.totalTiles == 0);
                    // Output ownership can outlive this job's first softmax.
                    hasRunningO = false;
                }
                if (outputWork.last()) {
                    if (outputWork.totalTiles != 0) {
                        // Nonfinal updates were consumed between the softmax
                        // VFs. Only the final tile's update remains here.
                        if (outputWork.tile != 0 && consumeRescaleDecision(outputWork.slot())) {
                            consumeOutput(outputWork.slot(), outputWork.tile, hasRunningO);
                        }
                        consumeFinalOutput(
                            output, outputWork.row(), outputWork.tile, outputWork.job, outputWork.jobSlot, hasRunningO);
                    }
                    const uint32_t queryRow = outputWork.queryRow(cfg);
                    const uint32_t batch = queryRow / cfg.queryLength;
                    const uint32_t query = queryRow % cfg.queryLength;
                    finishJobVector(
                        output, lse, outputWork.row(), batch, query, outputWork.head(cfg), cfg, outputWork.job,
                        outputWork.jobSlot, outputWork.totalTiles == 0);
                }
            }
            // A deferred index VF follows softmax, so publish before waiting
            // for those indices. Otherwise gather while the predicate runs.
            if (deferIndices && softmaxWork.valid && softmaxWork.totalTiles != 0 && softmaxWork.tile != 0) {
                publishRescaleDecision(softmaxWork.slot());
            }
            // The three-entry validity ring follows KV ownership. Softmax has
            // consumed the oldest entry before the next gather replaces it.
            if (next.valid && next.totalTiles != 0) {
                valid[next.kvSlot()] = beginKvGather(
                    next.tile < next.mainTiles ? cache : extraCache, next.kvSlot(), next.tile, next.metadataSlot,
                    next.metadataPart);
            }
            if (!deferIndices && softmaxWork.valid && softmaxWork.totalTiles != 0 && softmaxWork.tile != 0) {
                publishRescaleDecision(softmaxWork.slot());
            }
        }
        if (outputWork.valid && outputWork.last()) {
            stampJob(outputWork.job, 1);
        }
        outputWork = softmaxWork;
        softmaxWork = scoreWork;
        scoreWork = current;
        current = next;
        next = cursor.take();
    }
}

__global__ AICORE void flashMlaBf16Kernel(
    __gm__ bfloat16_t* output, __gm__ float* lse, __gm__ bfloat16_t* q, __gm__ uint8_t* cache, __gm__ float* sink,
    __gm__ int32_t* indices, __gm__ int32_t* topkLength, __gm__ uint8_t* extraCache, __gm__ int32_t* extraIndices,
    __gm__ int32_t* extraTopkLength, FlashMLABf16Config cfg, uint32_t logicalJobs, uint32_t launchCount)
{
    // Physical credits persist across all assigned jobs, including empty jobs.
    initAicPipeline();
    initVectorPipeline();
    runCrossJobPipeline(
        output, lse, q, cache, sink, indices, topkLength, extraCache, extraIndices, extraTopkLength, cfg, logicalJobs,
        launchCount);
    drainAicPipeline();
    drainVectorPipeline();
}

void validateLaunch(const FlashMLABf16Config& cfg, uint32_t launchCount)
{
    constexpr uint64_t MAX_TOKENS = std::numeric_limits<int32_t>::max();
    if (cfg.batchSize <= 0 || cfg.queryLength <= 0 || (cfg.numHeads != 64 && cfg.numHeads != 128) || cfg.headDim != D ||
        cfg.pageSize <= 0 || cfg.numPages <= 0 || cfg.topK < 0 || cfg.extraTopK < 0 || launchCount == 0) {
        throw std::invalid_argument("Invalid FlashMLA BF16 dimensions or launch count");
    }
    if (!std::isfinite(cfg.softmaxScale)) {
        throw std::invalid_argument("FlashMLA softmax scale must be finite");
    }
    if (static_cast<uint64_t>(cfg.pageSize) * cfg.numPages > MAX_TOKENS) {
        throw std::invalid_argument("FlashMLA page capacity exceeds the signed token index range");
    }
    if (cfg.extraTopK > 0 && (cfg.extraPageSize <= 0 || cfg.extraNumPages <= 0 ||
                              static_cast<uint64_t>(cfg.extraPageSize) * cfg.extraNumPages > MAX_TOKENS)) {
        throw std::invalid_argument("Invalid FlashMLA extra cache capacity");
    }
    const uint64_t jobs = static_cast<uint64_t>(cfg.batchSize) * cfg.queryLength * (cfg.numHeads / M);
    if (jobs > std::numeric_limits<uint32_t>::max() || launchCount > jobs) {
        throw std::invalid_argument("FlashMLA job count exceeds launch bounds");
    }
}
} // namespace

extern "C" void LaunchFlashMLABf16(
    void* output, float* lse, const uint16_t* q, const uint8_t* cache, const float* sink, const int32_t* indices,
    const int32_t* topkLength, const uint8_t* extraCache, const int32_t* extraIndices, const int32_t* extraTopkLength,
    const FlashMLABf16Config& config, uint32_t launchCount, void* stream)
{
    validateLaunch(config, launchCount);
    if (output == nullptr || lse == nullptr || q == nullptr ||
        (config.topK > 0 && (cache == nullptr || indices == nullptr)) ||
        (config.extraTopK > 0 && (extraCache == nullptr || extraIndices == nullptr))) {
        throw std::invalid_argument("Missing required FlashMLA BF16 tensor");
    }
    const uint32_t logicalJobs = static_cast<uint64_t>(config.batchSize) * config.queryLength * (config.numHeads / M);
    flashMlaBf16Kernel<<<launchCount, nullptr, stream>>>(
        (__gm__ bfloat16_t*)output, (__gm__ float*)lse, (__gm__ bfloat16_t*)q, (__gm__ uint8_t*)cache,
        (__gm__ float*)sink, (__gm__ int32_t*)indices, (__gm__ int32_t*)topkLength, (__gm__ uint8_t*)extraCache,
        (__gm__ int32_t*)extraIndices, (__gm__ int32_t*)extraTopkLength, config, logicalJobs, launchCount);
}
