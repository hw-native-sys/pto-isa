/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Communication Kernel (Vec Arch) for GEMM + AllReduce — HCCL backend
//
// Overlapped RS/AG kernel.
// RS uses TPUT<AtomicAdd> to accumulate directly at the owner's reduced_output,
// then publishes owner-local ready counters so AG can drain completed subtiles
// without a device-wide RS→AG barrier.
//
// Signal matrix layout in HCCL window (per rank):
//   [0 .. MAX_RANKS-1]                Reserved legacy cross-rank barrier counters
//   [MAX_RANKS]                       Reserved legacy local broadcast flag slot
//   [MAX_RANKS + 1]                   Reserved legacy intra-rank arrival slot
//   [G_SIGNAL_SUBTILE_READY_OFFSET .. G_SIGNAL_AG_SUMMARY_OFFSET-1]
//                                     Owner-local subtile-ready counters
//   [G_SIGNAL_AG_SUMMARY_OFFSET .. G_SIGNAL_TOTAL_SLOTS-1]
//                                     Per-AG-block summary wakeup counters

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <cstddef>
#include <cstdint>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include <pto/pto-inst.hpp>

#include "common.hpp"
#include "ready_queue.hpp"

#include "gemm_ar_config.h"
#include "kernel_launchers.h"

// Signal matrix layout (per rank, in HCCL RDMA window):
//   [0 .. G_SIGNAL_RS_DONE_SLOTS-1]             Reserved legacy cross-rank barrier counters
//   [G_SIGNAL_LOCAL_FLAG_OFFSET]                Reserved legacy local broadcast flag slot
//   [G_SIGNAL_INTRA_RANK_COUNTER_OFFSET]        Reserved legacy intra-rank arrival slot
//   [G_SIGNAL_SUBTILE_READY_OFFSET .. G_SIGNAL_AG_SUMMARY_OFFSET-1]
//                                              Owner-local subtile-ready counters
//   [G_SIGNAL_AG_SUMMARY_OFFSET .. G_SIGNAL_TOTAL_SLOTS-1]
//                                              Per-AG-block summary wakeup counters

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using RsSubtileData = pto::Tile<pto::TileType::Vec, half, G_COMM_SUB_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;
using AgSubtileData = RsSubtileData;

static constexpr size_t RS_SUBTILE_UB_BYTES = ((G_COMM_SUB_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;
static constexpr size_t AG_SUBTILE_UB_OFFSET = RS_SUBTILE_UB_BYTES * 2;

// ============================================================================
// RS helpers — broken out to reduce cyclomatic complexity
// ============================================================================

struct RsPendingMeta {
    int owner;
    int local_subtile_id;
    int ag_summary_block;
};

AICORE inline uint64_t RsSubtileRowOffset(int tile_idx, int stripe_id)
{
    const uint32_t mi = tile_idx / G_N_TILES;
    const uint32_t ni = tile_idx % G_N_TILES;
    const uint64_t tile_base = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;
    return tile_base + (uint64_t)stripe_id * G_COMM_SUB_M * G_N;
}

AICORE inline int RsOwnerLocalSubtileId(int tile_idx, int nranks, int stripe_id)
{
    const int safe_nranks = (nranks > 0) ? nranks : 1;
    const int owner_local_tile = tile_idx / safe_nranks;
    return owner_local_tile * static_cast<int>(G_COMM_SUBTILES_PER_TILE) + stripe_id;
}

// Subtile -> owning comm block mapping.
// Reversed-stripe layout: subtile k goes to block (num_comm_blocks-1-k%num_comm_blocks).
// Rationale: combined with A's rsN balancing (blocks 0..remainder-1 get the
// heavier rsN=ceil and blocks remainder.. get the lighter rsN=floor), reversing
// AG makes AG's "heavy" blocks (agN=ceil) land on RS's "light" blocks, so the
// total (rsN+agN) workload is flatter. Must stay consistent with
// AgInitAssignedState starting index.
AICORE inline int AgSummaryBlockForSubtile(int local_subtile_id, int num_comm_blocks)
{
    return (num_comm_blocks > 0) ? (num_comm_blocks - 1 - (local_subtile_id % num_comm_blocks)) : 0;
}

AICORE inline __gm__ int32_t *AgSummarySlotPtr(__gm__ int32_t *signal_base, int summary_block)
{
    return signal_base + G_SIGNAL_AG_SUMMARY_OFFSET + summary_block * G_SIGNAL_AG_SUMMARY_STRIDE;
}

AICORE inline void RsNotifySubtileReady(__gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int my_rank,
                                        const RsPendingMeta &meta)
{
    __gm__ int32_t *counter = signal_base + G_SIGNAL_SUBTILE_READY_OFFSET + meta.local_subtile_id;
    if (meta.owner != my_rank) {
        counter = HcclRemotePtr(hcclCtx, counter, meta.owner);
    }
    pto::comm::Signal sig(counter);
    pto::comm::TNOTIFY(sig, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
}

AICORE inline void RsNotifyAgSummary(__gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int my_rank,
                                     const RsPendingMeta &meta)
{
    __gm__ int32_t *counter = AgSummarySlotPtr(signal_base, meta.ag_summary_block);
    if (meta.owner != my_rank) {
        counter = HcclRemotePtr(hcclCtx, counter, meta.owner);
    }
    pto::comm::Signal sig(counter);
    pto::comm::TNOTIFY(sig, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
}

AICORE inline void RsPublishSubtileReady(__gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int my_rank,
                                         const RsPendingMeta &meta)
{
    // Match the proven allgather_gemm protocol: flush local pipeline state and
    // commit DDR visibility before publishing the doorbell for this subtile.
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    RsNotifySubtileReady(hcclCtx, signal_base, my_rank, meta);
    RsNotifyAgSummary(hcclCtx, signal_base, my_rank, meta);
}

// Legacy cross-rank barrier used by the staged validation flow. The current
// publish-only experiment only uses phase 0, which maps to the reserved legacy
// slots at the front of signal_matrix.
AICORE inline void DeviceBarrier(__gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int phase, int my_rank,
                                 int nranks, int comm_core_idx, int num_comm_blocks, int32_t expected = 1)
{
    pipe_barrier(PIPE_ALL);
    // A5 needs a stronger phase-completion fence here: all prior RS/AG window
    // writes must become globally visible before this block advertises barrier
    // arrival, otherwise later phases or the next iteration can observe partial
    // data with stale/late signal updates.
    dsb(DSB_DDR);

    __gm__ int32_t *arrival = signal_base + G_SIGNAL_INTRA_RANK_COUNTER_OFFSET + phase;
    {
        pto::comm::Signal arrSig(arrival);
        pto::comm::TNOTIFY(arrSig, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
    }

    if (comm_core_idx == 0) {
        {
            pto::comm::Signal arrSig(arrival);
            pto::comm::TWAIT(arrSig, static_cast<int32_t>(num_comm_blocks), pto::comm::WaitCmp::GE);
        }

        __gm__ int32_t *phase_base = signal_base + phase * MAX_RANKS;
        for (int r = 0; r < nranks; ++r) {
            if (r == my_rank) {
                continue;
            }
            __gm__ int32_t *remote_sig = HcclRemotePtr(hcclCtx, phase_base + my_rank, r);
            pto::comm::Signal sig(remote_sig);
            pto::comm::TNOTIFY(sig, static_cast<int32_t>(1), pto::comm::NotifyOp::AtomicAdd);
        }

        for (int r = 0; r < nranks; ++r) {
            if (r == my_rank) {
                continue;
            }
            pto::comm::Signal sig(phase_base + r);
            pto::comm::TWAIT(sig, expected, pto::comm::WaitCmp::GE);
        }

        __gm__ int32_t *local_flag = signal_base + G_SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal local_sig(local_flag);
        pto::comm::TNOTIFY(local_sig, expected, pto::comm::NotifyOp::Set);
    } else {
        __gm__ int32_t *local_flag = signal_base + G_SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal local_sig(local_flag);
        pto::comm::TWAIT(local_sig, expected, pto::comm::WaitCmp::GE);
    }

    pipe_barrier(PIPE_ALL);
}

// Round-robin poll across assigned queues; returns tile index or -1.
AICORE inline int32_t RsPollQueues(volatile __gm__ MultiBlockQueueSet *qset, const int *my_queue_indices,
                                   int my_queue_count, int32_t *heads, const int32_t *queue_max_tiles,
                                   int &next_queue_offset)
{
    if (my_queue_count <= 0) {
        return -1;
    }
    const int queue_count = my_queue_count;
    for (int i = 0; i < queue_count; i++) {
        int local_idx = (next_queue_offset + i) % queue_count;
        int32_t q = my_queue_indices[local_idx];

        if (heads[q] >= queue_max_tiles[q])
            continue;

        volatile __gm__ PerBlockQueue *pq = GetMyBlockQueue(qset, q);
        int32_t tile = PerBlockQueueTryDequeue(pq, heads[q]);

        if (tile >= 0) {
            heads[q]++;
            next_queue_offset = (local_idx + 1) % queue_count;
            return tile;
        }
    }
    return -1;
}

// Ping-pong pipeline: load current tile, optionally store previous tile.
AICORE inline void RsPipelineStep(RsSubtileData &pingTile, RsSubtileData &pongTile, Global &pp_pending_dst,
                                  RsPendingMeta &pp_pending_meta, Global &dstG, const RsPendingMeta &curMeta,
                                  Global &srcG, int pp_count, __gm__ HcclDeviceContext *hcclCtx,
                                  __gm__ int32_t *signal_base, int my_rank)
{
    bool use_ping = (pp_count % 2 == 0);
    RsSubtileData &curTile = use_ping ? pingTile : pongTile;
    event_t curEv = use_ping ? EVENT_ID0 : EVENT_ID1;

    if (pp_count == 0) {
        TLOAD(curTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
    } else {
        RsSubtileData &prevTile = use_ping ? pongTile : pingTile;
        event_t prevEv = use_ping ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEv);
        TSTORE_IMPL<RsSubtileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, prevTile);
        TLOAD(curTile, srcG);
        set_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
        RsPublishSubtileReady(hcclCtx, signal_base, my_rank, pp_pending_meta);
    }

    pp_pending_dst = dstG;
    pp_pending_meta = curMeta;
}

// Drain the last tile still in the pipeline after the RS loop.
AICORE inline void RsFlushPipeline(RsSubtileData &pingTile, RsSubtileData &pongTile, Global &pp_pending_dst,
                                   const RsPendingMeta &pp_pending_meta, int pp_count,
                                   __gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int my_rank)
{
    if (pp_count <= 0)
        return;

    bool last_was_ping = ((pp_count - 1) % 2 == 0);
    RsSubtileData &lastTile = last_was_ping ? pingTile : pongTile;
    event_t lastEv = last_was_ping ? EVENT_ID0 : EVENT_ID1;
    wait_flag(PIPE_MTE2, PIPE_MTE3, lastEv);
    TSTORE_IMPL<RsSubtileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, lastTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
    wait_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
    RsPublishSubtileReady(hcclCtx, signal_base, my_rank, pp_pending_meta);
}

AICORE inline void RsProcessTileStripes(__gm__ half *gemm_output, __gm__ half *reduced_output,
                                        __gm__ int32_t *signal_base, __gm__ HcclDeviceContext *hcclCtx, int my_rank,
                                        int safe_nranks, int32_t tile_idx, const ShapeDyn &subtileShape,
                                        const StrideDyn &subtileStride, RsSubtileData &pingTile,
                                        RsSubtileData &pongTile, Global &pp_pending_dst, RsPendingMeta &pp_pending_meta,
                                        int &pp_count, int num_comm_blocks)
{
    const int owner = tile_idx % safe_nranks;
    for (int stripe_id = 0; stripe_id < static_cast<int>(G_COMM_SUBTILES_PER_TILE); ++stripe_id) {
        const uint64_t row_offset = RsSubtileRowOffset(tile_idx, stripe_id);
        const int local_subtile_id = RsOwnerLocalSubtileId(tile_idx, safe_nranks, stripe_id);
        const int ag_summary_block = AgSummaryBlockForSubtile(local_subtile_id, num_comm_blocks);
        Global srcG(gemm_output + row_offset, subtileShape, subtileStride);

        __gm__ half *dst_ptr = (owner == my_rank) ? reduced_output + row_offset :
                                                    HcclRemotePtr(hcclCtx, reduced_output, owner) + row_offset;
        Global dstG(dst_ptr, subtileShape, subtileStride);

        RsPendingMeta curMeta{owner, local_subtile_id, ag_summary_block};
        RsPipelineStep(pingTile, pongTile, pp_pending_dst, pp_pending_meta, dstG, curMeta, srcG, pp_count, hcclCtx,
                       signal_base, my_rank);
        pp_count++;
    }
}

struct RsWaitTarget {
    int queue_idx;
    int head;
    int target;
};

AICORE inline bool RsGetWaitTarget(const int *my_queue_indices, int my_queue_count, const int32_t *heads,
                                   const int32_t *queue_max_tiles, int next_queue_offset, RsWaitTarget &target)
{
    target.queue_idx = -1;
    target.head = -1;
    target.target = -1;
    if (my_queue_count <= 0) {
        return false;
    }
    const int queue_count = my_queue_count;
    for (int i = 0; i < queue_count; i++) {
        int local_idx = (next_queue_offset + i) % queue_count;
        int32_t q = my_queue_indices[local_idx];
        if (heads[q] < queue_max_tiles[q]) {
            target.queue_idx = q;
            target.head = heads[q];
            target.target = heads[q] + 1;
            return true;
        }
    }
    return false;
}

// Block on the first non-exhausted queue via TWAIT.
AICORE inline void RsWaitOnQueue(volatile __gm__ MultiBlockQueueSet *qset, const int *my_queue_indices,
                                 int my_queue_count, const int32_t *heads, const int32_t *queue_max_tiles,
                                 int next_queue_offset)
{
    RsWaitTarget target;
    if (!RsGetWaitTarget(my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset, target)) {
        return;
    }
    volatile __gm__ PerBlockQueue *pq = GetMyBlockQueue(qset, target.queue_idx);
    pto::comm::Signal sig(const_cast<__gm__ int32_t *>(&pq->count));
    pto::comm::TWAIT(sig, target.target, pto::comm::WaitCmp::GE);
}

// Build the per-block queue assignment and tile counts for this RS block.
// Returns total expected tiles; writes my_queue_count via out-param.
AICORE inline int RsInitQueueState(int comm_core_idx, int num_compute_blocks, int num_comm_blocks,
                                   int *my_queue_indices, int &my_queue_count, int32_t *queue_max_tiles)
{
    my_queue_count = 0;
    if (num_compute_blocks <= 0 || num_comm_blocks <= 0) {
        return 0;
    }
    if (comm_core_idx < 0 || comm_core_idx >= num_comm_blocks) {
        return 0;
    }

    for (int q = comm_core_idx; q < num_compute_blocks; q += num_comm_blocks) {
        my_queue_indices[my_queue_count++] = q;
    }

    const int total_tiles = G_NUM_TILES;

    int my_expected_tiles = 0;
    for (int i = 0; i < my_queue_count; i++) {
        int q = my_queue_indices[i];
        int block_tiles = GEMM_AR_BLOCK_TILE_COUNT(q, total_tiles, num_compute_blocks);
        queue_max_tiles[q] = block_tiles;
        my_expected_tiles += block_tiles;
    }
    return my_expected_tiles;
}

AICORE inline bool RsTryProcessOneTile(__gm__ half *gemm_output, __gm__ half *reduced_output,
                                       __gm__ int32_t *signal_base, volatile __gm__ MultiBlockQueueSet *qset,
                                       __gm__ HcclDeviceContext *hcclCtx, int my_rank, int nranks,
                                       const ShapeDyn &subtileShape, const StrideDyn &subtileStride,
                                       RsSubtileData &pingTile, RsSubtileData &pongTile, Global &pp_pending_dst,
                                       RsPendingMeta &pp_pending_meta, const int *my_queue_indices, int my_queue_count,
                                       int32_t *heads, const int32_t *queue_max_tiles, int &next_queue_offset,
                                       int &pp_count, int32_t &tiles_sent, int32_t my_expected_tiles,
                                       int num_comm_blocks)
{
    if (tiles_sent >= my_expected_tiles) {
        return false;
    }

    int32_t tile_idx = RsPollQueues(qset, my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset);
    if (tile_idx < 0) {
        return false;
    }

    const int safe_nranks = (nranks > 0) ? nranks : 1;
    RsProcessTileStripes(gemm_output, reduced_output, signal_base, hcclCtx, my_rank, safe_nranks, tile_idx,
                         subtileShape, subtileStride, pingTile, pongTile, pp_pending_dst, pp_pending_meta, pp_count,
                         num_comm_blocks);

    tiles_sent++;
    if (tiles_sent >= my_expected_tiles) {
        RsFlushPipeline(pingTile, pongTile, pp_pending_dst, pp_pending_meta, pp_count, hcclCtx, signal_base, my_rank);
        pp_count = 0;
    }
    return true;
}

// ============================================================================
// Phase 1: ReduceScatter — TPUT with AtomicAdd to owner's reduced_output
//
// Only block 0..(num_compute_blocks-1) participate.
// Blocks >= num_compute_blocks skip straight to the barrier.
// ============================================================================
AICORE inline void ReduceScatterPhase(__gm__ half *gemm_output, __gm__ half *reduced_output,
                                      __gm__ int32_t *signal_base, __gm__ MultiBlockQueueSet *queue_set,
                                      __gm__ HcclDeviceContext *hcclCtx, int my_rank, int nranks,
                                      int num_compute_blocks, int comm_core_idx, int num_comm_blocks)
{
    if (num_comm_blocks <= 0 || comm_core_idx >= num_comm_blocks)
        return;

    volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;

    ShapeDyn subtileShape(1, 1, 1, G_COMM_SUB_M, G_BASE_N);
    StrideDyn subtileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    RsSubtileData pingTile(G_COMM_SUB_M, G_BASE_N);
    RsSubtileData pongTile(G_COMM_SUB_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, RS_SUBTILE_UB_BYTES);
    int32_t heads[MAX_COMPUTE_BLOCKS];
    for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++)
        heads[b] = 0;

    int my_queue_indices[MAX_COMPUTE_BLOCKS];
    int32_t queue_max_tiles[MAX_COMPUTE_BLOCKS];
    int my_queue_count = 0;
    int my_expected_tiles = RsInitQueueState(comm_core_idx, num_compute_blocks, num_comm_blocks, my_queue_indices,
                                             my_queue_count, queue_max_tiles);

    int next_queue_offset = 0;
    int pp_count = 0;
    int32_t tiles_sent = 0;
    Global pp_pending_dst(gemm_output, subtileShape, subtileStride);
    RsPendingMeta pp_pending_meta{0, 0, 0};

    while (tiles_sent < my_expected_tiles) {
        int32_t tile_idx =
            RsPollQueues(qset, my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset);
        if (tile_idx < 0) {
            RsWaitOnQueue(qset, my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset);
            continue;
        }

        const int safe_nranks = (nranks > 0) ? nranks : 1;
        RsProcessTileStripes(gemm_output, reduced_output, signal_base, hcclCtx, my_rank, safe_nranks, tile_idx,
                             subtileShape, subtileStride, pingTile, pongTile, pp_pending_dst, pp_pending_meta, pp_count,
                             num_comm_blocks);
        tiles_sent++;
    }

    RsFlushPipeline(pingTile, pongTile, pp_pending_dst, pp_pending_meta, pp_count, hcclCtx, signal_base, my_rank);
    pipe_barrier(PIPE_ALL);
    // Ensure all owner-window AtomicAdd writes are globally visible before the
    // RS->AG handoff barrier. Without this, A5 can enter AG or the next host
    // iteration while late RS writes/doorbells are still in flight.
    dsb(DSB_DDR);
}

// Transfer a contiguous sub-tile of rows from local reduced_output to a remote rank.
AICORE inline void AgTransferRows(__gm__ half *reduced_output, __gm__ HcclDeviceContext *hcclCtx,
                                  const StrideDyn &tileStride, int r, uint64_t row_offset, int nrows)
{
    ShapeDyn subShape(1, 1, 1, nrows, G_BASE_N);
    Global srcG(reduced_output + row_offset, subShape, tileStride);

    AgSubtileData subTile(nrows, G_BASE_N);
    TASSIGN(subTile, AG_SUBTILE_UB_OFFSET);

    TLOAD(subTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);

    __gm__ half *dst_ptr = HcclRemotePtr(hcclCtx, reduced_output, r) + row_offset;
    Global dstG(dst_ptr, subShape, tileStride);
    TSTORE_IMPL<AgSubtileData, Global, pto::AtomicType::AtomicNone>(dstG, subTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
}

// ============================================================================
// Phase 2 helpers: map owner-local subtile ids back to the global reduced_output.
// A communication tile is logically split along M into fixed-height subtile rows.
// ============================================================================
AICORE inline int AgGetMyTileCount(int total_tiles, int my_rank, int nranks)
{
    const int safe_nranks = (nranks > 0) ? nranks : 1;
    if (safe_nranks == 1) {
        return total_tiles;
    }
    const int tiles_per_owner = (total_tiles + safe_nranks - 1) / safe_nranks;
    const int remainder = total_tiles % safe_nranks;
    return (my_rank < remainder || remainder == 0) ? tiles_per_owner : (total_tiles / safe_nranks);
}

AICORE inline bool AgDecodeLocalSubtile(int local_subtile_id, int my_rank, int nranks, int total_tiles,
                                        uint64_t &row_offset)
{
    const int owner_local_tile = local_subtile_id / static_cast<int>(G_COMM_SUBTILES_PER_TILE);
    const int stripe_id = local_subtile_id % static_cast<int>(G_COMM_SUBTILES_PER_TILE);
    const int global_tile = my_rank + owner_local_tile * nranks;
    if (global_tile >= total_tiles) {
        return false;
    }

    const uint32_t mi = global_tile / G_N_TILES;
    const uint32_t ni = global_tile % G_N_TILES;
    const uint64_t tile_base = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;
    row_offset = tile_base + (uint64_t)stripe_id * G_COMM_SUB_M * G_N;
    return true;
}

AICORE inline void AgTransferSubtileToAll(__gm__ half *reduced_output, __gm__ HcclDeviceContext *hcclCtx,
                                          const StrideDyn &tileStride, int my_rank, int nranks, int total_tiles,
                                          int local_subtile_id)
{
    uint64_t row_offset = 0;
    if (!AgDecodeLocalSubtile(local_subtile_id, my_rank, nranks, total_tiles, row_offset)) {
        return;
    }

    if (nranks <= 1) {
        return;
    }

    const int start_step = 1 + (local_subtile_id % (nranks - 1));
    for (int peer_offset = 0; peer_offset < nranks - 1; ++peer_offset) {
        const int logical_step = 1 + ((start_step - 1 + peer_offset) % (nranks - 1));
        const int r = (my_rank + logical_step) % nranks;
        AgTransferRows(reduced_output, hcclCtx, tileStride, r, row_offset, G_COMM_SUB_M);
    }
}

// ============================================================================
// Phase 2: AllGather — owner-local subtile decomposition
//
// Each owner-local tile is split into G_COMM_SUBTILES_PER_TILE fixed subtile
// rows. AIV blocks partition the local subtile id range evenly and each task
// broadcasts one subtile to all remote ranks.
// ============================================================================
AICORE inline void AllGatherPhase(__gm__ half *reduced_output, __gm__ HcclDeviceContext *hcclCtx, int my_rank,
                                  int nranks, int comm_core_idx, int num_comm_blocks)
{
    const int total_tiles = G_NUM_TILES;
    const int my_tile_count = AgGetMyTileCount(total_tiles, my_rank, nranks);
    const int total_local_subtiles = my_tile_count * static_cast<int>(G_COMM_SUBTILES_PER_TILE);

    if (total_local_subtiles <= 0 || nranks <= 1) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    const int subtiles_per_block = (total_local_subtiles + num_comm_blocks - 1) / num_comm_blocks;
    int subtile_start = comm_core_idx * subtiles_per_block;
    int subtile_end = (comm_core_idx + 1) * subtiles_per_block;
    if (subtile_end > total_local_subtiles) {
        subtile_end = total_local_subtiles;
    }

    for (int local_subtile_id = subtile_start; local_subtile_id < subtile_end; ++local_subtile_id) {
        AgTransferSubtileToAll(reduced_output, hcclCtx, tileStride, my_rank, nranks, total_tiles, local_subtile_id);
    }

    pipe_barrier(PIPE_ALL);
    // Flush all AG remote stores before kernel return so host-side reset/copy
    // cannot race with late window traffic from the previous iteration.
    dsb(DSB_DDR);
}

// ============================================================================
// Phase 2 helpers: static AG work assignment plus subtile-ready probing.
// ============================================================================
struct AgAssignedState {
    uint16_t ids[G_SIGNAL_MAX_LOCAL_SUBTILES];
    uint8_t done[G_SIGNAL_MAX_LOCAL_SUBTILES];
    int count;
    int completed;
    int probe_cursor;
    int summary_slot;
    int summary_ack_count;
};

AICORE inline void AgInitAssignedState(AgAssignedState &state, int total_local_subtiles, int comm_core_idx,
                                       int num_comm_blocks)
{
    state.count = 0;
    state.completed = 0;
    state.probe_cursor = 0;
    state.summary_slot = comm_core_idx;
    state.summary_ack_count = 0;
    // Start from (num_comm_blocks - 1 - comm_core_idx) to match the reversed
    // AgSummaryBlockForSubtile mapping. Stride stays num_comm_blocks.
    const int start_id = (num_comm_blocks > 0) ? (num_comm_blocks - 1 - comm_core_idx) : 0;
    for (int id = start_id; id < total_local_subtiles; id += num_comm_blocks) {
        if (id < 0)
            continue;
        state.ids[state.count] = static_cast<uint16_t>(id);
        state.done[state.count] = 0;
        state.count++;
    }
}

AICORE inline bool AgAllDone(const AgAssignedState &state)
{
    return state.completed >= state.count;
}

AICORE inline bool AgDrainReadyAssignedSubtiles(__gm__ half *reduced_output, __gm__ int32_t *signal_base,
                                                __gm__ HcclDeviceContext *hcclCtx, const StrideDyn &tileStride,
                                                int my_rank, int nranks, int total_tiles, AgAssignedState &state)
{
    if (AgAllDone(state) || nranks <= 1 || state.count <= 0) {
        return false;
    }

    bool acquired = false;
    bool drained_any = false;
    int last_idx = state.probe_cursor;

    for (int i = 0; i < state.count; ++i) {
        const int idx = (state.probe_cursor + i) % state.count;
        if (state.done[idx]) {
            continue;
        }

        const int local_subtile_id = static_cast<int>(state.ids[idx]);
        pto::comm::Signal sig(signal_base + G_SIGNAL_SUBTILE_READY_OFFSET + local_subtile_id);
        if (!pto::comm::TTEST(sig, nranks, pto::comm::WaitCmp::GE)) {
            continue;
        }

        if (!acquired) {
            // One acquire fence is enough to consume all currently-ready subtiles
            // published before this drain pass.
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
            acquired = true;
        }
        AgTransferSubtileToAll(reduced_output, hcclCtx, tileStride, my_rank, nranks, total_tiles, local_subtile_id);
        state.done[idx] = 1;
        state.completed++;
        state.summary_ack_count++;
        last_idx = idx;
        drained_any = true;
    }

    if (drained_any) {
        state.probe_cursor = (state.count > 0) ? ((last_idx + 1) % state.count) : 0;
    }
    return drained_any;
}

AICORE inline void AgWaitAssignedSummary(__gm__ int32_t *signal_base, int nranks, AgAssignedState &state)
{
    if (AgAllDone(state) || state.count <= 0) {
        return;
    }

    pto::comm::Signal sig(AgSummarySlotPtr(signal_base, state.summary_slot));
    pto::comm::TWAIT(sig, state.summary_ack_count + 1, pto::comm::WaitCmp::GE);
}

struct RsPipelineState {
    int32_t heads[MAX_COMPUTE_BLOCKS];
    int my_queue_indices[MAX_COMPUTE_BLOCKS];
    int32_t queue_max_tiles[MAX_COMPUTE_BLOCKS];
    int my_queue_count;
    int my_expected_tiles;
    int next_queue_offset;
    int pp_count;
    int32_t tiles_sent;
    Global pp_pending_dst;
    RsPendingMeta pp_pending_meta;

    AICORE inline RsPipelineState(__gm__ half *gemm_output, const ShapeDyn &subtileShape,
                                  const StrideDyn &subtileStride, int comm_core_idx, int num_compute_blocks,
                                  int num_comm_blocks)
        : my_queue_count(0),
          my_expected_tiles(0),
          next_queue_offset(0),
          pp_count(0),
          tiles_sent(0),
          pp_pending_dst(gemm_output, subtileShape, subtileStride),
          pp_pending_meta{0, 0, 0}
    {
        for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++) {
            heads[b] = 0;
        }
        my_expected_tiles = RsInitQueueState(comm_core_idx, num_compute_blocks, num_comm_blocks, my_queue_indices,
                                             my_queue_count, queue_max_tiles);
    }
};

AICORE inline bool GemmCommTryRs(__gm__ half *gemm_output, __gm__ half *reduced_output, __gm__ int32_t *signal_matrix,
                                 volatile __gm__ MultiBlockQueueSet *qset, __gm__ HcclDeviceContext *hcclCtx,
                                 int my_rank, int nranks, const ShapeDyn &subtileShape, const StrideDyn &subtileStride,
                                 RsSubtileData &pingTile, RsSubtileData &pongTile, RsPipelineState &rsState,
                                 int num_comm_blocks, bool &rs_done)
{
    bool did_work = RsTryProcessOneTile(
        gemm_output, reduced_output, signal_matrix, qset, hcclCtx, my_rank, nranks, subtileShape, subtileStride,
        pingTile, pongTile, rsState.pp_pending_dst, rsState.pp_pending_meta, rsState.my_queue_indices,
        rsState.my_queue_count, rsState.heads, rsState.queue_max_tiles, rsState.next_queue_offset, rsState.pp_count,
        rsState.tiles_sent, rsState.my_expected_tiles, num_comm_blocks);
    rs_done = (rsState.tiles_sent >= rsState.my_expected_tiles && rsState.pp_count == 0);
    return did_work;
}

AICORE inline bool GemmCommTryAg(__gm__ half *reduced_output, __gm__ int32_t *signal_matrix,
                                 __gm__ HcclDeviceContext *hcclCtx, const StrideDyn &subtileStride, int my_rank,
                                 int nranks, int total_tiles, AgAssignedState &agState, bool &ag_done)
{
    bool ag_progress = AgDrainReadyAssignedSubtiles(reduced_output, signal_matrix, hcclCtx, subtileStride, my_rank,
                                                    nranks, total_tiles, agState);
    ag_done = AgAllDone(agState) || nranks <= 1;
    return ag_progress;
}

AICORE inline void GemmCommWaitForWork(volatile __gm__ MultiBlockQueueSet *qset, __gm__ int32_t *signal_matrix,
                                       int nranks, RsPipelineState &rsState, bool rs_done, bool ag_done,
                                       AgAssignedState &agState)
{
    if (!rs_done) {
        RsWaitOnQueue(qset, rsState.my_queue_indices, rsState.my_queue_count, rsState.heads, rsState.queue_max_tiles,
                      rsState.next_queue_offset);
    } else if (!ag_done) {
        AgWaitAssignedSummary(signal_matrix, nranks, agState);
    }
}

// Barrier-separated AG validation phase. RS still fully completes before AG
// starts, and AG consumes only the per-subtile ready counters. This removes the
// summary-slot wakeup path so we can isolate whether the hang comes from the
// summary/probe logic or from the ready-counter mapping itself.
AICORE inline void AllGatherPhaseViaReadyWaits(__gm__ half *reduced_output, __gm__ int32_t *signal_base,
                                               __gm__ HcclDeviceContext *hcclCtx, int my_rank, int nranks,
                                               int comm_core_idx, int num_comm_blocks)
{
    const int total_tiles = G_NUM_TILES;
    const int my_tile_count = AgGetMyTileCount(total_tiles, my_rank, nranks);
    const int total_local_subtiles = my_tile_count * static_cast<int>(G_COMM_SUBTILES_PER_TILE);

    if (total_local_subtiles <= 0 || nranks <= 1) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);
    AgAssignedState agState;
    AgInitAssignedState(agState, total_local_subtiles, comm_core_idx, num_comm_blocks);
    // Diagnostic threshold: if waiting for a single publish still hangs, the
    // problem is the ready-slot mapping itself rather than missing cross-rank
    // fan-in to nranks.
    const int ready_wait_threshold = 1;

    for (int idx = 0; idx < agState.count; ++idx) {
        const int local_subtile_id = static_cast<int>(agState.ids[idx]);
        pto::comm::Signal ready_sig(signal_base + G_SIGNAL_SUBTILE_READY_OFFSET + local_subtile_id);
        pto::comm::TWAIT(ready_sig, ready_wait_threshold, pto::comm::WaitCmp::GE);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        AgTransferSubtileToAll(reduced_output, hcclCtx, tileStride, my_rank, nranks, total_tiles, local_subtile_id);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Mixed subtile pipeline:
//   RS produces owner-local subtile-ready counters
//   AG consumes ready subtile counters without a global RS->AG barrier
// ============================================================================
AICORE inline void GemmCommAllImpl(__gm__ half *gemm_output, __gm__ half *reduced_output, __gm__ int32_t *signal_matrix,
                                   __gm__ MultiBlockQueueSet *queue_set, __gm__ HcclDeviceContext *hcclCtx, int rank,
                                   int nranks, int num_compute_blocks, int comm_core_idx, int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;
    const int total_tiles = G_NUM_TILES;
    const int my_tile_count = AgGetMyTileCount(total_tiles, my_rank, nranks);
    const int total_local_subtiles = my_tile_count * static_cast<int>(G_COMM_SUBTILES_PER_TILE);

    volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;
    ShapeDyn subtileShape(1, 1, 1, G_COMM_SUB_M, G_BASE_N);
    StrideDyn subtileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    RsSubtileData pingTile(G_COMM_SUB_M, G_BASE_N);
    RsSubtileData pongTile(G_COMM_SUB_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, RS_SUBTILE_UB_BYTES);

    RsPipelineState rsState(gemm_output, subtileShape, subtileStride, comm_core_idx, num_compute_blocks,
                            num_comm_blocks);
    AgAssignedState agState;
    AgInitAssignedState(agState, total_local_subtiles, comm_core_idx, num_comm_blocks);

    bool rs_done = (rsState.tiles_sent >= rsState.my_expected_tiles);
    bool ag_done = AgAllDone(agState) || nranks <= 1;

    while (!rs_done || !ag_done) {
        bool did_work = false;

        if (!rs_done) {
            did_work =
                GemmCommTryRs(gemm_output, reduced_output, signal_matrix, qset, hcclCtx, my_rank, nranks, subtileShape,
                              subtileStride, pingTile, pongTile, rsState, num_comm_blocks, rs_done);
        }

        if (!ag_done) {
            did_work = did_work || GemmCommTryAg(reduced_output, signal_matrix, hcclCtx, subtileStride, my_rank, nranks,
                                                 total_tiles, agState, ag_done);
        }

        if (!did_work) {
            GemmCommWaitForWork(qset, signal_matrix, nranks, rsState, rs_done, ag_done, agState);
        }
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel entry point
// ============================================================================
__global__ AICORE void GemmCommAllKernel(__gm__ uint8_t *gemm_output, __gm__ uint8_t *reduced_output,
                                         __gm__ uint8_t *signal_matrix, __gm__ uint8_t *queue_set,
                                         __gm__ uint8_t *hcclCtx, int rank, int nranks, int num_compute_blocks,
                                         int num_comm_blocks)
{
    GemmCommAllImpl(reinterpret_cast<__gm__ half *>(gemm_output), reinterpret_cast<__gm__ half *>(reduced_output),
                    reinterpret_cast<__gm__ int32_t *>(signal_matrix),
                    reinterpret_cast<__gm__ MultiBlockQueueSet *>(queue_set),
                    reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx), rank, nranks, num_compute_blocks,
                    get_block_idx(), num_comm_blocks);
}

// ============================================================================
// Host-side kernel launcher
// ============================================================================
void launchGemmCommAll(uint8_t *gemm_output, uint8_t *reduced_output, uint8_t *signal_matrix, uint8_t *queue_set,
                       uint8_t *hcclCtx, int rank, int nranks, void *stream, int num_compute_blocks)
{
    GemmCommAllKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(gemm_output, reduced_output, signal_matrix, queue_set,
                                                           hcclCtx, rank, nranks, num_compute_blocks, COMM_BLOCK_NUM);
}
