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
// Two-phase kernel: RS (AtomicAdd) → device barrier → AG
// RS uses TPUT<AtomicAdd> to accumulate directly at the owner's reduced_output,
// eliminating the separate Reduce phase and its barrier.
//
// Signal matrix layout in HCCL window (per rank):
//   [0 .. MAX_RANKS-1]   Phase 0 cross-rank counters (RS done)
//   [MAX_RANKS]           Phase 0 local broadcast flag
//
// Only block_idx==0 performs cross-rank TNOTIFY/TWAIT signaling.

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
//   [0 .. MAX_RANKS-1]              Phase 0 cross-rank counters (RS done)
//   [MAX_RANKS]                     Phase 0 local broadcast flag (block 0 -> all blocks)
//   [MAX_RANKS+1]                   Intra-rank block arrival counter (for RS completion sync)
static constexpr int SIGNAL_LOCAL_FLAG_OFFSET = MAX_RANKS;
static constexpr int SIGNAL_INTRA_RANK_COUNTER = MAX_RANKS + 1;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;

static constexpr size_t TILE_UB_BYTES = ((G_BASE_M * G_BASE_N * sizeof(half) + 1023) / 1024) * 1024;

// ============================================================================
// Device-side cross-rank barrier using TNOTIFY/TWAIT
//
// Block 0: performs cross-rank signaling then sets a local broadcast flag.
// Other blocks: wait on the local broadcast flag via TWAIT.
// ============================================================================
AICORE inline void DeviceBarrier(__gm__ HcclDeviceContext *hcclCtx, __gm__ int32_t *signal_base, int phase, int my_rank,
                                 int nranks, int block_idx, int num_comm_blocks, int32_t expected = 1)
{
    // Intra-rank barrier: all blocks must arrive before block 0 sends cross-rank TNOTIFY.
    // Non-zero blocks atomically increment the arrival counter;
    // block 0 waits until all (num_comm_blocks - 1) other blocks have arrived.
    // NOTE: The caller (ReduceScatterPhase) already ends with pipe_barrier(PIPE_ALL),
    //       so we don't need another one at entry.
    __gm__ int32_t *intra_counter = signal_base + SIGNAL_INTRA_RANK_COUNTER + phase;
    if (block_idx != 0) {
        pto::comm::Signal arrSig(intra_counter);
        pto::comm::TNOTIFY(arrSig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    } else {
        if (num_comm_blocks > 1) {
            pto::comm::Signal arrSig(intra_counter);
            pto::comm::TWAIT(arrSig, (int32_t)(num_comm_blocks - 1), pto::comm::WaitCmp::GE);
        }
    }
    pipe_barrier(PIPE_ALL);

    if (block_idx == 0) {
        __gm__ int32_t *phase_base = signal_base + phase * MAX_RANKS;

        for (int r = 0; r < nranks; r++) {
            if (r == my_rank)
                continue;
            __gm__ int32_t *remote_sig = HcclRemotePtr(hcclCtx, phase_base + my_rank, r);
            pto::comm::Signal sig(remote_sig);
            pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
        }

        for (int r = 0; r < nranks; r++) {
            if (r == my_rank)
                continue;
            pto::comm::Signal sig(phase_base + r);
            pto::comm::TWAIT(sig, expected, pto::comm::WaitCmp::GE);
        }

        __gm__ int32_t *local_flag = signal_base + SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal localSig(local_flag);
        pto::comm::TNOTIFY(localSig, expected, pto::comm::NotifyOp::Set);
    } else {
        __gm__ int32_t *local_flag = signal_base + SIGNAL_LOCAL_FLAG_OFFSET + phase;
        pto::comm::Signal localSig(local_flag);
        pto::comm::TWAIT(localSig, expected, pto::comm::WaitCmp::GE);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// RS helpers — broken out to reduce cyclomatic complexity
// ============================================================================

// Round-robin poll across assigned queues; returns tile index or -1.
AICORE inline int32_t RsPollQueues(volatile __gm__ MultiBlockQueueSet *qset, const int *my_queue_indices,
                                   int my_queue_count, int32_t *heads, const int32_t *queue_max_tiles,
                                   int &next_queue_offset)
{
    for (int i = 0; i < my_queue_count; i++) {
        int local_idx = (next_queue_offset + i) % my_queue_count;
        int32_t q = my_queue_indices[local_idx];

        if (heads[q] >= queue_max_tiles[q])
            continue;

        volatile __gm__ PerBlockQueue *pq = GetMyBlockQueue(qset, q);
        int32_t tile = PerBlockQueueTryDequeue(pq, heads[q]);

        if (tile >= 0) {
            heads[q]++;
            next_queue_offset = (local_idx + 1) % my_queue_count;
            return tile;
        }
    }
    return -1;
}

// Ping-pong pipeline: load current tile, optionally store previous tile.
AICORE inline void RsPipelineStep(TileData &pingTile, TileData &pongTile, Global &pp_pending_dst, Global &dstG,
                                  Global &srcG, int pp_count)
{
    bool use_ping = (pp_count % 2 == 0);
    TileData &curTile = use_ping ? pingTile : pongTile;
    event_t curEv = use_ping ? EVENT_ID0 : EVENT_ID1;

    if (pp_count == 0) {
        TLOAD(curTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
    } else {
        TileData &prevTile = use_ping ? pongTile : pingTile;
        event_t prevEv = use_ping ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEv);
        TSTORE_IMPL<TileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, prevTile);
        TLOAD(curTile, srcG);
        set_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
        set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
    }

    pp_pending_dst = dstG;
}

// Drain the last tile still in the pipeline after the RS loop.
AICORE inline void RsFlushPipeline(TileData &pingTile, TileData &pongTile, Global &pp_pending_dst, int pp_count)
{
    if (pp_count <= 0)
        return;

    bool last_was_ping = ((pp_count - 1) % 2 == 0);
    TileData &lastTile = last_was_ping ? pingTile : pongTile;
    event_t lastEv = last_was_ping ? EVENT_ID0 : EVENT_ID1;
    wait_flag(PIPE_MTE2, PIPE_MTE3, lastEv);
    TSTORE_IMPL<TileData, Global, pto::AtomicType::AtomicAdd>(pp_pending_dst, lastTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
    wait_flag(PIPE_MTE3, PIPE_MTE2, lastEv);
}

// Block on the first non-exhausted queue via TWAIT.
AICORE inline void RsWaitOnQueue(volatile __gm__ MultiBlockQueueSet *qset, const int *my_queue_indices,
                                 int my_queue_count, const int32_t *heads, const int32_t *queue_max_tiles,
                                 int next_queue_offset)
{
    for (int i = 0; i < my_queue_count; i++) {
        int local_idx = (next_queue_offset + i) % my_queue_count;
        int32_t q = my_queue_indices[local_idx];
        if (heads[q] < queue_max_tiles[q]) {
            volatile __gm__ PerBlockQueue *pq = GetMyBlockQueue(qset, q);
            pto::comm::Signal sig(const_cast<__gm__ int32_t *>(&pq->count));
            pto::comm::TWAIT(sig, heads[q] + 1, pto::comm::WaitCmp::GE);
            return;
        }
    }
}

// Build the per-block queue assignment and tile counts for this RS block.
// Returns total expected tiles; writes my_queue_count via out-param.
AICORE inline int RsInitQueueState(int block_idx, int num_compute_blocks, int *my_queue_indices, int &my_queue_count,
                                   int32_t *queue_max_tiles)
{
    my_queue_count = 0;
    if (num_compute_blocks <= 0) {
        return 0;
    }

    for (int q = 0; q < num_compute_blocks; q++) {
        if ((num_compute_blocks > 0) && (q % num_compute_blocks == block_idx)) {
            my_queue_indices[my_queue_count++] = q;
        }
    }

    const int total_tiles = G_NUM_TILES;
    const int tiles_per_cb =
        (num_compute_blocks > 0) ? (total_tiles + num_compute_blocks - 1) / num_compute_blocks : total_tiles;

    int my_expected_tiles = 0;
    for (int i = 0; i < my_queue_count; i++) {
        int q = my_queue_indices[i];
        int block_end_tile = (q + 1) * tiles_per_cb;
        if (block_end_tile > total_tiles)
            block_end_tile = total_tiles;
        int block_tiles = block_end_tile - q * tiles_per_cb;
        if (block_tiles < 0)
            block_tiles = 0;
        queue_max_tiles[q] = block_tiles;
        my_expected_tiles += block_tiles;
    }
    return my_expected_tiles;
}

// ============================================================================
// Phase 1: ReduceScatter — TPUT with AtomicAdd to owner's reduced_output
//
// Only block 0..(num_compute_blocks-1) participate.
// Blocks >= num_compute_blocks skip straight to the barrier.
// ============================================================================
AICORE inline void ReduceScatterPhase(__gm__ half *gemm_output, __gm__ half *reduced_output,
                                      __gm__ MultiBlockQueueSet *queue_set, __gm__ HcclDeviceContext *hcclCtx,
                                      int my_rank, int nranks, int num_compute_blocks, int block_idx)
{
    if (block_idx >= num_compute_blocks)
        return;

    volatile __gm__ MultiBlockQueueSet *qset = (volatile __gm__ MultiBlockQueueSet *)queue_set;

    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, TILE_UB_BYTES);

    int32_t heads[MAX_COMPUTE_BLOCKS];
    for (int b = 0; b < MAX_COMPUTE_BLOCKS; b++)
        heads[b] = 0;

    int my_queue_indices[MAX_COMPUTE_BLOCKS];
    int32_t queue_max_tiles[MAX_COMPUTE_BLOCKS];
    int my_queue_count = 0;
    int my_expected_tiles =
        RsInitQueueState(block_idx, num_compute_blocks, my_queue_indices, my_queue_count, queue_max_tiles);

    int next_queue_offset = 0;
    int pp_count = 0;
    int32_t tiles_sent = 0;
    Global pp_pending_dst(gemm_output, tileShape, tileStride);

    while (tiles_sent < my_expected_tiles) {
        int32_t tile_idx =
            RsPollQueues(qset, my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset);
        if (tile_idx < 0) {
            RsWaitOnQueue(qset, my_queue_indices, my_queue_count, heads, queue_max_tiles, next_queue_offset);
            continue;
        }

        uint32_t mi = tile_idx / G_N_TILES;
        uint32_t ni = tile_idx % G_N_TILES;
        uint64_t tile_offset = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

        Global srcG(gemm_output + tile_offset, tileShape, tileStride);

        int owner = tile_idx % nranks;
        __gm__ half *dst_ptr = (owner == my_rank) ? reduced_output + tile_offset :
                                                    HcclRemotePtr(hcclCtx, reduced_output, owner) + tile_offset;
        Global dstG(dst_ptr, tileShape, tileStride);

        RsPipelineStep(pingTile, pongTile, pp_pending_dst, dstG, srcG, pp_count);
        pp_count++;
        tiles_sent++;
    }

    RsFlushPipeline(pingTile, pongTile, pp_pending_dst, pp_count);
    pipe_barrier(PIPE_ALL);
}

// Transfer a contiguous sub-tile of rows from local reduced_output to a remote rank.
AICORE inline void AgTransferRows(__gm__ half *reduced_output, __gm__ HcclDeviceContext *hcclCtx,
                                  const StrideDyn &tileStride, int r, uint64_t row_offset, int nrows)
{
    ShapeDyn subShape(1, 1, 1, nrows, G_BASE_N);
    Global srcG(reduced_output + row_offset, subShape, tileStride);

    using SubTile = pto::Tile<pto::TileType::Vec, half, G_BASE_M, G_BASE_N, pto::BLayout::RowMajor, -1, -1>;
    SubTile subTile(nrows, G_BASE_N);
    TASSIGN(subTile, 0x0);

    TLOAD(subTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    __gm__ half *dst_ptr = HcclRemotePtr(hcclCtx, reduced_output, r) + row_offset;
    Global dstG(dst_ptr, subShape, tileStride);
    TSTORE_IMPL<SubTile, Global, pto::AtomicType::AtomicNone>(dstG, subTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// ============================================================================
// Phase 2: AllGather — row-level flattened decomposition
//
// Flatten all AG work into rows:
//   total_rows = my_tile_count * (nranks-1) * G_BASE_M
// Each AIV block handles an equal-sized contiguous slice of rows.
// Within each work row we recover (tile_owner_idx, remote_rank, row_in_tile)
// and issue sub-tile TLOAD/TSTORE covering only the assigned rows.
//
// This ensures every AIV transfers the exact same amount of data,
// eliminating the ±1 work-item imbalance of tile-level decomposition.
// ============================================================================
AICORE inline void AllGatherPhase(__gm__ half *reduced_output, __gm__ HcclDeviceContext *hcclCtx, int my_rank,
                                  int nranks, int block_idx, int num_comm_blocks)
{
    const int total_tiles = G_NUM_TILES;
    const int tiles_per_owner = (total_tiles + nranks - 1) / nranks;
    const int my_tile_count =
        (my_rank < total_tiles % nranks || total_tiles % nranks == 0) ? tiles_per_owner : (total_tiles / nranks);

    const int remotes = nranks - 1;
    constexpr int ROWS_PER_TILE = G_BASE_M;
    const int total_rows = my_tile_count * ROWS_PER_TILE * remotes;

    if (total_rows <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    StrideDyn tileStride(G_BASE_M * G_N, G_BASE_M * G_N, G_BASE_M * G_N, G_N, 1);

    const int rows_per_block = (total_rows + num_comm_blocks - 1) / num_comm_blocks;
    int row_start = block_idx * rows_per_block;
    int row_end = (block_idx + 1) * rows_per_block;
    if (row_end > total_rows)
        row_end = total_rows;

    int cur_row = row_start;
    while (cur_row < row_end) {
        int flat_transfer = cur_row / ROWS_PER_TILE;
        int row_in_tile = cur_row % ROWS_PER_TILE;
        int t = my_rank + (flat_transfer / remotes) * nranks;
        if (t >= total_tiles)
            break;

        int r = flat_transfer % remotes;
        if (r >= my_rank)
            r++;

        int nrows = ROWS_PER_TILE - row_in_tile;
        if (nrows > row_end - cur_row)
            nrows = row_end - cur_row;

        uint32_t mi = t / G_N_TILES;
        uint32_t ni = t % G_N_TILES;
        uint64_t tile_base = (uint64_t)(mi * G_BASE_M) * G_N + ni * G_BASE_N;

        AgTransferRows(reduced_output, hcclCtx, tileStride, r, tile_base + (uint64_t)row_in_tile * G_N, nrows);
        cur_row += nrows;
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Two-phase AllReduce kernel: RS (AtomicAdd) + AG in a single kernel launch
// ============================================================================
AICORE inline void GemmCommAllImpl(__gm__ half *gemm_output, __gm__ half *reduced_output, __gm__ int32_t *signal_matrix,
                                   __gm__ MultiBlockQueueSet *queue_set, __gm__ HcclDeviceContext *hcclCtx, int rank,
                                   int nranks, int num_compute_blocks, int block_idx, int num_comm_blocks)
{
    int my_rank = hcclCtx->rankId;

    ReduceScatterPhase(gemm_output, reduced_output, queue_set, hcclCtx, my_rank, nranks, num_compute_blocks, block_idx);

    DeviceBarrier(hcclCtx, signal_matrix, 0, my_rank, nranks, block_idx, num_comm_blocks);

    AllGatherPhase(reduced_output, hcclCtx, my_rank, nranks, block_idx, num_comm_blocks);
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
