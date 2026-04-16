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

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
#include "pto/comm/pto_comm_inst.hpp"
#endif

constexpr int MAX_RING_RANKS = 16;

// ============================================================================
// Streaming Pipeline Configuration
// ============================================================================
// Dynamic TILE_SIZE: auto-computed from num_blocks_per_src to keep
// tile count within [TARGET_TILES_MIN, TARGET_TILES_MAX].
constexpr int TARGET_TILES_MIN = 64;
constexpr int TARGET_TILES_MAX = 128;
constexpr int MIN_TILE_SIZE = 4;
constexpr int MAX_TILE_SIZE = 64;

inline int ComputeOptimalTileSize(int num_blocks_per_src)
{
    if (num_blocks_per_src <= 0)
        return MIN_TILE_SIZE;

    int tile_size = MIN_TILE_SIZE;
    int tile_count = (num_blocks_per_src + tile_size - 1) / tile_size;

    if (tile_count > TARGET_TILES_MAX) {
        tile_size = (num_blocks_per_src + TARGET_TILES_MAX - 1) / TARGET_TILES_MAX;
        tile_size = ((tile_size + 3) / 4) * 4;
        if (tile_size > MAX_TILE_SIZE)
            tile_size = MAX_TILE_SIZE;
    }

    return tile_size;
}

#ifndef STREAMING_TILE_SIZE
#define STREAMING_TILE_SIZE 4
#endif
constexpr int TILE_SIZE = STREAMING_TILE_SIZE;

// ============================================================================
// TileFlagMatrix: Tile-based flags for streaming pipeline
//
//   - Each tile contains TILE_SIZE blocks
//   - Communication sets tile flag after transfer completes
//   - Computation waits for tile flag before processing
//   - Non-competing: each (src_rank, tile_idx) has unique writer/reader
//
// Layout: [num_ranks][num_tiles_per_src] with 64-byte alignment per row
// ============================================================================
struct alignas(64) TileFlagMatrix {
    int32_t num_ranks;
    int32_t num_tiles_per_src;
    int32_t num_blocks_per_src;
    int32_t tile_size;
    int32_t stride;     // Aligned stride for each rank's tile flags
    int32_t my_rank;    // Local rank id
    int32_t epoch;      // Monotonically increasing generation counter (starts at 1)
    int32_t padding[9]; // Pad header to 64 bytes
    // Followed by int32_t tile_flags[num_ranks * stride]
};

inline size_t TileFlagMatrixSize(int num_ranks, int num_blocks_per_src, int tile_size)
{
    int actual_tile_size = (tile_size > 0) ? tile_size : ComputeOptimalTileSize(num_blocks_per_src);
    int num_tiles = (num_blocks_per_src + actual_tile_size - 1) / actual_tile_size;
    int stride = ((num_tiles + 15) / 16) * 16;
    return sizeof(TileFlagMatrix) + static_cast<size_t>(num_ranks) * stride * sizeof(int32_t);
}

// Summary region sits right after the flag matrix; one int32 per src rank.
inline size_t TileFlagMatrixSummaryOffset(int num_ranks, int num_blocks_per_src, int tile_size)
{
    return TileFlagMatrixSize(num_ranks, num_blocks_per_src, tile_size);
}
inline size_t TileFlagMatrixWithSummarySize(int num_ranks, int num_blocks_per_src, int tile_size)
{
    return TileFlagMatrixSummaryOffset(num_ranks, num_blocks_per_src, tile_size) +
           static_cast<size_t>(num_ranks) * sizeof(int32_t);
}

inline void TileFlagMatrixInit(TileFlagMatrix *flags, int num_ranks, int num_blocks_per_src, int tile_size)
{
    int actual_tile_size = (tile_size > 0) ? tile_size : ComputeOptimalTileSize(num_blocks_per_src);
    int num_tiles = (num_blocks_per_src + actual_tile_size - 1) / actual_tile_size;
    int stride = ((num_tiles + 15) / 16) * 16;

    flags->num_ranks = num_ranks;
    flags->num_tiles_per_src = num_tiles;
    flags->num_blocks_per_src = num_blocks_per_src;
    flags->tile_size = actual_tile_size;
    flags->stride = stride;
    flags->my_rank = -1;
    flags->epoch = 1;
    for (int i = 0; i < 9; i++)
        flags->padding[i] = 0;

    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(TileFlagMatrix));
    for (int i = 0; i < num_ranks * stride; ++i) {
        base[i] = 0;
    }
}

inline void TileFlagMatrixSummaryInit(int32_t *summary_base, int num_ranks)
{
    for (int i = 0; i < num_ranks; ++i)
        summary_base[i] = 0;
}

inline void TileFlagMatrixReset(TileFlagMatrix *flags)
{
    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(TileFlagMatrix));
    for (int i = 0; i < flags->num_ranks * flags->stride; ++i) {
        base[i] = 0;
    }
}

inline void TileFlagMatrixSetLocalReady(TileFlagMatrix *flags, int my_rank)
{
    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(TileFlagMatrix));
    int offset = my_rank * flags->stride;
    for (int c = 0; c < flags->num_tiles_per_src; ++c) {
        base[offset + c] = 1;
    }
}

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
// ============================================================================
// TileFlagMatrix device-side functions
// ============================================================================

AICORE inline size_t TileFlagMatrixBytes(volatile __gm__ TileFlagMatrix *flags)
{
    return sizeof(TileFlagMatrix) +
           static_cast<size_t>(flags->num_ranks) * static_cast<size_t>(flags->stride) * sizeof(int32_t);
}

AICORE inline volatile __gm__ int32_t *GetTileFlagPtr(volatile __gm__ TileFlagMatrix *flags, int32_t src_rank,
                                                      int32_t tile_idx)
{
    int32_t stride = flags->stride;
    int32_t idx = src_rank * stride + tile_idx;
    volatile __gm__ int32_t *base = reinterpret_cast<volatile __gm__ int32_t *>(
        reinterpret_cast<volatile __gm__ uint8_t *>(flags) + sizeof(TileFlagMatrix));
    return base + idx;
}

// Summary base address: right after flag matrix, one int32 doorbell per src rank.
AICORE inline volatile __gm__ int32_t *GetSummaryBase(volatile __gm__ TileFlagMatrix *flags)
{
    return reinterpret_cast<volatile __gm__ int32_t *>(reinterpret_cast<volatile __gm__ uint8_t *>(flags) +
                                                       TileFlagMatrixBytes(flags));
}

// Uses TNOTIFY AtomicAdd for hardware atomic semantics.
AICORE inline void SetTileFlagReady(volatile __gm__ TileFlagMatrix *flags, int32_t src_rank, int32_t tile_idx)
{
    volatile __gm__ int32_t *ptr = GetTileFlagPtr(flags, src_rank, tile_idx);
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ int32_t *>(ptr)));
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
}

// Notify remote rank that tile data is available.
// If remote_summary_src_ptr is provided, also increments the summary doorbell.
AICORE inline void SetRemoteTileFlagReady(__gm__ TileFlagMatrix *remote_flags, int32_t src_rank, int32_t tile_idx,
                                          __gm__ int32_t *remote_summary_src_ptr = nullptr)
{
    volatile __gm__ TileFlagMatrix *r = reinterpret_cast<volatile __gm__ TileFlagMatrix *>(remote_flags);
    if (src_rank < 0 || src_rank >= r->num_ranks || tile_idx < 0 || tile_idx >= r->num_tiles_per_src) {
        return;
    }
    volatile __gm__ int32_t *ptr = GetTileFlagPtr(r, src_rank, tile_idx);
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ int32_t *>(ptr)));
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    if (remote_summary_src_ptr != nullptr) {
        pto::comm::Signal sumSig(remote_summary_src_ptr);
        pto::comm::TNOTIFY(sumSig, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

AICORE inline void SetLocalSummaryReady(volatile __gm__ int32_t *summary_base, int32_t src_rank, int32_t value)
{
    if (summary_base == nullptr || src_rank < 0)
        return;
    volatile __gm__ int32_t *ptr = summary_base + src_rank;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    *ptr = value;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
}

AICORE inline bool IsTileReady(volatile __gm__ TileFlagMatrix *flags, int32_t src_rank, int32_t tile_idx)
{
    volatile __gm__ int32_t *ptr = GetTileFlagPtr(flags, src_rank, tile_idx);
    int32_t epoch = flags->epoch;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return (*ptr >= epoch);
}

// First-level poll: check if any tile from this src is ready.
AICORE inline bool IsAnyReadyFromSrc(volatile __gm__ int32_t *summary_base, int32_t src_rank)
{
    if (summary_base == nullptr || src_rank < 0)
        return false;
    volatile __gm__ int32_t *ptr = summary_base + src_rank;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return (*ptr >= 1);
}

AICORE inline int32_t GetReadyCountFromSrc(volatile __gm__ int32_t *summary_base, int32_t src_rank)
{
    if (summary_base == nullptr || src_rank < 0)
        return 0;
    volatile __gm__ int32_t *ptr = summary_base + src_rank;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return *ptr;
}

// Blocking wait until summary[src_rank] >= expected.
AICORE inline void WaitReadyCountFromSrc(volatile __gm__ int32_t *summary_base, int32_t src_rank, int32_t expected)
{
    if (summary_base == nullptr || src_rank < 0 || expected <= 0)
        return;
    __gm__ int32_t *ptr = const_cast<__gm__ int32_t *>(summary_base + src_rank);
    pto::comm::Signal sig(ptr);
    pto::comm::TWAIT(sig, expected, pto::comm::WaitCmp::GE);
}

#endif
