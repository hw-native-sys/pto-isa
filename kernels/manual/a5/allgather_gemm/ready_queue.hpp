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

#include <stddef.h>
#include <stdint.h>

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
#include "pto/comm/pto_comm_inst.hpp"
#endif

constexpr int MAX_RING_RANKS = 16;

// ============================================================================
// Streaming Pipeline Configuration
// ============================================================================
// Dynamic CHUNK_SIZE: auto-computed from num_tiles_per_src to keep
// chunk count within [TARGET_CHUNKS_MIN, TARGET_CHUNKS_MAX].
constexpr int TARGET_CHUNKS_MIN = 64;
constexpr int TARGET_CHUNKS_MAX = 128;
constexpr int MIN_CHUNK_SIZE = 4;
constexpr int MAX_CHUNK_SIZE = 64;

inline int ComputeOptimalChunkSize(int num_tiles_per_src)
{
    if (num_tiles_per_src <= 0)
        return MIN_CHUNK_SIZE;

    int chunk_size = MIN_CHUNK_SIZE;
    int chunk_count = (num_tiles_per_src + chunk_size - 1) / chunk_size;

    if (chunk_count > TARGET_CHUNKS_MAX) {
        chunk_size = (num_tiles_per_src + TARGET_CHUNKS_MAX - 1) / TARGET_CHUNKS_MAX;
        chunk_size = ((chunk_size + 3) / 4) * 4;
        if (chunk_size > MAX_CHUNK_SIZE)
            chunk_size = MAX_CHUNK_SIZE;
    }

    return chunk_size;
}

#ifndef STREAMING_CHUNK_SIZE
#define STREAMING_CHUNK_SIZE 4
#endif
constexpr int CHUNK_SIZE = STREAMING_CHUNK_SIZE;

// ============================================================================
// ChunkFlagMatrix: Chunk-based flags for streaming pipeline
//
//   - Each chunk contains CHUNK_SIZE tiles
//   - Communication sets chunk flag after transfer completes
//   - Computation waits for chunk flag before processing
//   - Non-competing: each (src_rank, chunk_idx) has unique writer/reader
//
// Layout: [num_ranks][num_chunks_per_src] with 64-byte alignment per row
// ============================================================================
struct alignas(64) ChunkFlagMatrix {
    int32_t num_ranks;
    int32_t num_chunks_per_src;
    int32_t num_tiles_per_src;
    int32_t chunk_size;
    int32_t stride;     // Aligned stride for each rank's chunk flags
    int32_t my_rank;    // Local rank id
    int32_t epoch;      // Monotonically increasing generation counter (starts at 1)
    int32_t padding[9]; // Pad header to 64 bytes
    // Followed by int32_t chunk_flags[num_ranks * stride]
};

inline size_t ChunkFlagMatrixSize(int num_ranks, int num_tiles_per_src, int chunk_size)
{
    int actual_chunk_size = (chunk_size > 0) ? chunk_size : ComputeOptimalChunkSize(num_tiles_per_src);
    int num_chunks = (num_tiles_per_src + actual_chunk_size - 1) / actual_chunk_size;
    int stride = ((num_chunks + 15) / 16) * 16;
    return sizeof(ChunkFlagMatrix) + static_cast<size_t>(num_ranks) * stride * sizeof(int32_t);
}

// Summary region sits right after the flag matrix; one int32 per src rank.
inline size_t ChunkFlagMatrixSummaryOffset(int num_ranks, int num_tiles_per_src, int chunk_size)
{
    return ChunkFlagMatrixSize(num_ranks, num_tiles_per_src, chunk_size);
}
inline size_t ChunkFlagMatrixWithSummarySize(int num_ranks, int num_tiles_per_src, int chunk_size)
{
    return ChunkFlagMatrixSummaryOffset(num_ranks, num_tiles_per_src, chunk_size) +
           static_cast<size_t>(num_ranks) * sizeof(int32_t);
}

inline void ChunkFlagMatrixInit(ChunkFlagMatrix *flags, int num_ranks, int num_tiles_per_src, int chunk_size)
{
    int actual_chunk_size = (chunk_size > 0) ? chunk_size : ComputeOptimalChunkSize(num_tiles_per_src);
    int num_chunks = (num_tiles_per_src + actual_chunk_size - 1) / actual_chunk_size;
    int stride = ((num_chunks + 15) / 16) * 16;

    flags->num_ranks = num_ranks;
    flags->num_chunks_per_src = num_chunks;
    flags->num_tiles_per_src = num_tiles_per_src;
    flags->chunk_size = actual_chunk_size;
    flags->stride = stride;
    flags->my_rank = -1;
    flags->epoch = 1;
    for (int i = 0; i < 9; i++)
        flags->padding[i] = 0;

    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(ChunkFlagMatrix));
    for (int i = 0; i < num_ranks * stride; ++i) {
        base[i] = 0;
    }
}

inline void ChunkFlagMatrixSummaryInit(int32_t *summary_base, int num_ranks)
{
    for (int i = 0; i < num_ranks; ++i)
        summary_base[i] = 0;
}

inline void ChunkFlagMatrixReset(ChunkFlagMatrix *flags)
{
    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(ChunkFlagMatrix));
    for (int i = 0; i < flags->num_ranks * flags->stride; ++i) {
        base[i] = 0;
    }
}

inline void ChunkFlagMatrixSetLocalReady(ChunkFlagMatrix *flags, int my_rank)
{
    int32_t *base = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(flags) + sizeof(ChunkFlagMatrix));
    int offset = my_rank * flags->stride;
    for (int c = 0; c < flags->num_chunks_per_src; ++c) {
        base[offset + c] = 1;
    }
}

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
// ============================================================================
// ChunkFlagMatrix device-side functions
// ============================================================================

AICORE inline size_t ChunkFlagMatrixBytes(volatile __gm__ ChunkFlagMatrix *flags)
{
    return sizeof(ChunkFlagMatrix) +
           static_cast<size_t>(flags->num_ranks) * static_cast<size_t>(flags->stride) * sizeof(int32_t);
}

AICORE inline volatile __gm__ int32_t *GetChunkFlagPtr(volatile __gm__ ChunkFlagMatrix *flags, int32_t src_rank,
                                                       int32_t chunk_idx)
{
    int32_t stride = flags->stride;
    int32_t idx = src_rank * stride + chunk_idx;
    volatile __gm__ int32_t *base = reinterpret_cast<volatile __gm__ int32_t *>(
        reinterpret_cast<volatile __gm__ uint8_t *>(flags) + sizeof(ChunkFlagMatrix));
    return base + idx;
}

// Summary base address: right after flag matrix, one int32 doorbell per src rank.
AICORE inline volatile __gm__ int32_t *GetSummaryBase(volatile __gm__ ChunkFlagMatrix *flags)
{
    return reinterpret_cast<volatile __gm__ int32_t *>(reinterpret_cast<volatile __gm__ uint8_t *>(flags) +
                                                       ChunkFlagMatrixBytes(flags));
}

// Uses TNOTIFY AtomicAdd for hardware atomic semantics.
AICORE inline void SetChunkFlagReady(volatile __gm__ ChunkFlagMatrix *flags, int32_t src_rank, int32_t chunk_idx)
{
    volatile __gm__ int32_t *ptr = GetChunkFlagPtr(flags, src_rank, chunk_idx);
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ int32_t *>(ptr)));
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
}

// Notify remote rank that chunk data is available.
// If remote_summary_src_ptr is provided, also increments the summary doorbell.
AICORE inline void SetRemoteChunkFlagReady(__gm__ ChunkFlagMatrix *remote_flags, int32_t src_rank, int32_t chunk_idx,
                                           __gm__ int32_t *remote_summary_src_ptr = nullptr)
{
    volatile __gm__ ChunkFlagMatrix *r = reinterpret_cast<volatile __gm__ ChunkFlagMatrix *>(remote_flags);
    if (src_rank < 0 || src_rank >= r->num_ranks || chunk_idx < 0 || chunk_idx >= r->num_chunks_per_src) {
        return;
    }
    volatile __gm__ int32_t *ptr = GetChunkFlagPtr(r, src_rank, chunk_idx);
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

AICORE inline bool IsChunkReady(volatile __gm__ ChunkFlagMatrix *flags, int32_t src_rank, int32_t chunk_idx)
{
    volatile __gm__ int32_t *ptr = GetChunkFlagPtr(flags, src_rank, chunk_idx);
    int32_t epoch = flags->epoch;
    dcci((__gm__ void *)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return (*ptr >= epoch);
}

// First-level poll: check if any chunk from this src is ready.
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
