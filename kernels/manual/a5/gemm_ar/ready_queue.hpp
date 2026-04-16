/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * Ready Queue - Per-block lock-free queue for tile scheduling
 *
 * Each compute block has its own queue (no contention on enqueue).
 * Comm kernel polls all queues via TTEST/TWAIT hardware instructions.
 * Single-producer, single-consumer design — no atomic operations needed.
 */

#pragma once

#include <cstdint>

#ifndef __CCE_KT_TEST__
#include "pto/comm/pto_comm_inst.hpp"
#endif

constexpr int MAX_COMPUTE_BLOCKS = 32;

struct alignas(64) PerBlockQueue {
    volatile int32_t tail;
    volatile int32_t count;
    int32_t capacity;
    int32_t block_id;
    int32_t padding[4];
    int32_t data[1];
};

struct alignas(64) MultiBlockQueueSet {
    int32_t num_blocks;
    int32_t total_tiles;
    int32_t tiles_per_block;
    int32_t consumed_count;
    int32_t padding[4];
    int32_t queue_offsets[MAX_COMPUTE_BLOCKS];
};

// ============================================================================
// Size calculations
// ============================================================================

constexpr size_t PerBlockQueueSize(int capacity)
{
    return sizeof(PerBlockQueue) - sizeof(int32_t) + capacity * sizeof(int32_t);
}

inline size_t MultiBlockQueueSetSize(int num_blocks, int tiles_per_block)
{
    size_t header_size = sizeof(MultiBlockQueueSet);
    size_t per_queue_size = PerBlockQueueSize(tiles_per_block);
    per_queue_size = ((per_queue_size + 63) / 64) * 64;
    return header_size + num_blocks * per_queue_size;
}

// ============================================================================
// Host-side initialization
// ============================================================================

inline void PerBlockQueueInit(PerBlockQueue *queue, int capacity, int block_id)
{
    queue->tail = 0;
    queue->count = 0;
    queue->capacity = capacity;
    queue->block_id = block_id;
    for (int i = 0; i < 4; i++)
        queue->padding[i] = 0;
    for (int i = 0; i < capacity; i++)
        queue->data[i] = -1;
}

inline void MultiBlockQueueSetInit(MultiBlockQueueSet *qset, int num_blocks, int total_tiles)
{
    qset->num_blocks = num_blocks;
    qset->total_tiles = total_tiles;
    qset->tiles_per_block = (num_blocks == 0) ? 0 : (total_tiles + num_blocks - 1) / num_blocks;
    qset->consumed_count = 0;
    for (int i = 0; i < 4; i++)
        qset->padding[i] = 0;

    size_t per_queue_size = PerBlockQueueSize(qset->tiles_per_block);
    per_queue_size = ((per_queue_size + 63) / 64) * 64;
    size_t base_offset = sizeof(MultiBlockQueueSet);

    for (int b = 0; b < num_blocks; b++) {
        qset->queue_offsets[b] = static_cast<int32_t>(base_offset + b * per_queue_size);
        PerBlockQueue *pq =
            reinterpret_cast<PerBlockQueue *>(reinterpret_cast<uint8_t *>(qset) + qset->queue_offsets[b]);
        PerBlockQueueInit(pq, qset->tiles_per_block, b);
    }
    for (int b = num_blocks; b < MAX_COMPUTE_BLOCKS; b++) {
        qset->queue_offsets[b] = 0;
    }
}

// ============================================================================
// Device-side functions (AICORE only — excluded from host unit-test builds)
// ============================================================================

#ifndef __CCE_KT_TEST__

AICORE inline volatile __gm__ PerBlockQueue *GetMyBlockQueue(volatile __gm__ MultiBlockQueueSet *qset, int queue_idx)
{
    dcci((__gm__ void *)&qset->queue_offsets[queue_idx], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    int32_t offset = qset->queue_offsets[queue_idx];
    return reinterpret_cast<volatile __gm__ PerBlockQueue *>(reinterpret_cast<volatile __gm__ uint8_t *>(qset) +
                                                             offset);
}

// Enqueue: caller tracks slot position, reducing 5 dcci → 2 dcci.
// Consumer only reads count (via TTEST) and data[head], never tail.
AICORE inline void PerBlockQueueEnqueueFast(volatile __gm__ PerBlockQueue *queue, int32_t tile_idx, int32_t local_slot)
{
    queue->data[local_slot] = tile_idx;
    dcci((__gm__ void *)&queue->data[local_slot], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");

    queue->tail = local_slot + 1;

    queue->count = local_slot + 1;
    dcci((__gm__ void *)&queue->count, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

AICORE inline void MultiBlockEnqueueFast(volatile __gm__ PerBlockQueue *cached_queue, int32_t tile_idx,
                                         int32_t local_slot)
{
    PerBlockQueueEnqueueFast(cached_queue, tile_idx, local_slot);
}

// Dequeue: uses TTEST hardware instruction for non-blocking poll.
AICORE inline int32_t PerBlockQueueTryDequeue(volatile __gm__ PerBlockQueue *queue, int32_t local_head)
{
    pto::comm::Signal sig(const_cast<__gm__ int32_t *>(&queue->count));
    if (!pto::comm::TTEST(sig, local_head + 1, pto::comm::WaitCmp::GE)) {
        return -1;
    }
    dcci((__gm__ void *)&queue->data[local_head], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    return queue->data[local_head];
}

#endif // __CCE_KT_TEST__
