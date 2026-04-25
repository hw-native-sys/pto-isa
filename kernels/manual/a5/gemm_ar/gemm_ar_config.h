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

#ifndef CONFIG_G_M
#define CONFIG_G_M 5416
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 6144
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 1408
#endif

static constexpr uint32_t G_ORIG_M = CONFIG_G_M;
static constexpr uint32_t G_ORIG_K = CONFIG_G_K;
static constexpr uint32_t G_ORIG_N = CONFIG_G_N;

#ifndef CONFIG_G_BASE_M
#define CONFIG_G_BASE_M 128
#endif
#ifndef CONFIG_G_BASE_K
#define CONFIG_G_BASE_K 64
#endif
#ifndef CONFIG_G_BASE_N
#define CONFIG_G_BASE_N 256
#endif

static constexpr uint32_t G_BASE_M = CONFIG_G_BASE_M;
static constexpr uint32_t G_BASE_K = CONFIG_G_BASE_K;
static constexpr uint32_t G_BASE_N = CONFIG_G_BASE_N;

static constexpr uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}
static constexpr uint32_t AlignUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

static constexpr uint32_t G_M = AlignUp(G_ORIG_M, G_BASE_M);
static constexpr uint32_t G_K = G_ORIG_K;
static constexpr uint32_t G_N = AlignUp(G_ORIG_N, G_BASE_N);
static constexpr uint32_t G_M_TILES = G_M / G_BASE_M;
static constexpr uint32_t G_N_TILES = G_N / G_BASE_N;
static constexpr uint32_t G_NUM_TILES = G_M_TILES * G_N_TILES;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 24
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 24
#endif
static constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
static constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;
static constexpr int MAX_RANKS = 8;

#ifndef CONFIG_COMM_SUB_M
#define CONFIG_COMM_SUB_M 64
#endif
static constexpr uint32_t G_COMM_SUB_M = CONFIG_COMM_SUB_M;
static_assert(G_COMM_SUB_M > 0, "CONFIG_COMM_SUB_M must be positive");
static_assert(G_BASE_M % G_COMM_SUB_M == 0, "CONFIG_COMM_SUB_M must divide G_BASE_M");
static constexpr uint32_t G_COMM_SUBTILES_PER_TILE = G_BASE_M / G_COMM_SUB_M;

// Per-rank signal_matrix layout:
//   [0 .. MAX_RANKS-1]                reserved legacy cross-rank barrier counters
//   [MAX_RANKS]                       reserved legacy local broadcast flag slot
//   [MAX_RANKS + 1]                   reserved legacy intra-rank arrival slot
//   [G_SIGNAL_SUBTILE_READY_OFFSET .. G_SIGNAL_AG_SUMMARY_OFFSET-1]
//                                     owner-local subtile-ready counters
//   [G_SIGNAL_AG_SUMMARY_OFFSET .. G_SIGNAL_TOTAL_SLOTS-1]
//                                     per-AG-block summary wakeup counters,
//                                     padded so each slot has its own 64B line
static constexpr uint32_t G_SIGNAL_RS_DONE_SLOTS = MAX_RANKS;
static constexpr uint32_t G_SIGNAL_LOCAL_FLAG_OFFSET = G_SIGNAL_RS_DONE_SLOTS;
static constexpr uint32_t G_SIGNAL_INTRA_RANK_COUNTER_OFFSET = G_SIGNAL_LOCAL_FLAG_OFFSET + 1;
static constexpr uint32_t G_SIGNAL_LEGACY_BARRIER_SLOTS = G_SIGNAL_INTRA_RANK_COUNTER_OFFSET + 1;
static constexpr uint32_t G_SIGNAL_SUBTILE_READY_OFFSET = G_SIGNAL_LEGACY_BARRIER_SLOTS;
static constexpr uint32_t G_SIGNAL_MAX_LOCAL_SUBTILES = G_NUM_TILES * G_COMM_SUBTILES_PER_TILE;
static constexpr uint32_t G_SIGNAL_AG_SUMMARY_OFFSET = G_SIGNAL_SUBTILE_READY_OFFSET + G_SIGNAL_MAX_LOCAL_SUBTILES;
static constexpr uint32_t G_SIGNAL_AG_SUMMARY_STRIDE = 16;
static_assert(G_SIGNAL_AG_SUMMARY_STRIDE * sizeof(int32_t) >= 64, "AG summary slot must be at least one cache line");
static constexpr uint32_t G_SIGNAL_AG_SUMMARY_SLOTS = COMM_BLOCK_NUM * G_SIGNAL_AG_SUMMARY_STRIDE;
static constexpr uint32_t G_SIGNAL_TOTAL_SLOTS = G_SIGNAL_AG_SUMMARY_OFFSET + G_SIGNAL_AG_SUMMARY_SLOTS;

static constexpr int WARMUP_ITERS = 5;
static constexpr int MEASURE_ITERS = 10;
static constexpr int COMPUTE_ONLY_ITERS = 5;

// Balanced tile distribution across compute/comm blocks.
// When total_tiles is not divisible by num_blocks, blocks get either floor(T/N)
// or ceil(T/N) tiles, with the remainder assigned to the first blocks.
#define GEMM_AR_BLOCK_TILE_COUNT(block_idx, total_tiles, num_blocks)                            \
    (((num_blocks) <= 0 || (block_idx) < 0 || (block_idx) >= (num_blocks)) ?                    \
         0 :                                                                                    \
         (((block_idx) < ((total_tiles) % (num_blocks))) ? ((total_tiles) / (num_blocks) + 1) : \
                                                           ((total_tiles) / (num_blocks))))

#define GEMM_AR_BLOCK_START_TILE(block_idx, total_tiles, num_blocks)                 \
    (((num_blocks) <= 0 || (block_idx) < 0 || (block_idx) >= (num_blocks)) ?         \
         0 :                                                                         \
         (((block_idx) < ((total_tiles) % (num_blocks))) ?                           \
              ((block_idx) * ((total_tiles) / (num_blocks) + 1)) :                   \
              (((total_tiles) % (num_blocks)) * ((total_tiles) / (num_blocks) + 1) + \
               ((block_idx) - ((total_tiles) % (num_blocks))) * ((total_tiles) / (num_blocks)))))
