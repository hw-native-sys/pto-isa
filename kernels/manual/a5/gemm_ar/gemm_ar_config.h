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

static constexpr int WARMUP_ITERS = 5;
static constexpr int MEASURE_ITERS = 10;
static constexpr int COMPUTE_ONLY_ITERS = 5;
