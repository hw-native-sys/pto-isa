/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef GEMM_CONFIG_H_
#define GEMM_CONFIG_H_

#include <cstdint>

#ifndef CONFIG_G_M
#define CONFIG_G_M 2048
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 2048
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 1024
#endif

constexpr uint32_t G_M = CONFIG_G_M;
constexpr uint32_t G_K = CONFIG_G_K;
constexpr uint32_t G_N = CONFIG_G_N;

#ifndef CONFIG_G_BASE_M
#define CONFIG_G_BASE_M 128
#endif
#ifndef CONFIG_G_BASE_N
#define CONFIG_G_BASE_N 256
#endif

// Must match G_STEP_KA in allgather_gemm_compute_kernel.cpp (comm K-chunk = G_BASE_K * G_STEP_KA).
constexpr uint32_t G_STEP_KA_PACK = 4;
static_assert(CONFIG_G_BASE_N % G_STEP_KA_PACK == 0,
              "CONFIG_G_BASE_N must be divisible by G_STEP_KA_PACK (see allgather_gemm_compute_kernel G_STEP_KA)");

constexpr uint32_t G_BASE_M = CONFIG_G_BASE_M;
constexpr uint32_t G_BASE_N = CONFIG_G_BASE_N;
constexpr uint32_t G_BASE_K = CONFIG_G_BASE_N / G_STEP_KA_PACK;

static_assert(CONFIG_G_M % CONFIG_G_BASE_M == 0, "G_M must be divisible by CONFIG_G_BASE_M");
static_assert(CONFIG_G_K % CONFIG_G_BASE_N == 0, "G_K must be divisible by CONFIG_G_BASE_N");
static_assert(CONFIG_G_N % CONFIG_G_BASE_N == 0, "G_N must be divisible by CONFIG_G_BASE_N");

#endif // GEMM_CONFIG_H_
