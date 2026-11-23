/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef FLASH_MLA_BF16_PROFILE_H_
#define FLASH_MLA_BF16_PROFILE_H_

#include "flash_mla_common.h"

namespace flash_mla_bf16 {
enum class StampStage : uint32_t { JOB = 0, Q_LOAD, DEQUANT, QK, SOFTMAX, PV, GU, OUTPUT };

// Stage windows encode logical tile ordinals modulo 50.
// Trace order disambiguates wraps and persistent jobs.
constexpr uint32_t STAMP_STAGE_STRIDE = 500;
constexpr uint32_t STAMP_TILE_STRIDE = 10;
constexpr uint32_t STAMP_TILE_PERIOD = 50;
constexpr uint32_t STAMP_PART_COUNT = 8;
#if defined(__CCE__)
AICORE
#endif
constexpr uint16_t stampId(StampStage stage, uint64_t ordinal, uint32_t part)
{
    return static_cast<uint16_t>(
        static_cast<uint32_t>(stage) * STAMP_STAGE_STRIDE + ordinal % STAMP_TILE_PERIOD * STAMP_TILE_STRIDE + part);
}
static_assert(
    STAMP_TILE_PERIOD * STAMP_TILE_STRIDE == STAMP_STAGE_STRIDE,
    "Each stamp stage must contain one complete wrapped tile window");
static_assert(STAMP_PART_COUNT <= STAMP_TILE_STRIDE, "Stamp parts must fit within one tile's ID range");
// Check the full expression before stampId narrows it to uint16_t.
static_assert(
    static_cast<uint64_t>(StampStage::OUTPUT) * STAMP_STAGE_STRIDE +
            uint64_t{STAMP_TILE_PERIOD - 1} * STAMP_TILE_STRIDE + STAMP_PART_COUNT - 1 <
        4096,
    "The largest profiling stamp must fit the 12-bit mark_stamp ID field");
// Dequant parts use the global interleaved slice IDs for both AIVs.
#if defined(__CCE__)
AICORE
#endif
constexpr uint32_t dequantStampPart(uint32_t slice) { return slice; }
static_assert(N / K_CHUNK <= STAMP_PART_COUNT, "Dequant slice IDs must fit the stamp part range");
static_assert(KV_REGIONS * (D / QK_K) <= STAMP_PART_COUNT, "QK region/depth IDs must fit the stamp part range");
static_assert((D / PV_MMAD_N) * (N / PV_K) <= STAMP_PART_COUNT, "PV column/token IDs must fit the stamp part range");

#if defined(__CCE__)
template <pipe_t Pipe, StampStage Stage>
AICORE inline void stampCompute(uint64_t ordinal, uint32_t part = 0)
{
    static_assert(Pipe == PIPE_M || Pipe == PIPE_V, "Compute stamps require the Cube or vector pipe");
    static_assert(
        Stage >= StampStage::DEQUANT && Stage <= StampStage::OUTPUT,
        "Compute stamps require a compute stage from DEQUANT through OUTPUT");
    if constexpr (FLASH_MLA_MARK_STAMP_COMPUTE_PIPE != 0) {
        // Record the source stage's endpoint on its compute pipe.
        bisheng::cce::mark_stamp<Pipe>(stampId(Stage, ordinal, part));
    }
}

template <pipe_t Pipe, StampStage Stage>
AICORE inline void stampData(uint64_t ordinal, uint32_t part = 0)
{
    static_assert(
        Pipe == PIPE_MTE1 || Pipe == PIPE_MTE2 || Pipe == PIPE_MTE3 || Pipe == PIPE_FIX,
        "Data stamps require an MTE or FIX pipe");
    static_assert(
        Stage >= StampStage::Q_LOAD && Stage <= StampStage::OUTPUT,
        "Data stamps require a transfer stage from Q_LOAD through OUTPUT");
    if constexpr (FLASH_MLA_MARK_STAMP_DATA_PIPE != 0) {
        // CANN's intrinsic inserts a same-pipe barrier for MTE1/2/3 on A5.
        // Producer/credit signals precede this endpoint.
        bisheng::cce::mark_stamp<Pipe>(stampId(Stage, ordinal, part));
    }
}

AICORE inline void stampJob(uint64_t job, uint32_t part)
{
    if constexpr (FLASH_MLA_MARK_STAMP_COMPUTE_PIPE != 0 || FLASH_MLA_MARK_STAMP_DATA_PIPE != 0) {
        // Record the job's scalar dispatch position.
        bisheng::cce::mark_stamp<PIPE_S>(stampId(StampStage::JOB, job, part));
    }
}
#endif
} // namespace flash_mla_bf16
#endif // FLASH_MLA_BF16_PROFILE_H_
