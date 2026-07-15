/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_CONFIG_HPP
#define PTO_PERF_SIM_CONFIG_HPP

#include <cstdint>
#include <string>

namespace pto::perf_sim {

// ── Architecture constants ──
constexpr uint32_t VEC_CORES_PER_AIC = 2; // VecCores per AI Core (1AIC:2AIV)

// ── Costmodel backend selector ──

enum class CostModelBackend : uint8_t {
    LightweightFormula, // profiling-fitted formula (lightweight_costmodel)
    CceMock,            // CCE mock trace
};

struct PerfSimConfig {
    // Architecture
    std::string arch_name = "a2a3";

    // Costmodel backend
    CostModelBackend costmodel_backend = CostModelBackend::LightweightFormula;

    // Buffer sizes (bytes)
    uint32_t ub_size = 192 * 1024; // 192 KB
    uint32_t l1_size = 512 * 1024; // 512 KB
    uint32_t l0a_size = 64 * 1024;
    uint32_t l0b_size = 64 * 1024;
    uint32_t l0c_size = 128 * 1024;

    // L2 Cache
    bool has_l2 = false;
    uint32_t l2_size = 0;                 // 0 = no L2 model
    uint32_t l2_cache_line = 512;         // bytes (pypto default)
    uint32_t l2_hit_latency = 50;         // cycles (pypto CacheConfig default)
    uint32_t l2_miss_extra_latency = 150; // cycles (pypto CacheConfig default)
    uint32_t l2_read_bw = 256;            // bytes/cycle
    uint32_t l2_write_bw = 256;           // bytes/cycle

    // GM bandwidth
    uint32_t gm_read_bw = 64;  // bytes/cycle
    uint32_t gm_write_bw = 64; // bytes/cycle

    // Launch config (set by SetLaunchConfig)
    uint32_t block_dim = 1;
    uint32_t cross_core_channel_count = 16; // semaphore IDs for cross-core sync

    // Output
    std::string output_dir = "./perf_sim_output"; // JSON swimlane output directory
};

inline PerfSimConfig& GetConfig()
{
    static thread_local PerfSimConfig cfg;
    return cfg;
}

// Called by LAUNCH_KERNEL macro to parse <<<blk_dim, l2_ptr, stream>>>
inline void SetLaunchConfig(uint32_t blk_dim, void* l2_ptr, void* /*stream*/)
{
    auto& cfg = GetConfig();
    cfg.block_dim = blk_dim;
    cfg.has_l2 = (l2_ptr != nullptr);
}

} // namespace pto::perf_sim

#endif
