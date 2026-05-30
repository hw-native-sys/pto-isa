/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_REPORTER_HPP
#define PTO_PERF_SIM_REPORTER_HPP

#include "pipe_model.hpp"
#include "tile_dep_tracker.hpp"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

namespace pto::perf_sim {

// ── Simulation report ──

struct SimReport {
    std::string op_name;
    uint32_t num_cores = 1;
    PipeTimeline timeline;            // single-core path
    MultiCoreTimeline multi_timeline; // multi-core path
    uint64_t instr_count = 0;
    uint64_t sync_count = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    double cache_hit_rate = 0.0;
};

#include "reporter_impl.inl"

} // namespace pto::perf_sim

#endif
