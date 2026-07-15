/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

#include "op_kernel/dispatch_mega_combine_tiling.h"
#include "runtime_context.hpp"

struct CaseConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t topk = 0;
    uint32_t expert_per_rank = 0;
    uint32_t world_size = 0;
    uint32_t max_output_size = 0;
    uint32_t aic_num = 0;
    uint32_t aiv_num = 0;
    double compare_atol = 1e-3;
    double compare_rtol = 1e-3;
    double input_tokens_all_ranks = 0.0;
    double routed_tokens_all_ranks = 0.0;
    double remote_routed_tokens_all_ranks = 0.0;
    double compute_flops_all_ranks = 0.0;
    double comm_bytes_all_ranks = 0.0;
};

struct MegaMoeBuildResult {
    MegaMoeTilingData tiling{};
    uint32_t block_dim = 1;
    uint64_t workspace_bytes = 0;
};

MegaMoeBuildResult BuildMegaMoeTiling(const CaseConfig& cfg, const StandaloneRankRuntime& runtime);
