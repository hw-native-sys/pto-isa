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

#include <stddef.h>
#include <stdint.h>

constexpr size_t kMegaMoeProfileEntryBytes = 4096U;
constexpr size_t kMegaMoeProfileEntriesPerBlock = 3U; // cube + two vec subblocks
constexpr size_t kMegaMoeProfileBytesPerBlock = kMegaMoeProfileEntryBytes * kMegaMoeProfileEntriesPerBlock;

constexpr size_t kMegaMoeProfileKernelStart = 0U;
constexpr size_t kMegaMoeProfileKernelEnd = 1U;

struct MegaMoeLaunchArgs {
    void* ffts = nullptr;
    void* x = nullptr;
    void* weight1 = nullptr;
    void* weight2 = nullptr;
    void* expert_idx = nullptr;
    void* scale1 = nullptr;
    void* scale2 = nullptr;
    void* probs = nullptr;
    void* out = nullptr;
    void* expert_token_nums = nullptr;
    void* workspace = nullptr;
    void* tiling = nullptr;
    void* profile_data = nullptr;
    uint32_t block_dim = 1;
    uint32_t start_sync_debug = 0;
};

void launchMegaMoe(const MegaMoeLaunchArgs& args, void* stream);
