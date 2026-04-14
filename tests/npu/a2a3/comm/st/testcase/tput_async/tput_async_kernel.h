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

#include <cstddef>
#include <cstdint>

// 1D Vector Tile tests (root puts to all other ranks)
template <typename T, size_t count>
bool RunPutAsyncRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Configurable SdmaBaseConfig tests (custom block_bytes / comm_block_offset / queue_num)
template <typename T, size_t count>
bool RunPutAsyncWithConfig(int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint64_t blockBytes,
                           uint64_t commBlockOffset, uint32_t queueNum);

// Multi-core tests (blockDim > 1). multiCoreMode: 0 = split, 1 = independent
template <typename T, size_t count>
bool RunPutAsyncMultiCore(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int blockDim,
                          int multiCoreMode);
