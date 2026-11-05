/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in tnotify_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// Test AtomicAdd mode: multiple ranks perform atomic add to same counter
bool RunNotifyAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Test Set mode: set remote signal value (ring pattern)
bool RunNotifySet(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Test Scoreboard mode: each rank notifies its slot in rank 0's scoreboard
template <size_t numSlots>
bool RunNotifyScoreboard(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Test runtime-specified NotifyOp (Set operation)
bool RunNotifyRuntimeOp(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
