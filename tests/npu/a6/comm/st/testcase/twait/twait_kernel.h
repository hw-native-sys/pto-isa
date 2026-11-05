/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in twait_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// TWAIT Basic: Rank 0 sends signal to rank 1, rank 1 waits (blocking)
bool RunTWaitBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Compare: Test different comparison operators (GE, LE, etc.)
bool RunTWaitCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t notifyValue);

// TWAIT Atomic: Wait for atomic counter to reach threshold
bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Matrix: Wait on 2D signal matrix
template <int Rows, int Cols>
bool RunTWaitMatrix(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT SubRegion: Wait on a sub-region of 2D signal matrix
template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TWAIT Multi-Phase: rank 0 updates signal in phases, rank 1 waits in phases
bool RunTWaitMultiPhase(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
