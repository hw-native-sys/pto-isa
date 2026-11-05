/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in ttest_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// TTEST True: Test returns true when condition is met
bool RunTTestTrue(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST False: Test returns false when condition is not met
bool RunTTestFalse(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST Compare: Test with different comparison operators
bool RunTTestCompare_GE(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue, int32_t cmpValue,
    bool expectedResult);
bool RunTTestCompare_GT(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue, int32_t cmpValue,
    bool expectedResult);
bool RunTTestCompare_LE(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue, int32_t cmpValue,
    bool expectedResult);
bool RunTTestCompare_LT(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue, int32_t cmpValue,
    bool expectedResult);

// TTEST Polling with Timeout
bool RunTTestPollingTimeout(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunTTestPollingTimeoutMiss(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST NE: Test not-equal comparison
bool RunTTestNE(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// TTEST SubRegion: Test sub-region signal matrix
template <int FullCols, int SubRows, int SubCols>
bool RunTTestSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
