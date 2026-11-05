/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in treduce_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// Basic (small tile) tests
bool RunReduceFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceFloat256SumWithRoot(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
bool RunReduceEmptyFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
bool RunReduceInt32_4096_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_512_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_256_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_256_Min(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Large shape chunked tests
bool RunReduceLargeShape_Int32_128x32_tile16_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceLargeShape_Float_256x64_tile32_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceLargeShape_Int32_128x32_tile16_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceLargeShape_Int32_512x32_tile64_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Ping-pong tests
bool RunReducePingPong_Int32_128x32_tile16_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReducePingPong_Float_256x64_tile32_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReducePingPong_Int32_128x32_tile16_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
