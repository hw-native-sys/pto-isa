/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in tbroadcast_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// Basic (small tile) tests
template <typename T, size_t count>
bool RunBroadCast(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

// Large shape chunked tests
bool RunBroadCastLargeShape_Int32_128x32_tile16(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
bool RunBroadCastLargeShape_Float_256x64_tile32(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
bool RunBroadCastLargeShape_Int32_512x32_tile64(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

// Ping-pong tests
bool RunBroadCastPingPong_Int32_128x32_tile16(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
bool RunBroadCastPingPong_Float_256x64_tile32(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
