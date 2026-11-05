/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in tget_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// Declaration of 1D test functions
template <typename T, size_t count>
bool RunGetRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of 2D test functions
template <typename T, size_t rows, size_t cols>
bool RunGetRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of large shape chunked test functions
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of multi-dimensional chunked test functions
template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
bool RunGetRingMultiDim(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of irregular shape chunked test functions
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingIrregularShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Sliding tests (both rows and cols exceed tile, auto 2D chunking)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRing2DSliding(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Ping-Pong tests (double buffering: TGET with two staging tiles)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRingPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
