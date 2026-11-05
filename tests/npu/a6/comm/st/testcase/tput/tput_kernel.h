/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// G.EXP.05-CPP: Shared header for declarations implemented in tput_kernel.cpp
#pragma once

#include <cstddef>
#include <cstdint>

// 1D Vector Tile tests
template <typename T, size_t count>
bool RunPutRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Matrix Tile tests
template <typename T, size_t rows, size_t cols>
bool RunPutRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// AtomicAdd tests
template <typename T, size_t count>
bool RunPutAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Large Shape Chunked tests (GlobalTensor > UB tile, auto-chunked by TPUT_IMPL)
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunPutRingLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Multi-Dimensional Chunked tests (outer dims > 1, auto-chunked by TPUT_IMPL)
template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
bool RunPutRingMultiDim(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Irregular Shape Chunked tests (total_rows % tile_rows != 0, partial last chunk)
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunPutRingIrregularShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Sliding tests (both rows and cols exceed tile, auto 2D chunking)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunPutRing2DSliding(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Ping-Pong tests (double buffering: TPUT with two staging tiles)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunPutRingPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
