/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "pto/pto-inst.hpp"
#include "pto/common/pto_tile.hpp"

#include <pto/common/constants.hpp>
#include "pto/comm/comm_types.hpp"
using namespace pto::comm;

// ============================================================================
// TREDUCE Test Kernel
// Tests the TREDUCE collective - root gathers and reduces data from all ranks
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TReduceKernelImpl(__gm__ T *src0, __gm__ T *src1, __gm__ T *output, int nranks,
                                         pto::comm::ReduceOp op)
{
    constexpr size_t total_count = total_rows * cols;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = 0;

    // Full shape for the large GlobalTensor
    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    Global outputG(output, fullShape, fullStride);

    // Create ParallelGroup: each tensor points to that rank's input buffer
    Global tensors[2];
    int actual_nranks = 2;
    tensors[0] = Global(src0, fullShape, fullStride);
    tensors[1] = Global(src1, fullShape, fullStride);

    ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    // Allocate UB tiles — TREDUCE_IMPL will auto-chunk using these
    TileData accTile(tile_rows, cols);
    TileData recvTile(tile_rows, cols);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    // Only root executes TREDUCE
    if (my_rank == 0) {
        TREDUCE(pg, outputG, accTile, recvTile, op);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTReduce(T *src0, T *src1, T *dst, pto::comm::ReduceOp op, void *stream)
{
    TReduceKernelImpl<T, kGRows_, kGCols_, kTRows_>(src0, src1, dst, 2, op);
}

template void LaunchTReduce<int32_t, 64, 64, 64, 64>(int32_t *src0, int32_t *src1, int32_t *dst, pto::comm::ReduceOp op,
                                                     void *stream);
