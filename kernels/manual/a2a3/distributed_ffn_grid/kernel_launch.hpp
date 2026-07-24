/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details.
You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
#define DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP

#include <cstdint>

// Four distributed-FFN GridPipe examples, one per interface-under-test.  Each call
// launches `blockCount` blocks of one `phase`; the kernel maps the wave-local
// block id to a global cell via (rowStart, colStart, waveCols):
//   row = rowStart + blk/waveCols, col = colStart + blk%waveCols.
//
// ReduceSum pattern (I split across all 32 cells, x broadcast, full-H down partial
// per cell, partials reduced EAST 8-way then SOUTH 4-way):
//   - TREDUCE  : the fused TREDUCE<Dir, Sum> collective.
//   - TPUSH    : the explicit TPOP<Dir> + TADD + TPUSH<Dir> lowering of TREDUCE.
//
// AllGather pattern (I split across all 32 cells, hidden AllGathered before down):
//   - TBROADCAST : the TBROADCAST<GridGroup> MPSC collective (every cell broadcasts).
//   - TPUSH      : a nearest-neighbor TPUSH/TPOP relay (fan-in-1 DAG) gather.

// --- ReduceSum pattern: TREDUCE variant (fused receive-combine-forward). ---
void launchDistributedFfnGridTreduceReduceSumMixedKernel(
    uint8_t *ffts, uint8_t *reduceWindow, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards, uint8_t *wDownShards,
    uint8_t *partialBuf, uint8_t *rowPartialBuf, uint8_t *yFull, uint8_t *gatePartialBuf, uint8_t *upPartialBuf,
    uint8_t *hiddenBuf, uint8_t *hcclCtx, int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
    int blockCount, void *stream);

// --- ReduceSum pattern: TPUSH variant (explicit TPOP + TADD + TPUSH). ---
void launchDistributedFfnGridTpushReduceSumMixedKernel(
    uint8_t *ffts, uint8_t *reduceWindow, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards, uint8_t *wDownShards,
    uint8_t *partialBuf, uint8_t *rowPartialBuf, uint8_t *yFull, uint8_t *gatePartialBuf, uint8_t *upPartialBuf,
    uint8_t *hiddenBuf, uint8_t *hcclCtx, int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
    int blockCount, void *stream);

// --- AllGather pattern: TBROADCAST variant (MPSC group broadcast).  Each ready/
//     free lane owns a full cache line (kBcastLaneStride), so the MPSC gather no
//     longer needs a cube keep-alive (doneFlags removed). ---
void launchDistributedFfnGridTbroadcastAllGatherMixedKernel(
    uint8_t *ffts, uint8_t *p1Window, uint8_t *p2Window, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards,
    uint8_t *wDownShards, uint8_t *hiddenShardBuf, uint8_t *rowBlockBuf, uint8_t *hiddenFullBuf,
    uint8_t *gatePartialBuf, uint8_t *upPartialBuf, uint8_t *yFull, uint8_t *p1Ctx, uint8_t *p2Ctx,
    int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
    int blockCount, void *stream);

// --- AllGather pattern: TPUSH variant (nearest-neighbor relay).  The relay is a
//     fan-in-1 DAG, so the vec-only gather waves need no cube keep-alive. ---
void launchDistributedFfnGridTpushAllGatherMixedKernel(
    uint8_t *ffts, uint8_t *p1Window, uint8_t *p2Window, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards,
    uint8_t *wDownShards, uint8_t *hiddenShardBuf, uint8_t *rowBlockBuf, uint8_t *hiddenFullBuf,
    uint8_t *gatePartialBuf, uint8_t *upPartialBuf, uint8_t *yFull, uint8_t *p1Ctx, uint8_t *p2Ctx,
    int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
    int blockCount, void *stream);

#endif // DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
