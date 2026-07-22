/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
#define DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP

#include <cstdint>

// ReduceSum mixed Cube/Vec kernel for the single-device multi-block FFN path.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each block uses
// get_block_idx() as its row-major cell id.  Inputs/weights arrive from the
// GM-simulated Batcher (see batcher.hpp): xFull is per-row (broadcast across
// cols), the w*Shards regions are per-col (shared across rows), and yFull is the
// Batcher output region the EAST reduce writes per row.
//
// The kernel is compiled for dav-c220 mixed Cube/Vector.  Cube and Vec branches
// run concurrently and exchange gate/up/hidden/down intermediates through
// regular A2/A3 TPipe ready/free synchronization.  The final row-local EAST
// reduce still uses GridPipe windows.
void launchDistributedFfnGridMixedKernel(uint8_t *ffts, uint8_t *reducePipeWindow, uint8_t *xFull,
                                         uint8_t *wGateShards, uint8_t *wUpShards, uint8_t *wDownShards,
                                         uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
                                         uint8_t *downPartial, uint8_t *yFull, uint8_t *hcclCtx, int gridRows,
                                         int gridCols, void *stream);

// AllGather split variant.  The GridPipe window carries fp16 hidden shards
// [T, Fi] across columns, then each column computes and stores its [T, Hc]
// output shard directly into the Batcher yFull region (H-sharded).
void launchDistributedFfnGridAllGatherMixedKernel(uint8_t *ffts, uint8_t *gatherPipeWindow, uint8_t *xFull,
                                                  uint8_t *wGateShards, uint8_t *wUpShards, uint8_t *wDownShards,
                                                  uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
                                                  uint8_t *downPartial, uint8_t *yFull, uint8_t *hcclCtx, int gridRows,
                                                  int gridCols, void *stream);

#endif // DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
