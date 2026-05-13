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

// Vec kernel for the single-device multi-block FFN path.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each block uses
// get_block_idx() as its row-major cell id.
//
// phase == 0:
//   hiddenOut[block] = fp16(PReLU(gatePartial[block]) * upPartial[block])
// phase == 1:
//   all columns run in one launch and synchronize through GridPipe flags:
//     col > 0           : TPOP<EAST> and add upstream partial
//     col + 1 < gridCols: TPUSH<EAST> to next column
//     final column      : write yOutput[row]
//
// reducePipeWindow points at gridRows*gridCols local GridPipe windows, one per
// cell, and hcclCtx is a fake same-device HcclDeviceContext whose windowsIn[]
// entries point into that contiguous allocation.
void launchDistributedFfnGridCommKernel(uint8_t *reducePipeWindow,
                                        uint8_t *gatePartial,
                                        uint8_t *upPartial,
                                        uint8_t *hiddenOut,
                                        uint8_t *downPartial,
                                        uint8_t *yOutput,
                                        uint8_t *hcclCtx,
                                        uint8_t *chunkFlags,
                                        int gridRows,
                                        int gridCols,
                                        int phase,
                                        void *stream);

// Cube kernel for the single-device multi-block FFN path.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each block uses
// get_block_idx() as its row-major cell id.
//
// phase == 0:
//   gatePartial[block] = x[block] @ wGate
//   upPartial[block]   = x[block] @ wUp
// phase == 1:
//   yOutput[block] = hiddenIn[block] @ wDown
//
// downPartial receives phase-1 per-cell partials. yOutput is written by the
// comm reduce phase, not by this kernel.
void launchDistributedFfnGridComputeKernel(uint8_t *x,
                                           uint8_t *wGate,
                                           uint8_t *wUp,
                                           uint8_t *wDown,
                                           uint8_t *gatePartial,
                                           uint8_t *upPartial,
                                           uint8_t *hiddenIn,
                                           uint8_t *downPartial,
                                           uint8_t *yOutput,
                                           uint8_t *chunkFlags,
                                           int gridRows,
                                           int gridCols,
                                           int myRankId,
                                           int phase,
                                           void *stream);

#endif  // DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
