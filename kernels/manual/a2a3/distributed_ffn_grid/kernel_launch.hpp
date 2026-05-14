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

// Mixed Cube/Vec kernel for the single-device multi-block FFN path.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each block uses
// get_block_idx() as its row-major cell id.
//
// The kernel is compiled for dav-c220 mixed Cube/Vector.  Cube and Vec branches
// run concurrently and exchange gate/up/hidden/down intermediates through
// regular A2/A3 TPipe ready/free synchronization.  The final row-local EAST
// reduce still uses GridPipe windows.
void launchDistributedFfnGridMixedKernel(uint8_t *ffts, uint8_t *reducePipeWindow, uint8_t *x, uint8_t *wGate,
                                         uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial, uint8_t *upPartial,
                                         uint8_t *hiddenIn, uint8_t *downPartial, uint8_t *yOutput, uint8_t *hcclCtx,
                                         int gridRows, int gridCols, void *stream);

#endif // DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
