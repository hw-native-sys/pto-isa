/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef BCAST_SMOKE_LAUNCH_HPP
#define BCAST_SMOKE_LAUNCH_HPP

#include <cstdint>

// GridPipe single-source row/column broadcast smoke kernel.
//
// gridRows*gridCols blocks form a single-device logical grid.  The single source
// cell on each span (BCAST_SRC along the active axis) loads its stamped input
// tile and TPUSH<Span>-es (broadcast overload) it to every other cell on the span in one multicast;
// every other cell pops/stores the broadcast it received.
void launchBcastSmokeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *inBuf, uint8_t *outBuf, uint8_t *hcclCtx,
                            int gridRows, int gridCols, void *stream);

#endif // BCAST_SMOKE_LAUNCH_HPP
