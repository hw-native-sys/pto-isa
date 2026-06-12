/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef KHOP_SMOKE_LAUNCH_HPP
#define KHOP_SMOKE_LAUNCH_HPP

#include <cstdint>

// GridPipe routed K-hop unicast smoke kernel.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each cell loads
// its stamped input tile, pushes it KHOP_DIST hops EAST (if a target exists) and
// pops/stores a tile from its KHOP_DIST-hop EAST upstream (if one exists).  The
// hop distance is the compile-time KHOP_DIST constant baked into the kernel.
void launchKHopSmokeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *inBuf, uint8_t *outBuf, uint8_t *hcclCtx,
                           int gridRows, int gridCols, void *stream);

#endif // KHOP_SMOKE_LAUNCH_HPP
