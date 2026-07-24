/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FA_CUSTOM_KERNEL_H
#define FA_CUSTOM_KERNEL_H

#include <cstddef>
#include <cstdint>

// Shared defaults for FA performance kernels and host driver
constexpr int kFaCvFifoSize = 8;
constexpr int kFaCvFifoConsSyncPeriod = kFaCvFifoSize / 2;
constexpr int kFaCubeS0 = 128;
constexpr int kFaCubeS1 = 128;
constexpr int kFaTileS1 = 256;
constexpr int kFaQkPreload = 4;
constexpr std::size_t kFaProfileBytesPerBlock = 1024 * 3; // cube + two vec subblocks
constexpr std::size_t kFaCvCommSlotBytes = 512U;

#endif // FA_CUSTOM_KERNEL_H
