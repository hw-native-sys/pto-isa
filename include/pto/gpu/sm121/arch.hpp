/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_SM121_ARCH_HPP
#define PTO_GPU_SM121_ARCH_HPP

namespace pto::gpu::sm121 {

constexpr int kComputeCapability = 121;
constexpr bool kPreferInlinePtx = true;
constexpr unsigned kDefaultWarpTileM = 16;
constexpr unsigned kDefaultWarpTileN = 16;
constexpr unsigned kDefaultWarpTileK = 16;

} // namespace pto::gpu::sm121

#endif
