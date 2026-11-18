/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DMA_ENGINE_HPP
#define PTO_COMM_DMA_ENGINE_HPP

#include <cstdint>

namespace pto {
namespace comm {

// ==========================================================================
// DmaEngine: DMA constraints for data transfer.
//
// Kept in its own header because comm_types.hpp decorates member functions with
// AICORE, which a host compiler cannot parse; a host-only facade can include
// this and still name the engine.
// ==========================================================================

enum class DmaEngine : uint8_t {
    SDMA = 0, // Supports 1D transfer
    URMA = 1, // Supports 1D transfer (HCCP V2 Jetty, NPU_ARCH 3510 only)
    RDMA = 2, // RDMA engine; RdmaBackend identifies the NIC implementation compiled into the binary
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_DMA_ENGINE_HPP
