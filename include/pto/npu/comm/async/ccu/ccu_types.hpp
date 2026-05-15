/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_TYPES_HPP
#define PTO_COMM_ASYNC_CCU_CCU_TYPES_HPP

// CCU backend types — POD structs shared between host and AIV code.
// Safe to include from both g++ and bisheng compilation units.

#include <cstdint>

namespace pto {
namespace comm {
namespace ccu {

// Gate descriptor: minimal ABI struct describing one CKE-based gate that a
// CCU stream is waiting on. Produced by the CCU kernel (GeneArgs → Publish)
// and consumed by the AIV trigger entry (pto::comm::ccu::CkeTrigger) or
// host trigger path.
struct CcuGateDescriptor {
    uint32_t dieId;    // Physical die index (matches driver udie_idx, typically 0 or 1)
    uint32_t ckeId;    // CKE entry index (0-based, resolved during Translate)
    uint32_t mask;     // 16-bit signal-event mask (low 16 bits used)
    uint64_t mmioAddr; // Per-CKE 8B slot VA from rtGetDevResAddress(dieId, ckeId)
};

static_assert(sizeof(CcuGateDescriptor) == 24, "CcuGateDescriptor ABI size changed: update consumers");

constexpr uint32_t CCU_GATE_MASK = 1u << 0;
constexpr uint32_t CCU_DONE_MASK = 1u << 0;

constexpr const char *CCU_GATE_ENV = "HCCL_PTO_GATE_REDUCE";

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_TYPES_HPP
