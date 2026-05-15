/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP
#define PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP

// PTO-native CKE trigger.
//
// The PTO CCU dispatch chain (TREDUCE<CollEngine::CCU>, TSCATTER<...>, ...)
// terminates here.  The implementation is a single scalar store into the
// CCU gate slot, which is mapped as device memory on this architecture so
// the write is globally visible without explicit cache maintenance.
//
// Self-contained: this header pulls in only <cstdint>.  No AscendC, no
// kernel_operator.h, no TPipe / DataCopy.  CCU test kernel translation
// units therefore do not need to drag the AscendC runtime into their
// compile graph, and HcclCcuKernelRegister still succeeds against the
// resulting .so (verified end-to-end on all four collective ST tests).

#include <cstdint>

namespace pto {
namespace comm {
namespace ccu {

// CKE gate slot encoding: the most-significant bit marks the entry as
// valid; the low bits carry the rank mask.  Producer (this) writes the
// bit set; CCU hardware clears it after the trigger fires.
static constexpr uint64_t kCkeValidBit = 1ULL << 63;

PTO_INTERNAL void CkeTrigger(uint64_t ckeSlotVA, uint32_t mask, __ubuf__ uint8_t * /* ubScratch */)
{
    auto *p = reinterpret_cast<__gm__ uint64_t *>(ckeSlotVA);
    *p = static_cast<uint64_t>(mask) | kCkeValidBit;
}

// Convenience wrapper used by the T{REDUCE,SCATTER,BROADCAST,GATHER}_CCU_IMPL
// fan-outs in a5/T*.hpp.  Resolves the UB scratch pointer from a tile then
// forwards to CkeTrigger; collapses two lines (reinterpret_cast + trigger)
// at every call site into one.
template <typename TileData>
PTO_INTERNAL void CkeTriggerFromTile(uint64_t ckeSlotVA, uint32_t mask, TileData &tile)
{
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(tile.data());
    CkeTrigger(ckeSlotVA, mask, ub);
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_CCU_TRIGGER_HPP
