/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TNOTIFY_HPP
#define PTO_COMM_TNOTIFY_HPP

#include <pto/common/pto_tile.hpp>
#include <pto/comm/comm_types.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
namespace comm {

using NotifyOp = ::pto::comm::NotifyOp;

template <typename GlobalSignalData>
void TNotify_Impl(typename GlobalSignalData::DType *dstSignaData, int32_t value, NotifyOp op)
{
    switch (op) {
        case NotifyOp::AtomicAdd:
            dstSignaData[0] = dstSignaData[0] + value;
            break;
        case NotifyOp::Set:
            dstSignaData[0] = value;
            break;
        default:
            break;
    }
}

template <typename GlobalSignalData>
PTO_INTERNAL void TNOTIFY_IMPL(GlobalSignalData &dstSignaData, int32_t value, NotifyOp op)
{
    TNotify_Impl<GlobalSignalData>(dstSignaData.data(), value, op);
}

} // namespace comm
} // namespace pto
#endif
