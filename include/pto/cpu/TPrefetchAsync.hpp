/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TPREFETCH_ASYNC_HPP
#define PTO_CPU_TPREFETCH_ASYNC_HPP

// CPU-sim no-op stubs for TPREFETCH_ASYNC. The L2 cache concept does not exist
// in CPU simulation, so this keeps API surface compatibility while returning
// an empty AsyncEvent. Functional correctness on CPU is checked by simply
// running TLOAD afterwards (cache state is irrelevant).

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"

namespace pto {
namespace comm {

PTO_INTERNAL bool AsyncEvent::Wait(const AsyncSession& /*session*/) const { return true; }

PTO_INTERNAL bool AsyncEvent::Test(const AsyncSession& /*session*/) const { return true; }

} // namespace comm

struct PrefetchAsyncContext {
    __gm__ uint8_t* workspace{nullptr};
    comm::AsyncSession session;

    constexpr PrefetchAsyncContext() = default;
    constexpr explicit PrefetchAsyncContext(__gm__ uint8_t* workspace_) : workspace(workspace_) {}
};

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_ASYNC_IMPL(GlobalData& /*src*/, PrefetchAsyncContext& /*ctx*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

} // namespace pto

#endif // PTO_CPU_TPREFETCH_ASYNC_HPP
