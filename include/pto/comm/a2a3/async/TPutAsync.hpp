/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_HPP
#define PTO_COMM_TPUT_ASYNC_HPP

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"

namespace pto {
namespace comm {

// ============================================================================
// Main TPUT_ASYNC_IMPL with DmaEngine template parameter
// A2/A3: only SDMA engine is supported
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_IMPL(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_SDMA_IMPL(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
    } else {
        static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC: only SDMA engine is supported on A2/A3");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_HPP
