/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_ASYNC_HPP
#define PTO_COMM_TGET_ASYNC_HPP

#include "pto/comm/async_common/TGetAsyncCommonDetail.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/npu/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {
namespace detail {

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_URMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const urma::UrmaExecContext &execCtx)
{
    (void)TGetAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    PTO_ASSERT(TGetAsyncIsFlatContiguous1D(srcGlobalData),
               "TGET_ASYNC URMA: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by URMA async path.");
    PTO_ASSERT(TGetAsyncIsFlatContiguous1D(dstGlobalData),
               "TGET_ASYNC URMA: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by URMA async path.");

    const uint32_t srcElems = TGetAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TGetAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TGET_ASYNC URMA: dst buffer too small for src data");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t transferSize = static_cast<uint64_t>(srcElems) * sizeof(T);
    PTO_ASSERT(transferSize <= UINT32_MAX, "TGET_ASYNC URMA: transfer size exceeds SGE length limit (4GB)");

    const uint64_t eventHandle =
        urma::__urma_get_async(reinterpret_cast<__gm__ uint8_t *>(dstGlobalData.data()),
                               reinterpret_cast<__gm__ uint8_t *>(srcGlobalData.data()), transferSize, execCtx);
    return AsyncEvent(eventHandle, DmaEngine::URMA);
}
#endif

} // namespace detail

// ============================================================================
// Main TGET_ASYNC_IMPL with DmaEngine template parameter
// A5: SDMA and URMA engines are supported
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                        const AsyncSession &session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TGET_ASYNC_SDMA_IMPL(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TGET_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData, session.urmaSession.execCtx);
#else
        static_assert(engine != DmaEngine::URMA, "TGET_ASYNC: URMA engine requires NPU_ARCH 3510");
        return AsyncEvent(0, engine);
#endif
    } else {
        PTO_ASSERT(false, "TGET_ASYNC: unsupported engine");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_ASYNC_HPP
