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

#include "pto/common/constants.hpp"
#include "pto/common/type.hpp"
#include "pto/common/debug.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/npu/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {

// ============================================================================
// TGET_ASYNC_IMPL: Asynchronous remote read operation implementation
//
// Directly transfers data from remote NPU's GM to local GM without UB staging.
// Returns AsyncEvent for synchronization with TSYNC.
//
// Data flow: srcGlobalData (remote GM) -> DMA Engine -> dstGlobalData (local GM)
// ============================================================================

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TGetAsyncIsFlatContiguous1D(GlobalData &globalData)
{
    const int shp0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int shp1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int shp2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int shp3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int shp4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int step0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int step1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int step2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int step3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int step4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool packedLayout = (step4 == 1) && (step3 == shp4) && (step2 == shp3 * step3) && (step1 == shp2 * step2) &&
                              (step0 == shp1 * step1);
    const bool oneDimLogical = (shp0 == 1 && shp1 == 1 && shp2 == 1 && shp3 == 1);
    return packedLayout && oneDimLogical;
}

template <typename GlobalData>
PTO_INTERNAL uint32_t TGetAsyncGetTotalElemCount(GlobalData &globalData)
{
    const uint32_t d0 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint32_t d1 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint32_t d2 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint32_t d3 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint32_t d4 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    return (((d0 * d1) * d2) * d3) * d4;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TGetAsyncCheckTensorCompatibility()
{
    using SrcElem = typename GlobalSrcData::RawDType;
    static_assert(std::is_same_v<SrcElem, typename GlobalDstData::RawDType>,
                  "TGET_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGET_ASYNC: src/dst layout mismatch");
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TGetAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (dstGlobalData.data() == nullptr || srcGlobalData.data() == nullptr) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    if (!TGetAsyncIsFlatContiguous1D(srcGlobalData) || !TGetAsyncIsFlatContiguous1D(dstGlobalData)) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint32_t srcElems = TGetAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TGetAsyncGetTotalElemCount(dstGlobalData);
    if (dstElems < srcElems) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_get_async(dstGlobalData.data(), srcGlobalData.data(), srcElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_URMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const urma::UrmaExecContext &execCtx)
{
    (void)TGetAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (!TGetAsyncIsFlatContiguous1D(srcGlobalData) || !TGetAsyncIsFlatContiguous1D(dstGlobalData)) {
        return AsyncEvent(0, DmaEngine::URMA);
    }

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
