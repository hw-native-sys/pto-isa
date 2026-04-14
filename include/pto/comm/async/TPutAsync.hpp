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

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/npu/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {

// ============================================================================
// TPUT_ASYNC_IMPL: Asynchronous remote write operation implementation
//
// Directly transfers data from local GM to remote NPU's GM without UB staging.
// Returns AsyncEvent for synchronization with TSYNC.
//
// Data flow: srcGlobalData (local GM) -> DMA Engine -> dstGlobalData (remote GM)
// ============================================================================

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TPutAsyncIsFlatContiguous1D(GlobalData &globalData)
{
    const int dim0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int dim1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int dim2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int dim3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int dim4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int pitch0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int pitch1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int pitch2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int pitch3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int pitch4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool hasPackedLayout = (pitch4 == 1) && (pitch3 == dim4) && (pitch2 == dim3 * pitch3) &&
                                 (pitch1 == dim2 * pitch2) && (pitch0 == dim1 * pitch1);
    const bool isSingleLine = (dim0 == 1 && dim1 == 1 && dim2 == 1 && dim3 == 1);
    return hasPackedLayout && isSingleLine;
}

template <typename GlobalData>
PTO_INTERNAL uint32_t TPutAsyncGetTotalElemCount(GlobalData &globalData)
{
    const uint32_t d0 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint32_t d1 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint32_t d2 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint32_t d3 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint32_t d4 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    return (((d0 * d1) * d2) * d3) * d4;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TPutAsyncCheckTensorCompatibility()
{
    using SrcElem = typename GlobalSrcData::RawDType;
    static_assert(std::is_same_v<SrcElem, typename GlobalDstData::RawDType>,
                  "TPUT_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TPUT_ASYNC: src/dst layout mismatch");
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (srcGlobalData.data() == nullptr || dstGlobalData.data() == nullptr) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    if (!TPutAsyncIsFlatContiguous1D(srcGlobalData) || !TPutAsyncIsFlatContiguous1D(dstGlobalData)) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    if (dstElems < srcElems) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_put_async(dstGlobalData.data(), srcGlobalData.data(), srcElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

// ============================================================================
// TPUT_ASYNC_MTE_FALLBACK: Synchronous MTE fallback for platforms where SDMA
// does not support PUT direction (e.g. A5).
//
// Uses the session's UB scratch buffer (tmpBuf) as staging to perform a
// chunked GM → UB → GM transfer via MTE2/MTE3 pipelines. The operation
// completes synchronously; the returned AsyncEvent has handle=0 (already done).
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_MTE_FALLBACK(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                                const sdma::SdmaExecContext &execCtx)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (dstGlobalData.data() == nullptr || srcGlobalData.data() == nullptr) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    if (!TPutAsyncIsFlatContiguous1D(srcGlobalData) || !TPutAsyncIsFlatContiguous1D(dstGlobalData)) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    if (dstElems < srcElems) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    using T = typename GlobalSrcData::RawDType;
    const uint64_t totalBytes = static_cast<uint64_t>(srcElems) * sizeof(T);
    if (totalBytes == 0) {
        return AsyncEvent(0, DmaEngine::SDMA);
    }

    __ubuf__ uint8_t *ubBuf = execCtx.tmpBuf.addr;
    const uint32_t ubSize = execCtx.tmpBuf.size;
    PTO_ASSERT(ubBuf != nullptr && ubSize > 0, "TPUT_ASYNC MTE fallback: tmpBuf is invalid");

    __gm__ uint8_t *srcPtr = reinterpret_cast<__gm__ uint8_t *>(srcGlobalData.data());
    __gm__ uint8_t *dstPtr = reinterpret_cast<__gm__ uint8_t *>(dstGlobalData.data());

    uint64_t offset = 0;
    while (offset < totalBytes) {
        const uint64_t remaining = totalBytes - offset;
        const uint32_t chunkBytes = static_cast<uint32_t>((remaining < ubSize) ? remaining : ubSize);

        copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint8_t *>(ubBuf),
                                 reinterpret_cast<__gm__ uint8_t *>(srcPtr + offset), 0, 1, chunkBytes, 0, 0, false, 0,
                                 chunkBytes, chunkBytes);
        set_flag(PIPE_MTE2, PIPE_MTE3, execCtx.syncId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, execCtx.syncId);

        copy_ubuf_to_gm_align_v2(reinterpret_cast<__gm__ uint8_t *>(dstPtr + offset),
                                 reinterpret_cast<__ubuf__ uint8_t *>(ubBuf), 0, 1, chunkBytes, 0, chunkBytes,
                                 chunkBytes);
        set_flag(PIPE_MTE3, PIPE_MTE2, execCtx.syncId);
        wait_flag(PIPE_MTE3, PIPE_MTE2, execCtx.syncId);

        offset += chunkBytes;
    }

    return AsyncEvent(0, DmaEngine::SDMA);
}

#ifdef PTO_URMA_SUPPORTED
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_URMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const urma::UrmaExecContext &execCtx)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    if (!TPutAsyncIsFlatContiguous1D(srcGlobalData) || !TPutAsyncIsFlatContiguous1D(dstGlobalData)) {
        return AsyncEvent(0, DmaEngine::URMA);
    }

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC URMA: dst buffer too small for src data");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t transferSize = static_cast<uint64_t>(srcElems) * sizeof(T);
    PTO_ASSERT(transferSize <= UINT32_MAX, "TPUT_ASYNC URMA: transfer size exceeds SGE length limit (4GB)");

    const uint64_t eventHandle =
        urma::__urma_put_async(reinterpret_cast<__gm__ uint8_t *>(dstGlobalData.data()),
                               reinterpret_cast<__gm__ uint8_t *>(srcGlobalData.data()), transferSize, execCtx);
    return AsyncEvent(eventHandle, DmaEngine::URMA);
}
#endif

} // namespace detail

// ============================================================================
// Main TPUT_ASYNC_IMPL with DmaEngine template parameter
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                        const AsyncSession &session)
{
    if constexpr (engine == DmaEngine::SDMA) {
#ifdef PTO_NPU_ARCH_A5
        return detail::TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
#else
        return detail::TPUT_ASYNC_SDMA_IMPL(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
#endif
    } else if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        return detail::TPUT_ASYNC_URMA_IMPL(dstGlobalData, srcGlobalData, session.urmaSession.execCtx);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC: URMA engine requires NPU_ARCH 3510");
        return AsyncEvent(0, engine);
#endif
    } else {
        PTO_ASSERT(false, "TPUT_ASYNC: unsupported engine");
        return AsyncEvent(0, engine);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_HPP
