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
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {
namespace detail {

// ============================================================================
// TPUT_ASYNC_MTE_FALLBACK: Synchronous MTE fallback for A5 platforms where
// SDMA does not support PUT direction.
//
// Uses the session's UB scratch buffer (tmpBuf) as staging to perform a
// chunked GM -> UB -> GM transfer via MTE2/MTE3 pipelines. The operation
// completes synchronously; the returned AsyncEvent has handle=0 (already done).
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_MTE_FALLBACK(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                                const sdma::SdmaExecContext &execCtx)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    PTO_ASSERT(dstGlobalData.data() != nullptr && srcGlobalData.data() != nullptr,
               "TPUT_ASYNC MTE fallback: src and dst tensor pointers must not be null.");

    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(srcGlobalData),
               "TPUT_ASYNC MTE fallback: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported.");
    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(dstGlobalData),
               "TPUT_ASYNC MTE fallback: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported.");

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC MTE fallback: dst buffer too small for src data.");

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

    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(srcGlobalData),
               "TPUT_ASYNC URMA: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by URMA async path.");
    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(dstGlobalData),
               "TPUT_ASYNC URMA: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by URMA async path.");

    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC URMA: dst buffer too small for src data");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t transferSize = static_cast<uint64_t>(srcElems) * sizeof(T);
    PTO_ASSERT(transferSize > 0 && transferSize <= urma::kUrmaMaxWqeTransferBytes,
               "TPUT_ASYNC URMA: transfer size must be in (0, 256MB] per single WQE");

    const uint64_t eventHandle =
        urma::__urma_put_async(reinterpret_cast<__gm__ uint8_t *>(dstGlobalData.data()),
                               reinterpret_cast<__gm__ uint8_t *>(srcGlobalData.data()), transferSize, execCtx);
    return AsyncEvent(eventHandle, DmaEngine::URMA);
}
#endif

} // namespace detail

// ============================================================================
// Main TPUT_ASYNC_IMPL with DmaEngine template parameter
// A5: SDMA uses MTE fallback for PUT direction; URMA is also supported
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                        const AsyncSession &session)
{
    if constexpr (engine == DmaEngine::SDMA) {
        return detail::TPUT_ASYNC_MTE_FALLBACK(dstGlobalData, srcGlobalData, session.sdmaSession.execCtx);
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
