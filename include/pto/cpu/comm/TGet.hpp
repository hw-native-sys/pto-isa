/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGET_HPP
#define TGET_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#include "pto/cpu/comm/urma_cpu_sim_nic.hpp"
#include "pto/cpu/comm/urma_async_event.hpp"
#endif

namespace pto {
namespace comm {

// Total payload element count for an async transfer (flat 5-D product).
template <typename GlobalData>
PTO_INTERNAL uint64_t AsyncElementCount(GlobalData& tensor)
{
    return static_cast<uint64_t>(tensor.GetShape(GlobalTensorDim::DIM_0)) *
           static_cast<uint64_t>(tensor.GetShape(GlobalTensorDim::DIM_1)) *
           static_cast<uint64_t>(tensor.GetShape(GlobalTensorDim::DIM_2)) *
           static_cast<uint64_t>(tensor.GetShape(GlobalTensorDim::DIM_3)) *
           static_cast<uint64_t>(tensor.GetShape(GlobalTensorDim::DIM_4));
}

// Total payload bytes for an async transfer, with the same src/dst validation
// the shared URMA path performs (TGetAsync.hpp / TPutAsyncCommonDetail.hpp):
// non-null pointers and dst at least as large as src. The shared path uses
// PTO_ASSERT, which compiles to a no-op in release, so use PTO_CPU_ASSERT here
// to actually catch misuse on CPU-SIM (there is no NIC to reject a bad
// descriptor, so an undersized dst would otherwise be written out of bounds).
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL uint64_t AsyncTransferSize(GlobalDstData& dst, GlobalSrcData& src)
{
    PTO_CPU_ASSERT(
        src.data() != nullptr && dst.data() != nullptr, "URMA async: src and dst tensor pointers must not be null.");
    PTO_CPU_ASSERT(AsyncElementCount(dst) >= AsyncElementCount(src), "URMA async: dst buffer too small for src data.");
    return AsyncElementCount(src) * sizeof(typename GlobalSrcData::DType);
}

template <typename GlobalDstData, typename GlobalSrcData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void Copy_Data(GlobalDstData& dstTensor, GlobalSrcData& srcTensor)
{
    typename GlobalDstData::DType* dst = dstTensor.data();
    typename GlobalSrcData::DType* src = srcTensor.data();
    int64_t shape[] = {
        dstTensor.GetShape(0), dstTensor.GetShape(1), dstTensor.GetShape(2), dstTensor.GetShape(3),
        dstTensor.GetShape(4)};
    int64_t dstStride[] = {
        dstTensor.GetStride(0), dstTensor.GetStride(1), dstTensor.GetStride(2), dstTensor.GetStride(3),
        dstTensor.GetStride(4)};
    int64_t srcStride[] = {
        srcTensor.GetStride(0), srcTensor.GetStride(1), srcTensor.GetStride(2), srcTensor.GetStride(3),
        srcTensor.GetStride(4)};

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                for (size_t l = 0; l < shape[3]; l++) {
                    for (size_t m = 0; m < shape[4]; m++) {
                        const int64_t dstIndex = i * dstStride[0] + j * dstStride[1] + k * dstStride[2] +
                                                 l * dstStride[3] + m * dstStride[4];
                        const int64_t srcIndex = i * srcStride[0] + j * srcStride[1] + k * srcStride[2] +
                                                 l * srcStride[3] + m * srcStride[4];
                        if constexpr (atomicType == AtomicType::AtomicNone) {
                            dst[dstIndex] = src[srcIndex];
                        } else {
                            dst[dstIndex] += src[srcIndex];
                        }
                    }
                }
            }
        }
    }
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& src1)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& ping, TileData& pong)
{
    Copy_Data(dst, src);
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_IMPL(GlobalDstData& dst, GlobalSrcData& src, const AsyncSession& session)
{
    if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        const uint64_t transferSize = AsyncTransferSize(dst, src);
        const urma::detail::UrmaMultiPostResult result = urma::__urma_get_async(
            reinterpret_cast<__gm__ uint8_t*>(dst.data()), reinterpret_cast<__gm__ uint8_t*>(src.data()), transferSize,
            session);
#ifdef __CPU_SIM
        uint32_t peer = 0U;
        uint32_t unusedBb = 0U;
        urma::detail::DecodeHandle(result.handle, peer, unusedBb);
        for (uint32_t j = 0U; j < result.jettyCount; ++j) {
            if (!urma::detail::CpuSimUrmaDrainPost(
                    session, peer, static_cast<uint32_t>(result.jettyBase) + j, result.targetBb[j])) {
                return AsyncEvent::Failed(engine);
            }
        }
#endif
        return urma::MakeUrmaMultiJettyEvent(result);
#else
        static_assert(engine != DmaEngine::URMA, "TGET_ASYNC: URMA requires PTO_URMA_SUPPORTED.");
        return AsyncEvent(0, engine);
#endif
    }
    Copy_Data(dst, src);
    return AsyncEvent(0, engine);
}

template <DmaEngine engine, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TGET_ASYNC_IMPL(GlobalDstData& dst, GlobalSrcData& src, const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        const uint64_t transferSize = AsyncTransferSize(dst, src);
        const urma::detail::UrmaMultiPostResult result = urma::__urma_get_async(
            reinterpret_cast<__gm__ uint8_t*>(dst.data()), reinterpret_cast<__gm__ uint8_t*>(src.data()), transferSize,
            session, peer);
#ifdef __CPU_SIM
        for (uint32_t j = 0U; j < result.jettyCount; ++j) {
            if (!urma::detail::CpuSimUrmaDrainPost(
                    session, peer, static_cast<uint32_t>(result.jettyBase) + j, result.targetBb[j])) {
                return AsyncEvent::Failed(engine);
            }
        }
#endif
        return urma::MakeUrmaMultiJettyEvent(result);
#else
        static_assert(engine != DmaEngine::URMA, "TGET_ASYNC: URMA requires PTO_URMA_SUPPORTED.");
        return AsyncEvent(0, engine);
#endif
    }
    return TGET_ASYNC_IMPL<engine>(dst, src, session);
}

} // namespace comm
} // namespace pto
#endif
