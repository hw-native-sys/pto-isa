/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUT_HPP
#define TPUT_HPP

#include <pto/cpu/comm/TGet.hpp>
#include <pto/cpu/comm/TNotify.hpp>

namespace pto {
namespace comm {
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& src1)
{
    Copy_Data<GlobalDstData, GlobalSrcData, atomicType>(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& ping, TileData& pong)
{
    Copy_Data<GlobalDstData, GlobalSrcData, atomicType>(dst, src);
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_IMPL(GlobalDstData& dst, GlobalSrcData& src, const AsyncSession& session)
{
    if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        const uint64_t transferSize = AsyncTransferSize(dst, src);
        const urma::detail::UrmaMultiPostResult result = urma::__urma_put_async(
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
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC: URMA requires PTO_URMA_SUPPORTED.");
        return AsyncEvent(0, engine);
#endif
    }
    Copy_Data(dst, src);
    return AsyncEvent(0, engine);
}

template <DmaEngine engine, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent
TPUT_ASYNC_IMPL(GlobalDstData& dst, GlobalSrcData& src, const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        const uint64_t transferSize = AsyncTransferSize(dst, src);
        const urma::detail::UrmaMultiPostResult result = urma::__urma_put_async(
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
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC: URMA requires PTO_URMA_SUPPORTED.");
        return AsyncEvent(0, engine);
#endif
    }
    return TPUT_ASYNC_IMPL<engine>(dst, src, session);
}

template <DmaEngine engine, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_NOTIFY_IMPL(
    GlobalDstData& dst, GlobalSrcData& src, GlobalSignalData& dstSignalData, int32_t signalValue, NotifyOp notifyOp,
    const AsyncSession& session, uint32_t peer)
{
    if constexpr (engine == DmaEngine::URMA) {
#ifdef PTO_URMA_SUPPORTED
        const uint64_t transferSize = AsyncTransferSize(dst, src);
        const urma::detail::UrmaPostResult result = urma::__urma_put_async_notify(
            reinterpret_cast<__gm__ uint8_t*>(dst.data()), reinterpret_cast<__gm__ uint8_t*>(src.data()), transferSize,
            reinterpret_cast<__gm__ int32_t*>(dstSignalData.data()), signalValue, notifyOp, session, peer);
#ifdef __CPU_SIM
        uint32_t drainPeer = 0U;
        uint32_t targetBb = 0U;
        urma::detail::DecodeHandle(result.handle, drainPeer, targetBb);
        if (!urma::detail::CpuSimUrmaDrainPost(session, drainPeer, result.jetty, targetBb)) {
            return AsyncEvent::Failed(engine);
        }
#endif
        return urma::MakeUrmaNotifyEvent(result);
#else
        static_assert(engine != DmaEngine::URMA, "TPUT_ASYNC_NOTIFY: URMA requires PTO_URMA_SUPPORTED.");
        return AsyncEvent::Failed(engine);
#endif
    }
    (void)peer;
    AsyncEvent event = TPUT_ASYNC_IMPL<engine>(dst, src, session);
    TNOTIFY_IMPL(dstSignalData, signalValue, notifyOp);
    return event;
}

} // namespace comm
} // namespace pto
#endif
