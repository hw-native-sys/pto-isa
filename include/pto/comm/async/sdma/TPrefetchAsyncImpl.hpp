/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_TPREFETCH_ASYNC_IMPL_HPP
#define PTO_COMM_ASYNC_SDMA_TPREFETCH_ASYNC_IMPL_HPP

// TPREFETCH_ASYNC - L2 cache prefetch via SDMA CMO (opcode = 6).
//
// This instruction is logically a *memory access* instruction (it stages data
// from GM/HBM into the on-chip L2 cache so that subsequent TLOADs hit warm
// lines). It happens to *implement* itself by submitting an SDMA CMO SQE from
// the AI Core, which means it has to depend on the SDMA infrastructure that
// also backs TPUT_ASYNC / TGET_ASYNC. The implementation is arch-neutral (all
// A2A3/A5 differences live inside the SDMA backend headers), so it is defined
// once here next to that SDMA stack and included by the thin per-arch wrappers
// pto/npu/a2a3/TPrefetchAsync.hpp and pto/npu/a5/TPrefetchAsync.hpp, which is
// what users see at the API surface (a memory-access instruction in `pto::`).

#include "pto/common/type.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"

namespace pto {
namespace detail {

struct PrefetchAsyncContextBase {
    __gm__ uint8_t* workspace{nullptr};
    comm::AsyncSession session;
    comm::AsyncSession* externalSession{nullptr};

    AICORE PrefetchAsyncContextBase() = default;
    AICORE explicit PrefetchAsyncContextBase(__gm__ uint8_t* workspace_) : workspace(workspace_) {}
    AICORE PrefetchAsyncContextBase(__gm__ uint8_t* workspace_, comm::AsyncSession* externalSession_)
        : workspace(workspace_), externalSession(externalSession_)
    {}

    AICORE comm::AsyncSession& GetSession() { return externalSession != nullptr ? *externalSession : session; }
    AICORE const comm::AsyncSession& GetSession() const
    {
        return externalSession != nullptr ? *externalSession : session;
    }
};

} // namespace detail
} // namespace pto

#ifdef __PTO_AUTO__
// ---------------------------------------------------------------------------
// Auto mode stub: the CCE tile_size type system prevents extracting a raw
// __ubuf__ pointer from Tile::data(), and auto-mode scheduling makes manual
// prefetch unnecessary. Provide a no-op implementation so the public API
// compiles but does nothing.
// ---------------------------------------------------------------------------
namespace pto {

struct PrefetchAsyncContext : detail::PrefetchAsyncContextBase {
    using detail::PrefetchAsyncContextBase::PrefetchAsyncContextBase;
};

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_ASYNC_IMPL(GlobalData& /*src*/, PrefetchAsyncContext& /*ctx*/)
{
    return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
}

} // namespace pto

#else // !__PTO_AUTO__ -- full manual-mode implementation
// ---------------------------------------------------------------------------

#include "pto/comm/async/sdma/sdma_async_intrin.hpp"
#include "pto/comm/async/sdma/sdma_cmo_intrin.hpp"

namespace pto {

struct PrefetchAsyncContext : detail::PrefetchAsyncContextBase {
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;
    using detail::PrefetchAsyncContextBase::PrefetchAsyncContextBase;

    ScratchTile scratchTile;
};

namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TPrefetchAsyncIsFlatContiguous1D(GlobalData& globalData)
{
    const int dim0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int dim1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int dim2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int dim3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int dim4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int p0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int p1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int p2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int p3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int p4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool hasPackedLayout =
        (p4 == 1) && (p3 == dim4) && (p2 == dim3 * p3) && (p1 == dim2 * p2) && (p0 == dim1 * p1);
    const bool isSingleLine = (dim0 == 1 && dim1 == 1 && dim2 == 1 && dim3 == 1);
    return hasPackedLayout && isSingleLine;
}

template <typename GlobalData>
PTO_INTERNAL uint64_t TPrefetchAsyncGetTotalBytes(GlobalData& globalData)
{
    const uint64_t d0 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint64_t d1 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint64_t d2 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint64_t d3 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint64_t d4 = static_cast<uint64_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    using T = typename GlobalData::RawDType;
    return (((d0 * d1) * d2) * d3) * d4 * sizeof(T);
}

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPrefetchAsyncSdmaImpl(GlobalData& srcGlobalData, const comm::sdma::SdmaSession& session)
{
    if (srcGlobalData.data() == nullptr) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    if (!TPrefetchAsyncIsFlatContiguous1D(srcGlobalData)) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    const uint64_t totalBytes = TPrefetchAsyncGetTotalBytes(srcGlobalData);
    if (totalBytes == 0) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }

    return comm::sdma::__sdma_cmo_prefetch(srcGlobalData.data(), totalBytes, session);
}

template <typename Context>
PTO_INTERNAL bool InitPrefetchAsyncSession(Context& ctx, comm::AsyncSession& session)
{
    if (ctx.workspace == nullptr) {
        session.valid = false;
        return false;
    }

    constexpr uint32_t syncId = 0U;
    constexpr comm::sdma::SdmaBaseConfig baseConfig{comm::sdma::kDefaultSdmaBlockBytes, 0, 1};
    session.engine = comm::DmaEngine::SDMA;
    session.valid = comm::sdma::BuildSdmaSession(
        ctx.scratchTile, ctx.workspace, session.sdmaSession, syncId, baseConfig, comm::sdma::kAutoChannelGroupIdx);
    return session.valid;
}

} // namespace detail

template <typename GlobalData>
PTO_INTERNAL comm::AsyncEvent TPREFETCH_ASYNC_IMPL(GlobalData& srcGlobalData, PrefetchAsyncContext& ctx)
{
    comm::AsyncSession& session = ctx.GetSession();
    // Reuse an external session when supplied; otherwise build and cache the
    // context-owned session on the first call.
    if (!session.valid) {
        // Fully-qualified TASSIGN_IMPL avoids two-phase lookup: the public
        // TASSIGN wrapper in `pto/common/pto_instr.hpp` is not yet declared
        // when this template is defined, but the per-arch TASSIGN_IMPL is.
        TASSIGN_IMPL(ctx.scratchTile, 0x0);
        if (!detail::InitPrefetchAsyncSession(ctx, session)) {
            return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
        }
    }
    if (session.engine != comm::DmaEngine::SDMA) {
        return comm::AsyncEvent(0, comm::DmaEngine::SDMA);
    }
    return detail::TPrefetchAsyncSdmaImpl(srcGlobalData, session.sdmaSession);
}

} // namespace pto

#endif // __PTO_AUTO__

#endif // PTO_COMM_ASYNC_SDMA_TPREFETCH_ASYNC_IMPL_HPP
