/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_BATCH_COMMON_DETAIL_HPP
#define PTO_COMM_TPUT_ASYNC_BATCH_COMMON_DETAIL_HPP

#include <cstdint>
#include <type_traits>
#include "pto/common/debug.h"
#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"

namespace pto {
namespace comm {
namespace detail {

PTO_INTERNAL void DiscardPendingUrmaBatch(const AsyncSession& session)
{
    const uint32_t nextWqeSeq = session.rt.urma.nextWqeSeq - session.rt.urma.batchStagedWqeCount;
    session.rt.urma = {};
    session.rt.urma.nextWqeSeq = nextWqeSeq;
}

PTO_INTERNAL void DiscardPendingAsyncPutBatch(const AsyncSession& session)
{
    if (session.engine == DmaEngine::SDMA) {
        session.rt.sdma.batchStagedDataSqeCount = 0U;
        session.rt.sdma.batchStagedOperationCount = 0U;
        return;
    }
    if (session.engine == DmaEngine::URMA) {
        DiscardPendingUrmaBatch(session);
        return;
    }
    // Any other engine (RDMA) keeps no staged batch state, and session.rt holds
    // no alternative for it, so there is nothing to discard.
}

template <DmaEngine engine>
PTO_INTERNAL bool ValidateAsyncPutBatchSession(const AsyncSession& session)
{
    if (session.valid && session.engine == engine) {
        return true;
    }
    DiscardPendingAsyncPutBatch(session);
    PTO_ASSERT(false, "TPUT_ASYNC batch: session is invalid or its engine does not match the template engine.");
    return false;
}

template <typename GlobalData>
PTO_INTERNAL bool TPutAsyncBatchGetElemCount(GlobalData& data, uint64_t& count)
{
    count = 1U;
    for (uint32_t dim = 0U; dim < 5U; ++dim) {
        const int value = data.GetShape(static_cast<int>(dim));
        if (value < 0) {
            return false;
        }
        const uint64_t extent = static_cast<uint64_t>(value);
        if (extent != 0U && count > UINT64_MAX / extent) {
            return false;
        }
        count *= extent;
    }
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TPutAsyncBatchValidatePayload(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint64_t& totalBytes)
{
    using SrcElem = typename GlobalSrcData::RawDType;
    static_assert(
        std::is_same_v<SrcElem, typename GlobalDstData::RawDType>, "TPUT_ASYNC(DEFER): src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TPUT_ASYNC(DEFER): src/dst layout mismatch");

    uint64_t srcElems = 0U;
    if (!TPutAsyncBatchGetElemCount(srcGlobalData, srcElems) || srcElems > UINT64_MAX / sizeof(SrcElem)) {
        DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER): invalid source shape or byte-size overflow.");
        return false;
    }
    if (srcElems == 0U) {
        totalBytes = 0U;
        return true;
    }

    const bool pointersValid = srcGlobalData.data() != nullptr && dstGlobalData.data() != nullptr;
    if (!pointersValid) {
        DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER): src and dst tensor pointers must not be null.");
        return false;
    }
    const bool layoutsValid = TPutAsyncIsFlatContiguous1D(srcGlobalData) && TPutAsyncIsFlatContiguous1D(dstGlobalData);
    if (!layoutsValid) {
        DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER): src and dst tensors must be flat contiguous 1D.");
        return false;
    }

    uint64_t dstElems = 0U;
    const bool countsValid = TPutAsyncBatchGetElemCount(dstGlobalData, dstElems) && dstElems >= srcElems;
    if (!countsValid) {
        DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER): invalid shape, dst capacity, or byte-size overflow.");
        return false;
    }
    totalBytes = srcElems * sizeof(SrcElem);
    return true;
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_BATCH_COMMON_DETAIL_HPP
